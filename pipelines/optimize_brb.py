"""
基于现有 system_brb/module_brb 的轻量优化器，不依赖 brb_engine / brb_rules.yaml。
- 优化目标：模块层 7 条规则的权重（module_brb 原始权重被可调参数替换）。
- 系统层仍使用 system_brb.py 中的固定规则，权重不优化。
- 支持无监督（熵 + 置信度）和有监督（label_mod 监督）。

使用示例（使用 sim_spectrum 数据）:
    python optimize_brb.py --data_dir Output/sim_spectrum --maxiter 60
    python optimize_brb.py --data_dir Output/sim_spectrum --supervised --maxiter 80

使用示例（旧格式 CSV）:
    python optimize_brb.py --data feats.csv --label_col label_mod --supervised --maxiter 80
"""

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

CMA_SPEC = importlib.util.find_spec("cma")
if CMA_SPEC is not None:
    import cma
else:
    cma = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from sklearn.metrics import log_loss, accuracy_score

from BRB.system_brb import system_level_infer
from BRB.module_brb import (
    MODULE_LABELS, BOARD_MODULES,
    hierarchical_module_infer, set_hierarchical_params,
    _AMP_MODULES, _FREQ_MODULES, _REF_MODULES,
)

# V2 module label list (all modules from BOARD_MODULES)
V2_LABELS: List[str] = []
for board_modules in BOARD_MODULES.values():
    V2_LABELS.extend(board_modules)

# ------- 与 module_brb 一致的 labels 列表 -------
LABELS = list(MODULE_LABELS)

# Module label v2 mapping for labels.json format
_MODULE_V1_LOOKUP: Dict[str, str] = {}
try:
    from BRB.module_brb import MODULE_LABELS_V2, _MODULE_V1_TO_V2
    for v1, v2 in _MODULE_V1_TO_V2.items():
        _MODULE_V1_LOOKUP[v2] = v1
    for v1 in MODULE_LABELS:
        _MODULE_V1_LOOKUP[v1] = v1
except ImportError:
    pass

# Fault type → V2 module list (for label resolution)
_FAULT_V2_MODULES = {
    "amp_error": _AMP_MODULES,
    "freq_error": _FREQ_MODULES,
    "ref_error": _REF_MODULES,
}

# V2 module name → index in V2_LABELS
_V2_LABEL_INDEX = {m: i for i, m in enumerate(V2_LABELS)}


def _resolve_module_to_v2(label_entry: dict) -> Optional[str]:
    """Resolve module label from labels.json entry to V2 module name."""
    mod_v2 = label_entry.get("module_v2") or ""
    if mod_v2 in _V2_LABEL_INDEX:
        return mod_v2
    # Try V1→V2 mapping
    mod_v1 = label_entry.get("module") or ""
    if mod_v1 in _MODULE_V1_LOOKUP:
        v1_name = _MODULE_V1_LOOKUP[mod_v1]
        # Map v1→v2 
        try:
            from tools.label_mapping import module_v2_from_v1
            return module_v2_from_v1(v1_name)
        except (ImportError, KeyError):
            pass
    # Try partial matching
    for v2_name in V2_LABELS:
        if mod_v2 and (mod_v2 in v2_name or v2_name in mod_v2):
            return v2_name
    return None


def _normalize_feature(x, low, high):
    if x <= low:
        return 0.0
    if x >= high:
        return 1.0
    return (x - low) / (high - low)


def _aggregate_module_score(features):
    """Compute module-level anomaly score from features (matches module_brb logic)."""
    md_step_raw = max(
        features.get("step_score", 0.0),
        features.get("switch_step_err_max", 0.0),
        features.get("nonswitch_step_max", 0.0),
        features.get("X7", 0.0),
    )
    md_step = _normalize_feature(md_step_raw, 0.2, 1.5)
    md_slope = _normalize_feature(abs(features.get("res_slope", 0.0)), 1e-12, 1e-10)
    md_ripple = _normalize_feature(features.get("ripple_var", features.get("X6", 0.0)), 0.001, 0.02)
    md_df = _normalize_feature(abs(features.get("df", 0.0)), 1e6, 5e7)
    md_viol = _normalize_feature(features.get("viol_rate", features.get("X11", 0.0)), 0.02, 0.2)
    md_gain_bias = max(
        _normalize_feature(abs(features.get("bias", 0.0)), 0.1, 1.0),
        _normalize_feature(abs(features.get("gain", 1.0) - 1.0), 0.02, 0.2),
    )
    return float(np.mean([md_step, md_slope, md_ripple, md_df, md_viol, md_gain_bias]))


# ------- Hierarchical inference wrapper for optimization --------
def _infer_hierarchical(features: Dict, fault_type: str, params: np.ndarray) -> Dict[str, float]:
    """Call hierarchical_module_infer with given params set globally."""
    set_hierarchical_params(list(np.clip(params, 0.01, 10.0)))
    return hierarchical_module_infer(fault_type, features, use_board_prior=True)


def _fault_type_from_features(features: Dict) -> str:
    """Determine fault type from system-level BRB inference."""
    sys_p = system_level_infer(features)
    probs = sys_p.get("probabilities", sys_p)
    fault_map = {
        "幅度失准": "amp_error",
        "频率失准": "freq_error",
        "参考电平失准": "ref_error",
        "正常": "normal",
    }
    best_cn = max(probs, key=probs.get)
    return fault_map.get(best_cn, "normal")


# ------- 目标函数 -------
def supervised_objective(params, feats_rows, label_v2_names, fault_types):
    """Supervised objective using hierarchical_module_infer with V2 labels.
    
    Uses (1 - accuracy) + regularization to prevent overfitting.
    The regularization penalizes large deviations from default params (1.0).
    """
    clipped = np.clip(params, 0.01, 10.0)
    set_hierarchical_params(list(clipped))
    
    correct = 0
    total = 0
    for row, true_v2, ft in zip(feats_rows, label_v2_names, fault_types):
        f = dict(row)
        mod_probs = hierarchical_module_infer(ft, f, use_board_prior=True)
        if mod_probs:
            pred_v2 = max(mod_probs, key=mod_probs.get)
            if pred_v2 == true_v2:
                correct += 1
        total += 1
    
    acc = correct / max(total, 1)
    # Regularization: penalize large deviations from 1.0 (default)
    reg = 0.01 * float(np.sum((clipped - 1.0) ** 2))
    return (1.0 - acc) + reg


def unsupervised_objective(params, feats_rows, fault_types, w_entropy=0.6, w_conf=0.4):
    """Unsupervised objective using hierarchical_module_infer."""
    set_hierarchical_params(list(np.clip(params, 0.01, 10.0)))
    
    all_probs = []
    for row, ft in zip(feats_rows, fault_types):
        f = dict(row)
        mod_probs = hierarchical_module_infer(ft, f, use_board_prior=True)
        vals = list(mod_probs.values())
        all_probs.append(vals)
    
    probs = np.array(all_probs) + 1e-12
    ent = -np.sum(probs * np.log(probs), axis=1)
    mean_ent = float(np.nanmean(ent))
    mean_top1 = float(np.nanmean(np.max(probs, axis=1)))
    return w_entropy * mean_ent + w_conf * (1.0 - mean_top1)


# ------- CMA-ES 优化主流程 -------
def optimize(feats_rows, label_v2_names=None, fault_types=None,
             maxiter=80, popsize=None, seed=42, sigma0=0.3):
    # 16 params: [amp_priors(6), freq_priors(4), ref_priors(3), feat_sens(3)]
    x0 = np.ones(16, dtype=float)  # All scales start at 1.0 (unchanged priors)
    opts = {"seed": seed, "verbose": 1, "maxiter": maxiter}
    if popsize:
        opts["popsize"] = popsize
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    best_x, best_obj = None, float("inf")
    history = []
    it = 0
    while not es.stop():
        sols = es.ask()
        objs = []
        for x in sols:
            w = np.clip(x, 0.01, 10.0)
            try:
                if label_v2_names is not None:
                    obj = supervised_objective(w, feats_rows, label_v2_names, fault_types)
                else:
                    obj = unsupervised_objective(w, feats_rows, fault_types)
            except Exception:
                obj = float("inf")
            objs.append(obj)
            if obj < best_obj:
                best_obj, best_x = obj, w.copy()
        es.tell(sols, objs)
        es.disp()
        history.append({"generation": it, "best_obj": best_obj,
                         "mean_obj": float(np.mean(objs))})
        it += 1

    if best_x is None:
        best_x = np.clip(es.result.xbest, 0.01, 10.0)
        best_obj = float(es.result.fbest)

    print(f"\n[INFO] best_obj={best_obj:.6f}, best_params={best_x}")
    return best_x, best_obj, history


def _load_csv_rows(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, object]] = []
        for row in reader:
            parsed: Dict[str, object] = {}
            for key, val in row.items():
                if val is None:
                    parsed[key] = val
                    continue
                try:
                    parsed[key] = float(val)
                except (ValueError, TypeError):
                    parsed[key] = val
            rows.append(parsed)
    return rows


def _resolve_module_label(label_entry: dict) -> Optional[str]:
    """Resolve module label from labels.json entry to MODULE_LABELS v1 name."""
    mod = label_entry.get("module") or ""
    if mod in LABELS:
        return mod
    mod_v2 = label_entry.get("module_v2") or ""
    if mod_v2 in _MODULE_V1_LOOKUP:
        return _MODULE_V1_LOOKUP[mod_v2]
    if mod:
        for v1_name in LABELS:
            if v1_name in mod or mod in v1_name:
                return v1_name
    return None


def main():
    ap = argparse.ArgumentParser(
        description="CMA-ES optimization for BRB hierarchical module inference params")
    ap.add_argument("--data_dir", default=None,
                    help="数据目录 (包含 features_brb.csv 和 labels.json)")
    ap.add_argument("--data", default=None, help="特征 CSV (旧格式)")
    ap.add_argument("--output_dir", default=None, help="输出目录")
    ap.add_argument("--label_col", default=None,
                    help="有监督时的标签列名 (旧格式 CSV)")
    ap.add_argument("--supervised", action="store_true",
                    help="开启有监督优化")
    ap.add_argument("--generations", "--maxiter", type=int, default=80,
                    dest="maxiter")
    ap.add_argument("--population", "--popsize", type=int, default=None,
                    dest="popsize")
    ap.add_argument("--n_jobs", type=int, default=1,
                    help="并行度 (当前未使用)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sigma0", type=float, default=0.3)
    args = ap.parse_args()

    if cma is None:
        print("[WARN] cma not installed. Install with: pip install cma")
        print("[WARN] Saving default params as fallback.")
        best_w = np.ones(16, dtype=float)
        out_dir = Path(args.output_dir) if args.output_dir else Path(".")
        out_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "hierarchical_params": best_w.tolist(),
            "module_rule_weights": [0.8, 0.6, 0.7, 0.5, 0.5, 0.4, 0.15],
            "objective": None,
            "note": "Default params (cma not installed)",
        }
        out_path = out_dir / "best_params.json"
        out_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] Default params saved to {out_path}")
        return

    # Load data
    feats_rows: List[Dict] = []
    label_v2_names: Optional[List[str]] = None
    fault_types: List[str] = []

    # Fault class mapping
    _SYS_CN_TO_FAULT = {
        "amp_error": "amp_error",
        "freq_error": "freq_error",
        "ref_error": "ref_error",
        "幅度失准": "amp_error",
        "频率失准": "freq_error",
        "参考电平失准": "ref_error",
    }

    if args.data_dir:
        data_dir = Path(args.data_dir)
        features_path = data_dir / "features_brb.csv"
        labels_path = data_dir / "labels.json"

        if not features_path.exists():
            raise FileNotFoundError(
                f"features_brb.csv not found in {data_dir}")
        if not labels_path.exists():
            raise FileNotFoundError(
                f"labels.json not found in {data_dir}")

        feats_rows = _load_csv_rows(features_path)
        labels_dict = json.loads(labels_path.read_text(encoding="utf-8"))

        sample_feats: Dict[str, Dict] = {}
        for row in feats_rows:
            sid = row.get("sample_id", "")
            if isinstance(sid, float):
                sid = str(int(sid))
            sample_feats[str(sid)] = row

        if args.supervised:
            filtered_rows: List[Dict] = []
            filtered_labels: List[str] = []
            filtered_faults: List[str] = []
            for sid, label_entry in labels_dict.items():
                if str(sid) not in sample_feats:
                    continue
                sys_class = label_entry.get("system_fault_class", "normal")
                if sys_class == "normal":
                    continue
                fault_type = _SYS_CN_TO_FAULT.get(sys_class)
                if fault_type is None:
                    continue
                mod_v2 = _resolve_module_to_v2(label_entry)
                if mod_v2 is None:
                    continue
                filtered_rows.append(sample_feats[str(sid)])
                filtered_labels.append(mod_v2)
                filtered_faults.append(fault_type)

            if filtered_rows:
                feats_rows = filtered_rows
                label_v2_names = filtered_labels
                fault_types = filtered_faults
                print(f"[INFO] Supervised: {len(feats_rows)} samples "
                      f"with V2 module labels")
                from collections import Counter
                dist = Counter(label_v2_names)
                for mod, count in sorted(dist.items(), key=lambda x: -x[1]):
                    print(f"  {mod}: {count}")
            else:
                print("[WARN] No valid module labels. "
                      "Falling back to unsupervised.")
                label_v2_names = None
        
        # For unsupervised mode, still need fault_types
        if not fault_types:
            print(f"[INFO] Unsupervised: {len(feats_rows)} samples")
            for row in feats_rows:
                fault_types.append(_fault_type_from_features(dict(row)))

    elif args.data:
        data_path = Path(args.data)
        if not data_path.exists():
            raise FileNotFoundError(f"CSV not found: {data_path}")
        feats_rows = _load_csv_rows(data_path)
        for row in feats_rows:
            fault_types.append(_fault_type_from_features(dict(row)))
    else:
        raise ValueError("Please provide --data_dir or --data")

    print(f"\n[INFO] Starting CMA-ES optimization (hierarchical params)...")
    print(f"  Samples: {len(feats_rows)}")
    print(f"  Mode: {'supervised' if label_v2_names is not None else 'unsupervised'}")
    print(f"  Parameters: 16 (6 amp + 4 freq + 3 ref priors + 3 feat sensitivity)")
    print(f"  Max iterations: {args.maxiter}")
    if args.popsize:
        print(f"  Population size: {args.popsize}")

    best_w, best_obj, history = optimize(
        feats_rows=feats_rows,
        label_v2_names=label_v2_names,
        fault_types=fault_types,
        maxiter=args.maxiter,
        popsize=args.popsize,
        seed=args.seed,
        sigma0=args.sigma0,
    )

    # Save results
    out_dir = Path(args.output_dir) if args.output_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "hierarchical_params": best_w.tolist(),
        "module_rule_weights": [0.8, 0.6, 0.7, 0.5, 0.5, 0.4, 0.15],
        "objective": float(best_obj),
        "generations": len(history),
        "mode": ("supervised" if label_v2_names is not None
                 else "unsupervised"),
        "n_samples": len(feats_rows),
    }
    params_path = out_dir / "best_params.json"
    params_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[INFO] Best params saved to {params_path}")

    if history:
        log_path = out_dir / "optimization_log.csv"
        with log_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["generation", "best_obj", "mean_obj"])
            writer.writeheader()
            writer.writerows(history)
        print(f"[INFO] Optimization log saved to {log_path}")

    weights_path = out_dir / "optimized_hierarchical_params.txt"
    np.savetxt(str(weights_path), best_w, fmt="%.6f")
    print(f"[INFO] Params saved to {weights_path}")

    param_names = [
        "amp_ADC", "amp_Mixer1", "amp_Filter", "amp_Power", "amp_IF", "amp_DSP",
        "freq_RefDist", "freq_Mixer1", "freq_LO1", "freq_OCXO",
        "ref_CalSrc", "ref_CalStore", "ref_CalSwitch",
        "feat_sens_amp", "feat_sens_freq", "feat_sens_ref",
    ]
    print(f"\n{'='*50}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*50}")
    for i, (name, w) in enumerate(zip(param_names, best_w)):
        print(f"  {name}: {w:.4f}")
    print(f"  Objective: {best_obj:.6f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

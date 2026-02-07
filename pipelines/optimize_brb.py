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
from BRB.module_brb import MODULE_LABELS

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


# ------- 可调参数的模块推理 --------
def module_level_infer_param(features, sys_probs, rule_weights):
    """
    rule_weights: 长度 7，对应 module_brb 中 7 条 BRBRule 的 weight 系数
    """
    probs = sys_probs.get("probabilities", sys_probs)
    ref_prior = probs.get("参考电平失准", 0.3)
    amp_prior = probs.get("幅度失准", 0.3)
    freq_prior = probs.get("频率失准", 0.3)

    md = _aggregate_module_score(features)

    rules = [
        (rule_weights[0] * ref_prior,
         {"衰减器": 0.60, "校准源": 0.08, "存储器": 0.06, "校准信号开关": 0.16}),
        (rule_weights[1] * amp_prior,
         {"中频放大器": 0.35, "数字放大器": 0.30, "衰减器": 0.20, "ADC": 0.15}),
        (rule_weights[2] * freq_prior,
         {"时钟振荡器": 0.35, "时钟合成与同步网络": 0.35, "本振源（谐波发生器）": 0.15, "本振混频组件": 0.15}),
        (rule_weights[3] * freq_prior, {"高频段YTF滤波器": 0.60, "高频段混频器": 0.40}),
        (rule_weights[4] * freq_prior, {"低频段前置低通滤波器": 0.60, "低频段第一混频器": 0.40}),
        (rule_weights[5] * amp_prior, {"数字RBW": 0.30, "数字检波器": 0.35, "VBW滤波器": 0.25, "ADC": 0.10}),
        (rule_weights[6], {"电源模块": 1.0}),
    ]

    acts = []
    for w, bel in rules:
        act = w * md
        acts.append((act, bel))
    total = sum(a for a, _ in acts) + 1e-9
    out = {lab: 0.0 for lab in LABELS}
    for a, bel in acts:
        for lab in LABELS:
            out[lab] += (a / total) * bel.get(lab, 0.0)
    s = sum(out.values()) + 1e-9
    for lab in LABELS:
        out[lab] = out[lab] / s
    return out


# ------- 目标函数 -------
def unsupervised_objective(weights, feats_rows, w_entropy=0.6, w_conf=0.4):
    probs = []
    for row in feats_rows:
        f = dict(row)
        sys_p = system_level_infer(f)
        mod_p = module_level_infer_param(f, sys_p, weights)
        probs.append([mod_p[lab] for lab in LABELS])
    probs = np.array(probs)
    eps = 1e-12
    ent = -np.sum(probs * np.log(np.clip(probs, eps, 1.0)), axis=1)
    mean_ent = float(np.nanmean(ent))
    mean_top1 = float(np.nanmean(np.max(probs, axis=1)))
    return w_entropy * mean_ent + w_conf * (1.0 - mean_top1)


def supervised_objective(weights, feats_rows, label_indices):
    """Supervised objective using module label indices."""
    probs = []
    for row in feats_rows:
        f = dict(row)
        sys_p = system_level_infer(f)
        mod_p = module_level_infer_param(f, sys_p, weights)
        probs.append([mod_p[lab] for lab in LABELS])
    probs = np.array(probs)
    y_idx = np.array(label_indices)
    try:
        loss = log_loss(y_idx, probs, labels=list(range(len(LABELS))))
    except Exception:
        loss = 1.0 - accuracy_score(y_idx, np.argmax(probs, axis=1))
    return float(loss)


# ------- CMA-ES 优化主流程 -------
def optimize(feats_rows, label_indices=None, maxiter=80, popsize=None, seed=42, sigma0=0.3):
    # Initial weights: [ref, amp, freq, hf_filter, lf_filter, digital, power]
    x0 = np.array([0.8, 0.6, 0.7, 0.5, 0.5, 0.4, 0.15], dtype=float)
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
            w = np.clip(x, 0.01, 3.0)
            try:
                if label_indices is not None:
                    obj = supervised_objective(w, feats_rows, label_indices)
                else:
                    obj = unsupervised_objective(w, feats_rows)
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
        best_x = np.clip(es.result.xbest, 0.01, 3.0)
        best_obj = float(es.result.fbest)

    print(f"\n[INFO] best_obj={best_obj:.6f}, best_weights={best_x}")
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
        description="CMA-ES optimization for BRB module rule weights")
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
        print("[WARN] Saving default weights as fallback.")
        best_w = np.array([0.8, 0.6, 0.7, 0.5, 0.5, 0.4, 0.15],
                          dtype=float)
        out_dir = Path(args.output_dir) if args.output_dir else Path(".")
        out_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "module_rule_weights": best_w.tolist(),
            "objective": None,
            "note": "Default weights (cma not installed)",
        }
        out_path = out_dir / "best_params.json"
        out_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] Default params saved to {out_path}")
        return

    # Load data
    feats_rows: List[Dict] = []
    label_indices: Optional[List[int]] = None

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
            idx_map = {m: i for i, m in enumerate(LABELS)}
            filtered_rows: List[Dict] = []
            filtered_indices: List[int] = []
            for sid, label_entry in labels_dict.items():
                if str(sid) not in sample_feats:
                    continue
                mod_name = _resolve_module_label(label_entry)
                if mod_name is None or mod_name not in idx_map:
                    continue
                sys_class = label_entry.get("system_fault_class", "normal")
                if sys_class == "normal":
                    continue
                filtered_rows.append(sample_feats[str(sid)])
                filtered_indices.append(idx_map[mod_name])

            if filtered_rows:
                feats_rows = filtered_rows
                label_indices = filtered_indices
                print(f"[INFO] Supervised: {len(feats_rows)} samples "
                      f"with module labels")
                from collections import Counter
                dist = Counter(label_indices)
                for idx, count in sorted(dist.items()):
                    print(f"  {LABELS[idx]}: {count}")
            else:
                print("[WARN] No valid module labels. "
                      "Falling back to unsupervised.")
                label_indices = None
        else:
            print(f"[INFO] Unsupervised: {len(feats_rows)} samples")

    elif args.data:
        data_path = Path(args.data)
        if not data_path.exists():
            raise FileNotFoundError(f"CSV not found: {data_path}")
        feats_rows = _load_csv_rows(data_path)

        if args.supervised and args.label_col:
            idx_map = {m: i for i, m in enumerate(LABELS)}
            filtered_rows = []
            filtered_indices = []
            for row in feats_rows:
                label = row.get(args.label_col, "")
                if label in idx_map:
                    filtered_rows.append(row)
                    filtered_indices.append(idx_map[label])
            if filtered_rows:
                feats_rows = filtered_rows
                label_indices = filtered_indices
            else:
                print("[WARN] No valid labels. Falling back to unsupervised.")
    else:
        raise ValueError("Please provide --data_dir or --data")

    print(f"\n[INFO] Starting CMA-ES optimization...")
    print(f"  Samples: {len(feats_rows)}")
    print(f"  Mode: {'supervised' if label_indices is not None else 'unsupervised'}")
    print(f"  Max iterations: {args.maxiter}")
    if args.popsize:
        print(f"  Population size: {args.popsize}")

    best_w, best_obj, history = optimize(
        feats_rows=feats_rows,
        label_indices=label_indices,
        maxiter=args.maxiter,
        popsize=args.popsize,
        seed=args.seed,
        sigma0=args.sigma0,
    )

    # Save results
    out_dir = Path(args.output_dir) if args.output_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "module_rule_weights": best_w.tolist(),
        "objective": float(best_obj),
        "generations": len(history),
        "mode": ("supervised" if label_indices is not None
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

    weights_path = out_dir / "optimized_module_rule_weights.txt"
    np.savetxt(str(weights_path), best_w, fmt="%.6f")
    print(f"[INFO] Weights saved to {weights_path}")

    print(f"\n{'='*50}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*50}")
    for i, w in enumerate(best_w):
        print(f"  Rule {i}: weight = {w:.4f}")
    print(f"  Objective: {best_obj:.6f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

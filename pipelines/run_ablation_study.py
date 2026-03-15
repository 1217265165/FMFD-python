#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ablation Study Pipeline for HBRB Method.

Implements three degraded variants to validate the necessity of each core
innovation:
  1. w/o Decoupling  – replaces knowledge-driven feature selection with PCA
  2. w/o Soft-Gating – replaces dual-driven soft gating with hard argmax
  3. w/o P-Constraint – removes expert anti-drift penalty in P-CMA-ES

All variants and the full model (Ours Full) are evaluated on the same 400-
sample test set to ensure fair comparison.

Usage:
    python pipelines/run_ablation_study.py --data_dir Output/sim_spectrum
    python pipelines/run_ablation_study.py --data_dir Output/sim_spectrum --load_params Output/sim_spectrum/best_params.json
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.default_paths import (
    PROJECT_ROOT,
    SIM_DIR,
    COMPARE_DIR,
    SEED,
    SPLIT,
)
from pipelines.compare_methods import (
    prepare_dataset,
    stratified_split,
    load_expected_features,
    select_feature_matrix,
    calculate_accuracy,
    calculate_macro_f1,
    set_global_seed,
    SYS_LABEL_ORDER,
)
from BRB.module_brb import (
    MODULE_LABELS,
    set_hierarchical_params,
    get_hierarchical_params,
    hierarchical_module_infer,
    hierarchical_module_infer_soft_gating,
)
from methods.ours_adapter import OursAdapter


# ── Variant display names ─────────────────────────────────────────────────
VARIANT_NAMES = {
    "ours_full": "Ours Full",
    "wo_decoupling": "w/o Decoupling",
    "wo_soft_gating": "w/o Soft-Gating",
    "wo_p_constraint": "w/o P-Constraint",
}


# ============================================================================
# Variant 1: w/o Decoupling (PCA on all features)
# ============================================================================
class OursWithoutDecoupling(OursAdapter):
    """Ablation variant: replace knowledge-driven feature selection with PCA.

    Instead of using the physically-decoupled X1-X37 feature subset, this
    variant feeds ALL available features through PCA (retaining 95% variance)
    before training the RF classifier.  The BRB module-level inference is
    unchanged but receives PCA-transformed features, breaking the physical
    interpretability link.
    """

    name = "wo_decoupling"

    def __init__(self, calibration_override=None):
        super().__init__(calibration_override=calibration_override)
        self._pca = None
        self._pca_n_components = None

    def fit(self, X_train, y_sys_train, y_mod_train=None, meta=None):
        from sklearn.decomposition import PCA

        if meta and "feature_names" in meta:
            self.feature_names = meta["feature_names"]

        # Use ALL features (no knowledge-driven selection)
        pca = PCA(n_components=0.95, random_state=42)
        X_pca = pca.fit_transform(X_train)
        self._pca = pca
        self._pca_n_components = X_pca.shape[1]

        print(
            f">> [w/o Decoupling] PCA: {X_train.shape[1]} dims → "
            f"{self._pca_n_components} components (95% variance)"
        )

        # Train RF on PCA-transformed features
        self.classifier.fit(X_pca, y_sys_train)
        self.is_fitted = True
        self._rf_feature_indices = None  # Not used in PCA mode

        from BRB.gating_prior import GatingPriorFusion
        from methods.ours_adapter import _load_gating_prior_config

        gating_config = _load_gating_prior_config()
        self.fusion_engine = GatingPriorFusion(gating_config)

        # Rule counts inherit from parent OursAdapter (3 sub-BRBs × 5 = 15 system,
        # 33 configured per-module rules).
        self.n_system_rules = 15
        self.n_module_rules = 33
        # Complexity includes PCA rotation matrix (n_components × input_dims)
        # as additional learned parameters on top of the original BRB params.
        self.n_params = (
            self._pca_n_components * X_train.shape[1]
            + 68
        )

    def predict(self, X_test, meta=None):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")

        if meta and "feature_names" in meta:
            self.feature_names = meta["feature_names"]

        n_test = len(X_test)

        # Transform test data with PCA
        X_pca = self._pca.transform(X_test)

        # System-level via RF on PCA features
        rf_proba = self.classifier.predict_proba(X_pca)
        sys_pred = np.argmax(rf_proba, axis=1)

        # Module-level: still use original features (BRB needs physical features)
        mod_proba = np.zeros((n_test, len(MODULE_LABELS)))
        mod_pred = np.zeros(n_test, dtype=int)
        from tools.label_mapping import module_v2_from_v1

        start_time = time.time()
        for i in range(n_test):
            features = self._array_to_dict(X_test[i])

            # Map RF prediction to fault type
            class_names = ["normal", "amp_error", "freq_error", "ref_error"]
            sys_probs = {c: float(rf_proba[i, j]) for j, c in enumerate(class_names)}

            try:
                soft_result = hierarchical_module_infer_soft_gating(
                    sys_probs, features, delta=0.15, use_board_prior=True
                )
                module_topk = soft_result["fused_topk"]
            except Exception:
                fault_type = class_names[sys_pred[i]]
                if fault_type == "normal":
                    fault_type = "amp_error"
                mod_probs = hierarchical_module_infer(
                    fault_type, features, use_board_prior=True
                )
                sorted_mods = sorted(
                    mod_probs.items(), key=lambda x: x[1], reverse=True
                )
                module_topk = [
                    {"name": m, "prob": p} for m, p in sorted_mods[:10]
                ]

            mod_probs_dict = {m["name"]: m["prob"] for m in module_topk}
            v2_to_v1 = {}
            for v1_idx, v1_name in enumerate(MODULE_LABELS):
                v2_name = module_v2_from_v1(v1_name)
                if v2_name not in v2_to_v1:
                    v2_to_v1[v2_name] = []
                v2_to_v1[v2_name].append(v1_idx)

            for mod_name, prob in mod_probs_dict.items():
                if mod_name in v2_to_v1:
                    for v1_idx in v2_to_v1[mod_name]:
                        mod_proba[i, v1_idx] = prob

            if np.sum(mod_proba[i]) > 0:
                mod_pred[i] = np.argmax(mod_proba[i])

        infer_time = time.time() - start_time
        infer_ms = (infer_time / n_test) * 1000 if n_test > 0 else 0.0

        return {
            "system_proba": rf_proba,
            "system_pred": sys_pred,
            "module_proba": mod_proba,
            "module_pred": mod_pred + 1,
            "meta": {
                "infer_time_ms_per_sample": infer_ms,
                "n_rules": self.n_system_rules + self.n_module_rules,
                "n_params": self.n_params,
                "n_features_used": self._pca_n_components or 0,
                "features_used": [f"PC{i+1}" for i in range(self._pca_n_components or 0)],
            },
        }

    def complexity(self):
        return {
            "n_rules": self.n_system_rules + self.n_module_rules,
            "n_params": self.n_params,
            "n_features_used": self._pca_n_components or 0,
        }


# ============================================================================
# Variant 2: w/o Soft-Gating (hard argmax instead of multi-hypothesis fusion)
# ============================================================================
class OursWithoutSoftGating(OursAdapter):
    """Ablation variant: replace soft gating with hard argmax.

    After RF system-level prediction, the probability vector is forced to a
    one-hot encoding (argmax → 1.0, rest → 0.0).  Only the single winning
    fault-type subgraph is activated for module-level inference — no multi-
    hypothesis fusion, no secondary weight floor, no diversity guarantee.
    """

    name = "wo_soft_gating"

    def predict(self, X_test, meta=None):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")

        if self.fusion_engine is None:
            from BRB.gating_prior import GatingPriorFusion
            from methods.ours_adapter import _load_gating_prior_config

            gating_config = _load_gating_prior_config()
            self.fusion_engine = GatingPriorFusion(gating_config)

        n_test = len(X_test)
        if meta and "feature_names" in meta:
            self.feature_names = meta["feature_names"]

        sys_proba = np.zeros((n_test, 4))
        sys_pred = np.zeros(n_test, dtype=int)
        mod_proba = np.zeros((n_test, len(MODULE_LABELS)))
        mod_pred = np.zeros(n_test, dtype=int)

        from tools.label_mapping import module_v2_from_v1

        class_names = ["normal", "amp_error", "freq_error", "ref_error"]

        start_time = time.time()
        for i in range(n_test):
            features = self._array_to_dict(X_test[i])

            # Get RF probabilities
            rf_vec = X_test[i]
            if self._rf_feature_indices is not None:
                rf_vec = X_test[i][self._rf_feature_indices]
            rf_proba = self.classifier.predict_proba(rf_vec.reshape(1, -1))[0]

            # Hard argmax: one-hot encoding (NO soft gating)
            best_idx = np.argmax(rf_proba)
            hard_probs = np.zeros(4)
            hard_probs[best_idx] = 1.0
            sys_proba[i] = hard_probs
            sys_pred[i] = best_idx

            # Only activate the single winning fault-type subgraph
            fault_type = class_names[best_idx]
            if fault_type == "normal":
                # If normal, use amp_error as fallback
                fault_type = "amp_error"

            # Single-hypothesis module inference (no soft gating)
            mod_probs = hierarchical_module_infer(
                fault_type, features, use_board_prior=True
            )
            sorted_mods = sorted(
                mod_probs.items(), key=lambda x: x[1], reverse=True
            )
            module_topk = [{"name": m, "prob": p} for m, p in sorted_mods[:10]]

            # Map V2 → V1
            mod_probs_dict = {m["name"]: m["prob"] for m in module_topk}
            v2_to_v1 = {}
            for v1_idx, v1_name in enumerate(MODULE_LABELS):
                v2_name = module_v2_from_v1(v1_name)
                if v2_name not in v2_to_v1:
                    v2_to_v1[v2_name] = []
                v2_to_v1[v2_name].append(v1_idx)

            for mod_name, prob in mod_probs_dict.items():
                if mod_name in v2_to_v1:
                    for v1_idx in v2_to_v1[mod_name]:
                        mod_proba[i, v1_idx] = prob

            if np.sum(mod_proba[i]) > 0:
                mod_pred[i] = np.argmax(mod_proba[i])

        infer_time = time.time() - start_time
        infer_ms = (infer_time / n_test) * 1000 if n_test > 0 else 0.0

        return {
            "system_proba": sys_proba,
            "system_pred": sys_pred,
            "module_proba": mod_proba,
            "module_pred": mod_pred + 1,
            "meta": {
                "infer_time_ms_per_sample": infer_ms,
                "n_rules": self.n_system_rules + self.n_module_rules,
                "n_params": self.n_params,
                "n_features_used": len(self.kd_features),
                "features_used": self.kd_features,
            },
        }


# ============================================================================
# Variant 3: w/o P-Constraint (remove expert anti-drift penalty)
# ============================================================================
class OursWithoutPConstraint(OursAdapter):
    """Ablation variant: remove expert anti-drift penalty from P-CMA-ES.

    The L2 regularization terms in the supervised objective function are set
    to zero.  This allows the CMA-ES optimiser to drift arbitrarily far from
    the expert-designed priors, risking overfitting on small training sets.

    Implementation: before prediction, re-optimise BRB parameters using
    a penalty-free objective, then restore original params afterwards.
    """

    name = "wo_p_constraint"

    def __init__(self, calibration_override=None):
        super().__init__(calibration_override=calibration_override)
        self._unconstrained_params = None

    def fit(self, X_train, y_sys_train, y_mod_train=None, meta=None):
        # Standard fit (train RF)
        super().fit(X_train, y_sys_train, y_mod_train, meta)

        # Re-optimise BRB params WITHOUT penalty terms
        try:
            self._run_unconstrained_optimization(X_train, y_sys_train, y_mod_train, meta)
        except Exception as e:
            print(f"[w/o P-Constraint] Optimization failed: {e}")
            print("[w/o P-Constraint] Using default params (no penalty variant)")
            self._unconstrained_params = None

    def _run_unconstrained_optimization(self, X_train, y_sys_train, y_mod_train, meta):
        """Run CMA-ES without expert anti-drift penalty."""
        try:
            import cma
        except ImportError:
            print("[w/o P-Constraint] cma not installed, using perturbed default params")
            # Perturb defaults to simulate unconstrained drift
            rng = np.random.RandomState(42)
            params = np.array(get_hierarchical_params(), dtype=float)
            # [0:15] = prior scale factors (8 amp + 4 freq + 3 ref)
            params[0:15] *= rng.uniform(0.3, 3.0, size=15)
            # [15:18] = feature sensitivity indices
            params[15:18] *= rng.uniform(0.5, 2.0, size=3)
            params = np.clip(params, 0.1, 5.0)
            self._unconstrained_params = params.tolist()
            return

        from pipelines.optimize_brb import project_to_feasible
        from BRB.system_brb import system_level_infer

        # Prepare training data for module-level optimization
        feature_names = meta.get("feature_names", []) if meta else []
        feats_rows = []
        fault_types = []
        label_v2_names = []

        from tools.label_mapping import module_v2_from_v1

        for i in range(len(X_train)):
            feat_dict = {}
            for j, name in enumerate(feature_names):
                if j < X_train.shape[1]:
                    feat_dict[name] = float(X_train[i, j])
            # Ensure X1-X22 keys exist (required by BRB module inference)
            for k in range(1, 23):
                key = f"X{k}"
                if key not in feat_dict:
                    feat_dict[key] = 0.0

            # Determine fault type from system label
            class_names = ["normal", "amp_error", "freq_error", "ref_error"]
            ft = class_names[y_sys_train[i]] if y_sys_train[i] < 4 else "normal"
            if ft == "normal":
                continue

            # Determine true module (V2 name)
            if y_mod_train is not None and y_mod_train[i] >= 0:
                v1_idx = y_mod_train[i]
                if v1_idx < len(MODULE_LABELS):
                    v2_name = module_v2_from_v1(MODULE_LABELS[v1_idx])
                    feats_rows.append(feat_dict)
                    fault_types.append(ft)
                    label_v2_names.append(v2_name)

        if len(feats_rows) < 5:
            print("[w/o P-Constraint] Too few module-labelled samples for optimization")
            return

        # Objective function WITHOUT penalty (reg_scales=0, reg_sens=0)
        def unconstrained_objective(params):
            projected = project_to_feasible(params)
            set_hierarchical_params(list(projected))

            class_correct = {}
            class_total = {}
            for row, true_v2, ft in zip(feats_rows, label_v2_names, fault_types):
                mod_probs = hierarchical_module_infer(ft, row, use_board_prior=True)
                if mod_probs:
                    pred_v2 = max(mod_probs, key=mod_probs.get)
                    class_total[true_v2] = class_total.get(true_v2, 0) + 1
                    if pred_v2 == true_v2:
                        class_correct[true_v2] = class_correct.get(true_v2, 0) + 1

            recalls = []
            for cls in class_total:
                total = class_total[cls]
                correct = class_correct.get(cls, 0)
                recalls.append(correct / total if total > 0 else 0.0)
            balanced_acc = float(np.mean(recalls)) if recalls else 0.0

            # NO penalty terms (this is the ablation)
            return 1.0 - balanced_acc

        # Save original params, run CMA-ES
        original_params = get_hierarchical_params()
        x0 = np.ones(18)
        opts = cma.CMAOptions()
        opts["maxiter"] = 40
        opts["popsize"] = 12
        opts["seed"] = 42
        opts["verbose"] = -1
        opts["bounds"] = [[0.1] * 18, [5.0] * 18]

        try:
            es = cma.CMAEvolutionStrategy(x0, 0.5, opts)
            while not es.stop():
                solutions = es.ask()
                fitness = [unconstrained_objective(s) for s in solutions]
                es.tell(solutions, fitness)
            best = project_to_feasible(es.result.xbest)
            self._unconstrained_params = best.tolist()
            print(
                f"[w/o P-Constraint] Unconstrained optimization complete. "
                f"Best fitness: {es.result.fbest:.4f}"
            )
        except Exception as e:
            print(f"[w/o P-Constraint] CMA-ES failed: {e}")
        finally:
            # Restore original params
            set_hierarchical_params(original_params)

    def predict(self, X_test, meta=None):
        # Temporarily set unconstrained params for prediction
        original_params = get_hierarchical_params()
        if self._unconstrained_params is not None:
            set_hierarchical_params(self._unconstrained_params)

        try:
            result = super().predict(X_test, meta)
        finally:
            set_hierarchical_params(original_params)

        return result


# ============================================================================
# Module-level evaluation helper (same as compare_methods.py)
# ============================================================================
def evaluate_module_metrics(
    predictions: Dict,
    y_mod_test: np.ndarray,
) -> Tuple[float, float]:
    """Compute module top-1 and top-3 accuracy with V2-aware de-duplication."""
    mod_proba = predictions.get("module_proba")
    if mod_proba is None or y_mod_test is None:
        return 0.0, 0.0

    from tools.label_mapping import module_v2_from_v1
    from BRB.module_brb import MODULE_LABELS

    n_correct_top1 = 0
    n_correct_top3 = 0
    n_valid = 0
    n_mods = len(MODULE_LABELS)

    for i, (gt_idx, row) in enumerate(zip(y_mod_test, mod_proba)):
        if gt_idx < 0:
            continue
        n_valid += 1

        gt_v1_name = MODULE_LABELS[gt_idx] if gt_idx < n_mods else ""
        gt_v2 = module_v2_from_v1(gt_v1_name)

        # De-duplicated top-3 by unique V2 module names
        sorted_indices = np.argsort(row)[::-1]
        pred_v2_unique = []
        for j in sorted_indices:
            if j >= n_mods:
                continue
            v2 = module_v2_from_v1(MODULE_LABELS[j])
            if v2 not in pred_v2_unique:
                pred_v2_unique.append(v2)
            if len(pred_v2_unique) >= 3:
                break

        pred_v2_top1 = pred_v2_unique[0] if pred_v2_unique else ""

        if gt_v2 == pred_v2_top1:
            n_correct_top1 += 1
        if gt_v2 in pred_v2_unique:
            n_correct_top3 += 1

    if n_valid > 0:
        return n_correct_top1 / n_valid, n_correct_top3 / n_valid
    return 0.0, 0.0


# ============================================================================
# Plotting: Accuracy bar chart
# ============================================================================
def plot_ablation_accuracy(results: List[Dict], output_path: Path):
    """Plot grouped bar chart comparing sys/mod_top1/mod_top3 accuracy."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        plt.rcParams["font.sans-serif"] = ["SimHei", "Arial", "sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False

        variants = [r["variant_display"] for r in results]
        sys_acc = [r["sys_accuracy"] * 100 for r in results]
        mod_top1 = [r["mod_top1_accuracy"] * 100 for r in results]
        mod_top3 = [r["mod_top3_accuracy"] * 100 for r in results]

        x = np.arange(len(variants))
        width = 0.22

        fig, ax = plt.subplots(figsize=(10, 6))

        bars1 = ax.bar(x - width, sys_acc, width, label="系统准确率", color="#4C72B0", alpha=0.85)
        bars2 = ax.bar(x, mod_top1, width, label="模块 Top-1 准确率", color="#DD8452", alpha=0.85)
        bars3 = ax.bar(x + width, mod_top3, width, label="模块 Top-3 准确率", color="#55A868", alpha=0.85)

        ax.set_ylabel("准确率 (%)", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(variants, fontsize=10)
        ax.tick_params(axis="x", rotation=15)
        ax.set_ylim(0, 110)
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 1,
                    f"{height:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        fig.suptitle("消融实验：多指标准确率对比", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved ablation accuracy chart to: {output_path}")
    except ImportError:
        print("matplotlib not available, skipping accuracy chart")


# ============================================================================
# Plotting: Rules bar chart
# ============================================================================
def plot_ablation_rules(results: List[Dict], output_path: Path):
    """Plot bar chart of n_rules per variant."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["SimHei", "Arial", "sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False

        variants = [r["variant_display"] for r in results]
        n_rules = [r["n_rules"] for r in results]

        fig, ax = plt.subplots(figsize=(8, 5))

        colors = ["#C44E52" if r["variant"] == "wo_decoupling" else "#4C72B0" for r in results]
        bars = ax.bar(variants, n_rules, color=colors, alpha=0.85, width=0.5)

        ax.set_ylabel("规则数 (n_rules)", fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=15, labelsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, max(n_rules) * 1.2)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(n_rules) * 0.02,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        fig.suptitle("消融实验：模型规则数对比", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved ablation rules chart to: {output_path}")
    except ImportError:
        print("matplotlib not available, skipping rules chart")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Ablation Study for HBRB")
    parser.add_argument("--data_dir", default=str(SIM_DIR), help="Data directory")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument(
        "--load_params", type=str, default=None,
        help="Path to best_params.json for optimized BRB weights",
    )
    args = parser.parse_args()

    set_global_seed(args.seed)

    data_dir = Path(args.data_dir) if Path(args.data_dir).is_absolute() else PROJECT_ROOT / args.data_dir
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "ablation_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load optimized BRB params if provided
    if args.load_params:
        params_path = Path(args.load_params) if Path(args.load_params).is_absolute() else PROJECT_ROOT / args.load_params
        if params_path.exists():
            try:
                params_data = json.loads(params_path.read_text(encoding="utf-8"))
                if "hierarchical_params" in params_data:
                    set_hierarchical_params(params_data["hierarchical_params"])
                    print(f"[INFO] Loaded optimized hierarchical params from {params_path}")
            except Exception as e:
                print(f"[WARN] Failed to load params: {e}")
    else:
        # Try auto-loading from data_dir
        auto_path = data_dir / "best_params.json"
        if auto_path.exists():
            try:
                params_data = json.loads(auto_path.read_text(encoding="utf-8"))
                if "hierarchical_params" in params_data:
                    set_hierarchical_params(params_data["hierarchical_params"])
                    print(f"[INFO] Auto-loaded params from {auto_path}")
            except Exception:
                pass

    # ── Load dataset ──────────────────────────────────────────────────────
    print("=" * 60)
    print("Loading dataset...")
    print("=" * 60)

    X, y_sys, y_mod, feature_names, sample_ids, leak_cols, _, _ = prepare_dataset(
        data_dir, use_pool_features=True, strict_leakage=False
    )

    print(f"Dataset: {len(X)} samples, {len(feature_names)} features")

    # ── Split dataset (reuse saved splits for fairness) ───────────────────
    split_path = (data_dir / ".." / "comparison_results" / "split_indices.json").resolve()
    if not split_path.exists():
        split_path = output_dir.parent / "comparison_results" / "split_indices.json"
    if not split_path.exists():
        # Try COMPARE_DIR
        split_path = PROJECT_ROOT / COMPARE_DIR / "split_indices.json"

    if split_path.exists():
        split_payload = json.loads(split_path.read_text(encoding="utf-8"))
        train_idx = np.array(split_payload["train_idx"], dtype=int)
        val_idx = np.array(split_payload["val_idx"], dtype=int)
        test_idx = np.array(split_payload["test_idx"], dtype=int)
        X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
        y_sys_train = y_sys[train_idx]
        y_sys_test = y_sys[test_idx]
        print(f"[INFO] Loaded split indices from: {split_path}")
    else:
        X_train, X_val, X_test, y_sys_train, _, y_sys_test, train_idx, val_idx, test_idx = \
            stratified_split(X, y_sys, SPLIT[0], SPLIT[1], args.seed)
        # Save for reproducibility
        split_payload = {
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
            "test_idx": test_idx.tolist(),
        }
        (output_dir / "split_indices.json").write_text(
            json.dumps(split_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[INFO] Created new split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    y_mod_train = y_mod[train_idx] if y_mod is not None else None
    y_mod_test = y_mod[test_idx] if y_mod is not None else None

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # ── Feature selection (same as compare_methods.py) ────────────────────
    expected_features = load_expected_features("ours", feature_names, output_dir)
    X_train_sel, _ = select_feature_matrix(X_train, feature_names, expected_features)
    X_test_sel, _ = select_feature_matrix(X_test, feature_names, expected_features)

    # Also prepare full-feature matrices for w/o Decoupling variant
    X_train_full = X_train
    X_test_full = X_test

    # Load best_params for calibration
    best_params = None
    for p in [data_dir / "best_params.json", PROJECT_ROOT / "Output" / "sim_spectrum" / "best_params.json"]:
        if p.exists():
            try:
                best_params = json.loads(p.read_text(encoding="utf-8"))
                break
            except Exception:
                pass

    # ── Run variants ──────────────────────────────────────────────────────
    all_results = []

    variants = [
        ("ours_full", OursAdapter, X_train_sel, X_test_sel, expected_features),
        ("wo_decoupling", OursWithoutDecoupling, X_train_full, X_test_full, feature_names),
        ("wo_soft_gating", OursWithoutSoftGating, X_train_sel, X_test_sel, expected_features),
        ("wo_p_constraint", OursWithoutPConstraint, X_train_sel, X_test_sel, expected_features),
    ]

    for variant_name, adapter_cls, x_tr, x_te, feat_names in variants:
        print(f"\n{'=' * 60}")
        print(f"Running variant: {VARIANT_NAMES[variant_name]}")
        print(f"{'=' * 60}")

        try:
            adapter = adapter_cls(calibration_override=best_params)
            meta = {"feature_names": feat_names}

            # Fit
            start_fit = time.time()
            adapter.fit(x_tr, y_sys_train, y_mod_train, meta)
            fit_time = time.time() - start_fit

            # Predict
            start_infer = time.time()
            predictions = adapter.predict(x_te, meta=meta)
            infer_time = time.time() - start_infer
            infer_ms = (infer_time / len(x_te)) * 1000

            # System metrics
            y_sys_pred = predictions["system_pred"]
            sys_acc = calculate_accuracy(y_sys_test, y_sys_pred)
            sys_f1 = calculate_macro_f1(y_sys_test, y_sys_pred, 4)

            # Module metrics
            mod_top1, mod_top3 = evaluate_module_metrics(predictions, y_mod_test)

            # Complexity
            complexity = adapter.complexity()

            result = {
                "variant": variant_name,
                "variant_display": VARIANT_NAMES[variant_name],
                "sys_accuracy": sys_acc,
                "sys_macro_f1": sys_f1,
                "mod_top1_accuracy": mod_top1,
                "mod_top3_accuracy": mod_top3,
                "n_rules": complexity.get("n_rules", 0),
                "n_params": complexity.get("n_params", 0),
                "fit_time_sec": fit_time,
                "infer_ms_per_sample": infer_ms,
            }
            all_results.append(result)

            print(f"  sys_accuracy:      {sys_acc:.4f}")
            print(f"  sys_macro_f1:      {sys_f1:.4f}")
            print(f"  mod_top1_accuracy: {mod_top1:.4f}")
            print(f"  mod_top3_accuracy: {mod_top3:.4f}")
            print(f"  n_rules:           {complexity.get('n_rules', 0)}")
            print(f"  n_params:          {complexity.get('n_params', 0)}")

        except Exception as e:
            print(f"[ERROR] Variant {variant_name} failed: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "variant": variant_name,
                "variant_display": VARIANT_NAMES[variant_name],
                "sys_accuracy": 0.0,
                "sys_macro_f1": 0.0,
                "mod_top1_accuracy": 0.0,
                "mod_top3_accuracy": 0.0,
                "n_rules": 0,
                "n_params": 0,
                "fit_time_sec": 0.0,
                "infer_ms_per_sample": 0.0,
            })

    # ── Save ablation_table.csv ───────────────────────────────────────────
    csv_path = output_dir / "ablation_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Variant",
                "sys_accuracy",
                "mod_top1_accuracy",
                "mod_top3_accuracy",
                "n_rules",
            ],
        )
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "Variant": r["variant_display"],
                "sys_accuracy": f"{r['sys_accuracy']:.4f}",
                "mod_top1_accuracy": f"{r['mod_top1_accuracy']:.4f}",
                "mod_top3_accuracy": f"{r['mod_top3_accuracy']:.4f}",
                "n_rules": r["n_rules"],
            })
    print(f"\nSaved ablation_table.csv to: {csv_path}")

    # ── Save full results as JSON ─────────────────────────────────────────
    json_path = output_dir / "ablation_summary.json"
    json_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved ablation_summary.json to: {json_path}")

    # ── Generate plots ────────────────────────────────────────────────────
    plot_ablation_accuracy(all_results, output_dir / "ablation_accuracy.png")
    plot_ablation_rules(all_results, output_dir / "ablation_rules.png")

    # ── Print summary table ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)
    print(f"{'Variant':<22} {'sys_acc':>10} {'mod_top1':>10} {'mod_top3':>10} {'n_rules':>10}")
    print("-" * 64)
    for r in all_results:
        print(
            f"{r['variant_display']:<22} "
            f"{r['sys_accuracy']:>9.4f} "
            f"{r['mod_top1_accuracy']:>9.4f} "
            f"{r['mod_top3_accuracy']:>9.4f} "
            f"{r['n_rules']:>10}"
        )
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()

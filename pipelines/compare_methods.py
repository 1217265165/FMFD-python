#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comprehensive method comparison pipeline with unified training/testing interface.

This module implements the complete experimental validation framework for comparing
the proposed method against 5 baseline methods. All methods implement the MethodAdapter
interface with fit/predict methods.

Key features:
- Automatic dataset discovery and loading
- Stratified train/val/test split with fixed seeds
- Unified evaluation metrics (system + module level)
- Complexity analysis (rules, params, features)
- Small-sample adaptability experiments
- Comprehensive output generation (CSV tables + plots)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

# Apply unified plot style for Chinese font support (P5)
try:
    from utils.plot_style import apply as apply_plot_style
    apply_plot_style()
except ImportError:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

import numpy as np
from sklearn.metrics import roc_auc_score

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.default_paths import (
    PROJECT_ROOT,
    OUTPUT_DIR,
    SIM_DIR,
    COMPARE_DIR,
    SEED,
    SINGLE_BAND,
    DISABLE_PREAMP,
    SPLIT,
    build_run_snapshot,
)
from tools.check_features_integrity import generate_report as generate_feature_integrity_report
from BRB.module_brb import MODULE_LABELS
from features.feature_extraction import extract_system_features

# P2.1: Import unified module localization metrics
try:
    from metrics.module_localization_metrics import compute_mod_metrics
except ImportError:
    compute_mod_metrics = None

# P2.1: Import canonicalization utilities
try:
    from utils.canonicalize import modules_match as canonical_modules_match
except ImportError:
    canonical_modules_match = None

# Unified system label order (Normal, Amp, Freq, Ref)
SYS_LABEL_ORDER = ['正常', '幅度失准', '频率失准', '参考电平失准']

LEAK_PREFIXES = ("sys_", "label", "target", "gt_", "y_", "truth", "class_")
LEAK_SUBSTRINGS = ("label", "target", "truth")

EXPECTED_FEATURES_DIR = PROJECT_ROOT / "config" / "expected_features"


class LeakageError(RuntimeError):
    def __init__(self, columns: List[str]):
        super().__init__(f"Leakage columns detected: {columns}")
        self.columns = columns


def set_global_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import sklearn
        # sklearn doesn't have global seed, but individual estimators do
    except ImportError:
        pass


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return 0.5
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return 0.5


def _compute_feature_separation(
    X: np.ndarray,
    y_sys: np.ndarray,
    feature_names: List[str],
    label_order: List[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for idx, name in enumerate(feature_names):
        feature_col = X[:, idx]
        for class_idx, class_name in enumerate(label_order):
            cls_mask = y_sys == class_idx
            rest_mask = ~cls_mask
            cls_vals = feature_col[cls_mask]
            rest_vals = feature_col[rest_mask]
            cls_mean = float(np.mean(cls_vals)) if cls_vals.size else 0.0
            cls_std = float(np.std(cls_vals)) if cls_vals.size else 0.0
            rest_mean = float(np.mean(rest_vals)) if rest_vals.size else 0.0
            rest_std = float(np.std(rest_vals)) if rest_vals.size else 0.0
            fisher = (cls_mean - rest_mean) ** 2 / (cls_std ** 2 + rest_std ** 2 + 1e-9)
            auc = _safe_auc(cls_mask.astype(int), feature_col)
            rows.append(
                {
                    "feature": name,
                    "class": class_name,
                    "mean": cls_mean,
                    "std": cls_std,
                    "fisher_score": float(fisher),
                    "auc_ovr": auc,
                    "n": int(cls_vals.size),
                }
            )
    return rows


def _compute_baseline_feature_stats(baseline_npz: Path) -> Dict[str, Dict[str, float]]:
    if not baseline_npz.exists():
        return {}
    data = np.load(baseline_npz, allow_pickle=True)
    traces = data.get("traces")
    rrs = data.get("rrs")
    upper = data.get("upper")
    lower = data.get("lower")
    if traces is None or rrs is None or upper is None or lower is None:
        return {}
    stats: Dict[str, List[float]] = {}
    for trace in traces:
        feats = extract_system_features(trace, baseline_curve=rrs, envelope=(upper, lower))
        for key, value in feats.items():
            stats.setdefault(key, []).append(float(value))
    summary = {}
    for key, vals in stats.items():
        arr = np.array(vals, dtype=float)
        summary[key] = {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
    return summary


def _write_feature_separation_reports(
    output_dir: Path,
    feature_names: List[str],
    separation_rows: List[Dict[str, object]],
    baseline_stats: Dict[str, Dict[str, float]],
    X: np.ndarray,
    y_sys: np.ndarray,
    label_order: List[str],
) -> None:
    schema_path = output_dir / "feature_schema.json"
    prev_schema = None
    if schema_path.exists():
        prev_schema = json.loads(schema_path.read_text(encoding="utf-8"))
        (output_dir / "feature_schema_prev.json").write_text(
            json.dumps(prev_schema, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    schema_payload = {
        "feature_count": len(feature_names),
        "feature_names": feature_names,
    }
    schema_path.write_text(json.dumps(schema_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    prev_features = set(prev_schema.get("feature_names", [])) if prev_schema else set()
    curr_features = set(feature_names)
    schema_diff = {
        "prev_count": len(prev_features),
        "curr_count": len(curr_features),
        "missing_from_prev": sorted(prev_features - curr_features),
        "added_since_prev": sorted(curr_features - prev_features),
    }

    sim_normal_stats = {}
    for idx, name in enumerate(feature_names):
        vals = X[y_sys == 0, idx] if np.any(y_sys == 0) else np.array([])
        sim_normal_stats[name] = {
            "mean": float(np.mean(vals)) if vals.size else 0.0,
            "std": float(np.std(vals)) if vals.size else 0.0,
        }

    report_json = {
        "feature_schema": schema_payload,
        "schema_diff": schema_diff,
        "baseline_normal_stats": baseline_stats,
        "sim_normal_stats": sim_normal_stats,
        "separation": separation_rows,
    }
    json_path = output_dir / "baseline_vs_sim_feature_separation.json"
    json_path.write_text(json.dumps(report_json, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output_dir / "baseline_vs_sim_feature_separation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "feature",
            "class",
            "mean",
            "std",
            "fisher_score",
            "auc_ovr",
            "n",
            "baseline_mean",
            "baseline_std",
            "sim_normal_mean",
            "sim_normal_std",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in separation_rows:
            feature = row["feature"]
            base = baseline_stats.get(feature, {})
            sim_norm = sim_normal_stats.get(feature, {})
            writer.writerow(
                {
                    **row,
                    "baseline_mean": base.get("mean", 0.0),
                    "baseline_std": base.get("std", 0.0),
                    "sim_normal_mean": sim_norm.get("mean", 0.0),
                    "sim_normal_std": sim_norm.get("std", 0.0),
                }
            )


# ============================================================================
# Data Loading and Preparation
# ============================================================================

def discover_dataset_files(data_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Auto-discover features and labels files in data directory.
    
    Priority:
    1. features_brb.csv / labels.json
    2. First file containing 'features' / 'labels'
    3. Single CSV with label columns
    
    Returns:
        (features_path, labels_path) or (None, None) if not found
    """
    # Try standard names first
    features_path = data_dir / "features_brb.csv"
    labels_path = data_dir / "labels.json"
    
    if features_path.exists() and labels_path.exists():
        return features_path, labels_path
    
    # Search for pattern matches
    all_files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.json"))
    
    feat_file = None
    label_file = None
    
    for f in all_files:
        fname_lower = f.name.lower()
        if 'feature' in fname_lower and f.suffix == '.csv' and feat_file is None:
            feat_file = f
        if 'label' in fname_lower and label_file is None:
            label_file = f
    
    return feat_file, label_file


def load_labels(labels_path: Path) -> Dict:
    """Load labels from JSON or CSV file."""
    if labels_path.suffix == '.json':
        with open(labels_path, 'r', encoding='utf-8-sig') as f:
            return json.load(f)
    elif labels_path.suffix == '.csv':
        # Parse CSV labels
        labels_dict = {}
        with open(labels_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_id = row.get('sample_id') or row.get('id')
                if sample_id:
                    labels_dict[sample_id] = row
        return labels_dict
    else:
        raise ValueError(f"Unsupported label file format: {labels_path.suffix}")


def load_features_csv(features_path: Path) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """Load features from CSV file and preserve column order."""
    features_dict = {}
    feature_names: List[str] = []
    with open(features_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            feature_names = [name for name in reader.fieldnames if name not in ['sample_id', 'id']]
        for row in reader:
            sample_id = row.get('sample_id') or row.get('id')
            if sample_id:
                # Convert all values to float, skip non-numeric columns
                feat_row = {}
                for k, v in row.items():
                    if k not in ['sample_id', 'id']:
                        try:
                            feat_row[k] = float(v)
                        except (ValueError, TypeError):
                            pass
                features_dict[sample_id] = feat_row
    return features_dict, feature_names


def extract_system_label(entry: Dict) -> str:
    """Extract system-level label from entry.
    
    Supports:
    - entry['type'] = 'normal'/'fault' + entry['system_fault_class']
    - entry['system_label'] or entry['y_sys']
    
    Returns normalized label: '正常', '幅度失准', '频率失准', '参考电平失准'
    """
    # Check direct label fields
    if 'system_label' in entry:
        return str(entry['system_label'])
    if 'y_sys' in entry:
        label_val = entry['y_sys']
        if isinstance(label_val, (int, float)):
            # Map numeric to string
            mapping = {0: '正常', 1: '幅度失准', 2: '频率失准', 3: '参考电平失准'}
            return mapping.get(int(label_val), '正常')
        return str(label_val)
    
    # Parse from type + fault_class
    if entry.get('type') == 'normal':
        return '正常'
    elif entry.get('type') == 'fault':
        fault_cls = entry.get('system_fault_class', '')
        mapping = {
            'amp_error': '幅度失准',
            'freq_error': '频率失准',
            'ref_error': '参考电平失准',
        }
        return mapping.get(fault_cls, '正常')
    
    return '正常'


def extract_module_label(entry: Dict) -> Optional[int]:
    """Extract module-level label (single-label for now).
    
    Returns module ID (1-21) or None if not available.
    """
    if 'module_id' in entry and entry['module_id'] is not None:
        module_id = entry['module_id']
        if isinstance(module_id, int):
            return module_id
        if isinstance(module_id, (int, float)):
            return int(module_id)
        if isinstance(module_id, str):
            try:
                return int(module_id)
            except ValueError:
                if module_id in MODULE_LABELS:
                    return MODULE_LABELS.index(module_id)
    if 'y_mod' in entry:
        mod_val = entry['y_mod']
        if isinstance(mod_val, (int, float)):
            return int(mod_val)
    if 'module' in entry:
        if isinstance(entry['module'], (int, float)):
            return int(entry['module'])
        if isinstance(entry['module'], str) and entry['module'] in MODULE_LABELS:
            return MODULE_LABELS.index(entry['module'])
    return None


def detect_leakage_columns(feature_names: List[str]) -> List[str]:
    suspicious = []
    for name in feature_names:
        lower = name.lower()
        if lower.startswith(LEAK_PREFIXES):
            suspicious.append(name)
            continue
        if any(sub in lower for sub in LEAK_SUBSTRINGS):
            suspicious.append(name)
    return sorted(set(suspicious))


def load_expected_features(
    method_name: str,
    default_features: List[str],
    output_dir: Path,
) -> List[str]:
    expected_path = EXPECTED_FEATURES_DIR / f"{method_name}.json"
    if expected_path.exists():
        try:
            payload = json.loads(expected_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "features" in payload:
                features = payload["features"]
            else:
                features = payload
            if isinstance(features, list) and features:
                return [str(f) for f in features]
        except json.JSONDecodeError:
            pass
    fallback_path = output_dir / f"EXPECTED_FEATURES_{method_name}.json"
    if fallback_path.exists():
        try:
            payload = json.loads(fallback_path.read_text(encoding="utf-8"))
            features = payload.get("features", payload)
            if isinstance(features, list) and features:
                return [str(f) for f in features]
        except json.JSONDecodeError:
            pass
    return list(default_features)


def select_feature_matrix(
    X: np.ndarray,
    feature_names: List[str],
    expected_features: List[str],
    fill_value: float = 0.0,
) -> Tuple[np.ndarray, List[str]]:
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    missing = []
    selected = np.full((X.shape[0], len(expected_features)), fill_value, dtype=float)
    for j, name in enumerate(expected_features):
        idx = name_to_idx.get(name)
        if idx is None:
            missing.append(name)
            continue
        selected[:, j] = X[:, idx]
    return selected, missing


def prepare_dataset(
    data_dir: Path,
    use_pool_features: bool = False,
    strict_leakage: bool = True,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    List[str],
    List[str],
    List[str],
    List[str],
]:
    """Load and prepare dataset with features and labels.
    
    Args:
        data_dir: Directory containing dataset files
        use_pool_features: If True, use/generate pool features for broader feature set
        
    Returns:
        (X, y_sys, y_mod, feature_names, sample_ids)
    """
    # Discover files
    features_path, labels_path = discover_dataset_files(data_dir)
    
    if not features_path or not labels_path:
        raise FileNotFoundError(
            f"Could not find features or labels in {data_dir}. "
            f"Expected features_brb.csv and labels.json"
        )
    
    print(f"Loading features from: {features_path}")
    print(f"Loading labels from: {labels_path}")
    
    # Load data
    labels_dict = load_labels(labels_path)
    features_dict, csv_feature_names = load_features_csv(features_path)
    
    # If using pool features and raw curves available, augment features
    if use_pool_features:
        raw_curves_dir = data_dir / "raw_curves"
        if raw_curves_dir.exists():
            from features.feature_pool import augment_features_with_pool
            for sample_id, feats in features_dict.items():
                curve_path = raw_curves_dir / f"{sample_id}.csv"
                features_dict[sample_id] = augment_features_with_pool(feats, curve_path)
        else:
            # Synthesize pool features from existing features
            from features.feature_pool import _synthesize_pool_from_base
            for sample_id, feats in features_dict.items():
                pool_feats = _synthesize_pool_from_base(feats)
                features_dict[sample_id] = {**feats, **pool_feats}
    
    # Align samples (only use samples with both features and labels)
    feature_ids = set(features_dict.keys())
    label_ids = set(labels_dict.keys())
    common_ids = sorted(feature_ids & label_ids)
    missing_in_features = sorted(label_ids - feature_ids)
    missing_in_labels = sorted(feature_ids - label_ids)
    
    if not common_ids:
        raise ValueError("No common samples found between features and labels")
    
    print(f"Found {len(common_ids)} samples with both features and labels")
    if missing_in_features or missing_in_labels:
        print(f"[WARN] Missing in features: {len(missing_in_features)}")
        print(f"[WARN] Missing in labels: {len(missing_in_labels)}")
    
    # Build feature matrix and label vectors
    # Determine feature names from full union for consistent ordering
    if csv_feature_names:
        all_feature_names = list(csv_feature_names)
    else:
        all_feature_names = sorted({k for feats in features_dict.values() for k in feats.keys()})
    leak_columns = detect_leakage_columns(all_feature_names)
    if leak_columns and strict_leakage:
        raise LeakageError(leak_columns)
    if leak_columns:
        for feats in features_dict.values():
            for col in leak_columns:
                feats.pop(col, None)
    feature_names = [name for name in all_feature_names if name not in leak_columns]
    n_features = len(feature_names)
    n_samples = len(common_ids)
    
    X = np.zeros((n_samples, n_features))
    y_sys_list = []
    y_mod_list = []
    
    for i, sample_id in enumerate(common_ids):
        # Features
        feat_dict = features_dict[sample_id]
        for j, fname in enumerate(feature_names):
            X[i, j] = feat_dict.get(fname, 0.0)
        
        # Labels
        label_entry = labels_dict[sample_id]
        y_sys_list.append(extract_system_label(label_entry))
        y_mod_val = extract_module_label(label_entry)
        y_mod_list.append(y_mod_val if y_mod_val is not None else -1)
    
    # Convert system labels to numeric
    unique_sys_labels = list(SYS_LABEL_ORDER)
    sys_label_to_idx = {label: idx for idx, label in enumerate(unique_sys_labels)}
    y_sys = np.array([sys_label_to_idx.get(label, 0) for label in y_sys_list])
    
    # Module labels (may have -1 for missing)
    y_mod = np.array(y_mod_list)
    if np.all(y_mod == -1):
        y_mod = None  # No module labels available
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"System labels: {unique_sys_labels}")
    print(f"System label distribution: {np.bincount(y_sys, minlength=len(SYS_LABEL_ORDER))}")
    if y_mod is not None:
        print(f"Module labels available: {np.sum(y_mod >= 0)} samples")
    
    return (
        X,
        y_sys,
        y_mod,
        feature_names,
        common_ids,
        leak_columns,
        missing_in_features,
        missing_in_labels,
    )


def write_eval_audit(output_dir: Path, audit_info: Dict[str, object]) -> None:
    eval_audit_path = output_dir / "eval_audit.json"
    with open(eval_audit_path, 'w', encoding='utf-8') as f:
        audit_serializable = {}
        for k, v in audit_info.items():
            if isinstance(v, (list, dict, str, int, float, bool, type(None))):
                audit_serializable[k] = v
            elif hasattr(v, 'tolist'):
                audit_serializable[k] = v.tolist()
            else:
                audit_serializable[k] = str(v)
        json.dump(audit_serializable, f, indent=2, ensure_ascii=False)
    print(f"Saved evaluation audit to: {eval_audit_path}")


def _hash_ids(ids: List[str]) -> str:
    import hashlib

    h = hashlib.sha256()
    for item in ids:
        h.update(str(item).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _summarize_feature_by_class(
    X: np.ndarray,
    y_sys: np.ndarray,
    feature_names: List[str],
    labels: List[str],
    output_path: Path,
) -> None:
    key_features = [
        "gain",
        "bias",
        "comp",
        "df",
        "viol_rate",
        "step_score",
        "res_slope",
        "ripple_var",
        "X11",
        "X12",
        "X13",
        "X16",
        "X17",
        "X18",
    ]
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    summary = {}
    for class_idx, label in enumerate(labels):
        mask = y_sys == class_idx
        if not np.any(mask):
            continue
        summary[label] = {}
        for feat in key_features:
            idx = name_to_idx.get(feat)
            if idx is None:
                continue
            values = X[mask, idx]
            summary[label][feat] = {
                "mean": float(np.mean(values)),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
            }
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def stratified_split(X: np.ndarray, y: np.ndarray, 
                     train_size: float = 0.6, val_size: float = 0.2,
                     random_state: int = 42) -> Tuple[np.ndarray, ...]:
    """Stratified train/val/test split.
    
    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test, train_idx, val_idx, test_idx)
    """
    n_samples = len(X)
    n_classes = len(np.unique(y))
    
    # Create stratified splits
    indices = np.arange(n_samples)
    train_indices = []
    val_indices = []
    test_indices = []
    
    for class_label in np.unique(y):
        class_indices = indices[y == class_label]
        n_class = len(class_indices)
        
        # Shuffle class indices
        rng = np.random.RandomState(random_state)
        rng.shuffle(class_indices)
        
        # Split
        n_train = max(1, int(n_class * train_size))
        n_val = max(1, int(n_class * val_size))
        
        train_indices.extend(class_indices[:n_train])
        val_indices.extend(class_indices[n_train:n_train+n_val])
        test_indices.extend(class_indices[n_train+n_val:])
    
    # Convert to arrays and shuffle
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)
    
    rng = np.random.RandomState(random_state)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    
    return (
        X[train_indices], X[val_indices], X[test_indices],
        y[train_indices], y[val_indices], y[test_indices],
        train_indices, val_indices, test_indices
    )


# ============================================================================
# Metrics Calculation
# ============================================================================

def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy."""
    return float(np.mean(y_true == y_pred))


def calculate_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Calculate macro F1 score."""
    f1_scores = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    
    return float(np.mean(f1_scores))


def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Calculate confusion matrix."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    return cm


# ============================================================================
# Visualization
# ============================================================================

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                         output_path: Path, title: str):
    """Plot and save confusion matrix."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved confusion matrix to: {output_path}")
    except ImportError:
        print("matplotlib not available, skipping confusion matrix plot")


def plot_comparison_bar(results: List[Dict], output_dir: Path):
    """Plot comparison bar charts for rules, params, and inference time."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        methods = [r['method'] for r in results]
        rules = [r.get('n_rules', 0) for r in results]
        params = [r.get('n_params', 0) for r in results]
        infer_ms = [r.get('infer_ms_per_sample', 0) for r in results]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Rules
        axes[0].bar(methods, rules, color='skyblue')
        axes[0].set_ylabel('Number of Rules')
        axes[0].set_title('Model Complexity: Rules')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Params
        axes[1].bar(methods, params, color='lightcoral')
        axes[1].set_ylabel('Number of Parameters')
        axes[1].set_title('Model Complexity: Parameters')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Inference time
        axes[2].bar(methods, infer_ms, color='lightgreen')
        axes[2].set_ylabel('Inference Time (ms/sample)')
        axes[2].set_title('Inference Efficiency')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        output_path = output_dir / "compare_barplot.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved comparison bar plot to: {output_path}")
    except ImportError:
        print("matplotlib not available, skipping bar plot")


def plot_small_sample_curve(small_sample_results: List[Dict], output_dir: Path):
    """Plot small-sample learning curves."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Group by method
        methods_data = {}
        for entry in small_sample_results:
            method = entry['method']
            if method not in methods_data:
                methods_data[method] = {'sizes': [], 'means': [], 'stds': []}
            methods_data[method]['sizes'].append(entry['train_size'])
            methods_data[method]['means'].append(entry['mean_acc'])
            methods_data[method]['stds'].append(entry['std_acc'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for method, data in methods_data.items():
            sizes = np.array(data['sizes'])
            means = np.array(data['means'])
            stds = np.array(data['stds'])
            
            ax.plot(sizes, means, marker='o', label=method)
            ax.fill_between(sizes, means - stds, means + stds, alpha=0.2)
        
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Accuracy')
        ax.set_title('Small-Sample Adaptability')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        output_path = output_dir / "small_sample_curve.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved small-sample curve to: {output_path}")
    except ImportError:
        print("matplotlib not available, skipping small-sample curve plot")


def plot_comprehensive_comparison(all_results: List[Dict], output_dir: Path):
    """Plot comprehensive comparison visualization combining multiple metrics."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        methods = [r['method'] for r in all_results]
        accuracies = [r['sys_accuracy'] * 100 for r in all_results]  # Convert to percentage
        f1_scores = [r['sys_macro_f1'] * 100 for r in all_results]
        n_rules = [r['n_rules'] for r in all_results]
        n_params = [r['n_params'] for r in all_results]
        infer_ms = [r['infer_ms_per_sample'] for r in all_results]
        
        fig = plt.figure(figsize=(16, 10))
        
        # Create 2x3 grid
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Accuracy comparison
        ax1 = fig.add_subplot(gs[0, 0])
        bars1 = ax1.bar(methods, accuracies, color='steelblue', alpha=0.8)
        ax1.set_ylabel('System Accuracy (%)', fontsize=11)
        ax1.set_title('(a) Classification Accuracy', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        # Add value labels on bars
        for bar, val in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 2. F1-score comparison
        ax2 = fig.add_subplot(gs[0, 1])
        bars2 = ax2.bar(methods, f1_scores, color='coral', alpha=0.8)
        ax2.set_ylabel('Macro F1-Score (%)', fontsize=11)
        ax2.set_title('(b) F1-Score Performance', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars2, f1_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 3. Rules count
        ax3 = fig.add_subplot(gs[0, 2])
        bars3 = ax3.bar(methods, n_rules, color='mediumseagreen', alpha=0.8)
        ax3.set_ylabel('Number of Rules', fontsize=11)
        ax3.set_title('(c) Model Complexity (Rules)', fontsize=12, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars3, n_rules):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val}', ha='center', va='bottom', fontsize=9)
        
        # 4. Parameters count
        ax4 = fig.add_subplot(gs[1, 0])
        bars4 = ax4.bar(methods, n_params, color='mediumpurple', alpha=0.8)
        ax4.set_ylabel('Number of Parameters', fontsize=11)
        ax4.set_title('(d) Parameter Count', fontsize=12, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars4, n_params):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val}', ha='center', va='bottom', fontsize=9)
        
        # 5. Inference time
        ax5 = fig.add_subplot(gs[1, 1])
        bars5 = ax5.bar(methods, infer_ms, color='lightgreen', alpha=0.8)
        ax5.set_ylabel('Inference Time (ms/sample)', fontsize=11)
        ax5.set_title('(e) Inference Efficiency', fontsize=12, fontweight='bold')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars5, infer_ms):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 6. Accuracy vs Complexity scatter
        ax6 = fig.add_subplot(gs[1, 2])
        scatter = ax6.scatter(n_rules, accuracies, s=100, c=infer_ms, 
                             cmap='viridis', alpha=0.7, edgecolors='black', linewidth=1)
        ax6.set_xlabel('Number of Rules', fontsize=11)
        ax6.set_ylabel('Accuracy (%)', fontsize=11)
        ax6.set_title('(f) Accuracy vs Complexity', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        # Add method labels
        for i, method in enumerate(methods):
            ax6.annotate(method, (n_rules[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        # Add colorbar for inference time
        cbar = plt.colorbar(scatter, ax=ax6)
        cbar.set_label('Infer Time (ms)', fontsize=9)
        
        plt.suptitle('Comprehensive Method Comparison Results', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        output_path = output_dir / "comparison_results.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved comprehensive comparison plot to: {output_path}")
    except ImportError as e:
        print(f"matplotlib not available, skipping comprehensive plot: {e}")
    except Exception as e:
        print(f"Error creating comprehensive plot: {e}")


# ============================================================================
# Main Evaluation Pipeline
# ============================================================================

def evaluate_method(
    method,
    X_train,
    y_sys_train,
    y_mod_train,
    X_test,
    y_sys_test,
    y_mod_test,
    feature_names,
    n_sys_classes,
    sample_ids,
    test_idx,
    label_order,
    label_map,
):
    """Evaluate a single method."""
    print(f"\n{'='*60}")
    print(f"Evaluating method: {method.name}")
    print(f"{'='*60}")
    
    # Fit
    start_fit = time.time()
    meta_train = {'feature_names': feature_names}
    
    # V-E.3: Force training with explicit logging
    print(f"[FORCE] Training {method.name} with {len(X_train)} samples...")
    method.fit(X_train, y_sys_train, y_mod_train, meta_train)
    print(f"[FORCE] Training {method.name} complete in {time.time() - start_fit:.2f}s")
    
    fit_time = time.time() - start_fit
    
    # Predict
    start_infer = time.time()
    predictions = method.predict(X_test, meta={'feature_names': feature_names})
    infer_time_total = time.time() - start_infer
    infer_time_per_sample = (infer_time_total / len(X_test)) * 1000  # ms
    
    # Extract predictions
    y_sys_pred = predictions['system_pred']
    sys_proba = predictions.get('system_proba')
    y_mod_pred = predictions.get('module_pred', None)
    
    # System-level metrics
    sys_acc = calculate_accuracy(y_sys_test, y_sys_pred)
    sys_f1 = calculate_macro_f1(y_sys_test, y_sys_pred, n_sys_classes)
    sys_cm = calculate_confusion_matrix(y_sys_test, y_sys_pred, n_sys_classes)
    
    # Module-level metrics (if available)
    mod_acc = None
    mod_top3_acc = None
    if y_mod_pred is not None and y_mod_test is not None:
        valid_mask = y_mod_test >= 0
        if np.sum(valid_mask) > 0:
            mod_acc = calculate_accuracy(y_mod_test[valid_mask], y_mod_pred[valid_mask])
    
    # P2.2: Enhanced module metrics for "ours" method using module_topk predictions
    if method.name == 'ours' and 'module_proba' in predictions:
        mod_proba = predictions['module_proba']
        # Get top-3 module predictions per sample
        try:
            from BRB.module_brb import MODULE_LABELS_V2
            module_names = MODULE_LABELS_V2 if len(MODULE_LABELS_V2) > 0 else MODULE_LABELS
            
            # For each sample, convert gt module index to name and check matching
            n_correct_top1 = 0
            n_correct_top3 = 0
            n_valid = 0
            
            for i, (gt_idx, row) in enumerate(zip(y_mod_test, mod_proba)):
                if gt_idx < 0:
                    continue
                n_valid += 1
                
                # Get top-3 predicted indices
                top3_indices = np.argsort(row)[::-1][:3]
                
                # Simple index match for now
                if gt_idx in top3_indices[:1]:
                    n_correct_top1 += 1
                if gt_idx in top3_indices:
                    n_correct_top3 += 1
            
            if n_valid > 0:
                mod_acc = n_correct_top1 / n_valid
                mod_top3_acc = n_correct_top3 / n_valid
        except Exception:
            pass
    
    # Complexity metrics
    complexity = method.complexity()
    
    # Combine all results
    results = {
        'method': method.name,
        'sys_accuracy': sys_acc,
        'sys_macro_f1': sys_f1,
        'mod_top1_accuracy': mod_acc if mod_acc is not None else 0.0,
        'mod_top3_accuracy': mod_top3_acc if mod_top3_acc is not None else 0.0,
        'fit_time_sec': fit_time,
        'infer_ms_per_sample': infer_time_per_sample,
        'n_rules': complexity.get('n_rules', 0),
        'n_params': complexity.get('n_params', 0),
        'n_features_used': complexity.get('n_features_used', 0),
        'confusion_matrix': sys_cm,
        'sys_pred': y_sys_pred,
        'sys_proba': sys_proba,
    }
    
    # Update with method metadata
    if 'meta' in predictions:
        pred_meta = predictions['meta']
        results['features_used'] = pred_meta.get('features_used', [])
    
    print(f"System Accuracy: {sys_acc:.4f}")
    print(f"System Macro-F1: {sys_f1:.4f}")
    if mod_acc is not None:
        print(f"Module Top-1 Accuracy: {mod_acc:.4f}")
    print(f"Fit Time: {fit_time:.2f} sec")
    print(f"Inference Time: {infer_time_per_sample:.4f} ms/sample")
    print(f"Rules: {results['n_rules']}, Params: {results['n_params']}, Features: {results['n_features_used']}")

    if method.name == 'ours':
        print(f"Evaluating method: ours -> System Accuracy: {sys_acc:.4f}")

    # Mapping sanity check (sample a few predictions)
    rng = np.random.RandomState(2025)
    n_samples = min(10, len(y_sys_test))
    if n_samples > 0:
        indices = rng.choice(len(y_sys_test), size=n_samples, replace=False)
        print("\n[Mapping Check] sample predictions:")
        for idx in indices:
            true_label = label_order[y_sys_test[idx]]
            pred_label_raw = label_order[y_sys_pred[idx]]
            mapped_label = label_order[label_map.get(y_sys_pred[idx], y_sys_pred[idx])]
            sample_id = sample_ids[test_idx[idx]]
            print(
                f"  {sample_id}: true={true_label}, pred_raw={pred_label_raw}, mapped={mapped_label}"
            )
            if pred_label_raw != mapped_label:
                raise ValueError(
                    f"Label mapping mismatch for {method.name}: raw={pred_label_raw}, mapped={mapped_label}"
                )
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Comprehensive method comparison")
    parser.add_argument('--data_dir', default=SIM_DIR,
                       help='Directory containing dataset')
    parser.add_argument('--output_dir', default=COMPARE_DIR,
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    parser.add_argument('--train_size', type=float, default=SPLIT[0], help='Training set ratio')
    parser.add_argument('--val_size', type=float, default=SPLIT[1], help='Validation set ratio')
    parser.add_argument('--small_sample', action='store_true', 
                       help='Run small-sample adaptability experiments')
    parser.add_argument('--methods', type=str, default='all',
                       help='Comma-separated list of methods to run (e.g., "ours,hcf")')
    parser.add_argument('--input_dir', type=str, dest='data_dir',
                       help='Alias for --data_dir')
    parser.add_argument('--manifest', '-m', default=None,
                       help='Evaluation manifest path (if provided, only evaluate samples in manifest)')
    args = parser.parse_args()
    
    # Setup
    set_global_seed(args.seed)
    data_dir = Path(args.data_dir) if Path(args.data_dir).is_absolute() else PROJECT_ROOT / args.data_dir
    output_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    build_run_snapshot(output_dir)

    print(f"[INFO] project_root={PROJECT_ROOT}")
    print(f"[INFO] single_band={SINGLE_BAND}")
    print(f"[INFO] disable_preamp={DISABLE_PREAMP}")
    print(f"[INFO] seed={args.seed}")
    print(f"[INFO] output_dir={output_dir}")
    
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load dataset
    print("\n" + "="*60)
    print("Loading dataset...")
    print("="*60)
    audit_info = {
        'data_dir': str(data_dir),
        'output_dir': str(output_dir),
        'seed': args.seed,
        'single_band': SINGLE_BAND,
        'disable_preamp': DISABLE_PREAMP,
    }
    try:
        (
            X,
            y_sys,
            y_mod,
            feature_names,
            sample_ids,
            leak_columns,
            missing_in_features,
            missing_in_labels,
        ) = prepare_dataset(
            data_dir,
            use_pool_features=True,
            strict_leakage=True,
        )
    except LeakageError as exc:
        audit_info['leakage_detected'] = True
        audit_info['leakage_columns'] = exc.columns
        write_eval_audit(output_dir, audit_info)
        raise SystemExit(1) from exc
    else:
        audit_info['leakage_detected'] = False
        audit_info['leakage_columns'] = leak_columns
    
    # P0.2: Filter by manifest if provided
    manifest_sample_ids = None
    if args.manifest:
        manifest_path = Path(args.manifest) if Path(args.manifest).is_absolute() else PROJECT_ROOT / args.manifest
        if manifest_path.exists():
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
            manifest_sample_ids = set(manifest_data.get("sample_ids", []))
            print(f"[INFO] Manifest loaded: {manifest_path}")
            print(f"[INFO] Manifest sample count: {len(manifest_sample_ids)}")
            
            # Filter samples by manifest
            if manifest_sample_ids:
                mask = np.array([sid in manifest_sample_ids for sid in sample_ids])
                X = X[mask]
                y_sys = y_sys[mask]
                if y_mod is not None:
                    y_mod = y_mod[mask]
                sample_ids = [sid for sid, m in zip(sample_ids, mask) if m]
                print(f"[INFO] After manifest filter: {len(sample_ids)} samples")
        else:
            print(f"[WARN] Manifest file not found: {manifest_path}")
    
    n_sys_classes = len(SYS_LABEL_ORDER)
    
    # Audit tracking: count samples at each stage
    audit_info.update({
        'n_labels_total': len(sample_ids),
        'n_features_rows': len(X),
        'n_joined': len(sample_ids),  # After joining features and labels
        'feature_names': feature_names,
        'n_sys_classes': n_sys_classes,
        'missing_in_features': missing_in_features,
        'missing_in_labels': missing_in_labels,
    })

    raw_curves_dir = data_dir / "raw_curves"
    raw_curves_count = len(list(raw_curves_dir.glob("*.csv"))) if raw_curves_dir.exists() else 0

    def _hash_file(path: Path) -> str | None:
        if not path.exists():
            return None
        import hashlib
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    # Feature integrity report
    features_path = data_dir / "features_brb.csv"
    feature_report_path = output_dir / "feature_integrity_report.json"
    if features_path.exists():
        feature_report = generate_feature_integrity_report(features_path, feature_report_path)
        print(f"[INFO] Feature integrity report saved: {feature_report_path}")
    else:
        feature_report = {}
        print(f"[WARN] features_brb.csv not found at {features_path}")

    # Dataset feature summary for sanity checks
    dataset_feature_summary_path = output_dir / "dataset_feature_summary.json"
    _summarize_feature_by_class(X, y_sys, feature_names, SYS_LABEL_ORDER, dataset_feature_summary_path)
    print(f"[INFO] Dataset feature summary saved: {dataset_feature_summary_path}")

    baseline_stats = _compute_baseline_feature_stats(PROJECT_ROOT / "Output/baseline_artifacts.npz")
    separation_rows = _compute_feature_separation(X, y_sys, feature_names, SYS_LABEL_ORDER)
    _write_feature_separation_reports(
        output_dir,
        feature_names,
        separation_rows,
        baseline_stats,
        X,
        y_sys,
        SYS_LABEL_ORDER,
    )
    print(f"[INFO] Feature separation report saved: {output_dir / 'baseline_vs_sim_feature_separation.json'}")

    repro_summary = {
        "data_dir": str(data_dir),
        "features_rows": len(X),
        "labels_count": len(sample_ids),
        "raw_curves_count": raw_curves_count,
        "missing_in_features": missing_in_features[:50],
        "missing_in_labels": missing_in_labels[:50],
        "missing_in_features_count": len(missing_in_features),
        "missing_in_labels_count": len(missing_in_labels),
        "label_distribution": {
            label: int(np.sum(y_sys == idx))
            for idx, label in enumerate(SYS_LABEL_ORDER)
        },
        "class_index_map": {str(idx): label for idx, label in enumerate(SYS_LABEL_ORDER)},
        "feature_integrity_report": str(feature_report_path),
        "best_params": {
            "path": str(data_dir / "best_params.json"),
            "exists": (data_dir / "best_params.json").exists(),
            "sha256": _hash_file(data_dir / "best_params.json"),
        },
        "single_band": SINGLE_BAND,
        "disable_preamp": DISABLE_PREAMP,
        "seed": args.seed,
    }
    repro_path = output_dir / "repro_check_summary.json"
    repro_path.write_text(json.dumps(repro_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved repro_check_summary.json to: {repro_path}")
    
    # Split dataset (reuse saved splits for fairness)
    print("\n" + "="*60)
    print("Splitting dataset...")
    print("="*60)
    split_path = output_dir / "split_indices.json"
    if split_path.exists():
        split_payload = json.loads(split_path.read_text(encoding="utf-8"))
        train_idx = np.array(split_payload["train_idx"], dtype=int)
        val_idx = np.array(split_payload["val_idx"], dtype=int)
        test_idx = np.array(split_payload["test_idx"], dtype=int)
        X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
        y_sys_train, y_sys_val, y_sys_test = y_sys[train_idx], y_sys[val_idx], y_sys[test_idx]
        print(f"[INFO] Loaded split indices from: {split_path}")
    else:
        X_train, X_val, X_test, y_sys_train, y_sys_val, y_sys_test, train_idx, val_idx, test_idx = \
            stratified_split(X, y_sys, args.train_size, args.val_size, args.seed)
        split_payload = {
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
            "test_idx": test_idx.tolist(),
        }
        split_path.write_text(json.dumps(split_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] Saved split indices to: {split_path}")
    
    y_mod_train = y_mod[train_idx] if y_mod is not None else None
    y_mod_val = y_mod[val_idx] if y_mod is not None else None
    y_mod_test = y_mod[test_idx] if y_mod is not None else None
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Val set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Train hash: {_hash_ids([sample_ids[i] for i in train_idx])}")
    print(f"Val hash: {_hash_ids([sample_ids[i] for i in val_idx])}")
    print(f"Test hash: {_hash_ids([sample_ids[i] for i in test_idx])}")
    
    # Update audit info with split sizes
    audit_info['n_train'] = len(X_train)
    audit_info['n_val'] = len(X_val)
    audit_info['n_test'] = len(X_test)
    audit_info['train_indices'] = train_idx.tolist()
    audit_info['val_indices'] = val_idx.tolist()
    audit_info['test_indices'] = test_idx.tolist()
    audit_info['train_sample_ids'] = [sample_ids[i] for i in train_idx]
    audit_info['val_sample_ids'] = [sample_ids[i] for i in val_idx]
    audit_info['test_sample_ids'] = [sample_ids[i] for i in test_idx]
    repro_summary["splits"] = {
        "train_count": len(train_idx),
        "val_count": len(val_idx),
        "test_count": len(test_idx),
        "train_hash": _hash_ids(audit_info["train_sample_ids"]),
        "val_hash": _hash_ids(audit_info["val_sample_ids"]),
        "test_hash": _hash_ids(audit_info["test_sample_ids"]),
    }
    repro_path.write_text(json.dumps(repro_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # Import methods (will be implemented)
    print("\n" + "="*60)
    print("Importing methods...")
    print("="*60)
    
    # For now, create placeholder - will implement actual methods
    from methods.ours_adapter import OursAdapter
    from methods.hcf_adapter import HCFAdapter
    from methods.brb_p_adapter import BRBPAdapter
    from methods.brb_mu_adapter import BRBMUAdapter
    from methods.dbrb_adapter import DBRBAdapter
    from methods.a_ibrb_adapter import AIBRBAdapter
    
    best_params_path = data_dir / "best_params.json"
    best_params = None
    if best_params_path.exists():
        try:
            best_params = json.loads(best_params_path.read_text(encoding="utf-8"))
            print(f"[INFO] Loaded best params: {best_params_path}")
        except json.JSONDecodeError:
            print(f"[WARN] Failed to parse best params: {best_params_path}")
            best_params = None
    else:
        print(f"[INFO] No best_params.json found at {best_params_path}")

    methods = [
        OursAdapter(calibration_override=best_params),
        HCFAdapter(),
        BRBPAdapter(),
        BRBMUAdapter(),
        DBRBAdapter(),
        AIBRBAdapter(),
    ]
    
    # Filter methods based on --methods argument
    if args.methods != 'all':
        method_list = [m.strip() for m in args.methods.split(',')]
        methods = [m for m in methods if m.name in method_list]
        print(f"[INFO] Running selected methods: {[m.name for m in methods]}")
    
    # Evaluate all methods
    all_results = []
    method_feature_usage: Dict[str, Dict[str, object]] = {}
    for method in methods:
        try:
            expected_features = load_expected_features(method.name, feature_names, output_dir)
            X_train_sel, missing_train = select_feature_matrix(X_train, feature_names, expected_features)
            X_test_sel, missing_test = select_feature_matrix(X_test, feature_names, expected_features)
            missing_features = sorted(set(missing_train + missing_test))
            if missing_features:
                print(
                    f"[WARN] {method.name} missing {len(missing_features)} features, "
                    f"filling with 0.0"
                )
            print(
                f"[INFO] {method.name} features: {len(expected_features)} "
                f"(first 10: {expected_features[:10]})"
            )
            results = evaluate_method(
                method,
                X_train_sel,
                y_sys_train,
                y_mod_train,
                X_test_sel,
                y_sys_test,
                y_mod_test,
                expected_features,
                n_sys_classes,
                sample_ids,
                test_idx,
                SYS_LABEL_ORDER,
                {i: i for i in range(n_sys_classes)},
            )
            all_results.append(results)
            audit_info[f"{method.name}_features"] = expected_features
            audit_info[f"{method.name}_missing_features"] = missing_features
            audit_info[f"{method.name}_label_map"] = {i: i for i in range(n_sys_classes)}
            used = results.get("features_used", [])
            used_names: List[str] = []
            used_serializable: List[object] = []
            if used and all(isinstance(idx, (int, np.integer)) for idx in used):
                used_serializable = [int(idx) for idx in used]
                used_names = [
                    expected_features[int(idx)]
                    for idx in used_serializable
                    if int(idx) < len(expected_features)
                ]
            elif used:
                used_serializable = [str(name) for name in used]
                used_names = [str(name) for name in used]
            method_feature_usage[method.name] = {
                "expected_features": expected_features,
                "missing_features": missing_features,
                "features_used": used_serializable,
                "features_used_names": used_names,
            }
            
            # Validate confusion matrix sum
            cm = results.get('confusion_matrix', None)
            if cm is not None:
                cm_sum = np.sum(cm)
                if cm_sum != len(X_test):
                    print(f"WARNING: {method.name} confusion matrix sum ({cm_sum}) != test size ({len(X_test)})")
                    audit_info[f'{method.name}_cm_mismatch'] = {
                        'cm_sum': int(cm_sum),
                        'expected': len(X_test),
                        'difference': len(X_test) - int(cm_sum),
                    }
        except Exception as e:
            print(f"Error evaluating {method.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Update audit with test used count
    audit_info['n_test_used'] = len(X_test)
    audit_info['n_dropped'] = 0  # All samples used in current implementation
    audit_info['drop_reasons'] = []  # No drops in current implementation
    
    # Save comparison table
    print("\n" + "="*60)
    print("Saving results...")
    print("="*60)
    
    def _extract_row(result: Dict, fieldnames: List[str]) -> Dict:
        """Extract row data with default values for missing keys."""
        return {k: result.get(k, 0.0) for k in fieldnames}
    
    comparison_path = output_dir / "comparison_table.csv"
    with open(comparison_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['method', 'sys_accuracy', 'sys_macro_f1', 'mod_top1_accuracy', 'mod_top3_accuracy',
                     'fit_time_sec', 'infer_ms_per_sample', 
                     'n_rules', 'n_params', 'n_features_used']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(_extract_row(result, fieldnames))
    print(f"Saved comparison table to: {comparison_path}")

    performance_path = output_dir / "performance_table.csv"
    with open(performance_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['method', 'sys_accuracy', 'sys_macro_f1', 'mod_top1_accuracy', 'mod_top3_accuracy']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(_extract_row(result, fieldnames))
    print(f"Saved performance table to: {performance_path}")

    summary_json_path = output_dir / "comparison_summary.json"
    try:
        def _serialize(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            if isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_serialize(v) for v in obj]
            return obj

        summary_payload = {
            "methods": [_serialize(result) for result in all_results],
            "label_order": SYS_LABEL_ORDER,
        }
        summary_json_path.write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Saved comparison summary to: {summary_json_path}")
    except Exception as exc:
        print(f"[WARN] Failed to save comparison summary JSON: {exc}")

    if feature_report_path.exists():
        feature_report = json.loads(feature_report_path.read_text(encoding="utf-8"))
        feature_report["method_feature_usage"] = method_feature_usage
        feature_report_path.write_text(
            json.dumps(feature_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    
    # Plot confusion matrices
    sys_label_names = SYS_LABEL_ORDER[:n_sys_classes]
    for result in all_results:
        cm_path = output_dir / f"confusion_matrix_{result['method']}.png"
        plot_confusion_matrix(
            result['confusion_matrix'], 
            sys_label_names,
            cm_path,
            f"System-Level Confusion Matrix: {result['method']}"
        )
    
    # Plot comparison bars
    plot_comparison_bar(all_results, output_dir)
    
    # Plot comprehensive comparison
    plot_comprehensive_comparison(all_results, output_dir)
    
    # Small-sample experiments
    if args.small_sample:
        print("\n" + "="*60)
        print("Running small-sample adaptability experiments...")
        print("="*60)
        
        train_sizes = [5, 10, 20, 30]
        n_repeats = 5
        small_sample_results = []
        
        for train_size in train_sizes:
            if train_size > len(X_train):
                print(f"Skipping train_size={train_size} (exceeds available training data)")
                continue
            
            print(f"\nTrain size: {train_size}")
            
            for method in methods:
                method_accs = []
                expected_features = load_expected_features(method.name, feature_names, output_dir)
                X_train_sel, _ = select_feature_matrix(X_train, feature_names, expected_features)
                X_test_sel, _ = select_feature_matrix(X_test, feature_names, expected_features)
                
                for rep in range(n_repeats):
                    # Sample subset
                    rep_seed = args.seed + rep
                    rng = np.random.RandomState(rep_seed)
                    indices = rng.choice(len(X_train), size=train_size, replace=False)
                    X_small = X_train_sel[indices]
                    y_small = y_sys_train[indices]
                    
                    # Train and evaluate
                    try:
                        method_copy = method.__class__()  # Create fresh instance
                        method_copy.fit(X_small, y_small, None, {'feature_names': expected_features})
                        pred = method_copy.predict(X_test_sel, {'feature_names': expected_features})
                        acc = calculate_accuracy(y_sys_test, pred['system_pred'])
                        method_accs.append(acc)
                    except Exception as e:
                        print(f"Error in small-sample experiment for {method.name}: {e}")
                        method_accs.append(0.0)
                
                mean_acc = np.mean(method_accs)
                std_acc = np.std(method_accs)
                small_sample_results.append({
                    'method': method.name,
                    'train_size': train_size,
                    'mean_acc': mean_acc,
                    'std_acc': std_acc,
                })
                print(f"  {method.name}: {mean_acc:.4f} ± {std_acc:.4f}")
        
        # Save small-sample results
        small_sample_path = output_dir / "small_sample_curve.csv"
        with open(small_sample_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['method', 'train_size', 'mean_acc', 'std_acc']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(small_sample_results)
        print(f"Saved small-sample results to: {small_sample_path}")
        
        # Plot small-sample curve
        plot_small_sample_curve(small_sample_results, output_dir)
    
    # Save eval_audit.json (mandatory for sample tracking)
    print("\n" + "="*60)
    print("Saving audit information...")
    print("="*60)
    
    write_eval_audit(output_dir, audit_info)
    
    # Verify confusion matrix sum equals test size
    for result in all_results:
        cm = result.get('confusion_matrix', None)
        if cm is not None:
            cm_sum = int(np.sum(cm))
            expected = len(X_test)
            if cm_sum != expected:
                raise ValueError(
                    f"Confusion matrix sum mismatch for {result['method']}: "
                    f"got {cm_sum}, expected {expected}. "
                    f"See eval_audit.json for details."
                )
    
    # Save ours-specific outputs
    ours_result = None
    for result in all_results:
        if result['method'] == 'ours':
            ours_result = result
            break
    
    if ours_result is not None:
        # Save ours confusion matrix CSV
        cm_csv_path = output_dir / "ours_confusion_matrix.csv"
        cm = ours_result['confusion_matrix']
        with open(cm_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            sys_label_names = SYS_LABEL_ORDER[:n_sys_classes]
            writer.writerow(['Predicted\\Actual'] + sys_label_names)
            for i, row in enumerate(cm):
                writer.writerow([sys_label_names[i]] + list(row))
        print(f"Saved ours confusion matrix to: {cm_csv_path}")

        # Save per-sample predictions (system-level)
        predictions_path = output_dir / "predictions_ours.csv"
        sys_pred = ours_result.get('sys_pred')
        sys_proba = ours_result.get('sys_proba')
        if sys_pred is not None:
            with open(predictions_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    "sample_id",
                    "true_label",
                    "pred_label",
                    "true_index",
                    "pred_index",
                ]
                if sys_proba is not None:
                    fieldnames.extend([f"prob_{label}" for label in SYS_LABEL_ORDER[:n_sys_classes]])
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for idx, pred in enumerate(sys_pred):
                    true_idx = int(y_sys_test[idx])
                    row = {
                        "sample_id": sample_ids[test_idx[idx]],
                        "true_label": SYS_LABEL_ORDER[true_idx],
                        "pred_label": SYS_LABEL_ORDER[int(pred)],
                        "true_index": true_idx,
                        "pred_index": int(pred),
                    }
                    if sys_proba is not None:
                        probs = sys_proba[idx]
                        for j, label in enumerate(SYS_LABEL_ORDER[:n_sys_classes]):
                            row[f"prob_{label}"] = float(probs[j]) if j < len(probs) else 0.0
                    writer.writerow(row)
            print(f"Saved ours predictions to: {predictions_path}")
        
        # v7: Save confusion matrix counts (separate file for clarity)
        cm_counts_path = output_dir / "ours_confusion_matrix_counts.csv"
        with open(cm_counts_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            sys_label_names = SYS_LABEL_ORDER[:n_sys_classes]
            # Header row
            writer.writerow(['True\\Pred'] + sys_label_names + ['Total'])
            for i, label in enumerate(sys_label_names):
                row_total = int(np.sum(cm[i, :]))
                writer.writerow([label] + [int(cm[i, j]) for j in range(len(sys_label_names))] + [row_total])
            # Column totals
            col_totals = [int(np.sum(cm[:, j])) for j in range(len(sys_label_names))]
            writer.writerow(['Pred Total'] + col_totals + [int(np.sum(cm))])
        print(f"Saved ours confusion matrix counts to: {cm_counts_path}")
        
        # Save per-class metrics
        per_class_path = output_dir / "ours_per_class_metrics.csv"
        with open(per_class_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['class', 'precision', 'recall', 'f1', 'support'])
            writer.writeheader()
            for i, label in enumerate(sys_label_names):
                tp = cm[i, i]
                fp = np.sum(cm[:, i]) - tp
                fn = np.sum(cm[i, :]) - tp
                support = np.sum(cm[i, :])
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                writer.writerow({
                    'class': label,
                    'precision': f'{precision:.4f}',
                    'recall': f'{recall:.4f}',
                    'f1': f'{f1:.4f}',
                    'support': int(support),
                })
        print(f"Saved ours per-class metrics to: {per_class_path}")

        # Save ours error cases for debugging
        error_path = output_dir / "ours_error_cases.csv"
        try:
            from BRB.normal_anchor import compute_anchor_score, NormalAnchorConfig
        except ImportError:
            compute_anchor_score = None
            NormalAnchorConfig = None
        error_rows = []
        sys_pred = ours_result.get('sys_pred')
        sys_proba = ours_result.get('sys_proba')
        if sys_pred is not None:
            for idx, pred in enumerate(sys_pred):
                true_label = y_sys_test[idx]
                if pred == true_label:
                    continue
                sample_id = sample_ids[test_idx[idx]]
                row = {
                    "sample_id": sample_id,
                    "true_label": SYS_LABEL_ORDER[true_label],
                    "pred_label": SYS_LABEL_ORDER[pred],
                }
                if sys_proba is not None:
                    probs = sys_proba[idx]
                    pmax = float(np.max(probs))
                    margin = float(np.sort(probs)[-1] - np.sort(probs)[-2]) if len(probs) > 1 else pmax
                    row["pmax"] = pmax
                    row["margin"] = margin
                feature_dict = dict(zip(feature_names, X_test[idx]))
                if compute_anchor_score is not None:
                    anchor_result = compute_anchor_score(feature_dict, NormalAnchorConfig())
                    row["anchor_score"] = anchor_result.get("anchor_score", 0.0)
                    row["score_amp"] = anchor_result.get("score_amp", 0.0)
                    row["score_freq"] = anchor_result.get("score_freq", 0.0)
                    row["score_ref"] = anchor_result.get("score_ref", 0.0)
                if "X14" in feature_dict:
                    row["X14"] = feature_dict.get("X14", 0.0)
                error_rows.append(row)
        if error_rows:
            with open(error_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = sorted({k for row in error_rows for k in row.keys()})
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(error_rows)
            print(f"Saved ours error cases to: {error_path}")
    
    # Save dataset class distribution
    dist_path = output_dir / "dataset_class_distribution.csv"
    with open(dist_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'train_count', 'val_count', 'test_count', 'total'])
        sys_label_names = SYS_LABEL_ORDER
        for i, label in enumerate(sys_label_names):
            if i >= n_sys_classes:
                break
            train_count = np.sum(y_sys_train == i)
            val_count = np.sum(y_sys_val == i)
            test_count = np.sum(y_sys_test == i)
            total = train_count + val_count + test_count
            writer.writerow([label, train_count, val_count, test_count, total])
    print(f"Saved dataset class distribution to: {dist_path}")

    # Baseline fairness report
    fairness_dir = PROJECT_ROOT / "Output" / "baseline_audit"
    fairness_dir.mkdir(parents=True, exist_ok=True)
    fairness_path = fairness_dir / "baseline_fairness_report.md"
    with fairness_path.open("w", encoding="utf-8") as f:
        f.write("# Baseline Fairness Report\n\n")
        f.write(f"- data_dir: {data_dir}\n")
        f.write(f"- split_indices: {split_path}\n")
        f.write(f"- seed: {args.seed}\n\n")
        f.write("| method | sys_acc | macro_f1 | mod_top1 | mod_top3 | n_rules | n_params | n_features | missing_features |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for result in all_results:
            usage = method_feature_usage.get(result["method"], {})
            missing = usage.get("missing_features", [])
            f.write(
                f"| {result['method']} | {result.get('sys_accuracy', 0.0):.4f} | "
                f"{result.get('sys_macro_f1', 0.0):.4f} | {result.get('mod_top1_accuracy', 0.0):.4f} | "
                f"{result.get('mod_top3_accuracy', 0.0):.4f} | "
                f"{result.get('n_rules', 0)} | {result.get('n_params', 0)} | "
                f"{result.get('n_features_used', 0)} | {len(missing)} |\n"
            )
        f.write("\n## Confusion Matrices\n")
        for result in all_results:
            f.write(f"- confusion_matrix_{result['method']}.png\n")
    print(f"Saved baseline fairness report to: {fairness_path}")
    
    # P6: Generate provenance file for traceability
    provenance = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "manifest_path": getattr(args, 'manifest', None),
        "n_eval": len(X_test),
        "n_train": len(X_train),
        "seed": args.seed,
        "sample_ids_first10": sample_ids[:10] if sample_ids else [],
        "labels_path": str(data_dir / "labels.json"),
        "features_path": str(data_dir / "features_brb.csv"),
    }
    provenance_path = output_dir / "metrics_provenance.json"
    with open(provenance_path, 'w', encoding='utf-8') as f:
        json.dump(provenance, f, indent=2, ensure_ascii=False)
    print(f"Saved provenance to: {provenance_path}")
    
    print("\n" + "="*60)
    print("Comparison complete!")
    print(f"N_eval (test set): {len(X_test)}")
    print("="*60)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Discriminability Report

Analyzes feature discriminability per fault type using AUC, Fisher score, etc.

Usage:
    python tools/feature_discriminability_report.py --features <features.csv> --labels <labels.json>
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple


def compute_fisher_score(X: np.ndarray, y: np.ndarray, feature_idx: int) -> float:
    """Compute Fisher score for a single feature."""
    classes = np.unique(y)
    n_classes = len(classes)
    
    overall_mean = X[:, feature_idx].mean()
    overall_var = X[:, feature_idx].var()
    
    if overall_var < 1e-10:
        return 0.0
    
    between_class_var = 0.0
    within_class_var = 0.0
    
    for c in classes:
        mask = (y == c)
        n_c = mask.sum()
        if n_c == 0:
            continue
        
        class_mean = X[mask, feature_idx].mean()
        class_var = X[mask, feature_idx].var()
        
        between_class_var += n_c * (class_mean - overall_mean) ** 2
        within_class_var += n_c * class_var
    
    if within_class_var < 1e-10:
        return float('inf')
    
    return between_class_var / within_class_var


def compute_auc_per_class(X: np.ndarray, y: np.ndarray, feature_idx: int, target_class: int) -> float:
    """Compute AUC for a single feature vs one class (binary)."""
    try:
        from sklearn.metrics import roc_auc_score
        
        y_binary = (y == target_class).astype(int)
        feature_vals = X[:, feature_idx]
        
        # Handle constant features
        if feature_vals.std() < 1e-10:
            return 0.5
        
        return roc_auc_score(y_binary, feature_vals)
    except:
        return 0.5


def analyze_feature_discriminability(
    features_df: pd.DataFrame,
    labels_data: List[Dict],
    output_dir: Path
) -> Dict[str, Any]:
    """Analyze feature discriminability and generate report."""
    
    # Extract feature matrix
    feature_cols = [c for c in features_df.columns if c.startswith('X') or c.startswith('feat')]
    if not feature_cols:
        feature_cols = [c for c in features_df.columns if c not in ['sample_id', 'Unnamed: 0']]
    
    X = features_df[feature_cols].values
    n_features = X.shape[1]
    
    # Get labels
    label_map = {"normal": 0, "amp_error": 1, "freq_error": 2, "ref_error": 3}
    y = np.array([
        label_map.get(item.get("system_fault_class", item.get("fault_type", "normal")).lower().strip(), 0)
        for item in labels_data
    ])
    
    # Ensure matching lengths
    min_len = min(len(X), len(y))
    X = X[:min_len]
    y = y[:min_len]
    
    fault_types = ["normal", "amp_error", "freq_error", "ref_error"]
    
    report = {
        "feature_names": feature_cols,
        "n_features": n_features,
        "n_samples": len(y),
        "class_distribution": {ft: int((y == i).sum()) for i, ft in enumerate(fault_types)},
        "fisher_scores": {},
        "auc_per_class": {},
        "top_features_per_class": {},
        "layer_structure_suggestion": {},
    }
    
    # Compute Fisher scores
    fisher_scores = []
    for i in range(n_features):
        score = compute_fisher_score(X, y, i)
        fisher_scores.append(score)
        report["fisher_scores"][feature_cols[i]] = float(score) if not np.isinf(score) else 999.0
    
    # Compute AUC per class
    for class_idx, class_name in enumerate(fault_types):
        if class_name == "normal":
            continue
        
        auc_scores = []
        for i in range(n_features):
            auc = compute_auc_per_class(X, y, i, class_idx)
            auc_scores.append((feature_cols[i], auc))
        
        # Sort by AUC (descending)
        auc_scores.sort(key=lambda x: abs(x[1] - 0.5), reverse=True)
        
        report["auc_per_class"][class_name] = {name: float(auc) for name, auc in auc_scores}
        report["top_features_per_class"][class_name] = [name for name, _ in auc_scores[:5]]
    
    # Generate layer structure suggestion
    for class_name in ["amp_error", "freq_error", "ref_error"]:
        if class_name not in report["top_features_per_class"]:
            continue
        
        top_features = report["top_features_per_class"][class_name]
        
        report["layer_structure_suggestion"][class_name] = {
            "L1_strongest": top_features[:2] if len(top_features) >= 2 else top_features,
            "L2_secondary": top_features[2:4] if len(top_features) >= 4 else [],
            "L3_detail": top_features[4:] if len(top_features) > 4 else [],
        }
    
    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "feature_discriminability.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Generate markdown
    md_lines = [
        "# Feature Discriminability Report",
        "",
        f"**Samples**: {report['n_samples']}",
        f"**Features**: {report['n_features']}",
        "",
        "## Class Distribution",
        "",
    ]
    
    for ft, count in report["class_distribution"].items():
        md_lines.append(f"- {ft}: {count}")
    
    md_lines.extend([
        "",
        "## Top Features per Fault Type (by AUC)",
        "",
    ])
    
    for class_name in ["amp_error", "freq_error", "ref_error"]:
        if class_name not in report["top_features_per_class"]:
            continue
        
        md_lines.append(f"### {class_name}")
        md_lines.append("")
        md_lines.append("| Rank | Feature | AUC |")
        md_lines.append("|------|---------|-----|")
        
        auc_data = report["auc_per_class"][class_name]
        for i, (feat, auc) in enumerate(list(auc_data.items())[:10]):
            md_lines.append(f"| {i+1} | {feat} | {auc:.3f} |")
        
        md_lines.append("")
    
    md_lines.extend([
        "## Layer Structure Suggestion (DBRB-style)",
        "",
    ])
    
    for class_name, layers in report["layer_structure_suggestion"].items():
        md_lines.append(f"### {class_name}")
        md_lines.append(f"- **L1 (Strongest)**: {', '.join(layers['L1_strongest'])}")
        md_lines.append(f"- **L2 (Secondary)**: {', '.join(layers['L2_secondary']) or 'N/A'}")
        md_lines.append(f"- **L3 (Detail)**: {', '.join(layers['L3_detail']) or 'N/A'}")
        md_lines.append("")
    
    with open(output_dir / "feature_discriminability.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print('\n'.join(md_lines))
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Feature discriminability analysis")
    parser.add_argument("--features", default="Output/sim_spectrum/features_brb.csv",
                        help="Path to features CSV")
    parser.add_argument("--labels", default="Output/sim_spectrum/labels.json",
                        help="Path to labels JSON")
    parser.add_argument("--output", default="Output/debug",
                        help="Output directory")
    args = parser.parse_args()
    
    features_path = Path(args.features)
    labels_path = Path(args.labels)
    output_dir = Path(args.output)
    
    if not features_path.exists():
        print(f"Error: Features file not found: {features_path}")
        return 1
    
    if not labels_path.exists():
        print(f"Error: Labels file not found: {labels_path}")
        return 1
    
    print("Loading features and labels...")
    features_df = pd.read_csv(features_path)
    
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)
    
    print(f"Loaded {len(features_df)} feature rows, {len(labels_data)} labels")
    
    analyze_feature_discriminability(features_df, labels_data, output_dir)
    
    print(f"\nReport saved to: {output_dir}/feature_discriminability.md")
    return 0


if __name__ == "__main__":
    exit(main())

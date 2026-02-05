#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RF Probability Calibration Tool

Calibrates Random Forest probabilities using Isotonic Regression or Platt Scaling.

Usage:
    python tools/calibrate_rf_probs.py --features <features.csv> --labels <labels.json>
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import warnings

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    import joblib
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("sklearn not available, calibration will use simple scaling")


def load_rf_probs(features_path: Path, labels_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load RF probabilities and true labels."""
    import pandas as pd
    
    # Load features
    features_df = pd.read_csv(features_path)
    
    # Load labels
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)
    
    # Get RF predictions from artifact
    rf_artifact_path = Path("artifacts/rf_model.joblib")
    if rf_artifact_path.exists() and HAS_SKLEARN:
        rf_model = joblib.load(rf_artifact_path)
        
        # Extract feature columns
        feature_cols = [c for c in features_df.columns if c.startswith('X') or c.startswith('feat')]
        if not feature_cols:
            feature_cols = features_df.columns[1:].tolist()  # Skip sample_id
        
        X = features_df[feature_cols].values
        rf_probs = rf_model.predict_proba(X)
    else:
        # Fallback: use uniform probs
        n_samples = len(features_df)
        rf_probs = np.ones((n_samples, 4)) * 0.25
    
    # Get true labels
    label_map = {"normal": 0, "amp_error": 1, "freq_error": 2, "ref_error": 3}
    y_true = []
    
    for item in labels_data:
        fault_type = item.get("system_fault_class", item.get("fault_type", "normal"))
        fault_type = fault_type.lower().strip()
        y_true.append(label_map.get(fault_type, 0))
    
    return rf_probs, np.array(y_true)


def calibrate_isotonic(rf_probs: np.ndarray, y_true: np.ndarray) -> Dict[int, Any]:
    """Calibrate probabilities using Isotonic Regression (per class)."""
    if not HAS_SKLEARN:
        return {}
    
    n_classes = rf_probs.shape[1]
    calibrators = {}
    
    for c in range(n_classes):
        # Binary indicator for this class
        y_binary = (y_true == c).astype(int)
        probs_c = rf_probs[:, c]
        
        # Fit isotonic regression
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(probs_c, y_binary)
        calibrators[c] = iso
    
    return calibrators


def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    n_samples = len(y_true)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y_true)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
    
    return ece


def generate_calibration_report(
    rf_probs_before: np.ndarray,
    rf_probs_after: np.ndarray,
    y_true: np.ndarray,
    output_dir: Path
) -> Dict[str, Any]:
    """Generate calibration report with ECE and reliability diagram data."""
    
    ece_before = compute_ece(rf_probs_before, y_true)
    ece_after = compute_ece(rf_probs_after, y_true)
    
    # Reliability diagram data
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    reliability_data = {"before": [], "after": []}
    
    for label, probs in [("before", rf_probs_before), ("after", rf_probs_after)]:
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        accuracies = (predictions == y_true)
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if in_bin.sum() > 0:
                reliability_data[label].append({
                    "bin_center": (bin_boundaries[i] + bin_boundaries[i + 1]) / 2,
                    "avg_confidence": float(confidences[in_bin].mean()),
                    "avg_accuracy": float(accuracies[in_bin].mean()),
                    "count": int(in_bin.sum()),
                })
    
    report = {
        "ece_before": float(ece_before),
        "ece_after": float(ece_after),
        "ece_improvement": float(ece_before - ece_after),
        "reliability_diagram": reliability_data,
    }
    
    # Save report
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "rf_calibration_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    # Generate markdown
    md_lines = [
        "# RF Probability Calibration Report",
        "",
        "## Expected Calibration Error (ECE)",
        "",
        f"- **Before calibration**: {ece_before:.4f}",
        f"- **After calibration**: {ece_after:.4f}",
        f"- **Improvement**: {(ece_before - ece_after):.4f}",
        "",
        "## Reliability Diagram Data",
        "",
        "### Before Calibration",
        "| Bin Center | Avg Confidence | Avg Accuracy | Count |",
        "|------------|----------------|--------------|-------|",
    ]
    
    for bin_data in reliability_data["before"]:
        md_lines.append(
            f"| {bin_data['bin_center']:.2f} | {bin_data['avg_confidence']:.3f} | "
            f"{bin_data['avg_accuracy']:.3f} | {bin_data['count']} |"
        )
    
    md_lines.extend([
        "",
        "### After Calibration",
        "| Bin Center | Avg Confidence | Avg Accuracy | Count |",
        "|------------|----------------|--------------|-------|",
    ])
    
    for bin_data in reliability_data["after"]:
        md_lines.append(
            f"| {bin_data['bin_center']:.2f} | {bin_data['avg_confidence']:.3f} | "
            f"{bin_data['avg_accuracy']:.3f} | {bin_data['count']} |"
        )
    
    with open(output_dir / "rf_calibration_report.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print('\n'.join(md_lines))
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Calibrate RF probabilities")
    parser.add_argument("--features", default="Output/sim_spectrum/features_brb.csv",
                        help="Path to features CSV")
    parser.add_argument("--labels", default="Output/sim_spectrum/labels.json",
                        help="Path to labels JSON")
    parser.add_argument("--method", choices=["isotonic", "platt"], default="isotonic",
                        help="Calibration method")
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
    
    print("Loading RF probabilities and labels...")
    rf_probs, y_true = load_rf_probs(features_path, labels_path)
    
    print(f"Loaded {len(y_true)} samples")
    print(f"Class distribution: {np.bincount(y_true)}")
    
    print(f"\nCalibrating with {args.method} method...")
    
    if args.method == "isotonic" and HAS_SKLEARN:
        calibrators = calibrate_isotonic(rf_probs, y_true)
        
        # Apply calibration
        rf_probs_calibrated = np.zeros_like(rf_probs)
        for c, cal in calibrators.items():
            rf_probs_calibrated[:, c] = cal.predict(rf_probs[:, c])
        
        # Normalize
        row_sums = rf_probs_calibrated.sum(axis=1, keepdims=True)
        rf_probs_calibrated = rf_probs_calibrated / (row_sums + 1e-10)
        
        # Save calibrator
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        joblib.dump(calibrators, artifacts_dir / "rf_calibrator.joblib")
        
        with open(artifacts_dir / "rf_calib_meta.json", 'w') as f:
            json.dump({"method": "isotonic", "n_classes": 4}, f)
        
        print(f"Saved calibrator to artifacts/rf_calibrator.joblib")
    else:
        # Simple temperature scaling fallback
        temperature = 1.5
        rf_probs_calibrated = np.exp(np.log(rf_probs + 1e-10) / temperature)
        rf_probs_calibrated = rf_probs_calibrated / rf_probs_calibrated.sum(axis=1, keepdims=True)
    
    # Generate report
    generate_calibration_report(rf_probs, rf_probs_calibrated, y_true, output_dir)
    
    return 0


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P1.1: Train and export RandomForest classifier for system-level fault classification.

This script:
1. Loads training data from features_brb.csv
2. Trains a RandomForest classifier
3. Exports the model to artifacts/rf_system_classifier.joblib
4. Exports metadata to artifacts/rf_meta.json

Usage:
    python tools/train_and_export_rf.py --features Output/sim_spectrum/features_brb.csv --out_dir artifacts
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# Feature schema: X1-X22 in fixed order
FEATURE_SCHEMA = [f"X{i}" for i in range(1, 23)]

# Class mapping (int -> string)
CLASS_NAMES = ["normal", "amp_error", "freq_error", "ref_error"]
CLASS_NAME_TO_INT = {name: i for i, name in enumerate(CLASS_NAMES)}


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def load_training_data(features_path: Path, labels_path: Path = None):
    """Load training data from features CSV and labels JSON.
    
    Parameters
    ----------
    features_path : Path
        Path to features CSV (must contain X1-X22 columns and sample_id)
    labels_path : Path, optional
        Path to labels JSON. If None, attempts to find in same directory.
        
    Returns
    -------
    X : np.ndarray
        Feature matrix (n_samples, 22)
    y : np.ndarray
        Label vector (n_samples,) with int class indices
    sample_ids : list
        List of sample IDs
    """
    # Load features
    df = pd.read_csv(features_path)
    
    # Check for required columns
    missing_features = [f for f in FEATURE_SCHEMA if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    # Extract X1-X22
    X = df[FEATURE_SCHEMA].values
    
    # Get sample IDs
    if 'sample_id' in df.columns:
        sample_ids = df['sample_id'].tolist()
    else:
        sample_ids = [f"sample_{i}" for i in range(len(df))]
    
    # Load labels
    if labels_path is None:
        labels_path = features_path.parent / "labels.json"
    
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)
    
    # Handle dict or list format
    if isinstance(labels_data, dict):
        if "samples" in labels_data:
            labels_dict = {s["sample_id"]: s for s in labels_data["samples"]}
        else:
            labels_dict = labels_data
    else:
        labels_dict = {s["sample_id"]: s for s in labels_data}
    
    # Extract labels
    y = []
    for sid in sample_ids:
        label = labels_dict.get(sid, {})
        fault_type = label.get("system_fault_class", label.get("type", "normal"))
        if fault_type in CLASS_NAME_TO_INT:
            y.append(CLASS_NAME_TO_INT[fault_type])
        else:
            # Fallback mapping
            if "amp" in fault_type.lower():
                y.append(1)
            elif "freq" in fault_type.lower():
                y.append(2)
            elif "ref" in fault_type.lower():
                y.append(3)
            else:
                y.append(0)
    
    return X, np.array(y), sample_ids


def train_rf_classifier(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    """Train RandomForest classifier.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Label vector
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    clf : RandomForestClassifier
        Fitted classifier
    metrics : dict
        Training metrics (accuracy, cv_scores, confusion_matrix)
    """
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # Train classifier
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    
    # Validation metrics
    y_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, y_pred)
    conf_mat = confusion_matrix(y_val, y_pred)
    
    # Feature importances
    importances = clf.feature_importances_
    feature_importance = {
        FEATURE_SCHEMA[i]: float(importances[i])
        for i in range(len(FEATURE_SCHEMA))
    }
    
    metrics = {
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "val_accuracy": float(val_acc),
        "cv_mean_accuracy": float(cv_scores.mean()),
        "cv_std_accuracy": float(cv_scores.std()),
        "confusion_matrix": conf_mat.tolist(),
        "class_names": CLASS_NAMES,
        "feature_importances": feature_importance,
    }
    
    # Re-train on full dataset
    clf.fit(X, y)
    
    return clf, metrics


def export_rf_model(clf, metrics: dict, out_dir: Path, features_path: Path):
    """Export RF model and metadata.
    
    Parameters
    ----------
    clf : RandomForestClassifier
        Fitted classifier
    metrics : dict
        Training metrics
    out_dir : Path
        Output directory for artifacts
    features_path : Path
        Path to training features (for hash)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Export model
    model_path = out_dir / "rf_system_classifier.joblib"
    joblib.dump(clf, model_path)
    print(f"[INFO] Model saved to: {model_path}")
    
    # Export metadata
    meta = {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "sklearn_version": str(joblib.__version__),
        "training_data_hash": compute_file_hash(features_path),
        "feature_schema": FEATURE_SCHEMA,
        "class_names": CLASS_NAMES,
        "n_estimators": clf.n_estimators,
        "max_depth": clf.max_depth,
        **metrics
    }
    
    meta_path = out_dir / "rf_meta.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Metadata saved to: {meta_path}")
    
    return model_path, meta_path


def main():
    parser = argparse.ArgumentParser(
        description='Train and export RandomForest classifier for system-level fault classification'
    )
    parser.add_argument('--features', default='Output/sim_spectrum/features_brb.csv',
                        help='Path to features CSV')
    parser.add_argument('--labels', default=None,
                        help='Path to labels JSON (default: same dir as features)')
    parser.add_argument('--out_dir', default='artifacts',
                        help='Output directory for model artifacts')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    features_path = Path(args.features)
    out_dir = Path(args.out_dir)
    labels_path = Path(args.labels) if args.labels else None
    
    print("=" * 60)
    print("P1.1: Train and Export RF Classifier")
    print("=" * 60)
    print(f"Features: {features_path}")
    print(f"Output: {out_dir}")
    
    # Check features exist
    if not features_path.exists():
        print(f"[ERROR] Features file not found: {features_path}")
        return 1
    
    # Load data
    print("\nLoading training data...")
    try:
        X, y, sample_ids = load_training_data(features_path, labels_path)
        print(f"  Samples: {len(X)}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Class distribution: {np.bincount(y).tolist()}")
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return 1
    
    # Train
    print("\nTraining RandomForest classifier...")
    clf, metrics = train_rf_classifier(X, y, args.seed)
    
    print(f"\nTraining Results:")
    print(f"  Validation Accuracy: {metrics['val_accuracy']:.1%}")
    print(f"  CV Mean Accuracy: {metrics['cv_mean_accuracy']:.1%} (+/- {metrics['cv_std_accuracy']:.1%})")
    print(f"\n  Confusion Matrix:")
    conf_mat = np.array(metrics['confusion_matrix'])
    for i, row in enumerate(conf_mat):
        print(f"    {CLASS_NAMES[i]:12s}: {row.tolist()}")
    
    # Export
    print("\nExporting model...")
    model_path, meta_path = export_rf_model(clf, metrics, out_dir, features_path)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Metadata: {meta_path}")
    print(f"Validation Accuracy: {metrics['val_accuracy']:.1%}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

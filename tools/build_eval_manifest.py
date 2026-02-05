#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P0.1: Build unified evaluation manifest.

Creates a manifest file that all evaluation scripts must use to ensure
consistent sample sets across all paths.

Output: Output/debug/eval_manifest.json

Usage:
    python tools/build_eval_manifest.py --labels Output/sim_spectrum/labels.json --curves_dir Output/sim_spectrum/raw_curves
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def compute_file_hash(filepath: Path, short: bool = True) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:16] if short else hasher.hexdigest()


def compute_string_hash(s: str, short: bool = True) -> str:
    """Compute SHA256 hash of a string."""
    hasher = hashlib.sha256(s.encode('utf-8'))
    return hasher.hexdigest()[:16] if short else hasher.hexdigest()


def build_manifest(
    labels_path: Path,
    curves_dir: Optional[Path] = None,
    features_path: Optional[Path] = None,
) -> Dict:
    """
    Build unified evaluation manifest.
    
    Parameters
    ----------
    labels_path : Path
        Path to labels.json
    curves_dir : Path, optional
        Path to raw_curves directory
    features_path : Path, optional
        Path to features_brb.csv
        
    Returns
    -------
    Dict
        Manifest with sample_ids, hashes, filter reasons
    """
    manifest = {
        "version": "1.0",
        "dataset_root": str(labels_path.parent),
        "labels_path": str(labels_path),
        "labels_hash": compute_file_hash(labels_path),
    }
    
    # Load labels
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)
    
    # Handle dict or list format
    if isinstance(labels_data, dict):
        if "samples" in labels_data:
            samples = {s["sample_id"]: s for s in labels_data["samples"]}
        else:
            samples = labels_data
    else:
        samples = {s["sample_id"]: s for s in labels_data}
    
    manifest["total_labels"] = len(samples)
    
    # Check curves directory
    if curves_dir:
        curves_dir = Path(curves_dir)
        if curves_dir.exists():
            csv_files = sorted(curves_dir.glob("*.csv"))
            curve_ids = {f.stem for f in csv_files}
            manifest["curves_dir"] = str(curves_dir)
            manifest["curves_count"] = len(csv_files)
            manifest["curves_hash"] = compute_string_hash(",".join(sorted(f.name for f in csv_files)))
        else:
            curve_ids = set()
            manifest["curves_dir"] = str(curves_dir)
            manifest["curves_count"] = 0
            manifest["curves_hash"] = "N/A"
    else:
        curve_ids = None
    
    # Check features file
    if features_path:
        features_path = Path(features_path)
        if features_path.exists():
            manifest["features_path"] = str(features_path)
            manifest["features_hash"] = compute_file_hash(features_path)
            
            # Read sample IDs from features
            import pandas as pd
            df = pd.read_csv(features_path)
            if 'sample_id' in df.columns:
                feature_ids = set(df['sample_id'].tolist())
            else:
                feature_ids = None
        else:
            feature_ids = None
    else:
        feature_ids = None
    
    # Build final sample list with filter reasons
    sample_ids = []
    sample_details = []  # P1.1: Extended sample info
    filter_reasons = Counter()
    filter_details = []
    
    for sample_id, label in sorted(samples.items()):
        # Check curve availability
        if curve_ids is not None and sample_id not in curve_ids:
            filter_reasons["no_curve"] += 1
            filter_details.append({"sample_id": sample_id, "reason": "no_curve"})
            continue
        
        # Check feature availability
        if feature_ids is not None and sample_id not in feature_ids:
            filter_reasons["no_features"] += 1
            filter_details.append({"sample_id": sample_id, "reason": "no_features"})
            continue
        
        # Check required fields
        fault_type = label.get("system_fault_class", label.get("type", ""))
        if not fault_type:
            filter_reasons["no_fault_type"] += 1
            filter_details.append({"sample_id": sample_id, "reason": "no_fault_type"})
            continue
        
        # Valid sample - collect detailed info (P1.1)
        sample_ids.append(sample_id)
        
        # Canonical fault type
        from utils.canonicalize import canonical_fault_type, canonical_module_v2
        canonical_ft = canonical_fault_type(fault_type)
        
        # Module info
        module_v1 = label.get("module_cause", "")
        module_v2 = label.get("module_v2", "")
        if module_v2:
            canonical_mod = canonical_module_v2(module_v2)
        elif module_v1:
            canonical_mod = canonical_module_v2(module_v1)
        else:
            canonical_mod = ""
        
        # CSV path
        csv_path = ""
        if curves_dir and curve_ids:
            csv_path = str(curves_dir / f"{sample_id}.csv")
        
        sample_details.append({
            "sample_id": sample_id,
            "fault_type": canonical_ft,
            "module_v2": canonical_mod,
            "csv_path": csv_path,
            "is_fault": canonical_ft != "normal",
        })
    
    manifest["sample_ids"] = sample_ids
    manifest["samples"] = sample_details  # P1.1: Detailed sample list
    manifest["n_samples"] = len(sample_ids)
    manifest["n_fault"] = sum(1 for s in sample_details if s.get("is_fault", False))
    manifest["filter_reasons"] = dict(filter_reasons)
    manifest["filter_details"] = filter_details[:20]  # Keep first 20 for debugging
    
    # Compute manifest hash for reproducibility
    manifest_content = json.dumps({
        "labels_hash": manifest["labels_hash"],
        "sample_ids": sample_ids,
    }, sort_keys=True)
    manifest["manifest_hash"] = compute_string_hash(manifest_content)
    
    # Class distribution
    class_dist = Counter()
    for sid in sample_ids:
        label = samples.get(sid, {})
        fault_type = label.get("system_fault_class", label.get("type", "unknown"))
        class_dist[fault_type] += 1
    manifest["class_distribution"] = dict(class_dist)
    
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Build unified evaluation manifest')
    parser.add_argument('--labels', default='Output/sim_spectrum/labels.json',
                        help='Path to labels.json')
    parser.add_argument('--curves_dir', default='Output/sim_spectrum/raw_curves',
                        help='Path to raw_curves directory')
    parser.add_argument('--features', default='Output/sim_spectrum/features_brb.csv',
                        help='Path to features CSV')
    parser.add_argument('--out', default='Output/debug/eval_manifest.json',
                        help='Output manifest path')
    
    args = parser.parse_args()
    
    labels_path = Path(args.labels)
    curves_dir = Path(args.curves_dir) if args.curves_dir else None
    features_path = Path(args.features) if args.features else None
    out_path = Path(args.out)
    
    print("=" * 60)
    print("P0.1: Build Unified Evaluation Manifest")
    print("=" * 60)
    
    # Check labels exist
    if not labels_path.exists():
        print(f"[ERROR] Labels file not found: {labels_path}")
        return 1
    
    # Build manifest
    manifest = build_manifest(
        labels_path,
        curves_dir=curves_dir,
        features_path=features_path if features_path and features_path.exists() else None,
    )
    
    # Save manifest
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"\nManifest saved to: {out_path}")
    print(f"\nManifest Summary:")
    print(f"  manifest_hash: {manifest['manifest_hash']}")
    print(f"  n_samples: {manifest['n_samples']}")
    print(f"  labels_hash: {manifest['labels_hash']}")
    print(f"  class_distribution: {manifest['class_distribution']}")
    
    if manifest['filter_reasons']:
        print(f"  filter_reasons: {manifest['filter_reasons']}")
    
    print("\n" + "=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

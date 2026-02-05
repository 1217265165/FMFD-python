#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build unified evaluation manifests.

Outputs two manifests:
1. manifest_all_400.json - All 400 samples
2. manifest_fault_300.json - Only 300 fault samples (exclude normal)

Usage:
    python tools/build_unified_manifest.py --labels Output/sim_spectrum/labels.json --out Output/debug/
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.canonicalize import get_truth_fault_type, get_truth_module, is_normal_sample


def compute_hash(sample_ids: List[str]) -> str:
    """Compute hash of sample ID list for reproducibility verification."""
    content = ",".join(sorted(sample_ids))
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def build_unified_manifest(
    labels_path: Path,
    raw_curves_dir: Path = None,
    include_normal: bool = True,
) -> Dict:
    """
    Build a unified manifest from labels.
    
    Parameters
    ----------
    labels_path : Path
        Path to labels.json
    raw_curves_dir : Path, optional
        Path to raw_curves directory
    include_normal : bool
        Whether to include normal samples
        
    Returns
    -------
    Dict
        Manifest with sample details
    """
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    samples = []
    excluded_normal = 0
    
    for sample_id, label in sorted(labels.items()):
        fault_type = get_truth_fault_type(label)
        module_v2 = get_truth_module(label)
        
        # Check normal exclusion
        if not include_normal and fault_type == "normal":
            excluded_normal += 1
            continue
        
        # Build raw curve path
        raw_curve_path = ""
        if raw_curves_dir:
            curve_path = raw_curves_dir / f"{sample_id}.csv"
            if curve_path.exists():
                raw_curve_path = str(curve_path)
        
        samples.append({
            "sample_id": sample_id,
            "system_fault_class": fault_type,
            "module_v2": module_v2,
            "raw_curve_path": raw_curve_path,
            "is_normal": fault_type == "normal",
        })
    
    sample_ids = [s["sample_id"] for s in samples]
    
    # Class distribution
    class_dist = {}
    for s in samples:
        ft = s["system_fault_class"]
        class_dist[ft] = class_dist.get(ft, 0) + 1
    
    manifest = {
        "version": "1.0",
        "labels_path": str(labels_path),
        "raw_curves_dir": str(raw_curves_dir) if raw_curves_dir else None,
        "include_normal": include_normal,
        "n_samples": len(samples),
        "n_excluded_normal": excluded_normal,
        "sample_ids": sample_ids,
        "sample_id_hash": compute_hash(sample_ids),
        "sample_id_range": {
            "min": sample_ids[0] if sample_ids else "",
            "max": sample_ids[-1] if sample_ids else "",
        },
        "class_distribution": class_dist,
        "samples": samples,
        "truth_fields": {
            "system_truth_field": "system_fault_class",
            "module_truth_field": "module_v2",
            "module_eval_policy": "exclude_normal" if not include_normal else "include_all",
        },
    }
    
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Build unified evaluation manifests')
    parser.add_argument('--labels', default='Output/sim_spectrum/labels.json',
                        help='Path to labels.json')
    parser.add_argument('--raw_curves', default='',
                        help='Path to raw_curves directory (optional)')
    parser.add_argument('--out', default='Output/debug/',
                        help='Output directory')
    
    args = parser.parse_args()
    
    labels_path = Path(args.labels)
    raw_curves_dir = Path(args.raw_curves) if args.raw_curves and Path(args.raw_curves).exists() else None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Build Unified Evaluation Manifests")
    print("=" * 60)
    
    if not labels_path.exists():
        print(f"[ERROR] Labels file not found: {labels_path}")
        return 1
    
    # Build manifest_all_400 (include normal)
    print("\n[1/2] Building manifest_all_400.json...")
    manifest_all = build_unified_manifest(labels_path, raw_curves_dir, include_normal=True)
    all_path = out_dir / "manifest_all_400.json"
    with open(all_path, 'w', encoding='utf-8') as f:
        json.dump(manifest_all, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {all_path}")
    print(f"  N_samples: {manifest_all['n_samples']}")
    print(f"  sample_id_hash: {manifest_all['sample_id_hash']}")
    print(f"  class_distribution: {manifest_all['class_distribution']}")
    
    # Build manifest_fault_300 (exclude normal)
    print("\n[2/2] Building manifest_fault_300.json...")
    manifest_fault = build_unified_manifest(labels_path, raw_curves_dir, include_normal=False)
    fault_path = out_dir / "manifest_fault_300.json"
    with open(fault_path, 'w', encoding='utf-8') as f:
        json.dump(manifest_fault, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {fault_path}")
    print(f"  N_samples: {manifest_fault['n_samples']}")
    print(f"  N_excluded_normal: {manifest_fault['n_excluded_normal']}")
    print(f"  sample_id_hash: {manifest_fault['sample_id_hash']}")
    print(f"  class_distribution: {manifest_fault['class_distribution']}")
    
    print("\n" + "=" * 60)
    print("Manifests built successfully!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

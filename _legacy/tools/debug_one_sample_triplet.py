#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug one sample triplet - trace inference path for a single sample.

Outputs detailed debug info including:
- 22 features
- RF/BRB/fused probabilities
- System-level prediction
- Module-level topk per subgraph
- Ground truth labels

Usage:
    python tools/debug_one_sample_triplet.py --sample_id sim_00042
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_sample_features(sample_id: str, labels_path: Path, features_path: Path = None):
    """Load features for a sample."""
    # Try to load from features CSV
    if features_path and features_path.exists():
        import pandas as pd
        df = pd.read_csv(features_path)
        row = df[df['sample_id'] == sample_id]
        if not row.empty:
            features = {}
            for col in df.columns:
                if col.startswith('X') or col.startswith('x'):
                    features[col] = float(row[col].values[0])
            return features
    
    return None


def run_inference_with_trace(features: dict, verbose: bool = True):
    """Run inference and capture all intermediate values."""
    from methods.ours_adapter import infer_system_and_modules
    
    result = infer_system_and_modules(
        features,
        use_gating=True,
        allow_fallback=True,
    )
    
    trace = {
        "system_probs": result.get("system_probs", {}),
        "fault_type_pred": result.get("fault_type_pred", ""),
        "module_topk": result.get("module_topk", []),
        "debug": result.get("debug", {}),
    }
    
    return trace


def main():
    parser = argparse.ArgumentParser(description='Debug one sample triplet')
    parser.add_argument('--sample_id', default='sim_00042',
                        help='Sample ID to debug')
    parser.add_argument('--labels', default='Output/sim_spectrum/labels.json',
                        help='Path to labels.json')
    parser.add_argument('--features', default='Output/sim_spectrum/features_brb.csv',
                        help='Path to features CSV')
    parser.add_argument('--raw_dir', default='Output/sim_spectrum/raw_curves',
                        help='Path to raw_curves directory')
    parser.add_argument('--out', default='Output/debug/',
                        help='Output directory')
    
    args = parser.parse_args()
    
    sample_id = args.sample_id
    labels_path = Path(args.labels)
    features_path = Path(args.features) if Path(args.features).exists() else None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"Debug Sample Triplet: {sample_id}")
    print("=" * 60)
    
    # Load labels
    if not labels_path.exists():
        print(f"[ERROR] Labels not found: {labels_path}")
        return 1
    
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    if sample_id not in labels:
        print(f"[ERROR] Sample {sample_id} not found in labels")
        return 1
    
    label = labels[sample_id]
    
    # Get truth values
    from utils.canonicalize import get_truth_fault_type, get_truth_module
    
    truth = {
        "sample_id": sample_id,
        "system_fault_class": get_truth_fault_type(label),
        "system_fault_class_raw": label.get("system_fault_class", ""),
        "module_v2": get_truth_module(label),
        "module_v2_raw": label.get("module_v2", ""),
        "module_cause_raw": label.get("module_cause", ""),
    }
    
    print(f"\n[Truth Values]")
    print(f"  system_fault_class: {truth['system_fault_class']}")
    print(f"  module_v2: {truth['module_v2']}")
    
    # Load features
    features = load_sample_features(sample_id, labels_path, features_path)
    
    if not features:
        print(f"[WARN] Could not load features for {sample_id}")
        # Try to compute features from raw curve
        raw_curve_path = Path(args.raw_dir) / f"{sample_id}.csv"
        if raw_curve_path.exists():
            print(f"[INFO] Raw curve exists: {raw_curve_path}")
        else:
            print(f"[WARN] Raw curve not found: {raw_curve_path}")
        
        # Create dummy features for output
        features = {f"X{i}": 0.0 for i in range(1, 23)}
    
    print(f"\n[Features (22 dims)]")
    for k, v in sorted(features.items(), key=lambda x: int(x[0][1:]) if x[0][1:].isdigit() else 0):
        print(f"  {k}: {v:.6f}")
    
    # Run inference
    try:
        trace = run_inference_with_trace(features)
        
        print(f"\n[System-Level Inference]")
        print(f"  RF probs: {trace['debug'].get('rf_probs', 'N/A')}")
        print(f"  BRB probs: {trace['debug'].get('brb_probs', 'N/A')}")
        print(f"  Fused probs: {trace['debug'].get('fused_probs', 'N/A')}")
        print(f"  Gating status: {trace['debug'].get('gating_status', 'N/A')}")
        print(f"  Pred fault_type: {trace['fault_type_pred']}")
        
        print(f"\n[Module-Level Inference]")
        print(f"  Soft gating: {trace['debug'].get('soft_gating', 'N/A')}")
        print(f"  Module TopK:")
        for i, mod in enumerate(trace['module_topk'][:5]):
            name = mod.get('name', mod.get('module', ''))
            prob = mod.get('prob', mod.get('probability', 0))
            print(f"    {i+1}. {name}: {prob:.4f}")
        
        # Check match
        pred_ft = trace['fault_type_pred']
        true_ft = truth['system_fault_class']
        sys_match = pred_ft == true_ft
        
        pred_mod_top1 = trace['module_topk'][0].get('name', '') if trace['module_topk'] else ''
        true_mod = truth['module_v2']
        
        from utils.canonicalize import modules_match
        mod_top1_match = modules_match(pred_mod_top1, true_mod)
        mod_top3_match = any(modules_match(m.get('name', m.get('module', '')), true_mod) 
                            for m in trace['module_topk'][:3])
        
        print(f"\n[Evaluation]")
        print(f"  System match: {sys_match} (pred={pred_ft}, true={true_ft})")
        print(f"  Module top1 match: {mod_top1_match}")
        print(f"  Module top3 match: {mod_top3_match}")
        
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        trace = {"error": str(e)}
    
    # Build output
    output = {
        "sample_id": sample_id,
        "truth": truth,
        "features": features,
        "inference": trace,
        "evaluation": {
            "system_match": sys_match if 'sys_match' in dir() else None,
            "module_top1_match": mod_top1_match if 'mod_top1_match' in dir() else None,
            "module_top3_match": mod_top3_match if 'mod_top3_match' in dir() else None,
        },
    }
    
    # Save output
    out_path = out_dir / f"triplet_{sample_id}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n[INFO] Output saved to: {out_path}")
    print("\n" + "=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

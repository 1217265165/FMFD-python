#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Pool Implementation for FMFD.

Defines feature pools for different fault types:
- COMMON_POOL: Features shared across all fault types
- AMP_POOL: Amplitude-related features
- FREQ_POOL: Frequency-related features
- REF_POOL: Reference-level related features

Usage:
    from utils.feature_pool import get_feature_pool, FEATURE_POOLS
"""

from typing import Dict, List, Set, Optional
import json
from pathlib import Path

# Feature names mapping (X1-X22)
FEATURE_NAMES = {
    "X1": "abs_range_ok",
    "X2": "global_offset_rrs_db",
    "X3": "hf_std_rrs_db",
    "X4": "p95_abs_dev_rrs_db",
    "X5": "inside_env_frac",
    "X6": "abs_out_of_spec_0p4",
    "X7": "abs_cap_db",
    "X8": "peak_freq_mae_hz",
    "X9": "peak_freq_outlier_frac",
    "X10": "peak_track_type",
    # Additional features (X11-X22) - placeholder names
    "X11": "feature_11",
    "X12": "feature_12",
    "X13": "feature_13",
    "X14": "feature_14",
    "X15": "feature_15",
    "X16": "feature_16",
    "X17": "feature_17",
    "X18": "feature_18",
    "X19": "feature_19",
    "X20": "feature_20",
    "X21": "feature_21",
    "X22": "feature_22",
}

# Feature pools definition
# These pools define which features are most relevant for each fault type

COMMON_POOL = [
    "X1",   # abs_range_ok - general amplitude check
    "X5",   # inside_env_frac - envelope conformance
]

AMP_POOL = [
    "X2",   # global_offset_rrs_db - amplitude offset
    "X3",   # hf_std_rrs_db - high-frequency amplitude variation
    "X4",   # p95_abs_dev_rrs_db - 95th percentile deviation
    "X6",   # abs_out_of_spec_0p4 - out-of-spec amplitude
    "X7",   # abs_cap_db - amplitude cap
]

FREQ_POOL = [
    "X8",   # peak_freq_mae_hz - frequency error
    "X9",   # peak_freq_outlier_frac - frequency outliers
    "X10",  # peak_track_type - frequency tracking type
]

REF_POOL = [
    "X2",   # global_offset_rrs_db - reference offset (shared with AMP)
    "X11",  # Additional reference-related features
    "X12",
    "X13",
]

# All pools consolidated
FEATURE_POOLS = {
    "common": COMMON_POOL,
    "amp_error": COMMON_POOL + AMP_POOL,
    "freq_error": COMMON_POOL + FREQ_POOL,
    "ref_error": COMMON_POOL + REF_POOL,
    "normal": COMMON_POOL,  # Normal uses only common features
}

# Cross-pool features (used when soft-gating between fault types)
CROSS_POOL_MAP = {
    ("amp_error", "ref_error"): ["X2"],  # global_offset shared between amp and ref
    ("amp_error", "freq_error"): [],
    ("freq_error", "ref_error"): [],
}


def get_feature_pool(fault_type: str) -> List[str]:
    """
    Get the feature pool for a given fault type.
    
    Parameters
    ----------
    fault_type : str
        One of: normal, amp_error, freq_error, ref_error
        
    Returns
    -------
    List[str]
        List of feature names (X1, X2, etc.)
    """
    fault_type = fault_type.lower().strip()
    
    if fault_type in FEATURE_POOLS:
        return FEATURE_POOLS[fault_type]
    
    # Default to all features if unknown (with warning)
    import warnings
    warnings.warn(f"Unknown fault_type '{fault_type}', returning all features")
    return list(FEATURE_NAMES.keys())


def get_feature_indices(pool: List[str]) -> List[int]:
    """
    Convert feature names to indices (0-based).
    
    Parameters
    ----------
    pool : List[str]
        List of feature names (X1, X2, etc.)
        
    Returns
    -------
    List[int]
        List of 0-based indices
    """
    indices = []
    for name in pool:
        if name.startswith("X") and name[1:].isdigit():
            indices.append(int(name[1:]) - 1)  # X1 -> 0, X2 -> 1, etc.
    return sorted(indices)


def get_cross_pool_features(fault_type1: str, fault_type2: str) -> List[str]:
    """
    Get features shared between two fault types (for soft-gating).
    
    Parameters
    ----------
    fault_type1, fault_type2 : str
        Fault types to compare
        
    Returns
    -------
    List[str]
        Features shared between the two fault types
    """
    key = tuple(sorted([fault_type1, fault_type2]))
    return CROSS_POOL_MAP.get(key, [])


def get_all_feature_names() -> Dict[str, str]:
    """Get mapping of feature indices to names."""
    return FEATURE_NAMES.copy()


def get_pool_summary() -> Dict[str, Dict]:
    """
    Get a summary of all feature pools.
    
    Returns
    -------
    Dict
        Summary with pool names, features, and indices
    """
    summary = {}
    for pool_name, features in FEATURE_POOLS.items():
        summary[pool_name] = {
            "features": features,
            "indices": get_feature_indices(features),
            "count": len(features),
        }
    return summary


def export_pool_config(output_path: Optional[Path] = None) -> Dict:
    """
    Export feature pool configuration to JSON.
    
    Parameters
    ----------
    output_path : Path, optional
        If provided, save to this path
        
    Returns
    -------
    Dict
        Pool configuration
    """
    config = {
        "version": "1.0",
        "feature_names": FEATURE_NAMES,
        "pools": {
            "common": COMMON_POOL,
            "amp_error": AMP_POOL,
            "freq_error": FREQ_POOL,
            "ref_error": REF_POOL,
        },
        "combined_pools": FEATURE_POOLS,
        "cross_pool_map": {f"{k[0]}__{k[1]}": v for k, v in CROSS_POOL_MAP.items()},
    }
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    return config


if __name__ == "__main__":
    # Print pool summary
    print("Feature Pool Summary")
    print("=" * 60)
    
    summary = get_pool_summary()
    for pool_name, info in summary.items():
        print(f"\n{pool_name}:")
        print(f"  Features: {info['features']}")
        print(f"  Indices: {info['indices']}")
        print(f"  Count: {info['count']}")
    
    # Export config
    config = export_pool_config()
    print("\n" + "=" * 60)
    print("Exported configuration:")
    print(json.dumps(config, indent=2))

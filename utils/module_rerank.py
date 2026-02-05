#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module Rerank Utility

Provides lightweight reranking of TopK module candidates based on:
1. Subgraph consistency (module must match activated fault types)
2. Knowledge constraints (module-feature mapping)
3. Evidence consistency (feature pool support)

Usage:
    from utils.module_rerank import rerank_modules
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# Load coupling matrix if available
def load_coupling_matrix() -> Dict[str, Dict[str, float]]:
    """Load coupling matrix from config."""
    config_path = Path("config/coupling_matrix.json")
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("matrix", {})
    
    # Default coupling matrix
    return {
        "normal": {"normal": 1.0, "amp_error": 0.1, "freq_error": 0.1, "ref_error": 0.1},
        "amp_error": {"normal": 0.1, "amp_error": 1.0, "freq_error": 0.8, "ref_error": 1.5},
        "freq_error": {"normal": 0.1, "amp_error": 0.8, "freq_error": 1.0, "ref_error": 0.9},
        "ref_error": {"normal": 0.1, "amp_error": 1.5, "freq_error": 0.9, "ref_error": 1.0},
    }


# Module-to-fault-type mapping (which modules are associated with which fault types)
MODULE_FAULT_AFFINITY = {
    # Amplitude-related modules
    "前置放大器": ["amp_error"],
    "主放大器": ["amp_error"],
    "功率放大器": ["amp_error"],
    "衰减器": ["amp_error", "ref_error"],
    "增益控制": ["amp_error"],
    
    # Frequency-related modules
    "本振源": ["freq_error"],
    "混频器": ["freq_error"],
    "锁相环": ["freq_error"],
    "频率合成器": ["freq_error"],
    "VCO": ["freq_error"],
    
    # Reference-related modules
    "参考源": ["ref_error"],
    "基准电压": ["ref_error"],
    "电源模块": ["ref_error", "amp_error"],
    "温补电路": ["ref_error"],
}


def get_module_fault_affinity(module_name: str) -> List[str]:
    """Get fault types associated with a module."""
    # Exact match
    if module_name in MODULE_FAULT_AFFINITY:
        return MODULE_FAULT_AFFINITY[module_name]
    
    # Partial match
    for key, faults in MODULE_FAULT_AFFINITY.items():
        if key in module_name or module_name in key:
            return faults
    
    # Default: associated with all fault types
    return ["amp_error", "freq_error", "ref_error"]


def compute_subgraph_consistency_score(
    module_name: str,
    fault_type_weights: Dict[str, float]
) -> float:
    """
    Compute consistency score between module and activated fault types.
    
    Higher score = module is more consistent with high-weight fault types.
    """
    module_affinities = get_module_fault_affinity(module_name)
    
    score = 0.0
    for fault_type, weight in fault_type_weights.items():
        if fault_type in module_affinities:
            score += weight
    
    return score


def rerank_modules(
    topk_modules: List[Tuple[str, float]],
    fault_type_weights: Dict[str, float],
    feature_evidence: Optional[Dict[str, float]] = None,
    max_rerank: int = 10,
    consistency_weight: float = 0.3,
    evidence_weight: float = 0.2
) -> List[Tuple[str, float, Dict[str, float]]]:
    """
    Rerank TopK module candidates.
    
    Parameters
    ----------
    topk_modules : List[Tuple[str, float]]
        List of (module_name, original_score) pairs
    fault_type_weights : Dict[str, float]
        Weights for each fault type (e.g., from fused_probs)
    feature_evidence : Dict[str, float], optional
        Feature evidence scores
    max_rerank : int
        Only rerank top N candidates
    consistency_weight : float
        Weight for subgraph consistency scoring
    evidence_weight : float
        Weight for feature evidence scoring
        
    Returns
    -------
    List[Tuple[str, float, Dict]]
        Reranked list of (module_name, new_score, trace)
    """
    if len(topk_modules) == 0:
        return []
    
    candidates = topk_modules[:max_rerank]
    reranked = []
    
    # Validate weights
    if consistency_weight + evidence_weight > 1.0:
        total_w = consistency_weight + evidence_weight
        consistency_weight = consistency_weight / total_w * 0.9
        evidence_weight = evidence_weight / total_w * 0.9
    
    for module_name, original_score in candidates:
        # Compute consistency score
        consistency_score = compute_subgraph_consistency_score(module_name, fault_type_weights)
        
        # Compute evidence score (if available)
        evidence_score = 0.0
        if feature_evidence:
            module_affinities = get_module_fault_affinity(module_name)
            for ft in module_affinities:
                evidence_score += feature_evidence.get(f"{ft}_support", 0.0)
        
        # Combine scores
        new_score = (
            (1 - consistency_weight - evidence_weight) * original_score +
            consistency_weight * consistency_score +
            evidence_weight * evidence_score
        )
        
        trace = {
            "original_score": original_score,
            "consistency_score": consistency_score,
            "evidence_score": evidence_score,
            "new_score": new_score,
            "module_affinities": get_module_fault_affinity(module_name),
        }
        
        reranked.append((module_name, new_score, trace))
    
    # Add remaining candidates without reranking
    for module_name, original_score in topk_modules[max_rerank:]:
        trace = {"original_score": original_score, "new_score": original_score, "not_reranked": True}
        reranked.append((module_name, original_score, trace))
    
    # Sort by new score (descending)
    reranked.sort(key=lambda x: x[1], reverse=True)
    
    return reranked


def apply_coupling_activation(
    fused_probs: Dict[str, float],
    predicted_type: str,
    topk_types: int = 2
) -> Dict[str, float]:
    """
    Apply coupling matrix to compute soft activation weights.
    
    Parameters
    ----------
    fused_probs : Dict[str, float]
        Fused probabilities for each fault type
    predicted_type : str
        Primary predicted fault type
    topk_types : int
        Number of fault types to activate
        
    Returns
    -------
    Dict[str, float]
        Normalized activation weights per fault type
    """
    coupling_matrix = load_coupling_matrix()
    
    if predicted_type not in coupling_matrix:
        predicted_type = "amp_error"  # Fallback
    
    coupling_row = coupling_matrix.get(predicted_type, {})
    
    # Compute weighted activation
    activation_weights = {}
    for ft, prob in fused_probs.items():
        coupling_factor = coupling_row.get(ft, 1.0)
        activation_weights[ft] = prob * coupling_factor
    
    # Normalize
    total = sum(activation_weights.values())
    if total > 0:
        activation_weights = {k: v / total for k, v in activation_weights.items()}
    
    # Keep only topK
    sorted_weights = sorted(activation_weights.items(), key=lambda x: x[1], reverse=True)
    top_weights = dict(sorted_weights[:topk_types])
    
    # Re-normalize
    total = sum(top_weights.values())
    if total > 0:
        top_weights = {k: v / total for k, v in top_weights.items()}
    
    return top_weights


if __name__ == "__main__":
    # Test example
    print("Module Rerank Utility Test")
    print("=" * 50)
    
    # Example inputs
    topk = [
        ("前置放大器", 0.35),
        ("参考源", 0.28),
        ("本振源", 0.20),
        ("混频器", 0.17),
    ]
    
    fault_weights = {
        "amp_error": 0.45,
        "ref_error": 0.40,
        "freq_error": 0.10,
        "normal": 0.05,
    }
    
    print("\nOriginal TopK:")
    for name, score in topk:
        print(f"  {name}: {score:.3f}")
    
    print("\nFault Type Weights:")
    for ft, w in fault_weights.items():
        print(f"  {ft}: {w:.3f}")
    
    reranked = rerank_modules(topk, fault_weights)
    
    print("\nReranked TopK:")
    for name, score, trace in reranked:
        print(f"  {name}: {score:.3f}")
        print(f"    Consistency: {trace.get('consistency_score', 0):.3f}")
        print(f"    Affinities: {trace.get('module_affinities', [])}")

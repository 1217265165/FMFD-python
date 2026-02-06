#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P2.1: Unified module localization metrics.

This module provides the single source of truth for computing module-level
localization accuracy metrics. All evaluation scripts must use these functions.

Functions:
    compute_mod_topk: Compute Top-K accuracy for module localization
    canonicalize_module: Normalize module names for comparison
    modules_match: Check if two module names match
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.canonicalize import modules_match, canonical_module_v2


def compute_mod_topk(
    y_true_module: List[str],
    y_pred_topk_modules: List[List[str]],
    k: int = 1,
) -> Dict[str, float]:
    """
    Compute Top-K accuracy for module localization.
    
    Parameters
    ----------
    y_true_module : List[str]
        Ground truth module names (one per sample)
    y_pred_topk_modules : List[List[str]]
        Predicted modules for each sample, sorted by confidence.
        Each element is a list of predicted module names.
    k : int
        K for Top-K accuracy (1 for Top-1, 3 for Top-3)
        
    Returns
    -------
    Dict[str, float]
        Dictionary with:
        - 'accuracy': Top-K accuracy
        - 'correct': Number of correct predictions
        - 'total': Total samples evaluated
    """
    if len(y_true_module) != len(y_pred_topk_modules):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true_module)}, "
            f"y_pred={len(y_pred_topk_modules)}"
        )
    
    correct = 0
    total = 0
    
    for gt, pred_list in zip(y_true_module, y_pred_topk_modules):
        # Skip samples with no ground truth module
        if not gt:
            continue
        
        total += 1
        
        # Get top-k predictions
        topk_pred = pred_list[:k] if pred_list else []
        
        # Check if any prediction matches
        hit = any(modules_match(pred, gt) for pred in topk_pred)
        if hit:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
    }


def compute_mod_metrics(
    y_true_module: List[str],
    y_pred_topk_modules: List[List[str]],
) -> Dict[str, float]:
    """
    Compute all module localization metrics (Top-1 and Top-3).
    
    Parameters
    ----------
    y_true_module : List[str]
        Ground truth module names
    y_pred_topk_modules : List[List[str]]
        Predicted modules for each sample
        
    Returns
    -------
    Dict[str, float]
        Dictionary with mod_top1 and mod_top3
    """
    top1 = compute_mod_topk(y_true_module, y_pred_topk_modules, k=1)
    top3 = compute_mod_topk(y_true_module, y_pred_topk_modules, k=3)
    
    return {
        'mod_top1': top1['accuracy'],
        'mod_top3': top3['accuracy'],
        'mod_top1_correct': top1['correct'],
        'mod_top3_correct': top3['correct'],
        'mod_total': top1['total'],
    }


def compute_metrics_by_fault_type(
    y_true_module: List[str],
    y_pred_topk_modules: List[List[str]],
    fault_types: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute module metrics grouped by fault type.
    
    Parameters
    ----------
    y_true_module : List[str]
        Ground truth module names
    y_pred_topk_modules : List[List[str]]
        Predicted modules for each sample
    fault_types : List[str]
        Fault type for each sample
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary mapping fault_type -> metrics
    """
    from collections import defaultdict
    
    # Group by fault type
    by_fault = defaultdict(lambda: {'gt': [], 'pred': []})
    
    for gt, pred, ft in zip(y_true_module, y_pred_topk_modules, fault_types):
        if gt:  # Only include samples with module labels
            by_fault[ft]['gt'].append(gt)
            by_fault[ft]['pred'].append(pred)
    
    results = {}
    for ft, data in by_fault.items():
        metrics = compute_mod_metrics(data['gt'], data['pred'])
        results[ft] = metrics
    
    return results

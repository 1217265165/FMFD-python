#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module-level flat evaluation: force all baselines to classify ~20 module classes.

This script evaluates every baseline method on the module-level fault localization
task (y_mod, ~14-20 classes) using flat multi-class classification. Methods that
cannot handle the high-dimensional output space (e.g., due to memory explosion
or intractable rule generation) are recorded as "OOM" or "Failed".

Usage:
    python pipelines/evaluate_baselines_module.py \
      --data_dir Output/sim_spectrum \
      --output_dir Output/compare_methods_real_mod \
      --methods ours,hcf,dbrb,brb_mu,brb_p,a_ibrb \
      --load_params Output/optimization_results_pcmaes/best_params.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import signal
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.default_paths import (
    PROJECT_ROOT, SIM_DIR, COMPARE_DIR, SEED, SPLIT,
)
from pipelines.compare_methods import (
    prepare_dataset, stratified_split, set_global_seed,
    SYS_LABEL_ORDER, calculate_accuracy, calculate_macro_f1,
)
from BRB.module_brb import MODULE_LABELS

# Timeout for each method's fit+predict (seconds)
METHOD_TIMEOUT = 300


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Method exceeded time limit")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Module-level flat evaluation for all baselines"
    )
    parser.add_argument('--data_dir', default=str(SIM_DIR))
    parser.add_argument('--output_dir', default=str(COMPARE_DIR))
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--methods', type=str, default='all')
    parser.add_argument('--load_params', type=str, default=None)
    parser.add_argument('--timeout', type=int, default=METHOD_TIMEOUT,
                        help='Timeout per method in seconds')
    return parser.parse_args()


def _evaluate_flat_module(
    method,
    X_train: np.ndarray,
    y_mod_train: np.ndarray,
    X_test: np.ndarray,
    y_mod_test: np.ndarray,
    feature_names: List[str],
    timeout_sec: int,
) -> Dict:
    """Evaluate a single method on flat module-level classification.

    Passes y_mod as the training target (instead of y_sys).
    Catches MemoryError, TimeoutError, and other exceptions.
    """
    result = {
        'method': method.name,
        'mod_top1_accuracy': None,
        'mod_macro_f1': None,
        'fit_time_sec': None,
        'infer_time_ms': None,
        'status': 'OK',
        'error': None,
        'n_classes': int(len(np.unique(y_mod_train))),
    }

    # Set up timeout (Unix only; on Windows this is a no-op)
    use_alarm = hasattr(signal, 'SIGALRM')
    if use_alarm:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_sec)

    try:
        meta = {'feature_names': feature_names}

        # --- FIT (pass y_mod as y_sys so the adapter trains on module labels) ---
        t0 = time.time()
        method.fit(X_train, y_mod_train, None, meta)
        fit_time = time.time() - t0
        result['fit_time_sec'] = round(fit_time, 2)

        # --- PREDICT ---
        t1 = time.time()
        predictions = method.predict(X_test, meta)
        infer_time = time.time() - t1
        result['infer_time_ms'] = round((infer_time / len(X_test)) * 1000, 4)

        y_pred = predictions['system_pred']  # flat mode: system_pred IS module pred

        # --- METRICS ---
        valid = y_mod_test >= 0
        if np.sum(valid) > 0:
            acc = calculate_accuracy(y_mod_test[valid], y_pred[valid])
            n_mod_classes = int(max(np.max(y_mod_test[valid]), np.max(y_pred[valid]))) + 1
            f1 = calculate_macro_f1(y_mod_test[valid], y_pred[valid], n_mod_classes)
            result['mod_top1_accuracy'] = round(float(acc), 4)
            result['mod_macro_f1'] = round(float(f1), 4)
        else:
            result['status'] = 'NO_LABELS'

    except MemoryError:
        result['status'] = 'OOM'
        result['error'] = 'MemoryError: rule/parameter explosion with high-dimensional output'
    except TimeoutError:
        result['status'] = 'TIMEOUT'
        result['error'] = f'Exceeded {timeout_sec}s time limit'
    except Exception as e:
        result['status'] = 'FAILED'
        result['error'] = f'{type(e).__name__}: {e}'
        traceback.print_exc()
    finally:
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    return result


def _evaluate_ours_hierarchical(
    method,
    X_train: np.ndarray,
    y_sys_train: np.ndarray,
    y_mod_train: np.ndarray,
    X_test: np.ndarray,
    y_sys_test: np.ndarray,
    y_mod_test: np.ndarray,
    feature_names: List[str],
) -> Dict:
    """Evaluate OursAdapter using its native hierarchical module prediction."""
    result = {
        'method': 'ours',
        'mod_top1_accuracy': None,
        'mod_macro_f1': None,
        'fit_time_sec': None,
        'infer_time_ms': None,
        'status': 'OK',
        'error': None,
        'n_classes': int(len(np.unique(y_mod_train[y_mod_train >= 0]))),
    }

    try:
        meta = {'feature_names': feature_names}

        t0 = time.time()
        method.fit(X_train, y_sys_train, y_mod_train, meta)
        result['fit_time_sec'] = round(time.time() - t0, 2)

        t1 = time.time()
        predictions = method.predict(X_test, meta)
        result['infer_time_ms'] = round(((time.time() - t1) / len(X_test)) * 1000, 4)

        # Use module_proba from hierarchical inference
        mod_proba = predictions.get('module_proba')
        if mod_proba is not None:
            from tools.label_mapping import module_v2_from_v1
            n_mods = len(MODULE_LABELS)
            n_correct = 0
            n_valid = 0
            for i, (gt_idx, row) in enumerate(zip(y_mod_test, mod_proba)):
                if gt_idx < 0:
                    continue
                n_valid += 1
                pred_idx = int(np.argmax(row))
                gt_name = MODULE_LABELS[gt_idx] if gt_idx < n_mods else ""
                pred_name = MODULE_LABELS[pred_idx] if pred_idx < n_mods else ""
                gt_v2 = module_v2_from_v1(gt_name)
                pred_v2 = module_v2_from_v1(pred_name)
                if gt_v2 == pred_v2:
                    n_correct += 1
            if n_valid > 0:
                result['mod_top1_accuracy'] = round(n_correct / n_valid, 4)
        else:
            result['status'] = 'NO_MODULE_PROBA'

    except Exception as e:
        result['status'] = 'FAILED'
        result['error'] = f'{type(e).__name__}: {e}'
        traceback.print_exc()

    return result


def main():
    args = parse_args()
    set_global_seed(args.seed)

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / args.data_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load optimized params if provided
    if args.load_params:
        params_path = Path(args.load_params)
        if not params_path.is_absolute():
            params_path = PROJECT_ROOT / args.load_params
        if params_path.exists():
            try:
                params_data = json.loads(params_path.read_text(encoding='utf-8'))
                if 'hierarchical_params' in params_data:
                    from BRB.module_brb import set_hierarchical_params
                    set_hierarchical_params(params_data['hierarchical_params'])
                    print(f"[INFO] Loaded hierarchical params from {params_path}")
                if 'module_rule_weights' in params_data:
                    from BRB.module_brb import set_module_rule_weights
                    set_module_rule_weights(params_data['module_rule_weights'])
            except Exception as e:
                print(f"[WARN] Failed to load params: {e}")

    print("=" * 60)
    print("MODULE-LEVEL FLAT EVALUATION (All Baselines)")
    print("=" * 60)
    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Timeout: {args.timeout}s per method")

    # Load dataset
    X, y_sys, y_mod, feature_names, sample_ids, *_ = prepare_dataset(
        data_dir, use_pool_features=True, strict_leakage=False,
    )

    if y_mod is None or np.all(y_mod < 0):
        print("[ERROR] No module labels found. Cannot evaluate module-level.")
        sys.exit(1)

    # Filter to samples with valid module labels
    valid_mask = y_mod >= 0
    n_valid = np.sum(valid_mask)
    n_mod_classes = len(np.unique(y_mod[valid_mask]))
    print(f"\nSamples with module labels: {n_valid}/{len(y_mod)}")
    print(f"Module classes in data: {n_mod_classes}")
    print(f"Module label indices: {sorted(np.unique(y_mod[valid_mask]).astype(int))}")

    # Split
    X_train, X_val, X_test, y_sys_train, y_sys_val, y_sys_test, train_idx, val_idx, test_idx = \
        stratified_split(X, y_sys, SPLIT[0], SPLIT[1], args.seed)
    y_mod_train = y_mod[train_idx]
    y_mod_test = y_mod[test_idx]

    # For flat baselines: filter training/test to only samples with valid module labels
    flat_train_mask = y_mod_train >= 0
    flat_test_mask = y_mod_test >= 0
    X_train_mod = X_train[flat_train_mask]
    y_mod_train_raw = y_mod_train[flat_train_mask]
    X_test_mod = X_test[flat_test_mask]
    y_mod_test_raw = y_mod_test[flat_test_mask]

    # Remap sparse module indices (e.g. [2,3,6,7,...,19]) to contiguous [0,1,2,...,13]
    # This is needed because baseline adapters use np.bincount which expects contiguous labels
    all_mod_labels = np.unique(np.concatenate([y_mod_train_raw, y_mod_test_raw]))
    sparse_to_dense = {int(v): i for i, v in enumerate(all_mod_labels)}
    dense_to_sparse = {i: int(v) for i, v in enumerate(all_mod_labels)}
    y_mod_train_flat = np.array([sparse_to_dense[int(v)] for v in y_mod_train_raw])
    y_mod_test_flat = np.array([sparse_to_dense[int(v)] for v in y_mod_test_raw])

    n_flat_classes = len(all_mod_labels)
    print(f"\nFlat train: {len(X_train_mod)} samples, {n_flat_classes} classes (remapped to 0..{n_flat_classes-1})")
    print(f"Flat test: {len(X_test_mod)} samples")
    print(f"Label mapping: {sparse_to_dense}")

    # Import all methods
    from methods.ours_adapter import OursAdapter
    from methods.hcf_adapter import HCFAdapter
    from methods.dbrb_adapter import DBRBAdapter
    from methods.brb_mu_adapter import BRBMUAdapter
    from methods.brb_p_adapter import BRBPAdapter
    from methods.a_ibrb_adapter import AIBRBAdapter

    all_methods = {
        'ours': OursAdapter,
        'hcf': HCFAdapter,
        'dbrb': DBRBAdapter,
        'brb_mu': BRBMUAdapter,
        'brb_p': BRBPAdapter,
        'a_ibrb': AIBRBAdapter,
    }

    if args.methods != 'all':
        selected = [m.strip() for m in args.methods.split(',')]
        all_methods = {k: v for k, v in all_methods.items() if k in selected}

    print(f"\nMethods to evaluate: {list(all_methods.keys())}")

    # Evaluate each method
    results = []
    for name, MethodClass in all_methods.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {name} (flat {n_mod_classes}-class module classification)")
        print(f"{'='*60}")

        method = MethodClass()

        if name == 'ours':
            # OursAdapter uses hierarchical inference, not flat classification
            result = _evaluate_ours_hierarchical(
                method, X_train, y_sys_train, y_mod_train,
                X_test, y_sys_test, y_mod_test, feature_names,
            )
        else:
            # All baselines: flat classification with y_mod as target
            result = _evaluate_flat_module(
                method, X_train_mod, y_mod_train_flat,
                X_test_mod, y_mod_test_flat, feature_names, args.timeout,
            )

        # Print result
        status = result['status']
        acc = result['mod_top1_accuracy']
        if status == 'OK' and acc is not None:
            print(f"  -> Module Top-1: {acc*100:.1f}%")
        else:
            print(f"  -> Status: {status}")
            if result['error']:
                print(f"     Error: {result['error']}")

        results.append(result)

    # Save results
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")

    # Print Markdown table
    print()
    print("| Method | Module Top-1 Acc | Status | Fit Time | n_classes |")
    print("|--------|-----------------|--------|----------|-----------|")
    for r in results:
        acc_str = f"{r['mod_top1_accuracy']*100:.1f}%" if r['mod_top1_accuracy'] is not None else r['status']
        fit_str = f"{r['fit_time_sec']:.1f}s" if r['fit_time_sec'] is not None else "N/A"
        print(f"| {r['method']:6s} | {acc_str:>15s} | {r['status']:6s} | {fit_str:>8s} | {r['n_classes']:>9d} |")

    # Save CSV
    csv_path = output_dir / "module_level_comparison.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['method', 'mod_top1_accuracy', 'mod_macro_f1',
                      'fit_time_sec', 'infer_time_ms', 'status', 'error', 'n_classes']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\nSaved: {csv_path}")

    # Save JSON
    json_path = output_dir / "module_level_comparison.json"
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8',
    )
    print(f"Saved: {json_path}")


if __name__ == '__main__':
    main()

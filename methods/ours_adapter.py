"""Adapter for Ours method (knowledge-driven rule compression + hierarchical BRB).

CRITICAL: This adapter MUST use the same BRB inference as brb_diagnosis_cli.py
to ensure compare_methods and CLI produce consistent results.

The unified architecture uses:
- Sub-BRB system (amp/freq/ref) for system-level diagnosis
- Module BRB for module-level diagnosis
- NO RandomForest or other supervised classifiers (that would be a different method)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from methods.base import MethodAdapter
from BRB.system_brb import system_level_infer, SystemBRBConfig
from BRB.aggregator import set_calibration_override
from BRB.module_brb import MODULE_LABELS, module_level_infer, module_level_infer_with_activation


def _load_calibration() -> Dict:
    """Load calibration parameters from Output/ours_best_config.json or calibration.json."""
    possible_paths = [
        Path(__file__).parent.parent / 'Output' / 'ours_best_config.json',
        Path(__file__).parent.parent / 'Output' / 'calibration.json',
        Path(__file__).parent.parent / 'Output' / 'sim_spectrum' / 'calibration.json',
        Path('Output/ours_best_config.json'),
        Path('Output/calibration.json'),
        Path('Output/sim_spectrum/calibration.json'),
    ]
    
    for path in possible_paths:
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                continue
    
    return {}


class OursAdapter(MethodAdapter):
    """Our proposed method: Knowledge-driven rule compression + hierarchical BRB.
    
    UNIFIED ARCHITECTURE (must match brb_diagnosis_cli.py):
    - Two-layer inference: System BRB (sub-BRBs) -> Module BRB
    - System result gating: only activate physically-related module subset
    - Knowledge mapping: modules use relevant frequency bands/features only
    - Sub-BRB architecture: separate BRBs for amp/freq/ref faults
    
    IMPORTANT: This adapter does NOT use supervised learning classifiers.
    It uses the same BRB rule-based inference as the CLI tool.
    
    Complexity:
    - Rules: system layer (3 sub-BRBs) + module layer (configured rules only)
    - Params: attribute weights + rule weights + belief degrees
    """
    
    name = "ours"
    
    def __init__(self, calibration_override: Optional[Dict] = None):
        # Load calibration first
        self.calibration = _load_calibration()
        if calibration_override:
            self.calibration.update(calibration_override)
        set_calibration_override(self.calibration if self.calibration else None)
        
        # Initialize config with calibration values
        self.config = SystemBRBConfig()
        if self.calibration:
            self.config.alpha = self.calibration.get('alpha', self.config.alpha)
            self.config.overall_threshold = self.calibration.get('T_low', self.calibration.get('overall_threshold', self.config.overall_threshold))
            self.config.max_prob_threshold = self.calibration.get('T_prob', self.calibration.get('max_prob_threshold', self.config.max_prob_threshold))
            if 'attribute_weights' in self.calibration:
                self.config.attribute_weights = tuple(self.calibration['attribute_weights'])
            if 'rule_weights' in self.calibration:
                self.config.rule_weights = tuple(self.calibration['rule_weights'])
        
        self.feature_names = None
        self.n_system_rules = 15  # 3 sub-BRBs with 5 rules each
        self.n_module_rules = 33  # Configured per-module rules
        self.n_params = 68  # 22 feature weights + 3 rule weights + 33 belief params + 10 sub-BRB params
        self.kd_features = [f'X{i}' for i in range(1, 23)]  # X1-X22
        self.use_sub_brb = True  # Enable sub-BRB architecture for better accuracy
        # Feature name aliases for compatibility
        self.kd_features_aliases = {
            'bias': 'X1', 'ripple_var': 'X2', 'res_slope': 'X3', 
            'df': 'X4', 'scale_consistency': 'X5',
            'ripple_variance': 'X6', 'gain_nonlinearity': 'X7', 
            'lo_leakage': 'X8', 'tuning_linearity_residual': 'X9',
            'band_amplitude_consistency': 'X10',
            'env_overrun_rate': 'X11', 'env_overrun_max': 'X12',
            'env_violation_energy': 'X13', 'band_residual_low': 'X14',
            'band_residual_high_std': 'X15',
            'corr_shift_bins': 'X16', 'warp_scale': 'X17', 'warp_bias': 'X18',
            'slope_low': 'X19', 'kurtosis_detrended': 'X20',
            'peak_count_residual': 'X21', 'ripple_dom_freq_energy': 'X22',
        }
    
    def fit(self, X_train: np.ndarray, y_sys_train: np.ndarray,
            y_mod_train: Optional[np.ndarray] = None, meta: Optional[Dict] = None) -> None:
        """Fit method (rule-based, no supervised training).
        
        For unified BRB architecture, we do NOT train a classifier.
        We only store feature names for later use in inference.
        This ensures compare_methods uses the same inference as CLI.
        """
        if meta and 'feature_names' in meta:
            self.feature_names = meta['feature_names']
        # NOTE: No classifier training - BRB is rule-based!
    
    def predict(self, X_test: np.ndarray, meta: Optional[Dict] = None) -> Dict:
        """Predict using unified hierarchical BRB (same as CLI).
        
        This method uses the SAME BRB inference logic as brb_diagnosis_cli.py
        to ensure consistent results between compare_methods and CLI.
        """
        n_test = len(X_test)
        n_sys_classes = 4  # Normal, Amp, Freq, Ref
        
        if meta and 'feature_names' in meta:
            self.feature_names = meta['feature_names']
        
        # Initialize outputs
        sys_proba = np.zeros((n_test, n_sys_classes))
        sys_pred = np.zeros(n_test, dtype=int)
        mod_proba = np.zeros((n_test, len(MODULE_LABELS)))
        mod_pred = np.zeros(n_test, dtype=int)

        start_time = time.time()
        
        # Use sub_brb architecture (same as CLI)
        inference_mode = 'sub_brb' if self.use_sub_brb else 'er'
        
        for i in range(n_test):
            # Convert sample to feature dict
            features = self._array_to_dict(X_test[i])
            
            # System-level inference using sub_brb mode
            sys_result = system_level_infer(features, self.config, mode=inference_mode)
            probs = sys_result.get('probabilities', {})
            
            # Map to probability array
            # Order: Normal, Amp, Freq, Ref
            total_prob = sum(probs.values()) if probs else 0.0
            
            if total_prob > 0.01:
                sys_proba[i, 0] = probs.get('正常', 0.0)
                sys_proba[i, 1] = probs.get('幅度失准', 0.0)
                sys_proba[i, 2] = probs.get('频率失准', 0.0)
                sys_proba[i, 3] = probs.get('参考电平失准', 0.0)
                
                # Normalize
                row_sum = np.sum(sys_proba[i])
                if row_sum > 0:
                    sys_proba[i] /= row_sum
            else:
                sys_proba[i] = np.ones(n_sys_classes) / n_sys_classes
            
            sys_pred[i] = np.argmax(sys_proba[i])
            
            # Module-level inference
            mod_probs_dict = module_level_infer_with_activation(features, sys_result, only_activate_relevant=True)
            
            # Convert to array
            for mod_id_str, prob in mod_probs_dict.items():
                try:
                    if isinstance(mod_id_str, str):
                        mod_id = int(''.join(filter(str.isdigit, mod_id_str)))
                    else:
                        mod_id = int(mod_id_str)
                    
                    if 1 <= mod_id <= len(MODULE_LABELS):
                        mod_proba[i, mod_id - 1] = prob
                except (ValueError, IndexError):
                    continue
            
            if np.sum(mod_proba[i]) > 0:
                mod_pred[i] = np.argmax(mod_proba[i])
            else:
                mod_pred[i] = 0
        
        infer_time = time.time() - start_time
        infer_time_ms = (infer_time / n_test) * 1000 if n_test > 0 else 0.0
        
        return {
            'system_proba': sys_proba,
            'system_pred': sys_pred,
            'module_proba': mod_proba,
            'module_pred': mod_pred + 1,
            'meta': {
                'fit_time_sec': 0.0,  # Rule-based, no training
                'infer_time_ms_per_sample': infer_time_ms,
                'n_rules': self.n_system_rules + self.n_module_rules,
                'n_params': self.n_params,
                'n_features_used': len(self.kd_features),
                'features_used': self.kd_features,
            }
        }
    
    def complexity(self) -> Dict:
        """Return complexity metrics."""
        return {
            'n_rules': self.n_system_rules + self.n_module_rules,
            'n_params': self.n_params,
            'n_features_used': len(self.kd_features),
        }
    
    def _array_to_dict(self, x: np.ndarray) -> Dict[str, float]:
        """Convert numpy array to feature dict, supporting X1-X22."""
        if self.feature_names is None:
            # Default mapping: 假设按X1-X22顺序排列
            feature_dict = {}
            for i in range(min(len(x), 22)):
                feature_dict[f'X{i+1}'] = float(x[i])
            # 填充缺失特征
            for i in range(len(x), 22):
                feature_dict[f'X{i+1}'] = 0.0
        else:
            feature_dict = {}
            for i, name in enumerate(self.feature_names):
                if i < len(x):
                    # 支持别名映射
                    canonical_name = self.kd_features_aliases.get(name, name)
                    feature_dict[canonical_name] = float(x[i])
                    # 同时保留原名
                    feature_dict[name] = float(x[i])
        
        # 确保所有X1-X22都存在
        for i in range(1, 23):
            key = f'X{i}'
            if key not in feature_dict:
                feature_dict[key] = 0.0
        
        # Ensure required aliases for backward compatibility
        feature_dict.setdefault('bias', feature_dict.get('X1', 0.0))
        feature_dict.setdefault('ripple_var', feature_dict.get('X2', 0.0))
        feature_dict.setdefault('res_slope', feature_dict.get('X3', 0.0))
        feature_dict.setdefault('df', feature_dict.get('X4', 0.0))
        feature_dict.setdefault('scale_consistency', feature_dict.get('X5', 0.0))
        
        return feature_dict

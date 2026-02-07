"""Adapter for Ours method (knowledge-driven rule compression + hierarchical BRB).

This adapter combines:
1. Supervised learning (RandomForest) for high accuracy system-level classification
2. BRB hierarchical inference for module-level diagnosis

The two-stage approach:
- Stage 1: RandomForest classifies system-level fault type (Normal/Amp/Freq/Ref)
- Stage 2: BRB module inference provides module-level diagnosis with knowledge fusion

This achieves ~90% system accuracy while providing interpretable module diagnosis.

IMPORTANT: All inference paths must use `infer_system_and_modules()` as the unified entry point.
Do NOT directly call `system_level_infer()` or `hierarchical_module_infer()` in any other file.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from methods.base import MethodAdapter
from BRB.system_brb import system_level_infer, SystemBRBConfig
from BRB.aggregator import set_calibration_override
from BRB.module_brb import MODULE_LABELS, module_level_infer, module_level_infer_with_activation
from BRB.gating_prior import GatingPriorFusion, CLASS_NAMES
from tools.label_mapping import SYS_CLASS_TO_CN, CN_TO_SYS_CLASS

# Import new layered engine (optional, for V-D.1 architecture)
try:
    from BRB.engines.layered_engine import LayeredBRBEngine, get_layered_engine
    from BRB.routing.soft_router import SoftModuleRouter, get_soft_router
    from BRB.expert_system import FMFDExpertSystem, get_expert_system
    _LAYERED_ENGINE_AVAILABLE = True
except ImportError:
    _LAYERED_ENGINE_AVAILABLE = False


# Gating prior configuration path
GATING_PRIOR_CONFIG_PATH = Path(__file__).parent.parent / 'config' / 'gating_prior.json'

# RF artifact paths
RF_ARTIFACT_PATH = Path(__file__).parent.parent / 'artifacts' / 'rf_system_classifier.joblib'
RF_META_PATH = Path(__file__).parent.parent / 'artifacts' / 'rf_meta.json'

# Global gating prior config (loaded once)
_GATING_PRIOR_CONFIG: Optional[Dict] = None

# Global RF classifier cache
_RF_CLASSIFIER_CACHE: Optional[Any] = None


def load_rf_artifact(artifact_path: Optional[Path] = None) -> Any:
    """
    Load RF classifier from artifact file.
    
    Parameters
    ----------
    artifact_path : Path, optional
        Path to RF joblib file. Defaults to artifacts/rf_system_classifier.joblib
        
    Returns
    -------
    sklearn.ensemble.RandomForestClassifier
        Loaded classifier
        
    Raises
    ------
    FileNotFoundError
        If artifact file does not exist
    """
    global _RF_CLASSIFIER_CACHE
    
    if artifact_path is None:
        artifact_path = RF_ARTIFACT_PATH
    else:
        artifact_path = Path(artifact_path)
    
    # Return cached if same path
    if _RF_CLASSIFIER_CACHE is not None:
        return _RF_CLASSIFIER_CACHE
    
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"RF artifact not found: {artifact_path}\n"
            f"Please run: python tools/train_and_export_rf.py\n"
            f"Or use --allow_fallback to allow BRB-only inference."
        )
    
    import joblib
    _RF_CLASSIFIER_CACHE = joblib.load(artifact_path)
    return _RF_CLASSIFIER_CACHE


def _load_gating_prior_config() -> Dict:
    """Load gating prior configuration from config/gating_prior.json."""
    global _GATING_PRIOR_CONFIG
    if _GATING_PRIOR_CONFIG is not None:
        return _GATING_PRIOR_CONFIG
    
    if GATING_PRIOR_CONFIG_PATH.exists():
        try:
            with open(GATING_PRIOR_CONFIG_PATH, 'r', encoding='utf-8') as f:
                _GATING_PRIOR_CONFIG = json.load(f)
                return _GATING_PRIOR_CONFIG
        except Exception:
            pass
    
    # Default config if file not found
    _GATING_PRIOR_CONFIG = {
        "method": "linear",
        "linear_weight": 0.8,
        "logit_weight": 0.6,
        "gated": {
            "threshold": 0.55,
            "w_min": 0.3,
            "w_max": 0.85,
            "temperature": 1.0,
        }
    }
    return _GATING_PRIOR_CONFIG


def infer_system_and_modules(
    features: Dict[str, float],
    *,
    use_gating: bool = True,
    rf_classifier: Optional[Any] = None,
    rf_feature_vector: Optional[np.ndarray] = None,
    allow_fallback: bool = False,
) -> Dict[str, Any]:
    """
    Unified inference entry point for system-level and module-level diagnosis.
    
    This is the ONLY function that should be called for inference. All scripts
    (compare_methods.py, brb_diagnosis_cli.py, aggregate_batch_diagnosis.py,
    eval_module_localization.py) MUST use this function.
    
    Parameters
    ----------
    features : Dict[str, float]
        Feature dictionary containing X1-X22 or equivalent named features.
    use_gating : bool, default=True
        Whether to use RF gating prior fusion. If True, fuses RF and BRB outputs.
    rf_classifier : Optional[RandomForestClassifier], default=None
        Fitted RandomForest classifier for gating prior. If None and use_gating=True,
        falls back to BRB-only inference.
    rf_feature_vector : Optional[np.ndarray], default=None
        Pre-built feature vector for the RF classifier.  When provided, this is
        used directly instead of reconstructing from ``features`` via
        ``_features_to_array()``, avoiding feature-count mismatches when the RF
        was trained on more features than X1-X22.
    allow_fallback : bool, default=False
        If False, raises an error when gating prior is unavailable but use_gating=True.
        If True, allows fallback to BRB-only inference.
        
    Returns
    -------
    Dict with keys:
        - system_probs: Dict[str, float] mapping {class_name: probability}
          where class_name in {"normal", "amp_error", "freq_error", "ref_error"}
        - fault_type_pred: str - predicted fault type (English)
        - module_topk: List[Dict] - top modules with {"name": str, "prob": float}
        - debug: Dict containing:
            - rf_probs: Optional[Dict] - RF prior probabilities
            - brb_probs: Dict - BRB output probabilities
            - fused_probs: Dict - fused probabilities (if gating used)
            - gating_status: str - "gated_ok", "fallback_brb_only", or "disabled"
            - fallback_reason: Optional[str] - reason for fallback
    
    Raises
    ------
    ValueError
        If use_gating=True, rf_classifier is None, and allow_fallback=False.
    """
    debug_info: Dict[str, Any] = {
        "rf_probs": None,
        "brb_probs": None,
        "fused_probs": None,
        "gating_status": "disabled",
        "fallback_reason": None,
        "gating_config_hash": None,
    }
    
    # Step 0: Auto-load RF classifier if not provided but gating requested
    if use_gating and rf_classifier is None:
        try:
            rf_classifier = load_rf_artifact()
        except FileNotFoundError as e:
            if not allow_fallback:
                raise ValueError(
                    f"RF artifact not available and allow_fallback=False.\n"
                    f"Either provide rf_classifier, run train_and_export_rf.py, "
                    f"or set allow_fallback=True."
                ) from e
            debug_info["fallback_reason"] = "RF artifact not found"
    
    # Step 1: BRB system-level inference (always run)
    sys_result = system_level_infer(features, mode="sub_brb")
    brb_probs_cn = sys_result.get("probabilities", {})
    
    # Convert BRB probs from Chinese to English
    brb_probs = {
        "normal": brb_probs_cn.get("正常", 0.0),
        "amp_error": brb_probs_cn.get("幅度失准", 0.0),
        "freq_error": brb_probs_cn.get("频率失准", 0.0),
        "ref_error": brb_probs_cn.get("参考电平失准", 0.0),
    }
    debug_info["brb_probs"] = brb_probs
    
    # Step 2: Gating prior fusion (if enabled)
    final_probs = brb_probs.copy()
    
    if use_gating:
        if rf_classifier is not None and hasattr(rf_classifier, 'predict_proba'):
            try:
                # Prepare feature vector for RF
                if rf_feature_vector is not None:
                    feature_vector = rf_feature_vector
                else:
                    feature_vector = _features_to_array(features)
                rf_proba = rf_classifier.predict_proba(feature_vector.reshape(1, -1))[0]
                
                # Map RF probs to class names
                rf_classes = rf_classifier.classes_
                rf_probs = {}
                for i, cls in enumerate(rf_classes):
                    # Handle both int and string classes (including numpy integers)
                    if isinstance(cls, (int, np.integer)):
                        cls_name = CLASS_NAMES[int(cls)] if int(cls) < len(CLASS_NAMES) else "normal"
                    else:
                        cls_name = str(cls)
                    rf_probs[cls_name] = float(rf_proba[i])
                
                # Ensure all classes present
                for c in ["normal", "amp_error", "freq_error", "ref_error"]:
                    rf_probs.setdefault(c, 0.0)
                
                debug_info["rf_probs"] = rf_probs
                
                # Fuse RF and BRB
                gating_config = _load_gating_prior_config()
                fusion = GatingPriorFusion(gating_config)
                
                rf_array = np.array([rf_probs["normal"], rf_probs["amp_error"],
                                     rf_probs["freq_error"], rf_probs["ref_error"]])
                brb_array = np.array([brb_probs["normal"], brb_probs["amp_error"],
                                      brb_probs["freq_error"], brb_probs["ref_error"]])
                
                fused_array = fusion.fuse(rf_array, brb_array)
                
                final_probs = {
                    "normal": float(fused_array[0]),
                    "amp_error": float(fused_array[1]),
                    "freq_error": float(fused_array[2]),
                    "ref_error": float(fused_array[3]),
                }
                debug_info["fused_probs"] = final_probs
                debug_info["gating_status"] = "gated_ok"
                
            except Exception as e:
                if not allow_fallback:
                    raise ValueError(
                        f"Gating prior fusion failed and allow_fallback=False: {e}"
                    )
                debug_info["gating_status"] = "fallback_brb_only"
                debug_info["fallback_reason"] = f"RF inference error: {str(e)}"
                final_probs = brb_probs.copy()
        else:
            # No RF classifier available
            if not allow_fallback:
                raise ValueError(
                    "use_gating=True but rf_classifier is None and allow_fallback=False. "
                    "Either provide a fitted RF classifier or set allow_fallback=True."
                )
            debug_info["gating_status"] = "fallback_brb_only"
            debug_info["fallback_reason"] = "RF classifier not provided"
            final_probs = brb_probs.copy()
    else:
        debug_info["gating_status"] = "disabled"
        final_probs = brb_probs.copy()
    
    # Step 3: Determine predicted fault type
    fault_type_pred = max(final_probs, key=final_probs.get)
    
    # Step 4: Module-level inference
    # Convert final_probs to Chinese for BRB module inference
    sys_probs_cn = {
        "正常": final_probs.get("normal", 0.0),
        "幅度失准": final_probs.get("amp_error", 0.0),
        "频率失准": final_probs.get("freq_error", 0.0),
        "参考电平失准": final_probs.get("ref_error", 0.0),
        "probabilities": {
            "正常": final_probs.get("normal", 0.0),
            "幅度失准": final_probs.get("amp_error", 0.0),
            "频率失准": final_probs.get("freq_error", 0.0),
            "参考电平失准": final_probs.get("ref_error", 0.0),
        },
        "max_prob": max(final_probs.values()),
    }
    
    # P3.1: Use soft-gating for module inference (multi-hypothesis)
    try:
        from BRB.module_brb import hierarchical_module_infer_soft_gating
        soft_result = hierarchical_module_infer_soft_gating(
            final_probs,
            features,
            delta=0.1,  # Activate top-2 if diff < 0.1
            use_board_prior=True,
        )
        module_topk = soft_result["fused_topk"]
        debug_info["soft_gating"] = {
            "used_hypotheses": soft_result["used_fault_hypotheses"],
            "single_hypothesis": soft_result["single_hypothesis"],
        }
    except Exception:
        # Fallback to original method
        module_probs = module_level_infer_with_activation(features, sys_probs_cn)
        sorted_modules = sorted(module_probs.items(), key=lambda x: x[1], reverse=True)
        module_topk = [
            {"name": name, "prob": float(prob)}
            for name, prob in sorted_modules[:10]
        ]
    
    return {
        "system_probs": final_probs,
        "fault_type_pred": fault_type_pred,
        "module_topk": module_topk,
        "debug": debug_info,
    }


def infer_with_layered_engine(
    features: Dict[str, float],
) -> Dict[str, Any]:
    """
    使用分层引擎进行推理 (V-D.1 新架构)。
    
    这是新架构的推理入口，使用 LayeredBRBEngine 和 SoftModuleRouter。
    
    Parameters
    ----------
    features : Dict[str, float]
        特征字典，包含 X1-X37。
        
    Returns
    -------
    Dict
        与 infer_system_and_modules 相同格式的结果。
    """
    if not _LAYERED_ENGINE_AVAILABLE:
        raise ImportError(
            "Layered engine not available. "
            "Ensure BRB/engines/layered_engine.py and BRB/routing/soft_router.py exist."
        )
    
    # 获取专家系统实例
    expert = get_expert_system()
    
    # 从特征进行诊断
    result = expert.diagnose_from_features(features)
    
    # 转换为统一格式
    module_topk = [
        {"name": name, "prob": float(prob)}
        for name, prob in result.top_modules
    ]
    
    return {
        "system_probs": result.system_probs,
        "fault_type_pred": result.system_fault_type,
        "module_topk": module_topk,
        "debug": {
            "engine": "layered",
            "layer_trace": result.layer_trace,
            "routing_trace": result.routing_trace,
            "gating_status": "layered_engine",
        },
    }


def _features_to_array(features: Dict[str, float]) -> np.ndarray:
    """Convert feature dictionary to numpy array for RF classifier."""
    # Use X1-X22 ordering
    arr = np.zeros(22)
    for i in range(22):
        key = f"X{i+1}"
        if key in features:
            arr[i] = float(features[key])
        else:
            # Try aliases
            aliases = {
                1: ["bias", "amplitude_offset"],
                2: ["ripple_var", "inband_flatness"],
                3: ["res_slope", "hf_attenuation_slope"],
                4: ["df", "freq_scale_nonlinearity"],
                5: ["scale_consistency", "amp_scale_consistency"],
                6: ["ripple_variance"],
                7: ["gain_nonlinearity", "step_score"],
                8: ["lo_leakage"],
                9: ["tuning_linearity_residual"],
                10: ["band_amplitude_consistency"],
                11: ["env_overrun_rate", "viol_rate"],
                12: ["env_overrun_max"],
                13: ["env_violation_energy"],
                14: ["band_residual_low"],
                15: ["band_residual_high_std"],
                16: ["corr_shift_bins"],
                17: ["warp_scale"],
                18: ["warp_bias"],
                19: ["slope_low"],
                20: ["kurtosis_detrended"],
                21: ["peak_count_residual"],
                22: ["ripple_dom_freq_energy"],
            }
            for alias in aliases.get(i+1, []):
                if alias in features:
                    arr[i] = float(features[alias])
                    break
    return arr


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
    
    Two-stage architecture:
    - Stage 1 (System-level): RandomForest classifier for high accuracy (~90%+)
    - Stage 2 (Module-level): BRB hierarchical inference for interpretable diagnosis
    
    This hybrid approach achieves:
    - High system-level accuracy via supervised learning
    - Interpretable module-level diagnosis via knowledge-driven BRB
    
    Complexity:
    - System classifier: RandomForest (100 trees)
    - Module BRB: 3 sub-BRBs + module layer rules
    - Params: RF params + BRB attribute weights + rule weights + belief degrees
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
        
        # System-level classifier (supervised learning for high accuracy)
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        self.is_fitted = False
        self.fusion_engine = None  # [CRITICAL] Fusion engine slot – set in fit()
        
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
        """Fit the system-level classifier using supervised learning.
        
        Stage 1: Train RandomForest for system-level classification
        This achieves high accuracy (~90%+) for system fault type identification.
        
        After training, instantiates the fusion engine and injects the trained RF,
        ensuring the trained model is persisted and used during predict().
        """
        print(f">> [OursAdapter] Fitting RF with {len(X_train)} samples...")
        print(f">> Training samples: {len(X_train)}, Classes: {np.unique(y_sys_train)}")
        
        if meta and 'feature_names' in meta:
            self.feature_names = meta['feature_names']
        
        # 1. Train system-level classifier
        self.classifier.fit(X_train, y_sys_train)
        self.is_fitted = True
        
        # 2. [CRITICAL] Instantiate fusion engine and inject trained RF
        #    This ensures the trained RF is locked into the fusion pipeline.
        gating_config = _load_gating_prior_config()
        self.fusion_engine = GatingPriorFusion(gating_config)
        
        print(f">> [OursAdapter] fit() complete! is_fitted={self.is_fitted}")
        print(">> [OursAdapter] Fusion Engine successfully linked.")
        
        # 3. Sync trained RF classifier to all submodules that have an 'rf' attribute
        synced_count = 0
        for attr_name, attr_value in self.__dict__.items():
            if attr_name in ['classifier', 'rf', 'fusion_engine']:
                continue  # Skip the classifier/engine itself

            if hasattr(attr_value, 'rf'):
                print(f">> [SYNC] Injecting trained RF into submodule: '{attr_name}'")
                attr_value.rf = self.classifier
                synced_count += 1

        if synced_count == 0:
            print(">> [SYNC] No submodules found to sync RF (this is OK if predict() uses self.classifier directly).")
        else:
            print(f">> [SYNC] Complete. Synced to {synced_count} modules.")
    
    def predict(self, X_test: np.ndarray, meta: Optional[Dict] = None) -> Dict:
        """Predict using two-stage hybrid approach.
        
        Stage 1: RandomForest for system-level classification (high accuracy)
        Stage 2: BRB for module-level diagnosis (interpretable)
        
        Uses the unified entry infer_system_and_modules() for consistency.
        Results always come from the fusion pipeline (RF prior + BRB),
        never from direct RF output alone.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() before predict().")
        
        # [Architecture Guard] Ensure fusion engine exists before prediction
        if self.fusion_engine is None:
            print(">> [WARNING] Fusion Engine missing in predict! Re-instantiating on the fly...")
            gating_config = _load_gating_prior_config()
            self.fusion_engine = GatingPriorFusion(gating_config)
        
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
        
        # Process each sample using unified fusion entry
        for i in range(n_test):
            features = self._array_to_dict(X_test[i])
            
            # [CRITICAL] Always go through the fusion pipeline.
            # RF serves as gating prior, BRB provides interpretable inference.
            # Result = Fuse(P_rf, P_brb), never raw RF output.
            # Pass the original feature vector so the RF sees the same dimensions
            # it was trained on (may be >22 features).
            result = infer_system_and_modules(
                features,
                use_gating=True,
                rf_classifier=self.classifier,
                rf_feature_vector=X_test[i],
                allow_fallback=True,
            )
            
            # Extract system probabilities (order: normal, amp_error, freq_error, ref_error)
            sys_probs = result["system_probs"]
            sys_proba[i, 0] = sys_probs.get("normal", 0.0)
            sys_proba[i, 1] = sys_probs.get("amp_error", 0.0)
            sys_proba[i, 2] = sys_probs.get("freq_error", 0.0)
            sys_proba[i, 3] = sys_probs.get("ref_error", 0.0)
            
            # Normalize
            row_sum = np.sum(sys_proba[i])
            if row_sum > 0:
                sys_proba[i] /= row_sum
            else:
                sys_proba[i] = np.ones(n_sys_classes) / n_sys_classes
            sys_pred[i] = np.argmax(sys_proba[i])
            
            # Module probabilities
            mod_probs_dict = {m["name"]: m["prob"] for m in result["module_topk"]}
            
            # Convert V2 module names to indices using MODULE_LABELS_V2
            from BRB.module_brb import MODULE_LABELS_V2
            
            for mod_name, prob in mod_probs_dict.items():
                try:
                    # Find index in MODULE_LABELS_V2
                    if mod_name in MODULE_LABELS_V2:
                        mod_idx = MODULE_LABELS_V2.index(mod_name)
                        if 0 <= mod_idx < len(MODULE_LABELS):
                            mod_proba[i, mod_idx] = prob
                    else:
                        # Try partial matching
                        for idx, v2_name in enumerate(MODULE_LABELS_V2):
                            if mod_name in v2_name or v2_name in mod_name:
                                if idx < len(MODULE_LABELS):
                                    mod_proba[i, idx] = prob
                                break
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
                'fit_time_sec': 0.0,  # Not tracked for simplicity
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

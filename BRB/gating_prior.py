"""
门控先验融合模块 (Gating Prior Fusion Module)

将 RandomForest 作为门控先验，与系统级 BRB 输出融合。

架构:
    特征 → RandomForest → prior_probs ─┐
                                       ├→ fuse() → fused_probs
    特征 → system_level_infer → brb_probs ─┘

融合方案:
    S1: 线性加权融合
    S2: Logit 融合 (推荐)
    S3: 置信度门控融合 (论文推荐)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

# 融合配置
FUSION_CONFIG = {
    # 默认融合方法
    "method": "gated",  # "linear", "logit", "gated"
    
    # 线性融合权重
    "linear_weight": 0.7,  # w * rf + (1-w) * brb
    
    # Logit 融合权重
    "logit_weight": 0.6,
    
    # 置信度门控参数
    "gated": {
        "threshold": 0.55,  # 置信度阈值
        "w_min": 0.3,       # 最小 RF 权重
        "w_max": 0.85,      # 最大 RF 权重
        "temperature": 1.0,  # 温度校准
    }
}

# 类别映射
CLASS_NAMES = ["normal", "amp_error", "freq_error", "ref_error"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """温度校准的 Softmax"""
    x = np.asarray(x, dtype=np.float64)
    x = x / temperature
    x = x - np.max(x)  # 数值稳定
    exp_x = np.exp(x)
    return exp_x / (np.sum(exp_x) + 1e-10)


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """概率转 logit"""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


class GatingPriorFusion:
    """门控先验融合器"""
    
    def __init__(self, config: Optional[Dict] = None, rf_model=None):
        self.config = config or FUSION_CONFIG
        self.method = self.config.get("method", "gated")
        self.rf_model = rf_model
    
    def fuse(self, rf_probs: np.ndarray, brb_probs: np.ndarray) -> np.ndarray:
        """
        融合 RF 先验与 BRB 输出
        
        Args:
            rf_probs: RF 输出的概率分布 [p_normal, p_amp, p_freq, p_ref]
            brb_probs: 系统级 BRB 输出的概率分布
            
        Returns:
            fused_probs: 融合后的概率分布
        """
        rf_probs = np.asarray(rf_probs, dtype=np.float64)
        brb_probs = np.asarray(brb_probs, dtype=np.float64)
        
        # 归一化输入
        rf_probs = rf_probs / (np.sum(rf_probs) + 1e-10)
        brb_probs = brb_probs / (np.sum(brb_probs) + 1e-10)
        
        if self.method == "linear":
            return self.fuse_linear(rf_probs, brb_probs)
        elif self.method == "logit":
            return self.fuse_logit(rf_probs, brb_probs)
        elif self.method == "gated":
            return self.fuse_gated(rf_probs, brb_probs)
        else:
            raise ValueError(f"Unknown fusion method: {self.method}")
    
    def fuse_linear(self, rf_probs: np.ndarray, brb_probs: np.ndarray) -> np.ndarray:
        """
        S1: 线性加权融合
        
        p_fused = normalize(w * p_rf + (1-w) * p_brb)
        """
        w = self.config.get("linear_weight", 0.7)
        fused = w * rf_probs + (1 - w) * brb_probs
        return fused / (np.sum(fused) + 1e-10)
    
    def fuse_logit(self, rf_probs: np.ndarray, brb_probs: np.ndarray) -> np.ndarray:
        """
        S2: Logit 融合 (更像贝叶斯证据融合)
        
        logit(p_fused) = w * logit(p_rf) + (1-w) * logit(p_brb)
        """
        w = self.config.get("logit_weight", 0.6)
        
        rf_logits = logit(rf_probs)
        brb_logits = logit(brb_probs)
        
        fused_logits = w * rf_logits + (1 - w) * brb_logits
        return softmax(fused_logits)
    
    def fuse_gated(self, rf_probs: np.ndarray, brb_probs: np.ndarray) -> np.ndarray:
        """
        S3: 置信度门控融合 (论文推荐)
        
        1) 计算 RF 置信度 c = max(p_rf)
        2) 动态权重 w = clip((c - c0)/(1-c0), w_min, w_max)
        3) Logit 融合
        
        解释: RF 只在"它很确定时"提供强先验；不确定时退回 BRB 的可解释推理
        """
        gated_config = self.config.get("gated", {})
        threshold = gated_config.get("threshold", 0.55)
        w_min = gated_config.get("w_min", 0.3)
        w_max = gated_config.get("w_max", 0.85)
        temperature = gated_config.get("temperature", 1.0)
        
        # 温度校准
        if temperature != 1.0:
            rf_probs = softmax(logit(rf_probs), temperature)
        
        # 计算置信度
        confidence = np.max(rf_probs)
        
        # 动态权重
        if confidence >= threshold:
            # 线性映射到 [w_min, w_max]
            w = w_min + (confidence - threshold) / (1 - threshold) * (w_max - w_min)
            w = np.clip(w, w_min, w_max)
        else:
            w = w_min
        
        # Logit 融合
        rf_logits = logit(rf_probs)
        brb_logits = logit(brb_probs)
        fused_logits = w * rf_logits + (1 - w) * brb_logits
        
        return softmax(fused_logits)
    
    def get_fusion_info(self, rf_probs: np.ndarray, brb_probs: np.ndarray) -> Dict:
        """获取融合详情（用于调试和解释）"""
        rf_probs = np.asarray(rf_probs, dtype=np.float64)
        brb_probs = np.asarray(brb_probs, dtype=np.float64)
        
        fused_probs = self.fuse(rf_probs, brb_probs)
        
        # 计算置信度和权重
        confidence = float(np.max(rf_probs))
        gated_config = self.config.get("gated", {})
        threshold = gated_config.get("threshold", 0.55)
        w_min = gated_config.get("w_min", 0.3)
        w_max = gated_config.get("w_max", 0.85)
        
        if confidence >= threshold:
            w = w_min + (confidence - threshold) / (1 - threshold) * (w_max - w_min)
            w = float(np.clip(w, w_min, w_max))
        else:
            w = w_min
        
        return {
            "method": self.method,
            "rf_probs": rf_probs.tolist(),
            "brb_probs": brb_probs.tolist(),
            "fused_probs": fused_probs.tolist(),
            "rf_confidence": confidence,
            "fusion_weight": w,
            "rf_pred": CLASS_NAMES[int(np.argmax(rf_probs))],
            "brb_pred": CLASS_NAMES[int(np.argmax(brb_probs))],
            "fused_pred": CLASS_NAMES[int(np.argmax(fused_probs))],
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Execute batch fusion prediction: P_final = alpha * P_rf + (1-alpha) * P_brb.

        Uses the stored ``rf_model`` for the RF prior.  BRB likelihood is
        temporarily approximated as a uniform distribution until BRB inference
        is fully integrated.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            Feature matrix.

        Returns
        -------
        np.ndarray, shape (N,)
            Predicted class indices.
        """
        # 1. Get RF prior
        if self.rf_model is not None and hasattr(self.rf_model, 'predict_proba'):
            rf_probs = self.rf_model.predict_proba(X)
        else:
            print("> [Fusion Error] RF model missing! Returning random.")
            return np.random.randint(0, len(CLASS_NAMES), size=len(X))

        # 2. BRB likelihood – uniform placeholder until real BRB is wired in
        # TODO: replace with real BRB inference (brb_probs = self.brb.infer(X))
        n_classes = rf_probs.shape[1]
        brb_probs = np.full_like(rf_probs, 1.0 / n_classes)

        # 3. Linear fusion (weighted average)
        # Give RF high weight (0.8) to preserve accuracy baseline
        alpha = self.config.get("linear_weight", 0.8)
        final_probs = alpha * rf_probs + (1 - alpha) * brb_probs

        # Debug log (sampled)
        if np.random.rand() < 0.01:
            print(f"> [Fusion Debug] RF: {rf_probs[0]}, BRB: {brb_probs[0]} -> Final: {final_probs[0]}")

        return np.argmax(final_probs, axis=1)


def create_fusion_instance(method: str = "gated", **kwargs) -> GatingPriorFusion:
    """创建融合器实例"""
    config = FUSION_CONFIG.copy()
    config["method"] = method
    config.update(kwargs)
    return GatingPriorFusion(config)


# 便捷函数
def fuse_rf_brb(rf_probs: np.ndarray, brb_probs: np.ndarray, 
                method: str = "gated") -> np.ndarray:
    """便捷融合函数"""
    fusion = create_fusion_instance(method)
    return fusion.fuse(rf_probs, brb_probs)


if __name__ == "__main__":
    # 测试
    print("=== 门控先验融合测试 ===\n")
    
    # 模拟数据
    rf_probs = np.array([0.1, 0.7, 0.15, 0.05])   # RF 判断 amp_error
    brb_probs = np.array([0.3, 0.4, 0.2, 0.1])    # BRB 也倾向 amp_error 但不确定
    
    print(f"RF 输出: {rf_probs}")
    print(f"BRB 输出: {brb_probs}")
    print()
    
    for method in ["linear", "logit", "gated"]:
        fusion = create_fusion_instance(method)
        fused = fusion.fuse(rf_probs, brb_probs)
        info = fusion.get_fusion_info(rf_probs, brb_probs)
        
        print(f"方法: {method}")
        print(f"  融合结果: {fused}")
        print(f"  预测: {info['fused_pred']}")
        if method == "gated":
            print(f"  RF置信度: {info['rf_confidence']:.3f}")
            print(f"  融合权重: {info['fusion_weight']:.3f}")
        print()

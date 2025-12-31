"""
BIZRA PEAK MASTERPIECE - SNR Autonomous Engine
═══════════════════════════════════════════════════════════════════════════════
Elite Practitioner Pattern: Dynamic Cognitive Homeostasis
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, List
from core.snr_scorer import SNRMetrics, SNRLevel

logger = logging.getLogger("BIZRA_SNR_AUTO")

@dataclass
class CognitiveParameters:
    beam_width: int
    max_depth: int
    attention_mask_sharpness: float
    min_snr_threshold: float

class SNROptimizer:
    """
    Autonomous engine that maximize Signal-to-Noise Ratio.
    Tunes the hyperparameters of the GoT engine and Perceptual Buffer.
    """
    def __init__(self, initial_params: CognitiveParameters):
        self.params = initial_params
        self.history: List[SNRMetrics] = []
        self.target_snr = 0.85 # Goal: consistently HIGH SNR

    def update(self, latest_metrics: SNRMetrics) -> CognitiveParameters:
        """
        Adjust parameters based on latest telemetry.
        """
        self.history.append(latest_metrics)
        if len(self.history) > 10:
            self.history.pop(0)

        current_snr = latest_metrics.snr_score
        
        # 1. NOISE FLOOR REDUCTION
        # If noise is high (entropy/quantization), sharpen attention and reduce beam width
        if latest_metrics.noise_floor > 0.4:
            self.params.beam_width = max(3, self.params.beam_width - 1)
            self.params.attention_mask_sharpness = min(10.0, self.params.attention_mask_sharpness + 0.5)
            logger.info(f"[SNR_AUTO] Noise floor high. Sharpening attention. BeamWidth -> {self.params.beam_width}")

        # 2. SIGNAL AMPLIFICATION
        # If signal is weak, increase depth for deeper interdisciplinary connections
        if latest_metrics.signal_strength < 0.3:
            self.params.max_depth = min(10, self.params.max_depth + 1)
            logger.info(f"[SNR_AUTO] Signal weak. Increasing depth. MaxDepth -> {self.params.max_depth}")

        # 3. ETHICAL CONVERGENCE
        # If Ihsān drops, tighten the SNR threshold to avoid 'low-ethics' noise
        if latest_metrics.ihsan_metric < 0.95:
            self.params.min_snr_threshold = min(0.9, self.params.min_snr_threshold + 0.05)
            logger.info(f"[SNR_AUTO] Ihsān drop. Tightening SNR threshold -> {self.params.min_snr_threshold}")
        else:
            # Gradually relax threshold if stable
            self.params.min_snr_threshold = max(0.3, self.params.min_snr_threshold - 0.01)

        return self.params

    def get_status(self) -> Dict[str, Any]:
        return {
            "current_params": self.params.__dict__,
            "avg_history_snr": sum(m.snr_score for m in self.history) / len(self.history) if self.history else 0.0
        }

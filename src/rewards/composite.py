"""Composite reward registry and orchestrator for GRPO training."""

from typing import Any, Callable, Dict, List, Optional

from src.rewards.accuracy import (
    ClassWeightedAccuracyReward,
    compute_emotion_weights,
    reward_label_accuracy_weighted,
)
from src.rewards.acoustic import AcousticInventoryReward, reward_acoustic_inventory
from src.rewards.format import FormatReward, reward_format
from src.rewards.parser import parse_completion


def unified_gated_grpo_reward(
    prompts: List[str],
    completions: List[str],
    w_r1: float = 1.5,
    w_r2: float = 1.0,
    w_r3: float = 0.2,
    gate_floor: float = 0.15,
    normalize_weights: bool = False,
    scale_r2_to_unit: bool = True,
    r1_reward_func: Optional[Callable] = None,
    r2_reward_func: Optional[Callable] = None,
    r3_reward_func: Optional[Callable] = None,
    **kwargs: Any,
) -> List[float]:
    """Computes the unified GRPO reward with causal soft-gating normalized by per-sample R2_max.

    Formula:
      R_unified = w_r1 * [R1 * (gate_floor + (1 - gate_floor) * R2_gate)] + w_r2 * R2 + w_r3 * R3
    """
    if "parsed_completions" not in kwargs:
        kwargs["parsed_completions"] = [parse_completion(c) for c in completions]

    if r1_reward_func is not None:
        r1_scores = r1_reward_func(prompts, completions, **kwargs)
    elif "label" in kwargs:
        r1_scores = reward_label_accuracy_weighted(prompts, completions, **kwargs)
    else:
        r1_scores = [1.0] * len(completions)

    if r2_reward_func is not None:
        try:
            r2_out = r2_reward_func(prompts, completions, return_gate_scores=True, **kwargs)
            if isinstance(r2_out, tuple) and len(r2_out) == 2:
                r2_scores, r2_gate_scores = r2_out
            else:
                r2_scores = r2_out
                r2_gate_scores = r2_out
        except TypeError:
            r2_scores = r2_reward_func(prompts, completions, **kwargs)
            r2_gate_scores = r2_scores
    else:
        r2_scores, r2_gate_scores = reward_acoustic_inventory(
            prompts, completions, normalize=True, return_gate_scores=True, **kwargs
        )

    if r3_reward_func is not None:
        r3_scores = r3_reward_func(prompts, completions, **kwargs)
    else:
        r3_scores = reward_format(prompts, completions, **kwargs)

    total_weight = (w_r1 + w_r2 + w_r3) if normalize_weights else 1.0

    unified_scores: List[float] = []
    for r1, r2, r2_gate, r3 in zip(r1_scores, r2_scores, r2_gate_scores, r3_scores):
        r1_val = float(r1)
        r2_val = max(0.0, min(1.0, float(r2)))
        r2_gate_val = max(0.0, min(1.0, float(r2_gate)))
        r3_val = max(0.0, min(1.0, float(r3)))

        causal_gate = gate_floor + (1.0 - gate_floor) * r2_gate_val
        gated_r1 = r1_val * causal_gate

        total = (w_r1 * gated_r1 + w_r2 * r2_val + w_r3 * r3_val) / total_weight
        unified_scores.append(round(float(total), 4))

    return unified_scores


unified_grpo_reward = unified_gated_grpo_reward


class RewardManager:
    """Manages multi-objective reward functions for GRPO rollouts."""

    def __init__(
        self,
        class_weights: Optional[Dict[str, float]] = None,
        quantiles_ref: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
        audio_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
        normalize_acoustic: bool = True,
    ):
        self.r1_accuracy = ClassWeightedAccuracyReward(class_weights=class_weights)
        self.r2_acoustic = AcousticInventoryReward(
            quantiles_ref=quantiles_ref,
            audio_lookup=audio_lookup,
            normalize=normalize_acoustic,
        )
        self.r3_format = FormatReward()

    def get_reward_functions(self) -> List[Callable]:
        """Returns list of individual reward functions compatible with TRL GRPOTrainer."""
        return [
            self.r1_accuracy,
            self.r2_acoustic,
            self.r3_format,
        ]

    def get_unified_reward_function(
        self,
        w_r1: float = 1.5,
        w_r2: float = 1.0,
        w_r3: float = 0.2,
        normalize_weights: bool = True,
        scale_r2_to_unit: bool = True,
    ) -> Callable:
        """Returns a single unified reward function computing the normalized weighted sum of R1, R2, R3."""
        def _unified_reward(prompts: List[str], completions: List[str], **kwargs: Any) -> List[float]:
            return unified_gated_grpo_reward(
                prompts=prompts,
                completions=completions,
                w_r1=w_r1,
                w_r2=w_r2,
                w_r3=w_r3,
                normalize_weights=normalize_weights,
                scale_r2_to_unit=scale_r2_to_unit,
                r1_reward_func=self.r1_accuracy,
                r2_reward_func=self.r2_acoustic,
                r3_reward_func=self.r3_format,
                **kwargs,
            )

        return _unified_reward

    def update_class_weights(self, class_weights: Dict[str, float]) -> None:
        """Update R1 class rarity weights."""
        self.r1_accuracy.set_weights(class_weights)

    def update_quantiles(self, quantiles_ref: Dict[str, Dict[str, Dict[str, float]]]) -> None:
        """Update R2 empirical quantile thresholds."""
        self.r2_acoustic.set_quantiles(quantiles_ref)

    def update_audio_lookup(self, audio_lookup: Dict[str, Dict[str, Any]]) -> None:
        """Update R2 audio path feature lookup."""
        self.r2_acoustic.set_audio_lookup(audio_lookup)

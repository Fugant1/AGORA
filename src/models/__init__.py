"""Model loading, frozen SFT reference context, and 24 GB-safe vLLM rollout engine."""

from src.models.qwen_omni import generation_eos_ids, load_thinker_model, reference_policy
from src.models.vllm_engine import VLLMRolloutEngine

__all__ = [
    "generation_eos_ids",
    "load_thinker_model",
    "reference_policy",
    "VLLMRolloutEngine",
]

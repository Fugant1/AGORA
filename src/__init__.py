"""AGORA: Acoustic-Grounded Omni-modal Reasoning for Affect (V4 SFT-MT + Hybrid Cue-Split GRPO v3).

Modular Python package matching `data_processing.ipynb`, `ser_sft.ipynb`, and `ser_grpo_v3.ipynb`,
with colocated 24 GB-safe vLLM rollout generation for GRPO.
"""

__version__ = "0.3.0"

from src.config import (
    ACOUSTIC_CUES,
    CUE_COLUMNS,
    CUE_LABELS,
    CUES,
    DEFAULT_REWARD_WEIGHTS,
    EMOTIONS,
    GLOBAL_CUES,
    MODEL_ID,
    PROMPT_VERSION,
    TIMBRE_LABEL_MAP,
    VALID_EMOTIONS,
    DataProcessingConfig,
    ExperimentConfig,
    GRPOConfig,
    ModelConfig,
    RewardConfig,
    SFTConfig,
    VLLMConfig,
)
from src.pipeline import AGORAPipeline, run_data_processing

# Backward-compatible alias for previously released Python callers.
OPERAPipeline = AGORAPipeline
from src.prompts import (
    PERSONA,
    build_bridge_generation_prompt,
    format_sft_target,
    gender_context,
    inventory_text,
    messages_for,
    student_instruction,
)
from src.rewards import (
    GRPORewardCalibrator,
    parse_completion,
    parse_completion_strict,
    unified_gated_grpo_reward,
    validate_grpo_reward_mathematics,
)

__all__ = [
    "__version__",
    "MODEL_ID",
    "PROMPT_VERSION",
    "CUES",
    "CUE_COLUMNS",
    "CUE_LABELS",
    "GLOBAL_CUES",
    "EMOTIONS",
    "VALID_EMOTIONS",
    "TIMBRE_LABEL_MAP",
    "ACOUSTIC_CUES",
    "DEFAULT_REWARD_WEIGHTS",
    "DataProcessingConfig",
    "ModelConfig",
    "SFTConfig",
    "RewardConfig",
    "VLLMConfig",
    "GRPOConfig",
    "ExperimentConfig",
    "AGORAPipeline",
    "OPERAPipeline",
    "run_data_processing",
    "PERSONA",
    "gender_context",
    "inventory_text",
    "student_instruction",
    "build_bridge_generation_prompt",
    "format_sft_target",
    "messages_for",
    "parse_completion",
    "parse_completion_strict",
    "GRPORewardCalibrator",
    "unified_gated_grpo_reward",
    "validate_grpo_reward_mathematics",
]

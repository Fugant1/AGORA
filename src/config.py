"""Canonical constants and typed experiment configurations for AGORA (MELD SER SFT + GRPO v3).

Matches `data_processing.ipynb`, `ser_sft.ipynb`, and `ser_grpo_v3.ipynb` exactly,
with 24 GB VRAM + vLLM rollout defaults for local NVIDIA GPU execution.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Optional, Tuple

# ==============================================================================
# Canonical V4 Schema Constants (Exact Parity with Notebooks)
# ==============================================================================
MODEL_ID: str = "Qwen/Qwen2.5-Omni-7B"
PROMPT_VERSION: str = "gender_wxrate_4cue_srfirst_teacher_v5_short_concise"
TARGET_SR: int = 16000
TARGET_CHANNELS: int = 1

CUES: Tuple[str, ...] = ("speaking_rate", "pitch_level", "vocal_energy", "timbre")
CUE_COLUMNS: Dict[str, str] = {
    "speaking_rate": "whisperx_syllable_rate_cue",
    "pitch_level": "pitch_level_cue",
    "vocal_energy": "vocal_energy_cue",
    "timbre": "timbre_cue",
}
GLOBAL_CUES: FrozenSet[str] = frozenset({"speaking_rate"})

EMOTIONS: Tuple[str, ...] = (
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
)
VALID_EMOTIONS: FrozenSet[str] = frozenset(EMOTIONS)
VALID_EMOTIONS_LIST = list(EMOTIONS)

TIMBRE_LABEL_MAP: Dict[str, str] = {
    "low": "low (muffled/soft)",
    "moderate": "moderate (modal)",
    "high": "high (tense/bright)",
}

CUE_LABELS: Dict[str, Tuple[str, str, str]] = {
    "speaking_rate": ("low", "moderate", "high"),
    "pitch_level": ("low", "moderate", "high"),
    "vocal_energy": ("low", "moderate", "high"),
    "timbre": ("low (muffled/soft)", "moderate (modal)", "high (tense/bright)"),
}

# Full 5-cue openSMILE eGeMAPSv02 extraction map from data_processing.ipynb
ACOUSTIC_CUES: Dict[str, str] = {
    "pitch_level": "F0semitoneFrom27.5Hz_sma3nz_percentile50.0",
    "pitch_variation": "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
    "vocal_energy": "loudness_sma3_percentile50.0",
    "speaking_rate": "VoicedSegmentsPerSec",
    "timbre": "alphaRatioV_sma3nz_amean",
}
ACOUSTIC_FEATURE_COLS = list(ACOUSTIC_CUES.values())

DEFAULT_REWARD_WEIGHTS: Dict[str, float] = {"r1": 1.0, "r2": 1.0, "r3": 0.2}


@dataclass
class DataProcessingConfig:
    """Paths and hyperparameters for MELD preprocessing, EMO-TAG, eGeMAPSv02, and WhisperX rates."""

    meld_dir: Path = Path("raw_data/prep_meld")
    archive_path: Optional[Path] = None
    data_dir: Path = Path("data/meld_16k_mono")
    drive_data_dir: Optional[Path] = None
    rvad_repo_dir: Path = Path("third_party/rVAD")

    target_sr: int = TARGET_SR
    target_channels: int = TARGET_CHANNELS
    max_workers: int = 8
    overwrite_audio: bool = False

    # EMO-TAG (Mohamed et al., 2026) constants
    whisper_model_name: str = "large-v3"
    language: str = "en"
    pause_max_rms_dbfs: float = -35.0
    pause_max_peak_dbfs: float = -25.0
    boundary_margin_sec: float = 0.05
    max_consecutive_insertions: int = 2
    max_replacements: int = 3
    min_speech_duration_sec: float = 0.15

    # Training integrity bounds
    min_train_duration_sec: float = 0.5
    max_train_duration_sec: float = 20.0
    delete_rejected_wavs: bool = True

    # Batched WhisperX true syllable/word rate tracking
    whisperx_rate_batch_size: int = 32
    whisperx_rate_silence_pad_sec: float = 0.50
    whisperx_rate_checkpoint_every_batches: int = 25
    overwrite_whisperx_rate: bool = False

    @property
    def clean_dir(self) -> Path:
        return Path(self.data_dir) / "emo_tag_clean"

    @property
    def train_csv(self) -> Path:
        return self.clean_dir / "clean_train_set_with_cues.csv"

    @property
    def val_csv(self) -> Path:
        return self.clean_dir / "clean_val_set_with_cues.csv"

    @property
    def quartiles_json(self) -> Path:
        return self.clean_dir / "gender_acoustic_quartiles.json"


@dataclass
class ModelConfig:
    """Qwen2.5-Omni-7B Thinker loading, 4-bit QLoRA quantization, and LoRA config (24 GB VRAM safe)."""

    model_id: str = MODEL_ID
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    attn_implementation: Optional[str] = "sdpa"

    # LoRA hyperparameters (exact match to ser_sft.ipynb & ser_grpo_v3.ipynb)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_bias: str = "none"
    target_modules: str = (
        r"model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
        r"mlp\.(gate_proj|up_proj|down_proj))"
    )


@dataclass
class SFTConfig:
    """Stage-1 Multitask Cue-Weighted SFT configuration (exact match to ser_sft.ipynb)."""

    project_dir: Path = Path(".")
    data_dir: Path = Path("data/meld_16k_mono")
    drive_data_dir: Optional[Path] = None
    split_dir: Path = Path("data/sft_grpo_gender_wxrate_60_40_semi25_v5")
    run_name: str = "full_sft_gender_wxrate_multitask_alpha05_semi25_seed42_v5"
    partition_mode: str = "semi_balanced"  # 'semi_balanced' (all non-neutral + 25% neutral) | 'uniform'
    neutral_target_fraction: float = 0.25

    seed: int = 42
    epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_steps: int = 5
    save_total_limit: int = 2

    # Multitask cue weighting (V4 winning SFT-MT)
    cue_weight_alpha: float = 0.5
    equal_multitask_cue_loss: bool = True
    cue_task_share: float = 0.25

    generation_batch_size: int = 16
    eval_batch_size: int = 16
    max_new_tokens: int = 512
    evaluate_before: bool = True
    prompt_version: str = PROMPT_VERSION

    @property
    def output_dir(self) -> Path:
        return Path(self.project_dir) / "runs" / "ser_sft" / self.run_name

    @property
    def final_adapter_dir(self) -> Path:
        return self.output_dir / "final_sft_adapter"

    @property
    def sft_csv(self) -> Path:
        return Path(self.split_dir) / "sft_train.csv"

    @property
    def grpo_csv(self) -> Path:
        return Path(self.split_dir) / "grpo_train.csv"

    @property
    def split_meta(self) -> Path:
        return Path(self.split_dir) / "split_manifest.json"


@dataclass
class RewardConfig:
    """Multi-objective GRPO reward calibration flags (decoupled R1 emotion alpha vs R2 cue alpha)."""

    use_zero_mean: bool = False
    use_map: bool = False
    use_inverse_frequency: bool = True
    # R1 emotion exponent (applied OUTSIDE group_standardize so imbalanced splits preserve class weighting)
    r1_inverse_frequency_alpha: float = 0.5
    # R2 acoustic cue exponent (1.0 = full inverse-frequency, proven best on Emotion MF1 & UAR)
    inverse_frequency_alpha: float = 1.0
    use_distance_based_payoff: bool = False
    reward_weights: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_REWARD_WEIGHTS)
    )


@dataclass
class VLLMConfig:
    """Colocated vLLM rollout engine settings tuned for a 24 GB NVIDIA GPU."""

    enabled: bool = True
    # 'bitsandbytes' (~5.8 GB weights) fits comfortably alongside 4-bit QLoRA (~6 GB) on 24 GB VRAM.
    # For 48/80 GB GPUs or Ada/Hopper fp8, set quantization='fp8' or None.
    quantization: Optional[str] = "bitsandbytes"
    load_format: Optional[str] = "bitsandbytes"
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.35
    max_model_len: int = 2048
    max_num_seqs: int = 32
    max_lora_rank: int = 16
    enable_sleep_mode: bool = True
    sleep_level: int = 1
    enforce_eager: bool = False
    tensor_parallel_size: int = 1
    lora_sync_dir: Path = field(
        default_factory=lambda: Path(f"/dev/shm/agora_vllm_lora_{os.environ.get('USER', 'user')}")
    )
    use_for_validation: bool = True


@dataclass
class GRPOConfig:
    """Stage-2 Hybrid Cue-Split GRPO v3 configuration (exact match to ser_grpo_v3.ipynb)."""

    project_dir: Path = Path(".")
    data_dir: Path = Path("data/meld_16k_mono")
    drive_data_dir: Optional[Path] = None
    split_dir: Path = Path("data/sft_grpo_gender_wxrate_60_40_semi25_v5")
    sft_run_name: str = "full_sft_gender_wxrate_multitask_alpha05_semi25_seed42_v5"
    run_name: str = "grpo_v3_hybrid_invfreq_4cue_mt_sft_semi25_seed42_cuekl005_v5"
    partition_mode: str = "semi_balanced"  # 'semi_balanced' | 'uniform'
    resume_checkpoint_dir: Optional[str] = None

    use_sft_model: bool = True
    seed: int = 42

    # 1 full epoch over 420 grpo_train rows at G=8:
    # - effective batch = 4 * 8 = 32 completions (4 unique audios) per optimizer step -> 105 exact steps.
    # - gen_batch = 4 * 8 = 32 completions (4 unique audios x 8) -> 100% strictly on-policy.
    # Mathematically identical to (batch=2, steps_per_gen=8, grad_acc=16).
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    steps_per_generation: int = 8
    gradient_accumulation_steps: int = 16
    num_generations: int = 8
    temperature: float = 1.0
    learning_rate: float = 1e-5
    warmup_steps: int = 5
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0

    # Hybrid cue-split credit assignment + L/n-scaled cue-only KL anchor
    loss_type: str = "hybrid"
    use_split_advantage: bool = True
    use_split_r2: bool = True
    beta: float = 0.05
    cue_kl_only: bool = True

    max_completion_length: int = 256
    eval_max_new_tokens: int = 512
    eval_batch_size: int = 16
    save_steps: int = 25
    save_total_limit: int = 6
    logging_steps: int = 5
    prompt_version: str = PROMPT_VERSION

    @property
    def sft_adapter_path(self) -> Path:
        return Path(self.project_dir) / "runs" / "ser_sft" / self.sft_run_name / "final_sft_adapter"

    @property
    def output_dir(self) -> Path:
        return Path(self.project_dir) / "runs" / "ser_grpo" / self.run_name

    @property
    def final_adapter_dir(self) -> Path:
        return self.output_dir / "final_grpo_adapter"

    @property
    def sft_csv(self) -> Path:
        return Path(self.split_dir) / "sft_train.csv"

    @property
    def grpo_csv(self) -> Path:
        return Path(self.split_dir) / "grpo_train.csv"

    @property
    def split_meta(self) -> Path:
        return Path(self.split_dir) / "split_manifest.json"


@dataclass
class ExperimentConfig:
    """Unified end-to-end AGORA configuration."""

    preprocessing: DataProcessingConfig = field(default_factory=DataProcessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)

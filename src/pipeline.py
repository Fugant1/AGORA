"""End-to-end OPERA V4 pipeline orchestrator & CLI (Preprocess -> SFT-MT -> vLLM GRPO v3).

Run locally on a 24 GB NVIDIA GPU:
  python -m src.pipeline preprocess --meld-dir /path/to/prep_meld --data-dir data/meld_16k_mono
  python -m src.pipeline sft --run-name full_sft_gender_wxrate_multitask_alpha05_seed42_v4
  python -m src.pipeline grpo --sft-run-name full_sft_gender_wxrate_multitask_alpha05_seed42_v4 \\
      --run-name grpo_v3_hybrid_invfreq_4cue_mt_sft_seed42_cuekl005_v4
  python -m src.pipeline all
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.acoustics.features import (
    extract_egemaps_cues,
    filter_training_set_by_duration_and_nonzero_cues,
)
from src.acoustics.quartiles import (
    apply_gender_quartile_classification,
    compute_gender_acoustic_quartiles,
)
from src.config import (
    ACOUSTIC_CUES,
    DataProcessingConfig,
    ExperimentConfig,
    GRPOConfig,
    ModelConfig,
    RewardConfig,
    SFTConfig,
    VLLMConfig,
)
from src.data.emo_tag import EmoTagWhisperXCleaner, run_emo_tag_cleaning_on_split
from src.data.gender import classify_and_filter_gender
from src.data.loader import (
    all_clean_outputs_cached,
    export_split_16k_mono,
    load_meld,
    verify_split,
)
from src.data.whisperx_rate import (
    add_whisperx_speech_rates_to_clean_train,
    add_whisperx_speech_rates_to_clean_val,
)
from src.training.grpo import run_grpo_training
from src.training.sft import run_sft_training
from src.utils.logging import get_logger

logger = get_logger("opera.pipeline")


def run_data_processing(cfg: DataProcessingConfig) -> Dict[str, Any]:
    """Execute the full data_processing.ipynb pipeline (MELD -> Gender -> 16k -> EMO-TAG -> eGeMAPS -> Quartiles -> WhisperX rate)."""
    data_dir = Path(cfg.data_dir)
    clean_dir = cfg.clean_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load or skip raw MELD splits
    raw_csvs = {s: data_dir / f"raw_{s}_set.csv" for s in ("train", "val", "test")}
    if not cfg.overwrite_audio and all(p.is_file() for p in raw_csvs.values()):
        raw_train = pd.read_csv(raw_csvs["train"])
        raw_val = pd.read_csv(raw_csvs["val"])
        raw_test = pd.read_csv(raw_csvs["test"])
    elif not cfg.overwrite_audio and all_clean_outputs_cached(data_dir):
        raw_train = raw_val = raw_test = None
    else:
        raw_train = load_meld(cfg.meld_dir, "train", archive_path=cfg.archive_path)
        raw_val = load_meld(cfg.meld_dir, "val", archive_path=cfg.archive_path)
        raw_test = load_meld(cfg.meld_dir, "test", archive_path=cfg.archive_path)

    # 2. Gender classification & train filtering
    if isinstance(raw_train, pd.DataFrame):
        raw_train = classify_and_filter_gender(raw_train, "raw_train_set", drop_unknown=True)
    if isinstance(raw_val, pd.DataFrame):
        raw_val = classify_and_filter_gender(raw_val, "raw_val_set", drop_unknown=False)
    if isinstance(raw_test, pd.DataFrame):
        raw_test = classify_and_filter_gender(raw_test, "raw_test_set", drop_unknown=False)

    # 3. 16 kHz mono WAV standardization
    raw_train = export_split_16k_mono(raw_train, "train", data_dir, cfg.max_workers, cfg.overwrite_audio)
    raw_val = export_split_16k_mono(raw_val, "val", data_dir, cfg.max_workers, cfg.overwrite_audio)
    raw_test = export_split_16k_mono(raw_test, "test", data_dir, cfg.max_workers, cfg.overwrite_audio)
    if isinstance(raw_train, pd.DataFrame):
        verify_split(raw_train, "raw_train_set")
    if isinstance(raw_val, pd.DataFrame):
        verify_split(raw_val, "raw_val_set")

    # 4. EMO-TAG 5-step laughter & overlap cleaning
    cleaner = EmoTagWhisperXCleaner(whisper_model_name=cfg.whisper_model_name, language=cfg.language)
    clean_train, _ = run_emo_tag_cleaning_on_split(raw_train, "train", cleaner, clean_dir, cfg.overwrite_audio)
    clean_val, _ = run_emo_tag_cleaning_on_split(raw_val, "val", cleaner, clean_dir, cfg.overwrite_audio)
    clean_test, _ = run_emo_tag_cleaning_on_split(raw_test, "test", cleaner, clean_dir, cfg.overwrite_audio)

    # 5. openSMILE eGeMAPSv02 extraction + train integrity gate
    train_eg_csv = clean_dir / "train_egemaps.csv"
    val_eg_csv = clean_dir / "val_egemaps.csv"
    if train_eg_csv.exists() and val_eg_csv.exists() and not cfg.overwrite_audio:
        train_eg = pd.read_csv(train_eg_csv)
        val_eg = pd.read_csv(val_eg_csv)
    else:
        train_eg = extract_egemaps_cues(clean_train, "train", cfg.max_workers)
        val_eg = extract_egemaps_cues(clean_val, "val", cfg.max_workers)
        train_eg = filter_training_set_by_duration_and_nonzero_cues(
            train_eg,
            "train",
            min_dur=cfg.min_train_duration_sec,
            max_dur=cfg.max_train_duration_sec,
            delete_rejected_wavs_in_data_dir=cfg.delete_rejected_wavs,
            data_dir=data_dir,
        )
        train_eg.to_csv(train_eg_csv, index=False)
        val_eg.to_csv(val_eg_csv, index=False)

    # 6. Gender-specific quartile fitting (strictly on train) & classification
    if cfg.quartiles_json.exists() and cfg.train_csv.exists() and cfg.val_csv.exists() and not cfg.overwrite_audio:
        quartiles = json.loads(cfg.quartiles_json.read_text(encoding="utf-8"))
        train_cues = pd.read_csv(cfg.train_csv)
        val_cues = pd.read_csv(cfg.val_csv)
    else:
        quartiles = compute_gender_acoustic_quartiles(train_eg, ACOUSTIC_CUES)
        train_cues = apply_gender_quartile_classification(train_eg, quartiles, "clean_train_set", ACOUSTIC_CUES)
        val_cues = apply_gender_quartile_classification(val_eg, quartiles, "clean_val_set", ACOUSTIC_CUES)
        cfg.quartiles_json.write_text(json.dumps(quartiles, indent=2) + "\n", encoding="utf-8")
        train_cues.to_csv(cfg.train_csv, index=False)
        val_cues.to_csv(cfg.val_csv, index=False)

    # 7. Batched WhisperX true syllable & word rate tracking (train-fitted global cutoffs)
    train_cues = add_whisperx_speech_rates_to_clean_train(
        cfg.train_csv,
        drive_dir=cfg.drive_data_dir,
        batch_size=cfg.whisperx_rate_batch_size,
        overwrite=cfg.overwrite_whisperx_rate,
        whisper_model_name=cfg.whisper_model_name,
        asr_model=cleaner.asr_model,
        align_model=cleaner.align_model,
        align_metadata=cleaner.align_metadata,
        device=cleaner.device,
    )
    val_cues = add_whisperx_speech_rates_to_clean_val(
        cfg.val_csv,
        cfg.train_csv,
        split="val",
        drive_dir=cfg.drive_data_dir,
        batch_size=cfg.whisperx_rate_batch_size,
        overwrite=cfg.overwrite_whisperx_rate,
        whisper_model_name=cfg.whisper_model_name,
        asr_model=cleaner.asr_model,
        align_model=cleaner.align_model,
        align_metadata=cleaner.align_metadata,
        device=cleaner.device,
    )

    logger.info("Data processing complete: train=%d, val=%d", len(train_cues), len(val_cues))
    return {"train": train_cues, "val": val_cues, "test": clean_test, "quartiles": quartiles}


class OPERAPipeline:
    """Master orchestrator for OPERA V4 (Data Processing -> Multitask SFT -> vLLM GRPO v3)."""

    def __init__(self, config: Optional[ExperimentConfig] = None) -> None:
        self.config = config or ExperimentConfig()

    def run_preprocess(self) -> Dict[str, Any]:
        return run_data_processing(self.config.preprocessing)

    def run_sft(self) -> Dict[str, Any]:
        return run_sft_training(self.config.sft, self.config.model)

    def run_grpo(self) -> Dict[str, Any]:
        return run_grpo_training(
            self.config.grpo,
            self.config.model,
            self.config.reward,
            self.config.vllm,
        )

    def run_all(self) -> Dict[str, Any]:
        prep = self.run_preprocess()
        sft = self.run_sft()
        grpo = self.run_grpo()
        return {"preprocess": prep, "sft": sft, "grpo": grpo}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OPERA V4 SER Pipeline (Preprocess / SFT-MT / vLLM GRPO v3)")
    parser.add_argument("stage", choices=("preprocess", "sft", "grpo", "all"))
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--meld-dir", type=Path, default=Path("raw_data/prep_meld"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/meld_16k_mono"))
    parser.add_argument("--split-dir", type=Path, default=SFTConfig.split_dir)
    parser.add_argument("--partition-mode", choices=("semi_balanced", "uniform"), default="semi_balanced")
    parser.add_argument("--neutral-fraction", type=float, default=0.25)
    parser.add_argument("--sft-run-name", type=str, default=SFTConfig.run_name)
    parser.add_argument("--grpo-run-name", type=str, default=GRPOConfig.run_name)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--r1-alpha", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=1.0, help="R2 acoustic cue inverse-frequency alpha")
    parser.add_argument("--r2-weight", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps-per-gen", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--no-vllm", action="store_true", help="Fallback to HuggingFace generate() instead of vLLM.")
    parser.add_argument("--vllm-quantization", type=str, default="bitsandbytes", help="bitsandbytes | fp8 | none")
    parser.add_argument("--vllm-gpu-util", type=float, default=0.35)
    parser.add_argument("--resume", type=str, default=None)
    return parser


def main(argv: Optional[list] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    vllm_quant = None if str(args.vllm_quantization).lower() in ("none", "") else args.vllm_quantization
    vllm_load = "bitsandbytes" if vllm_quant == "bitsandbytes" else None

    cfg = ExperimentConfig(
        preprocessing=DataProcessingConfig(meld_dir=args.meld_dir, data_dir=args.data_dir),
        model=ModelConfig(),
        sft=SFTConfig(
            project_dir=args.project_dir,
            data_dir=args.data_dir,
            split_dir=args.split_dir,
            run_name=args.sft_run_name,
            partition_mode=args.partition_mode,
            neutral_target_fraction=args.neutral_fraction,
            seed=args.seed,
        ),
        reward=RewardConfig(
            r1_inverse_frequency_alpha=args.r1_alpha,
            inverse_frequency_alpha=args.alpha,
            reward_weights={"r1": 1.0, "r2": args.r2_weight, "r3": 0.2},
        ),
        vllm=VLLMConfig(
            enabled=not args.no_vllm,
            quantization=vllm_quant,
            load_format=vllm_load,
            gpu_memory_utilization=args.vllm_gpu_util,
        ),
        grpo=GRPOConfig(
            project_dir=args.project_dir,
            data_dir=args.data_dir,
            split_dir=args.split_dir,
            sft_run_name=args.sft_run_name,
            run_name=args.grpo_run_name,
            partition_mode=args.partition_mode,
            resume_checkpoint_dir=args.resume,
            seed=args.seed,
            beta=args.beta,
            per_device_train_batch_size=args.batch_size,
            steps_per_generation=args.steps_per_gen,
            gradient_accumulation_steps=args.grad_accum,
        ),
    )

    pipe = OPERAPipeline(cfg)
    if args.stage == "preprocess":
        pipe.run_preprocess()
    elif args.stage == "sft":
        pipe.run_sft()
    elif args.stage == "grpo":
        pipe.run_grpo()
    else:
        pipe.run_all()


if __name__ == "__main__":
    main()

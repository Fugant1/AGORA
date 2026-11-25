# AGORA: Acoustic-Grounded Omni Reasoning for Affect

**Acoustically Grounded Speech Emotion Recognition on `Qwen2.5-Omni-7B` via Hybrid Cue-Split GRPO**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## Research Overview

Large omni-modal language models frequently rely on lexical shortcuts or emit unfaithful prosodic explanations when performing Speech Emotion Recognition (SER)—claiming pitch, loudness, or voice-quality patterns that contradict the physical speech signal or misreading lower-register male speech as neutral.

**AGORA** trains `Qwen/Qwen2.5-Omni-7B` (`Qwen2_5OmniThinkerForConditionalGeneration`, 4-bit NF4 QLoRA) through an end-to-end **Jupyter Notebook pipeline** combining:
1. **Laughter-cleaned, gender-normalized physical acoustic labeling** (**EMO-TAG** + **WhisperX** + **openSMILE `eGeMAPSv02`**),
2. **Disjoint MILP-balanced Supervised Fine-Tuning (SFT)** with teacher-grounded prosodic reasoning, and
3. **Hybrid Cue-Split Group Relative Policy Optimization (RLVR / GRPO)** that assigns local, mean-centered physical acoustic rewards directly to each generated cue value token while optimizing standardized emotion advantages on downstream reasoning and classification tokens.

```text
                  Cleaned 16 kHz Mono Speech + Speaker Gender (F / M)
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
   Immutable Precomputed Cues                    Qwen2.5-Omni-7B Thinker (4-bit QLoRA)
   • speaking_rate (WhisperX syll/s, global)     • Policy Adapter + Frozen sft_reference
   • pitch_level   (eGeMAPS F0 p50, F/M Q25/Q75)                 │
   • vocal_energy  (eGeMAPS Loudness, F/M Q25/Q75)               ▼
   • timbre        (eGeMAPS alphaRatioV, F/M)      Rollout Sampling (G = 8)
                 │                                 <think>
                 │                                   [Acoustic Inventory]
                 │                                   speaking_rate: ...   ◄── Cue-local centered R2 (L/n_c)
                 │                                   pitch_level: ...     ◄── Cue-local centered R2 (L/n_c)
                 │                                   vocal_energy: ...    ◄── Cue-local centered R2 (L/n_c)
                 │                                   timbre: ...          ◄── Cue-local centered R2 (L/n_c)
                 │                                   [Prosodic Reasoning]
                 │                                   ...                  ◄── Standardized z(w1·R1 + w3·R3)
                 │                                 </think>
                 │                                 <answer>emotion</answer> ◄── Standardized z(w1·R1 + w3·R3)
                 └───────────────────────┬───────────────────────┘
                                         ▼
                        Hybrid Cue-Split GRPO Policy Update
```

---

## Notebook Pipeline

All experiments, preprocessing, training, diagnostics, and evaluations are self-contained in Jupyter notebooks designed for Google Colab (CUDA GPU + Google Drive persistence) or local GPU execution:

| Stage | Notebook | Purpose & Outputs |
| :--- | :--- | :--- |
| **1. Data & Acoustics** | [`data_processing.ipynb`](data_processing.ipynb) | Loads official MELD splits (`train` / `dev → val` / `test`), resolves speaker sex (`F` / `M`), converts audio to `16 kHz` mono WAV, runs **EMO-TAG** 5-step single-utterance cleaning & laugh-track removal (`rVAD` + two-pass `WhisperX`), extracts **WhisperX** syllable rates and **openSMILE `eGeMAPSv02`** functionals, and fits train-only quartiles (`clean_train_set_with_cues.csv`: `6,059` clips; `clean_val_set_with_cues.csv`: `816` clips). |
| **2. Cue Learnability** | [`cue_learnability.ipynb`](cue_learnability.ipynb) | Diagnostic QLoRA probe evaluating how accurately the frozen `Qwen2.5-Omni` audio encoder exposes individual prosodic cues (`pitch_level`, `vocal_energy`, `timbre`, `speaking_rate`, `pitch_variation`) across utterance duration bins and speaker genders. |
| **3. Disjoint SFT (V4)** | [`ser_sft.ipynb`](ser_sft.ipynb) | Solves a **HiGHS MILP** (`scipy.optimize.milp`) to build a `1,050`-clip `emotion × effective_gender` balanced subset preserving empirical cue marginals, partitions it `60/40` into strictly disjoint `sft_train.csv` (`630` clips) and `grpo_train.csv` (`420` clips), distills concise teacher `[Prosodic Reasoning]`, trains the V4 SFT QLoRA adapter, and evaluates greedy validation (`N = 816`). |
| **4. Hybrid GRPO (V3)** | [`ser_grpo_v3.ipynb`](ser_grpo_v3.ipynb) | **Primary RLVR notebook.** Loads the V4 SFT adapter + frozen `sft_reference`, trains 1-epoch **Hybrid Cue-Split GRPO** (`G = 8`, `105` steps) on the disjoint `420`-clip `grpo_train.csv`, logs live rollout & adapter telemetry, and runs post-training greedy validation (`N = 816`) + `model_health_report.md`. |
| *(Legacy Prototype)* | [`ser_grpo_v2.ipynb`](ser_grpo_v2.ipynb) | Earlier 3-cue GRPO v2 notebook retained for reward/credit-assignment regression tests. |

---

## Output Schema, Acoustic Cues, and Hybrid Rewards

### Structured Student Output (`PROMPT_VERSION = gender_wxrate_4cue_srfirst_teacher_v5_short_concise`)
Conditioned on the audio waveform and speaker vocal register (`female` or `male` via `effective_gender`), the model generates:

```xml
<think>
[Acoustic Inventory]
speaking_rate: low | moderate | high
pitch_level: low | moderate | high
vocal_energy: low | moderate | high
timbre: low (muffled/soft) | moderate (modal) | high (tense/bright)
[Prosodic Reasoning]
Concise prosodic deduction integrating acoustic cues and spoken content.
</think>
<answer>anger | disgust | fear | joy | neutral | sadness | surprise</answer>
```

### Physical Acoustic Grounding (4 Ordinal Cues)
All quartile thresholds (`Q25` / `Q75` $\rightarrow$ `25% low / 50% moderate / 25% high`) are fitted **exclusively on `clean_train_set`** (zero validation/test refitting):

| Inventory Field | Backing Physical Feature | Normalization |
| :--- | :--- | :--- |
| `speaking_rate` | WhisperX aligned `whisperx_syllable_rate` (syllables/s) | Global `clean_train` `Q25` / `Q75` |
| `pitch_level` | `eGeMAPSv02` `F0semitoneFrom27.5Hz_sma3nz_percentile50.0` (median $F_0$, st) | Gender-specific (`F` / `M`) `Q25` / `Q75` |
| `vocal_energy` | `eGeMAPSv02` `loudness_sma3_percentile50.0` (median loudness, sones) | Gender-specific (`F` / `M`) `Q25` / `Q75` |
| `timbre` | `eGeMAPSv02` `alphaRatioV_sma3nz_amean` (50–1000 Hz vs. 1–5 kHz ratio, dB) | Gender-specific (`F` / `M`) `Q25` / `Q75` |

### Reward Objectives & Credit Assignment (`LOSS_TYPE = "hybrid"`)
- **`R1` — Emotion Classification (`w1 = 1.0`)**: Empirical Inverse-Frequency payoff ($1 / (7 p_y)$ on exact match, $0$ on miss/invalid; `1.0` on balanced `grpo_train.csv`), standardized per audio group via GRPO `z`-score (`ddof=1`) so sparse minority-emotion hits retain full gradient leverage.
- **`R2` — 4-Cue Physical Calibration (`w2 = 1.0`)**: Parameter-free Inverse-Frequency payoff per cue ($1 / (3 p_{c,y})$ on exact match, $0$ on miss/invalid; optional ordinal distance payoff via `USE_DISTANCE_BASED_PAYOFF=True`). Under `loss_type="hybrid"`, each cue's reward is **mean-centered within the $G=8$ group without `std` division** (`w2 / 4` per cue) and applied **only to that cue's value tokens** with `L / n_c` length compensation so multi-token `timbre` labels receive identical sequence-level weight to single-token cues.
- **`R3` — 4-Slot Format Adherence (`w3 = 0.2`)**: `valid_slots / 4.0` verifying strict `<think>...</think><answer>...</answer>` ordering, all 4 canonical cue values, non-empty prosodic reasoning, and a valid canonical emotion label.
- **Disjoint Span Routing (`USE_SPLIT_ADVANTAGE=True`, `USE_SPLIT_R2=True`)**:
  - **Cue value tokens**: cue-local centered `R2` (`w2 / 4`, `L / n_c` compensated)
  - **Inventory structure tokens**: `w3 · z(R3)`
  - **`[Prosodic Reasoning]` + `</think>` + `<answer>` tokens**: `z(w1 · R1 + w3 · R3)`
  - **Fast `BETA = 0.0` loss path**: `AudioGRPOTrainer.compute_loss` skips the redundant frozen `sft_reference` forward pass during training when `BETA == 0.0`, while retaining `sft_reference` for post-hoc teacher-forcing KL/drift replay diagnostics.

---

## Validation Results (`N = 816` Official Cleaned MELD Dev → Val)

Greedy decoding (`do_sample=False`, `EVAL_MAX_NEW_TOKENS=512`) across the full Stage-1 ablation chain (`630` SFT clips + `420` disjoint GRPO clips, `105` steps, `seed=42`):

| Model | Accuracy | Weighted-F1 | Macro-F1 | **UAR (Macro-Recall)** | Invalid | Female MF1 | **Male MF1** | **`F − M` MF1 Gap** | 4-Cue Mean UAR | Pitch / Timbre $\rho$ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Base (`Qwen2.5-Omni-7B`)** | 53.43% | 47.84% | 32.53% | 31.23% | 1.10% | 36.13% | 27.16% | +8.97 pp | — | — |
| **SFT V4 (`ser_sft.ipynb`)** | 43.14% | 45.41% | 38.16% | 44.56% | 0.86% | 38.61% | 36.59% | **+2.02 pp** | 42.93% | — |
| **GRPO v3 `no-R2` (`w2=0`)** | 46.08% | 47.55% | 39.34% | 41.96% | 1.23% | **43.86%** | 32.11% | +11.75 pp | 41.03% | `+0.479` / `+0.278` |
| **GRPO v3 `with-R2` (`w2=1`)** | **46.45%** | **48.38%** | **40.77%** | **47.85%** | **0.74%** | 42.08% | **39.52%** | **+2.56 pp** | **46.75%** | **`+0.656` / `+0.448`** |

Key takeaways from the controlled `with-R2` vs. `no-R2` ablation:
- **Higher balanced & weighted SER**: Adding `R2` improves **UAR by `+5.89 pp`** (`47.85%` vs. `41.96%`, paired bootstrap 95% CI `[+0.49, +11.59] pp`, $P(\text{with-R2} > \text{no-R2}) = 98.4\%$), **Macro-F1 by `+1.43 pp`**, and **Weighted-F1 by `+0.83 pp`** (the only configuration surpassing both Base and SFT across Weighted-F1, Macro-F1, and UAR).
- **Prevents male-speaker bias regression**: Removing `R2` causes Male Macro-F1 to drop from `36.59%` (SFT) to `32.11%` (`+11.75 pp` `F − M` gap), whereas gender-relative `R2` lifts Male Macro-F1 to **`39.52%`** (`+2.56 pp` gap) alongside sharp gains in male pitch ($\rho_M: +0.331 \rightarrow +0.568$) and timbre ($\rho_M: +0.259 \rightarrow +0.436$) perception.
- **Prevents collateral acoustic forgetting**: Even when cue tokens receive `0.0` gradient (`no-R2`), shared LoRA updates degrade 4-cue UAR below SFT (`42.93% → 41.03%`), whereas `with-R2` raises 4-cue UAR to **`46.75%`** (`pitch_level` Macro-F1 `55.2%`, `timbre` Macro-F1 `45.6%`, `100%` valid 4-slot inventories).

---

## Running the Notebooks & Verification Tests

### 1. Notebook Execution Order
1. Run [`data_processing.ipynb`](data_processing.ipynb) once to generate `clean_train_set_with_cues.csv`, `clean_val_set_with_cues.csv`, `gender_acoustic_quartiles.json`, and `meld_16k_mono_clean.tar` on Drive.
2. *(Optional)* Run [`cue_learnability.ipynb`](cue_learnability.ipynb) to audit standalone cue learnability.
3. Run [`ser_sft.ipynb`](ser_sft.ipynb) to create the verified `60/40` MILP split (`sft_train.csv` / `grpo_train.csv`), distill teacher reasoning, and save `final_sft_adapter`.
4. Run [`ser_grpo_v3.ipynb`](ser_grpo_v3.ipynb) (`PER_DEVICE_TRAIN_BATCH_SIZE=2`, `STEPS_PER_GENERATION=8`, `GRADIENT_ACCUMULATION_STEPS=16`, `MAX_COMPLETION_LENGTH=256`, `EVAL_MAX_NEW_TOKENS=512`) to train `final_grpo_adapter` and export validation predictions, plots, and `model_health_report.md`.

### 2. Local Automated Test Suite
The notebook reward math, strict parsers, MILP partition invariants, WhisperX syllable-rate logic, hybrid cue-split token alignment, and health-report generators are verified directly against the `.ipynb` cells via `pytest`:

```bash
.venv/bin/pytest -v tests/test_ser_grpo_v3.py tests/test_sft_v2.py tests/test_data_processing_whisperx_rate.py tests/test_grpo_reward_math.py tests/test_split_r2.py tests/test_grpo_health_report.py
```

---

## Citation

```bibtex
@misc{agora2026,
  title={AGORA: Acoustic-Grounded Omni Reasoning for Speech Emotion Recognition via Hybrid Cue-Split GRPO},
  author={Fuganti et al.},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/Fugant1/AGORA}}
}
```

# AGORA: Acoustic-Grounded Omni Reasoning for Affect

**Can verifiable acoustic grounding improve audio-only emotion recognition through reinforcement learning?**

AGORA investigates this question on MELD with `Qwen/Qwen2.5-Omni-7B` Thinker. The implemented pipeline uses EMO-TAG cleaning, acoustic labels derived from openSMILE and WhisperX, supervised fine-tuning (SFT), and cue-split Group Relative Policy Optimization (GRPO). The model receives audio and `effective_gender` metadata; therefore results should be described as audio-conditioned with gender metadata, not strictly audio-only.

## Source pipeline

The executable implementation is under `src/`. Notebooks in the repository are research artifacts and are not the entry point documented here.

```text
MELD -> gender metadata -> 16 kHz mono -> EMO-TAG cleaning
     -> eGeMAPSv02 features -> train-fitted acoustic quartiles
     -> WhisperX syllable rates -> disjoint SFT/GRPO partitions
     -> teacher-grounded SFT -> hybrid cue-split GRPO -> validation
```

Run commands from the repository root in an isolated Python environment with the required dependencies installed. Keep data and outputs in persistent storage, and select a machine appropriate to the run's CPU, RAM, and GPU needs.

```bash
python -m src.pipeline preprocess --meld-dir raw_data/prep_meld --data-dir data/meld_16k_mono
python -m src.pipeline sft --data-dir data/meld_16k_mono
python -m src.pipeline grpo --data-dir data/meld_16k_mono
# Or execute all stages in sequence:
python -m src.pipeline all --meld-dir raw_data/prep_meld --data-dir data/meld_16k_mono
```

The CLI defaults to `semi_balanced` partitions, seed 42, SFT cue-weight exponent 0.5, R1 emotion inverse-frequency exponent 0.5, R2 cue exponent 1.0, `R2` weight 1.0, `R3` weight 0.2, `beta=0.05`, and vLLM rollouts. GRPO's CLI defaults are per-device batch size 2, 8 steps per generation, and gradient accumulation 16. Use `--no-vllm` to fall back to Hugging Face generation; this may be slower and is still subject to the selected model/backend's hardware and dependency requirements. `--vllm-quantization none` disables vLLM quantization. These options do not make the full pipeline CPU-only: QLoRA training requires a supported GPU setup.

Set `--project-dir` to the persistent project directory when it differs from the current directory. Data defaults to `data/meld_16k_mono`; split manifests and CSVs default to `data/sft_grpo_gender_wxrate_60_40_semi25_v5`; model runs are written under `runs/ser_sft/<run-name>` and `runs/ser_grpo/<run-name>`. Use `--resume` for a GRPO checkpoint. Keep the split manifest, experiment JSON, logs, adapters, and evaluation outputs together so runs can be audited and resumed. The preprocessing stage currently fits/classifies acoustic cues and adds WhisperX rates for train and validation; it returns cleaned test audio without creating a test cue CSV or applying frozen train thresholds to the test split. Do not treat that output as a fully prepared held-out cue-evaluation set.

The default `semi_balanced` partition keeps all available non-neutral utterances (rounded down by stratum to multiples of five) and selects neutral utterances to target a 0.25 neutral share, subject to available data. A cue-stratified MILP preserves emotion-by-effective-gender quotas while minimizing cue-marginal deviations; a second stratified selection divides the resulting pool 60/40 into disjoint SFT and GRPO utterances. Counts depend on the cleaned MELD data; historical notebook counts are not current source defaults. `--partition-mode uniform` is also available. Disjointness is at utterance/audio level, not a speaker- or dialogue-held-out split.

## Acoustic targets and SFT

The model predicts four ordinal inventory cues and one of seven MELD emotions. The student is prompted with the waveform and `effective_gender`; it is not given the transcript, gold emotion, or acoustic cue targets. For SFT rows only, a teacher target is generated using the transcript and precomputed labels.

| Cue | Measurement | Label thresholds |
| --- | --- | --- |
| `speaking_rate` | WhisperX-aligned syllables per second | Global train Q25/Q75 |
| `pitch_level` | eGeMAPSv02 median F0 in semitones | Effective-gender train Q25/Q75 |
| `vocal_energy` | eGeMAPSv02 median loudness | Effective-gender train Q25/Q75 |
| `timbre` | eGeMAPSv02 mean alpha ratio | Effective-gender train Q25/Q75 |

Values below Q25 are `low`, above Q75 are `high`, and ties/between-threshold values are `moderate`. Quartiles are fit on cleaned training data and then reused for validation. Ties mean the labels are not guaranteed to have exact 25/50/25 proportions. The active inventory has four cues; the fifth extracted eGeMAPS feature, pitch variation, is not an active prediction target. WhisperX rate uses syllable count over the aligned utterance span; if aligned words are unavailable, source code falls back to reference-transcript syllable count over full clip duration, so fallback-derived rates should be identified in analyses.

For cue class `k`, let `n_k` be its count among the `N` SFT rows and let `α` be the cue weighting exponent. The implemented tempered inverse-frequency weight is

\[
w_k = \frac{N n_k^{-\alpha}}{\sum_j n_j^{1-\alpha}}.
\]

Thus the expected weight under the empirical class distribution is one; `α=0` gives unit weights and `α=1` gives `N/(K n_k)`. By default `α=0.5`. Weights are fit separately for each cue using SFT rows; a missing cue class is an error.

With equal multitask weighting enabled, let `L` be the number of supervised assistant tokens, `s=0.25` the cue-task share, and `K_c=4` the number of cues. Only the first token of each cue value span receives cue loss weight:

\[
q_{c,1}=\frac{L s}{K_c}w_{c,y_c}, \qquad
q_t=\frac{L(1-s)}{L-K_c}\quad\text{for every other assistant token.}
\]

The response loss is the mean over examples of `Σ_active q_t CE_t / L`. Consequently, later tokens in a multi-token cue value are in the “other token” allocation. This is the implemented first-token decision weighting, not uniform weighting of every token in a cue span.

## GRPO reward and token credit

Each rollout receives three rewards: emotion correctness `R1`, four-cue acoustic calibration `R2`, and output format/reasoning quality `R3`. Priors for `R1` and `R2` are measured on the GRPO partition; they are not estimated from validation. For inverse-frequency exponent `α` and class prior `p_y`, the correct-label payoff is

\[
h_y=\frac{p_y^{-\alpha}}{\sum_j p_j^{1-\alpha}},
\]

with zero for an incorrect/invalid prediction in inverse-frequency mode. The default is `α=0.5` for emotion and `α=1` for each cue. At `α=1`, `h_y=1/(K p_y)`; a perfect prediction averaged over the empirical prior has payoff one. R2 may optionally use an ordinal distance payoff, but that is not the default.

For rollout groups of size `G=8`, `z(x)=(x - mean(x))/(s_x + 1e-8)`, where `s_x` is the sample standard deviation (`ddof=1`); tied groups therefore have zero standardized advantage. Let `w_y=h_y` be the truth-class emotion payoff, `w1=1`, `w2=1`, `w3=0.2`, and `r2,c` be the raw payoff for cue `c`. The hybrid advantages implemented in `src/training/credit_assignment.py` are

\[
A_{\mathrm{emotion}}=w_y\,z\!\left(w_1\frac{R_1}{w_y}+w_3R_3\right),\qquad
A_{\mathrm{format}}=w_3z(R_3),
\]

\[
A_c=\frac{w_2}{4}\left(r_{2,c}-\overline{r_{2,c}}\right).
\]

The cue mean is computed within each rollout group and is not divided by its standard deviation. Emotion and format advantages use group standardization. Cue value tokens receive their cue-specific `A_c`, while inventory-structure tokens receive `A_format` and reasoning/emotion tokens receive `A_emotion`. If a completion boundary cannot be located, source code has a fallback that applies the combined advantage to the completion; such cases should be monitored in rollout diagnostics.

For cue `c`, let `n_c` be its value-span token count and `L` the active completion-token count. The implemented cue span multiplier is

\[
\lambda_c=L\min\left(1,\frac{8}{n_c}\right).
\]

Thus spans up to eight tokens each get multiplier `L`; longer spans are capped to total scale `8L/n_c`. It is not `L/n_c` for short spans. The optional cue-only KL anchor uses the same cue-span multiplier and defaults to `β=0.05`; `--beta 0` disables the KL term. GRPO uses a score-function gradient path; its unit policy-ratio construction is not PPO clipping with an old-policy ratio.

The format reward is composed in `src/rewards/format.py`: tag order (0.20), valid cue slots (0.35 times valid slots/4), reasoning length/character checks (up to 0.25), and valid emotion (0.20), with the result rounded to three decimals. Consult that implementation when changing this rubric.

## Outputs and evaluation

The pipeline evaluates SFT and GRPO checkpoints on validation and writes predictions, classification metrics (accuracy, macro/weighted F1, macro recall/UAR, invalid rate), per-class and gender breakdowns, cue metrics, and diagnostics under each run directory. These are validation measurements, not final held-out test results. No result table is presented here as a current `src` reproduction; previously recorded notebook metrics are historical and should not be attributed to this pipeline without reproducing them.

The research target is IEEE Transactions on Affective Computing (TAFFC) or Interspeech. Publication claims should be based on held-out evaluation with matched baselines, confidence intervals, and checks that generated acoustic claims match the measured signals.

## References

- EMO-TAG, laughter/overlap-cleaned speech annotation: see the linked paper and implementation references in [`references/`](references/).
- EmotionThinker, audio emotion reasoning: see [`references/`](references/).
- Montreal Forced Aligner: McAuliffe et al., “Montreal Forced Aligner: Trainable Text-Speech Alignment Using Kaldi,” *Interspeech 2017*.
- WhisperX: Bain et al., “WhisperX: Time-Accurate Speech Transcription of Long-Form Audio,” *Interspeech 2023*.
- openSMILE/eGeMAPS: Eyben et al., “The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research and Affective Computing,” *IEEE Transactions on Affective Computing*, 2016.

For full paper metadata, consult the project references rather than treating notebook-era claims as sourced by this README.

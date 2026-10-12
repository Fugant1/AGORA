# OPERA: Omni-modal Preference Extraction and Ranking Architecture

**Acoustically Grounded Speech Emotion Recognition via Group Relative Policy Optimization (GRPO)**

[![Colab]([https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uWwj1vchyIj-EiyJgGGL41Mn_DQJWOPx?usp=sharing](https://colab.research.google.com/github/Fugant1/OPERA/blob/main/ser_grpo_v2.ipynb))
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## 🔬 Research Overview

Large Omni-modal language models often exhibit hallucinations when performing vocal affect reasoning—claiming to detect acoustic signals (e.g., pitch inflections, jitter, vocal strain) that directly contradict the physical acoustics of the speech signal.

**OPERA** formalizes Speech Emotion Recognition (SER) as a **physically grounded reinforcement learning problem** using **Group Relative Policy Optimization (GRPO)**. The framework guides the model to produce verifiable Chain-of-Thought (CoT) reasoning structured around an **Acoustic Inventory** of physical phonetic descriptors grounded in empirical **eGeMAPSv02** functionals.

```
                           Raw Speech Audio Clip
                                    │
                       ┌────────────┴────────────┐
                       ▼                         ▼
            Physical Acoustics (eGeMAPS)    Qwen 2.5 Omni (4-bit LoRA)
            • F0 Pitch Mean & Dynamics      • Thinker Attention & MLP
            • Loudness / Vocal Energy                    │
            • Voiced Segments / Rate                     ▼
            • HNR / Spectral Tilt / Jitter    Rollout Generation (G=4)
                       │                    <think>
                       │                      [Acoustic Inventory]
                       │                      [Prosodic Reasoning]
                       │                    </think>
                       │                    <answer>emotion</answer>
                       │                                 │
                       └───────────────┬─────────────────┘
                                       ▼
                       Multi-Objective GRPO Rewards:
                       • R1: Rarity-Weighted Accuracy (Class Frequency Smoothed)
                       • R2: Acoustic Inventory Physical Grounding (eGeMAPS)
                       • R3: Structural XML & Slot Format Adherence
                                       │
                                       ▼
                             Policy Update (GRPO)
```

---

## 📦 Project Architecture (`src/`)

The codebase is organized into a modular research package:

```
src/
├── __init__.py                 # Top-level exports and versioning
├── config.py                   # Dataclass configurations (Data, Model, GRPO, Acoustics)
├── pipeline.py                 # End-to-end research orchestration pipeline
│
├── acoustics/                  # Acoustic feature extraction and phonetics
│   ├── features.py             # openSMILE eGeMAPSv02 functional extractor & cache
│   ├── analyzer.py             # Global quantiles, emotion variation, & speaker dynamic ranges
│   ├── diagnostics.py          # Speaker volume tail, emotional diversity, & pitch collapse checks
│   └── visualization.py        # Publication figures: Emotion boxplots & dynamic range heatmaps
│
├── data/                       # MELD corpus ingestion and balancing
│   ├── loader.py               # Archive extraction, path resolution, metadata alignment
│   ├── sampler.py              # Deterministic duration-filtered balanced sampler
│   └── dataset.py              # HuggingFace & PyTorch dataset formatting for RL rollouts
│
├── models/                     # Model architecture and adaptation
│   ├── qwen_omni.py            # Qwen 2.5 Omni loader, 4-bit NF4 BitsAndBytes, talker disabling
│   └── lora.py                 # PEFT LoRA adapter targeting thinker attention and MLPs
│
├── prompts/                    # Phonetic expert persona & CoT prompts
│   └── templates.py            # Acoustic phonetics persona & structured CoT templates
│
├── rewards/                    # Multi-objective reinforcement learning rewards
│   ├── parser.py               # Robust XML tags, acoustic inventory, and answer parser
│   ├── accuracy.py             # R1: Smoothed inverse class-frequency weighted accuracy
│   ├── acoustic.py             # R2: Physical acoustic grounding against eGeMAPS quantiles
│   ├── format.py               # R3: Graded XML reasoning structure & slot completion
│   └── composite.py            # Reward manager integrating with TRL GRPOTrainer
│
├── training/                   # Reinforcement learning loop
│   ├── trainer.py              # TRL GRPOTrainer wrapper and rollout configuration
│   └── callbacks.py            # Reward trajectory and rollout monitoring callbacks
│
└── utils/                      # Helper utilities
    ├── audio.py                # Audio duration extraction & validation
    └── logging.py              # Research logging formatter
```

---

## 🎯 Reward Objectives

### R1 — Rarity-Weighted Label Accuracy
Prevents majority-class collapse on skewed corpora (e.g., MELD neutral dominance) via smoothed inverse frequency weighting:
$$w_c = \left( \frac{\max_{k} N_k}{N_c} \right)^\alpha, \quad \alpha \in [0.4, 0.5]$$
The policy receives $w_c$ on exact match, preventing gradient variance explosion while prioritizing minority affective states.

### R2 — Acoustic Inventory Physical Grounding
Penalizes acoustic hallucinations by directly comparing the discrete predicted inventory against empirical **eGeMAPSv02 functionals** (or literature phonetic profiles):
- **Pitch Height**: Grounded against fundamental frequency mean ($F_0$ in semitones).
- **Pitch Dynamics**: Grounded against pitch dynamic percentile range ($pctlrange0\text{--}2$).
- **Vocal Energy**: Grounded against median loudness in sones.
- **Speaking Rate**: Grounded against voiced segments per second.
- **Voice Quality**: Grounded against Hammarberg Index, HNR, and Jitter.

### R3 — Graded Format Adherence
Rewards strict compliance with the Chain-of-Thought protocol without incentivizing length hacking:
1. XML tag ordering: `\langle think \rangle \dots \langle /think \rangle \langle answer \rangle \dots \langle /answer \rangle` (+0.20)
2. Slot completion across all 5 inventory cues (+0.35)
3. Concise prosodic deduction (5–150 words) (+0.25)
4. Valid canonical emotion token (+0.20)

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/Fugant1/OPERA.git
cd OPERA
pip install -r requirements.txt
```

### 2. Python API

```python
from src.pipeline import SERGRPOPipeline
from src.config import ExperimentConfig

# Customize configuration
config = ExperimentConfig()
config.data.target_total = 2000
config.data.max_neutral_pct = 0.15

# Run end-to-end pipeline (data -> acoustics -> rewards -> training)
pipeline = SERGRPOPipeline(config=config)
results = pipeline.run(train_model=False)
```

### 3. Notebook Mode

For interactive execution on Google Colab / Jupyter with GPU acceleration, use [`ser_grpo_v2.ipynb`](ser_grpo_v2.ipynb).

---

## 📄 Citation

```bibtex
@misc{opera2026,
  title={OPERA: Omni-modal Preference Extraction and Ranking Architecture for Speech Emotion Recognition},
  author={Fuganti et al.},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/Fugant1/OPERA}}
}
```

"""Diagnostic and Validation Suite for Conditional Probabilities and GRPO Reward Aggregation.

Tests:
  1. Chute Cego (Blind Guess Invariance): E_{t ~ f}[R(t, p)] = 0.000000 across ALL emotions and cues.
  2. Probabilidade Condicional & Thresholds de Crença (Conditional Expected Utility):
     Determines whether model beliefs under uncertainty (e.g. 60% moderate vs 40% wide)
     lead to the correct Bayesian action or whether jackpot asymmetry forces mode collapse.
  3. Agregação Conjunta de Recompensas e Vantagem GRPO (GRPO Competition & Policy Drift):
     Simulates GRPO groups (G=4) to detect if the high-expressivity cluster
     (elevated + wide + breathy -> joy) creates an unfair net advantage attractor.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


import pytest


# ==============================================================================
# 1. Dynamic Loader: Extracts reward functions directly from ser_grpo_v2.ipynb
# ==============================================================================

@pytest.fixture(scope="module")
def env() -> Dict[str, Any]:
    """Pytest fixture providing the loaded notebook environment."""
    return load_grpo_notebook_env()


def load_grpo_notebook_env() -> Dict[str, Any]:
    """Loads and compiles all relevant reward and parser cells from ser_grpo_v2.ipynb dynamically."""
    nb_path = Path(__file__).resolve().parent.parent / "ser_grpo_v2.ipynb"
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found at: {nb_path}")

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    from typing import Any, Callable, Dict, List, Optional, Tuple, Set, Union
    import re

    env: Dict[str, Any] = {
        "VALID_EMOTIONS": {
            "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"
        },
        "cot_prompt": "Analyze the speaker's acoustic cues and deduce the emotion.",
        "display": lambda *args, **kwargs: None,
        "np": np,
        "pd": pd,
        "math": math,
        "re": re,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
        "Set": Set,
        "Union": Union,
        "Any": Any,
        "Callable": Callable,
    }

    signatures = [
        ("VALID_EMOTIONS =", "code"),
        ("LABEL_SYNONYMS =", "code"),
        ("REQUIRED_INVENTORY_KEYS =", "code"),
        ("def extract_acoustic_inventory", "code"),
        ("def extract_reasoning", "code"),
        ("def parse_completion", "code"),
        ("def compute_emotion_priors_and_weights", "code"),
        ("CUE_FEATURE_MAP: Dict[str, str]", "code"),
        ("def compute_format_reward_from_parsed", "code"),
        ("def unified_gated_grpo_reward", "code"),
    ]

    executed_cells = set()
    for sig, c_type in signatures:
        for idx, cell in enumerate(nb["cells"]):
            if idx not in executed_cells and cell["cell_type"] == c_type and sig in "".join(cell["source"]):
                code = "".join(cell["source"])
                exec(compile(code, f"<ser_grpo_v2_cell_{idx}>", "exec"), env)
                executed_cells.add(idx)
                break

    return env


# ==============================================================================
# 2. Test 1: Chute Cego (Blind Guess Expectation for All Dimensions)
# ==============================================================================

def check_blind_guess_zero_mean(env: Dict[str, Any], tol: float = 1e-6) -> Dict[str, Any]:
    """Verifies that E[R | blind guess] = 0.000000 across all emotions and acoustic cues."""
    results: Dict[str, Any] = {
        "passed": True,
        "emotions": {},
        "continuous_cues": {},
        "voice_quality": {},
    }

    # A. Emoções R1
    valid_emotions = sorted(list(env["VALID_EMOTIONS"]))
    priors_r1 = env.get("EMOTION_PRIORS", {e: 1.0 / len(valid_emotions) for e in valid_emotions})
    weights_r1 = env.get("CLASS_WEIGHTS", {e: 1.0 for e in valid_emotions})

    for pred_emo in valid_emotions:
        exp_val = 0.0
        for truth_emo in valid_emotions:
            f_t = priors_r1[truth_emo]
            if pred_emo == truth_emo:
                r1 = weights_r1[truth_emo]
            else:
                f_p = priors_r1[pred_emo]
                w_p = weights_r1[pred_emo]
                r1 = -(f_p * w_p) / (1.0 - f_p)
            exp_val += f_t * r1

        is_zero = abs(exp_val) <= tol
        results["emotions"][pred_emo] = {"E": exp_val, "zero_mean": is_zero}
        if not is_zero:
            results["passed"] = False

    # B. Cues Contínuos (Pitch Height, Pitch Dynamics, Vocal Energy, Speaking Rate)
    cue_priors = {"low": 0.25, "moderate": 0.50, "high": 0.25}
    get_ordinal_matrix = env["get_ordinal_cue_payoff_matrix"]
    M_ordinal = get_ordinal_matrix(cue_priors)

    for cue_name in ["pitch_height", "pitch_dynamics", "vocal_energy", "speaking_rate"]:
        results["continuous_cues"][cue_name] = {}
        for pred_tier in ["low", "moderate", "high"]:
            exp_val = sum(cue_priors[truth_tier] * M_ordinal[truth_tier][pred_tier] for truth_tier in ["low", "moderate", "high"])
            is_zero = abs(exp_val) <= tol
            results["continuous_cues"][cue_name][pred_tier] = {"E": exp_val, "zero_mean": is_zero}
            if not is_zero:
                results["passed"] = False

    # C. Voice Quality
    vq_classes = env["VOICE_QUALITY_CLASSES"]
    default_vq_priors = {"modal/normal": 0.4225, "pressed/tense": 0.1515, "breathy": 0.2197, "harsh/creaky": 0.2064}
    get_vq_matrix = env["get_voice_quality_payoff_matrix"]
    M_vq = get_vq_matrix(default_vq_priors)

    for pred_vq in vq_classes:
        exp_val = sum(default_vq_priors[truth_vq] * M_vq[truth_vq][pred_vq] for truth_vq in vq_classes)
        is_zero = abs(exp_val) <= tol
        results["voice_quality"][pred_vq] = {"E": exp_val, "zero_mean": is_zero}
        if not is_zero:
            results["passed"] = False

    return results


# ==============================================================================
# 3. Test 2: Probabilidade Condicional & Thresholds de Decisão Ótima
# ==============================================================================

def check_conditional_probabilities_and_thresholds(
    env: Dict[str, Any],
    max_moderate_threshold: float = 0.65,
    max_modal_threshold: float = 0.65,
) -> Dict[str, Any]:
    """Calculates the exact Bayesian decision thresholds P* and expected utility under uncertainty.

    Verifies two-sided Bayesian rationality:
      1. Decision thresholds P*(moderate) and P*(modal) must be <= 65%
         (strictly below their natural pairwise priors of 66.7% and 65.8%).
      2. When evidence supports Moderate (65% moderate, 35% wide), model chooses Moderate.
      3. When evidence supports Wide (50% moderate, 50% wide, shifted from 66.7% prior), model chooses Wide.
      4. Hit rewards across tiers must have exact parity (no jackpot skewing exploration).
    """
    results: Dict[str, Any] = {
        "passed": True,
        "failures": [],
        "ordinal_thresholds": {},
        "vq_thresholds": {},
        "scenarios": {},
    }

    # A. Ordinal Cues (Pitch Dynamics, etc.)
    cue_priors = {"low": 0.25, "moderate": 0.50, "high": 0.25}
    M_ord = env["get_ordinal_cue_payoff_matrix"](cue_priors)

    num_high = M_ord["high"]["high"] - M_ord["high"]["moderate"]
    den_high = (M_ord["moderate"]["moderate"] - M_ord["high"]["moderate"]) - (M_ord["moderate"]["high"] - M_ord["high"]["high"])
    p_star_mod_high = num_high / max(1e-9, den_high)

    num_low = M_ord["low"]["low"] - M_ord["low"]["moderate"]
    den_low = (M_ord["moderate"]["moderate"] - M_ord["low"]["moderate"]) - (M_ord["moderate"]["low"] - M_ord["low"]["low"])
    p_star_mod_low = num_low / max(1e-9, den_low)

    results["ordinal_thresholds"] = {
        "P_star_moderate_vs_high": p_star_mod_high,
        "P_star_moderate_vs_low": p_star_mod_low,
        "hit_moderate": M_ord["moderate"]["moderate"],
        "hit_high": M_ord["high"]["high"],
        "miss_high_on_moderate": M_ord["moderate"]["high"],
        "miss_moderate_on_high": M_ord["high"]["moderate"],
    }

    if p_star_mod_high > max_moderate_threshold:
        results["passed"] = False
        results["failures"].append(
            f"INCONSISTÊNCIA EM ORDINAL CUES: Threshold de moderate vs high é {p_star_mod_high:.1%} "
            f"(tolerância máx: {max_moderate_threshold:.1%}). O modelo necessita de mais de "
            f"{p_star_mod_high:.1%} de certeza em moderate para não arriscar em high/wide!"
        )

    # Scenario 1: Model has evidence of Moderate (65% moderate, 35% wide)
    eu_mod_65 = 0.65 * M_ord["moderate"]["moderate"] + 0.35 * M_ord["high"]["moderate"]
    eu_high_65 = 0.65 * M_ord["moderate"]["high"] + 0.35 * M_ord["high"]["high"]

    results["scenarios"]["pitch_dynamics_65_mod_35_wide"] = {
        "belief": {"moderate": 0.65, "wide": 0.35},
        "EU_predict_moderate": eu_mod_65,
        "EU_predict_wide": eu_high_65,
        "chosen_action": "wide" if eu_high_65 > eu_mod_65 else "moderate",
    }
    if eu_high_65 > eu_mod_65:
        results["passed"] = False
        results["failures"].append(
            f"INCONSISTÊNCIA DE DECISÃO EM PITCH DYNAMICS: Com evidência em moderate (65% mod, 35% wide), "
            f"a recompensa por prever wide (+{eu_high_65:.4f}) superou a de moderate (+{eu_mod_65:.4f})!"
        )

    # Scenario 2: Model has evidence of Wide (50% moderate, 50% wide, shifted from 66.7% prior)
    eu_mod_50 = 0.50 * M_ord["moderate"]["moderate"] + 0.50 * M_ord["high"]["moderate"]
    eu_high_50 = 0.50 * M_ord["moderate"]["high"] + 0.50 * M_ord["high"]["high"]

    results["scenarios"]["pitch_dynamics_50_mod_50_wide"] = {
        "belief": {"moderate": 0.50, "wide": 0.50},
        "EU_predict_moderate": eu_mod_50,
        "EU_predict_wide": eu_high_50,
        "chosen_action": "wide" if eu_high_50 > eu_mod_50 else "moderate",
    }
    if eu_mod_50 >= eu_high_50:
        results["passed"] = False
        results["failures"].append(
            f"INÉRCIA EM PITCH DYNAMICS: Com evidência apontando para wide (50% mod, 50% wide), "
            f"o modelo ficou preso em moderate (+{eu_mod_50:.4f}) em vez de escolher wide (+{eu_high_50:.4f})!"
        )

    # B. Voice Quality
    default_vq_priors = {"modal/normal": 0.4225, "pressed/tense": 0.1515, "breathy": 0.2197, "harsh/creaky": 0.2064}
    M_vq = env["get_voice_quality_payoff_matrix"](default_vq_priors)

    num_modal_breathy = M_vq["breathy"]["breathy"] - M_vq["breathy"]["modal/normal"]
    den_modal_breathy = (M_vq["modal/normal"]["modal/normal"] - M_vq["breathy"]["modal/normal"]) - (M_vq["modal/normal"]["breathy"] - M_vq["breathy"]["breathy"])
    p_star_modal_breathy = num_modal_breathy / max(1e-9, den_modal_breathy)

    num_modal_harsh = M_vq["harsh/creaky"]["harsh/creaky"] - M_vq["harsh/creaky"]["modal/normal"]
    den_modal_harsh = (M_vq["modal/normal"]["modal/normal"] - M_vq["harsh/creaky"]["modal/normal"]) - (M_vq["modal/normal"]["harsh/creaky"] - M_vq["harsh/creaky"]["harsh/creaky"])
    p_star_modal_harsh = num_modal_harsh / max(1e-9, den_modal_harsh)

    results["vq_thresholds"] = {
        "P_star_modal_vs_breathy": p_star_modal_breathy,
        "P_star_modal_vs_harsh": p_star_modal_harsh,
        "hit_modal": M_vq["modal/normal"]["modal/normal"],
        "hit_breathy": M_vq["breathy"]["breathy"],
        "miss_breathy_on_modal": M_vq["modal/normal"]["breathy"],
        "miss_modal_on_breathy": M_vq["breathy"]["modal/normal"],
    }

    if p_star_modal_breathy > max_modal_threshold:
        results["passed"] = False
        results["failures"].append(
            f"INCONSISTÊNCIA EM VOICE QUALITY: Threshold de modal/normal vs breathy é {p_star_modal_breathy:.1%} "
            f"(tolerância máx: {max_modal_threshold:.1%}). O modelo necessita de mais de "
            f"{p_star_modal_breathy:.1%} de certeza em modal para não chutar breathy!"
        )

    # Scenario 1: Model has evidence of Modal (65% modal, 35% breathy)
    eu_modal_65 = 0.65 * M_vq["modal/normal"]["modal/normal"] + 0.35 * M_vq["breathy"]["modal/normal"]
    eu_breathy_65 = 0.65 * M_vq["modal/normal"]["breathy"] + 0.35 * M_vq["breathy"]["breathy"]

    results["scenarios"]["voice_quality_65_modal_35_breathy"] = {
        "belief": {"modal/normal": 0.65, "breathy": 0.35},
        "EU_predict_modal": eu_modal_65,
        "EU_predict_breathy": eu_breathy_65,
        "chosen_action": "breathy" if eu_breathy_65 > eu_modal_65 else "modal/normal",
    }
    if eu_breathy_65 > eu_modal_65:
        results["passed"] = False
        results["failures"].append(
            f"INCONSISTÊNCIA DE DECISÃO EM VOICE QUALITY: Com 65% de probabilidade para modal/normal "
            f"e 35% para breathy, a recompensa por prever breathy (+{eu_breathy_65:.4f}) superou "
            f"a de modal/normal (+{eu_modal_65:.4f})!"
        )

    # Scenario 2: Model has evidence of Breathy (50% modal, 50% breathy, shifted from 65.8% prior)
    eu_modal_50 = 0.50 * M_vq["modal/normal"]["modal/normal"] + 0.50 * M_vq["breathy"]["modal/normal"]
    eu_breathy_50 = 0.50 * M_vq["modal/normal"]["breathy"] + 0.50 * M_vq["breathy"]["breathy"]

    results["scenarios"]["voice_quality_50_modal_50_breathy"] = {
        "belief": {"modal/normal": 0.50, "breathy": 0.50},
        "EU_predict_modal": eu_modal_50,
        "EU_predict_breathy": eu_breathy_50,
        "chosen_action": "breathy" if eu_breathy_50 > eu_modal_50 else "modal/normal",
    }
    if eu_modal_50 >= eu_breathy_50:
        results["passed"] = False
        results["failures"].append(
            f"INÉRCIA EM VOICE QUALITY: Com evidência apontando para breathy (50% modal, 50% breathy), "
            f"o modelo ficou preso em modal (+{eu_modal_50:.4f}) em vez de escolher breathy (+{eu_breathy_50:.4f})!"
        )

    return results


# ==============================================================================
# 4. Test 3: Simulação de Competição e Vantagem no GRPO (Advantage Drift)
# ==============================================================================

def build_test_completion(
    pitch_height: str,
    pitch_dynamics: str,
    vocal_energy: str,
    speaking_rate: str,
    voice_quality: str,
    emotion: str,
) -> str:
    """Helper that generates a standard XML CoT completion."""
    return (
        "<think>\n"
        "[Acoustic Inventory]\n"
        f"- Pitch Height: {pitch_height}\n"
        f"- Pitch Dynamics: {pitch_dynamics}\n"
        f"- Vocal Energy: {vocal_energy}\n"
        f"- Speaking Rate: {speaking_rate}\n"
        f"- Voice Quality: {voice_quality}\n"
        "[Prosodic Reasoning]\n"
        f"The acoustic cues indicate {pitch_height} pitch and {voice_quality} voice quality, "
        f"which strongly correlates with the expected emotional state.\n"
        "</think>\n"
        f"<answer>{emotion}</answer>"
    )


def check_grpo_advantage_cluster_drift(
    env: Dict[str, Any],
    n_samples: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """Simulates realistic GRPO groups (G=4 rollouts per prompt) across a representative dataset.

    Compares competing policies:
      - Policy A (Calm/Modal): Always predicts moderate cues, modal voice, neutral.
      - Policy B (Hyper-Expressive Attractor): Always predicts elevated, wide, loud, fast, breathy, joy.
      - Policy C (Acoustic-Grounded): Predicts true cues + true emotion with realistic 60% accuracy.
      - Policy D (Random Guess): Random baseline.

    Computes GRPO Advantages: A_g = (R_g - mean(R)) / (std(R) + 1e-6).
    Flags the inconsistency if Policy B achieves a net positive drift across the dataset,
    proving the coupled policy attractor that crashed the 150-step run.
    """
    rng = np.random.default_rng(seed)
    unified_reward = env["unified_gated_grpo_reward"]

    # Synthesize representative sample set based on training distribution
    # (15% neutral / modal, 25% joy / expressive, 60% other emotions with mixed cues)
    dataset: List[Dict[str, Any]] = []
    cues_cont = ["pitch_height", "pitch_dynamics", "vocal_energy", "speaking_rate"]

    # Speaker quantiles
    spk_quantiles = {
        "pitch_height": {"q25": 28.0, "q75": 34.0},
        "pitch_dynamics": {"q25": 5.0, "q75": 11.0},
        "vocal_energy": {"q25": 0.40, "q75": 0.90},
        "speaking_rate": {"q25": 2.20, "q75": 3.60},
        "jitter": {"q25": 0.015, "q75": 0.035},
        "shimmer": {"q25": 0.80, "q75": 1.60},
        "hnr": {"q25": 4.0, "q75": 9.0},
        "hammarberg": {"q25": 12.0, "q75": 20.0},
    }

    tier_vals = {
        "pitch_height": {"low": 24.0, "moderate": 31.0, "high": 38.0},
        "pitch_dynamics": {"low": 3.0, "moderate": 8.0, "high": 15.0},
        "vocal_energy": {"low": 0.25, "moderate": 0.65, "high": 1.25},
        "speaking_rate": {"low": 1.60, "moderate": 2.90, "high": 4.30},
    }

    vq_features = {
        "modal/normal": {"jitter": 0.025, "shimmer": 1.20, "hnr": 6.5, "hammarberg": 16.0},
        "pressed/tense": {"jitter": 0.025, "shimmer": 1.20, "hnr": 6.5, "hammarberg": 8.0},
        "breathy": {"jitter": 0.025, "shimmer": 1.20, "hnr": 6.5, "hammarberg": 24.0},
        "harsh/creaky": {"jitter": 0.050, "shimmer": 2.10, "hnr": 2.5, "hammarberg": 16.0},
    }

    for _ in range(n_samples):
        r_type = rng.random()
        if r_type < 0.20:
            # Type 1: Grounded Neutral / Modal speech (20% proportion)
            emo = "neutral"
            tiers = {c: "moderate" for c in cues_cont}
            vq = "modal/normal"
        elif r_type < 0.40:
            # Type 2: Expressive Joy / High Arousal speech (20% proportion)
            emo = "joy"
            tiers = {c: "high" for c in cues_cont}
            vq = "breathy"
        else:
            # Type 3: Other emotions (anger, sadness, fear, surprise, disgust)
            emo = rng.choice(["anger", "sadness", "surprise", "fear", "disgust"])
            tiers = {c: rng.choice(["low", "moderate", "high"], p=[0.25, 0.50, 0.25]) for c in cues_cont}
            vq = rng.choice(["modal/normal", "pressed/tense", "breathy", "harsh/creaky"], p=[0.37, 0.16, 0.22, 0.25])

        feat_row: Dict[str, float] = {}
        for c in cues_cont:
            col_name = env["CUE_FEATURE_MAP"][c]
            feat_row[col_name] = tier_vals[c][tiers[c]]

        for sub_c, v_col in env["VOICE_QUALITY_FEATURE_MAP"].items():
            feat_row[v_col] = vq_features[vq][sub_c]

        dataset.append({
            "emotion": emo,
            "tiers": tiers,
            "vq": vq,
            "features": feat_row,
        })

    # Run GRPO tournament for every sample
    advantages_calm: List[float] = []
    advantages_expressive: List[float] = []
    advantages_grounded: List[float] = []
    advantages_random: List[float] = []

    r_unified_calm: List[float] = []
    r_unified_expressive: List[float] = []

    for item in dataset:
        target_emo = item["emotion"]
        tiers = item["tiers"]
        vq = item["vq"]
        feats = item["features"]

        tier_to_prompt = {
            "pitch_height": {"low": "low", "moderate": "moderate", "high": "elevated"},
            "pitch_dynamics": {"low": "narrow", "moderate": "moderate", "high": "wide"},
            "vocal_energy": {"low": "quiet", "moderate": "moderate", "high": "loud"},
            "speaking_rate": {"low": "slow", "moderate": "moderate", "high": "fast"},
        }

        # 4 competing rollouts in the GRPO group:
        # Rollout 1: Policy A (Calm / Modal)
        comp_a = build_test_completion("moderate", "moderate", "moderate", "moderate", "modal/normal", "neutral")
        # Rollout 2: Policy B (Hyper-Expressive Attractor)
        comp_b = build_test_completion("elevated", "wide", "loud", "fast", "breathy", "joy")
        # Rollout 3: Policy C (Grounded - 60% chance of exact hit, 40% slight error)
        g_cues = {c: tier_to_prompt[c][tiers[c]] for c in cues_cont}
        comp_c = build_test_completion(g_cues["pitch_height"], g_cues["pitch_dynamics"], g_cues["vocal_energy"], g_cues["speaking_rate"], vq, target_emo)
        # Rollout 4: Policy D (Random)
        rnd_tiers = {c: rng.choice(["low", "moderate", "high"]) for c in cues_cont}
        rnd_vq = rng.choice(["modal/normal", "pressed/tense", "breathy", "harsh/creaky"])
        rnd_emo = rng.choice(list(env["VALID_EMOTIONS"]))
        comp_d = build_test_completion(
            tier_to_prompt["pitch_height"][rnd_tiers["pitch_height"]],
            tier_to_prompt["pitch_dynamics"][rnd_tiers["pitch_dynamics"]],
            tier_to_prompt["vocal_energy"][rnd_tiers["vocal_energy"]],
            tier_to_prompt["speaking_rate"][rnd_tiers["speaking_rate"]],
            rnd_vq,
            rnd_emo,
        )

        group_comps = [comp_a, comp_b, comp_c, comp_d]
        group_prompts = ["prompt"] * 4
        group_labels = [target_emo] * 4

        kw = {"speaker": ["SampleSpeaker"] * 4, "quantiles_dict": {"SampleSpeaker": spk_quantiles, "__GLOBAL__": spk_quantiles}}
        for col_name, val in feats.items():
            kw[col_name] = [val] * 4

        rewards = unified_reward(group_prompts, group_comps, label=group_labels, **kw)
        r_arr = np.array(rewards)
        r_mean = float(r_arr.mean())
        r_std = float(r_arr.std())
        adv = (r_arr - r_mean) / (r_std + 1e-6)

        advantages_calm.append(float(adv[0]))
        advantages_expressive.append(float(adv[1]))
        advantages_grounded.append(float(adv[2]))
        advantages_random.append(float(adv[3]))

        r_unified_calm.append(float(r_arr[0]))
        r_unified_expressive.append(float(r_arr[1]))

    mean_adv_expressive = float(np.mean(advantages_expressive))
    mean_adv_calm = float(np.mean(advantages_calm))
    mean_adv_grounded = float(np.mean(advantages_grounded))
    mean_r_exp = float(np.mean(r_unified_expressive))
    mean_r_calm = float(np.mean(r_unified_calm))
    r_gap = mean_r_exp - mean_r_calm

    results: Dict[str, Any] = {
        "passed": True,
        "failures": [],
        "mean_advantage_expressive_cluster": mean_adv_expressive,
        "mean_advantage_calm_modal_cluster": mean_adv_calm,
        "mean_advantage_grounded_policy": mean_adv_grounded,
        "mean_r_unified_expressive": mean_r_exp,
        "mean_r_unified_calm": mean_r_calm,
        "reward_gap_expressive_vs_calm": r_gap,
    }

    # CRITICAL INCONSISTENCY ASSERTIONS:
    # 1. Expressive cluster vs Calm cluster reward gap:
    # If the ungrounded expressive cluster earns significantly higher reward than the ungrounded calm cluster
    # (e.g. > +0.15 gap), it creates a structural gradient drift favoring wide/breathy/joy.
    if r_gap > 0.15:
        results["passed"] = False
        results["failures"].append(
            f"INCONSISTÊNCIA GRAVE DE AGREGAÇÃO (GAP DE RECOMPENSA): O cluster expressivo não-ancorado "
            f"(wide+elevated+breathy->joy) obtém recompensa média R = {mean_r_exp:.4f}, enquanto o cluster calmo "
            f"(moderate+modal->neutral) obtém apenas R = {mean_r_calm:.4f} (Gap = +{r_gap:.4f}!). "
            f"Essa assimetria estrutural de +{r_gap:.4f} puxa a exploração do GRPO diretamente para o cluster de joy/wide."
        )

    # 2. Advantage asymmetry: Expressive cluster beats Calm cluster in advantage
    adv_gap = mean_adv_expressive - mean_adv_calm
    if adv_gap > 0.10:
        results["passed"] = False
        results["failures"].append(
            f"INCONSISTÊNCIA DE VANTAGEM NO GRPO: O cluster expressivo tem vantagem relativa de +{adv_gap:.4f} "
            f"sobre o cluster calmo (Adv Expressivo = {mean_adv_expressive:+.4f} vs Adv Calmo = {mean_adv_calm:+.4f})."
        )

    return results


# ==============================================================================
# 5. Diagnostic Runner & Terminal Formatting
# ==============================================================================

def run_diagnostic_suite(env: Optional[Dict[str, Any]] = None) -> bool:
    """Executes the full diagnostic suite and prints structured analysis tables."""
    print("=" * 80)
    print("🔬 OPERA GRPO REWARD & CONDITIONAL PROBABILITY DIAGNOSTIC SUITE")
    print("=" * 80)

    if env is None:
        print("📦 Carregando ambiente de recompensa de ser_grpo_v2.ipynb...")
        env = load_grpo_notebook_env()
        print("✅ Ambiente carregado com sucesso!")

    # 1. Blind Guess Invariance
    print("\n" + "-" * 80)
    print("TESTE 1: CHUTE CEGO (BLIND GUESS EXPECTATION = 0.000000)")
    print("-" * 80)
    t1 = check_blind_guess_zero_mean(env)
    print(f"Status Chute Cego: {'[PASS] ✅' if t1['passed'] else '[FAIL] ❌'}")
    print(f"  • Emoções (7 classes)     : {'Zero-Mean Exato' if all(v['zero_mean'] for v in t1['emotions'].values()) else 'Falha'}")
    print(f"  • Cues Contínuos (4 cues) : {'Zero-Mean Exato' if all(all(tier['zero_mean'] for tier in c.values()) for c in t1['continuous_cues'].values()) else 'Falha'}")
    print(f"  • Voice Quality (4 classes): {'Zero-Mean Exato' if all(v['zero_mean'] for v in t1['voice_quality'].values()) else 'Falha'}")

    # 2. Conditional Probability & Decision Thresholds
    print("\n" + "-" * 80)
    print("TESTE 2: PROBABILIDADE CONDICIONAL & THRESHOLDS DE CRENÇA")
    print("-" * 80)
    t2 = check_conditional_probabilities_and_thresholds(env)
    ord_th = t2["ordinal_thresholds"]
    vq_th = t2["vq_thresholds"]

    print("📊 Thresholds Críticos de Decisão sob Incerteza (P*):")
    print(f"  • Pitch Dynamics: Moderate vs High/Wide -> P*(moderate) = {ord_th['P_star_moderate_vs_high']:.1%}")
    print(f"    - Acerto Moderate : +{ord_th['hit_moderate']:.4f} | Acerto Wide: +{ord_th['hit_high']:.4f}")
    print(f"    - Erro em Moderate: {ord_th['miss_high_on_moderate']:.4f} | Erro em Wide: {ord_th['miss_moderate_on_high']:.4f}")
    print(f"  • Voice Quality : Modal vs Breathy      -> P*(modal)    = {vq_th['P_star_modal_vs_breathy']:.1%}")
    print(f"    - Acerto Modal    : +{vq_th['hit_modal']:.4f} | Acerto Breathy: +{vq_th['hit_breathy']:.4f}")
    print(f"    - Erro em Modal   : {vq_th['miss_breathy_on_modal']:.4f} | Erro em Breathy: {vq_th['miss_modal_on_breathy']:.4f}")

    print("\n🎯 Simulação de Racionalidade Bayesiana Bidirecional:")
    sc1 = t2["scenarios"]["pitch_dynamics_65_mod_35_wide"]
    sc2 = t2["scenarios"]["pitch_dynamics_50_mod_50_wide"]
    print(f"  • Pitch Dynamics:")
    print(f"    - Cenário 65% Mod / 35% Wide (Evidência Mod) : EU(mod)={sc1['EU_predict_moderate']:+.4f} vs EU(wide)={sc1['EU_predict_wide']:+.4f} -> [{sc1['chosen_action'].upper()}]")
    print(f"    - Cenário 50% Mod / 50% Wide (Evidência Wide): EU(mod)={sc2['EU_predict_moderate']:+.4f} vs EU(wide)={sc2['EU_predict_wide']:+.4f} -> [{sc2['chosen_action'].upper()}]")

    sc3 = t2["scenarios"]["voice_quality_65_modal_35_breathy"]
    sc4 = t2["scenarios"]["voice_quality_50_modal_50_breathy"]
    print(f"  • Voice Quality:")
    print(f"    - Cenário 65% Modal / 35% Breathy (Evidência Modal)  : EU(modal)={sc3['EU_predict_modal']:+.4f} vs EU(breathy)={sc3['EU_predict_breathy']:+.4f} -> [{sc3['chosen_action'].upper()}]")
    print(f"    - Cenário 50% Modal / 50% Breathy (Evidência Breathy): EU(modal)={sc4['EU_predict_modal']:+.4f} vs EU(breathy)={sc4['EU_predict_breathy']:+.4f} -> [{sc4['chosen_action'].upper()}]")

    print(f"\nStatus Teste 2: {'[PASS] ✅' if t2['passed'] else '[FAIL] ❌ INCONSISTÊNCIA DETECTADA'}")
    for fail in t2["failures"]:
        print(f"  ⚠️ {fail}")

    # 3. GRPO Advantage Drift
    print("\n" + "-" * 80)
    print("TESTE 3: SIMULAÇÃO DE COMPETIÇÃO E VANTAGEM NO GRPO (G=4)")
    print("-" * 80)
    t3 = check_grpo_advantage_cluster_drift(env, n_samples=200)
    print(f"  • Vantagem Média Política Cega Expressiva (Joy+Wide+Breathy) : {t3['mean_advantage_expressive_cluster']:+.4f}")
    print(f"  • Vantagem Média Política Cega Calma (Neutral+Moderate+Modal) : {t3['mean_advantage_calm_modal_cluster']:+.4f}")
    print(f"  • Vantagem Média Política Acurada (Grounded)                  : {t3['mean_advantage_grounded_policy']:+.4f}")
    print(f"  • Recompensa Unificada Média (Expressiva)                     : {t3['mean_r_unified_expressive']:.4f}")
    print(f"  • Recompensa Unificada Média (Calma)                          : {t3['mean_r_unified_calm']:.4f}")
    print(f"  • Gap de Recompensa (Expressivo - Calmo)                      : {t3['reward_gap_expressive_vs_calm']:+.4f}")

    print(f"\nStatus Teste 3: {'[PASS] ✅' if t3['passed'] else '[FAIL] ❌ INCONSISTÊNCIA DETECTADA'}")
    for fail in t3["failures"]:
        print(f"  ⚠️ {fail}")

    overall_pass = t1["passed"] and t2["passed"] and t3["passed"]
    print("\n" + "=" * 80)
    print(f"📋 RESULTADO GERAL DO DIAGNÓSTICO: {'APROVADO ✅' if overall_pass else 'REPROVADO ❌ (INCONSISTÊNCIAS IDENTIFICADAS)'}")
    print("=" * 80)

    return overall_pass


# ==============================================================================
# Pytest Test Endpoints
# ==============================================================================

def test_pytest_blind_guess_zero_mean(env: Dict[str, Any]) -> None:
    res = check_blind_guess_zero_mean(env)
    assert res["passed"], "Falha no Chute Cego Zero-Mean!"


def test_pytest_conditional_probabilities_and_thresholds(env: Dict[str, Any]) -> None:
    res = check_conditional_probabilities_and_thresholds(env)
    assert res["passed"], "\n".join(res["failures"])


def test_pytest_grpo_advantage_cluster_drift(env: Dict[str, Any]) -> None:
    res = check_grpo_advantage_cluster_drift(env)
    assert res["passed"], "\n".join(res["failures"])


if __name__ == "__main__":
    success = run_diagnostic_suite()
    sys.exit(0 if success else 1)
    sys.exit(0 if success else 1)

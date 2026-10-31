"""Mathematical validation tests for the GRPO reward system in ser_grpo_v2.ipynb.

Validates Fix 1, Fix 2, and Fix 3 both in isolation (R1, per-cue s_c, R2, R3)
and combined in R_unified using sample physical eGeMAPS values and XML completions:
  - Fix 1: Quartile-Preserving (25% / 50% / 25%) Ordinal Zero-Mean Payoff Matrix
           for the 4 continuous prosodic cues + Zero-Mean categorical voice_quality.
  - Fix 2: Constant-denominator affine normalization (R2_NORM_SCALE = 2.0) in R2,
           eliminating Jensen's inequality / random-denominator bias.
  - Fix 3: Zero-Mean Soft-Causal Gating in unified_gated_grpo_reward (R_unified),
           ensuring no emotion or acoustic cue class has any blind-guess advantage.
"""

import json
from pathlib import Path
import unittest
from typing import Any, Dict, List
import numpy as np
import pandas as pd


def load_notebook_reward_env() -> Dict[str, Any]:
    """Loads and executes the reward cells from ser_grpo_v2.ipynb in an isolated namespace."""
    nb_path = Path(__file__).resolve().parent.parent / "ser_grpo_v2.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    env: Dict[str, Any] = {
        "VALID_EMOTIONS": {
            "anger",
            "disgust",
            "fear",
            "joy",
            "neutral",
            "sadness",
            "surprise",
        },
        "cot_prompt": "Analyze the speaker's acoustic cues and deduce the emotion.",
        "display": lambda *args, **kwargs: None,
    }

    # Execute cells 50-55 (parsers), 59 (R1), 61 (R2), 63 (R3), 65 (R_unified)
    for cell_idx in [50, 51, 52, 53, 54, 55, 59, 61, 63, 65]:
        code = "".join(nb["cells"][cell_idx]["source"])
        exec(compile(code, f"<ser_grpo_v2_cell_{cell_idx}>", "exec"), env)

    # Also load INVENTORY_DIMENSIONS and evaluate_single_prompt_acoustics from cell 94 via AST
    import ast
    cell_94_src = "".join(nb["cells"][94]["source"])
    tree_94 = ast.parse(cell_94_src)
    kept_nodes = [
        node
        for node in tree_94.body
        if (isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "INVENTORY_DIMENSIONS")
        or (isinstance(node, ast.FunctionDef) and node.name == "evaluate_single_prompt_acoustics")
    ]
    mod_94 = ast.Module(body=kept_nodes, type_ignores=[])
    exec(compile(mod_94, "<ser_grpo_v2_cell_94>", "exec"), env)

    return env


class TestGRPORewardMathematics(unittest.TestCase):
    """Tests isolated and combined GRPO reward functions with sample values."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = load_notebook_reward_env()
        cls.CUE_FEATURE_MAP = cls.env["CUE_FEATURE_MAP"]
        cls.VOICE_QUALITY_FEATURE_MAP = cls.env["VOICE_QUALITY_FEATURE_MAP"]
        cls.ALL_ACOUSTIC_FEATURES = cls.env["ALL_ACOUSTIC_FEATURES"]
        cls.R2_NORM_SCALE = cls.env["R2_NORM_SCALE"]
        cls.VALID_EMOTIONS = sorted(cls.env["VALID_EMOTIONS"])

        # Sample speaker quantiles (Q25 and Q75) representing scientific quartile boundaries
        cls.sample_quantiles = {
            "SampleSpeaker": {
                "pitch_height": {"q25": 28.0, "q75": 34.0},
                "pitch_dynamics": {"q25": 5.0, "q75": 11.0},
                "vocal_energy": {"q25": 0.40, "q75": 0.90},
                "speaking_rate": {"q25": 2.20, "q75": 3.60},
                "jitter": {"q25": 0.015, "q75": 0.035},
                "shimmer": {"q25": 0.80, "q75": 1.60},
                "hnr": {"q25": 4.0, "q75": 9.0},
                "hammarberg": {"q25": 12.0, "q75": 20.0},
            }
        }
        cls.sample_quantiles["__GLOBAL__"] = cls.sample_quantiles["SampleSpeaker"]

        # Sample physical values falling strictly inside Low (<= Q25), Moderate (Q25..Q75), High (> Q75)
        cls.cont_values = {
            "pitch_height": {"low": 24.0, "moderate": 31.0, "high": 38.0},
            "pitch_dynamics": {"low": 3.0, "moderate": 8.0, "high": 15.0},
            "vocal_energy": {"low": 0.25, "moderate": 0.65, "high": 1.25},
            "speaking_rate": {"low": 1.60, "moderate": 2.90, "high": 4.30},
        }
        cls.vq_values = {
            "modal/normal": {"jitter": 0.025, "shimmer": 1.20, "hnr": 6.5, "hammarberg": 16.0},
            "pressed/tense": {"jitter": 0.025, "shimmer": 1.20, "hnr": 6.5, "hammarberg": 8.0},
            "breathy": {"jitter": 0.025, "shimmer": 1.20, "hnr": 6.5, "hammarberg": 24.0},
            "harsh/creaky": {"jitter": 0.050, "shimmer": 2.10, "hnr": 2.5, "hammarberg": 16.0},
        }

        # Scientific Quartile cycle: 25% low, 50% moderate, 25% high (N=20 samples)
        cls.quartile_cycle = ["low", "moderate", "moderate", "high"]
        cls.vq_cycle = (
            ["modal/normal"] * 12
            + ["pressed/tense"] * 3
            + ["breathy"] * 3
            + ["harsh/creaky"] * 2
        )

        cls.acoustic_samples: List[Dict[str, Any]] = []
        for i in range(20):
            t_tier = cls.quartile_cycle[i % 4]
            t_vq = cls.vq_cycle[i]
            cls.acoustic_samples.append({
                "audio_path": f"sample_{i}.wav",
                "speaker": "SampleSpeaker",
                "truth_tier": t_tier,
                "truth_vq": t_vq,
                cls.CUE_FEATURE_MAP["pitch_height"]: cls.cont_values["pitch_height"][t_tier],
                cls.CUE_FEATURE_MAP["pitch_dynamics"]: cls.cont_values["pitch_dynamics"][t_tier],
                cls.CUE_FEATURE_MAP["vocal_energy"]: cls.cont_values["vocal_energy"][t_tier],
                cls.CUE_FEATURE_MAP["speaking_rate"]: cls.cont_values["speaking_rate"][t_tier],
                cls.VOICE_QUALITY_FEATURE_MAP["jitter"]: cls.vq_values[t_vq]["jitter"],
                cls.VOICE_QUALITY_FEATURE_MAP["shimmer"]: cls.vq_values[t_vq]["shimmer"],
                cls.VOICE_QUALITY_FEATURE_MAP["hnr"]: cls.vq_values[t_vq]["hnr"],
                cls.VOICE_QUALITY_FEATURE_MAP["hammarberg"]: cls.vq_values[t_vq]["hammarberg"],
            })

        cls.df_acoustic = pd.DataFrame(cls.acoustic_samples)

        # Imbalanced sample emotion distribution (N=20 samples)
        cls.emotion_counts = {
            "neutral": 7,
            "joy": 4,
            "anger": 3,
            "sadness": 2,
            "fear": 2,
            "disgust": 1,
            "surprise": 1,
        }
        cls.emotion_samples = [
            emo for emo, cnt in cls.emotion_counts.items() for _ in range(cnt)
        ]
        cls.df_emotions = pd.DataFrame({"emotion": cls.emotion_samples})

        # Calibrate priors using the actual notebook functions
        cls.emo_priors, cls.emo_weights = cls.env["compute_emotion_priors_and_weights"](
            cls.df_emotions, power=0.35
        )
        cls.cue_priors = cls.env["compute_acoustic_cue_priors"](
            cls.df_acoustic, cls.sample_quantiles
        )

        cls.env["EMOTION_PRIORS"].clear()
        cls.env["EMOTION_PRIORS"].update(cls.emo_priors)
        cls.env["CLASS_WEIGHTS"].clear()
        cls.env["CLASS_WEIGHTS"].update(cls.emo_weights)
        cls.env["ACOUSTIC_CUE_PRIORS"].clear()
        cls.env["ACOUSTIC_CUE_PRIORS"].update(cls.cue_priors)
        cls.env["SPEAKER_ACOUSTIC_QUANTILES"].clear()
        cls.env["SPEAKER_ACOUSTIC_QUANTILES"].update(cls.sample_quantiles)

    @staticmethod
    def build_completion(
        pitch_height: str = "moderate",
        pitch_dynamics: str = "moderate",
        vocal_energy: str = "moderate",
        speaking_rate: str = "moderate",
        voice_quality: str = "modal/normal",
        emotion: str = "neutral",
    ) -> str:
        return (
            "<think>\n"
            "[Acoustic Inventory]\n"
            f"- Pitch Height: {pitch_height}\n"
            f"- Pitch Dynamics: {pitch_dynamics}\n"
            f"- Vocal Energy: {vocal_energy}\n"
            f"- Speaking Rate: {speaking_rate}\n"
            f"- Voice Quality: {voice_quality}\n\n"
            "[Prosodic Reasoning]\n"
            "The speaker exhibits prosodic cues consistent with the acoustic inventory above.\n"
            "</think>\n"
            f"<answer>{emotion}</answer>"
        )

    def test_1_isolated_r1_zero_mean_all_emotions(self) -> None:
        """Validates R1 in isolation: blind guessing any emotion yields E[R1] == 0.000000."""
        reward_r1 = self.env["reward_label_accuracy_weighted"]
        prompts = ["prompt"] * len(self.emotion_samples)

        blind_means = {}
        for blind_emo in self.VALID_EMOTIONS:
            completions = [self.build_completion(emotion=blind_emo) for _ in self.emotion_samples]
            scores = reward_r1(prompts, completions, label=self.emotion_samples)
            mean_score = float(np.mean(scores))
            blind_means[blind_emo] = mean_score
            self.assertAlmostEqual(
                mean_score,
                0.0,
                places=9,
                msg=f"Blind emotion '{blind_emo}' has non-zero E[R1]={mean_score}",
            )

        # Spread between highest and lowest blind emotion expectation must be 0.0
        spread = max(blind_means.values()) - min(blind_means.values())
        self.assertAlmostEqual(spread, 0.0, places=9)

        # Oracle emotion prediction must yield strictly positive mean reward (> 1.0)
        oracle_comps = [self.build_completion(emotion=e) for e in self.emotion_samples]
        oracle_mean = float(np.mean(reward_r1(prompts, oracle_comps, label=self.emotion_samples)))
        self.assertGreater(oracle_mean, 1.10)

    def test_2_isolated_cues_fix1_ordinal_and_zero_mean(self) -> None:
        """Validates Fix 1 & Fix 2 per cue in isolation: E[s_c] == 0.0, E[norm_sc] == 0.50, and strict ordinality."""
        score_cue = self.env["score_cue_zero_mean"]
        eval_single = self.env["evaluate_single_prompt_acoustics"]

        prompt_options = {
            "pitch_height": ["low", "moderate", "elevated"],
            "pitch_dynamics": ["narrow", "moderate", "wide"],
            "vocal_energy": ["quiet", "moderate", "loud"],
            "speaking_rate": ["slow", "moderate", "fast"],
            "voice_quality": ["modal/normal", "pressed/tense", "breathy", "harsh/creaky"],
        }

        # Check each of the 5 cues in isolation via evaluate_single_prompt_acoustics
        for cue_name, options in prompt_options.items():
            for opt in options:
                raw_scores = []
                norm_scores = []
                for row in self.acoustic_samples:
                    kw = {
                        "pitch_height": "moderate",
                        "pitch_dynamics": "moderate",
                        "vocal_energy": "moderate",
                        "speaking_rate": "moderate",
                        "voice_quality": "modal/normal",
                    }
                    kw[cue_name] = opt
                    comp = self.build_completion(**kw)
                    res = eval_single(
                        completion=comp,
                        audio_path=row["audio_path"],
                        speaker=row["speaker"],
                        row_data=row,
                        quantiles_dict=self.sample_quantiles,
                        audio_lookup={},
                        priors_dict=self.cue_priors,
                    )
                    raw_scores.append(res["cue_raw_scores"][cue_name])
                    norm_scores.append(res["cue_scores"][cue_name])

                mean_raw = float(np.mean(raw_scores))
                mean_norm = float(np.mean(norm_scores))
                self.assertAlmostEqual(
                    mean_raw,
                    0.0,
                    places=9,
                    msg=f"Cue '{cue_name}' option '{opt}' has non-zero E[s_c]={mean_raw}",
                )
                self.assertAlmostEqual(
                    mean_norm,
                    0.50,
                    places=9,
                    msg=f"Cue '{cue_name}' option '{opt}' has E[norm_sc]={mean_norm} != 0.50",
                )

        # Verify strict topological ordinality for the 4 continuous quartile cues (Fix 1)
        for cue_name in self.CUE_FEATURE_MAP:
            # Ground truth = low: hit(low) > 0 > miss_1step(moderate) > miss_2step(high)
            s_ll, _, _ = score_cue("low", "low", cue_name, self.cue_priors)
            s_lm, _, _ = score_cue("low", "moderate", cue_name, self.cue_priors)
            s_lh, _, _ = score_cue("low", "high", cue_name, self.cue_priors)
            self.assertGreater(s_ll, 0.0)
            self.assertLess(s_lm, 0.0)
            self.assertGreater(
                s_lm,
                s_lh,
                msg=f"Fix 1 ordinality failed on truth=low for {cue_name}: mod={s_lm} <= high={s_lh}",
            )

            # Ground truth = high: hit(high) > 0 > miss_1step(moderate) > miss_2step(low)
            s_hh, _, _ = score_cue("high", "high", cue_name, self.cue_priors)
            s_hm, _, _ = score_cue("high", "moderate", cue_name, self.cue_priors)
            s_hl, _, _ = score_cue("high", "low", cue_name, self.cue_priors)
            self.assertGreater(s_hh, 0.0)
            self.assertLess(s_hm, 0.0)
            self.assertGreater(
                s_hm,
                s_hl,
                msg=f"Fix 1 ordinality failed on truth=high for {cue_name}: mod={s_hm} <= low={s_hl}",
            )

            # Ground truth = moderate: hit(moderate) > 0 > miss_1step(low) == miss_1step(high)
            s_mm, _, _ = score_cue("moderate", "moderate", cue_name, self.cue_priors)
            s_ml, _, _ = score_cue("moderate", "low", cue_name, self.cue_priors)
            s_mh, _, _ = score_cue("moderate", "high", cue_name, self.cue_priors)
            self.assertGreater(s_mm, 0.0)
            self.assertLess(s_ml, 0.0)
            self.assertAlmostEqual(s_ml, s_mh, places=9)

    def test_3_isolated_r2_fix2_constant_normalization_equal_probability(self) -> None:
        """Validates R2 in isolation: every blind acoustic strategy yields E[R2] == 0.500000."""
        reward_r2 = self.env["reward_acoustic_inventory"]
        prompts = ["prompt"] * len(self.acoustic_samples)
        ac_kwargs = {
            "speaker": [r["speaker"] for r in self.acoustic_samples],
            **{
                col: [r[col] for r in self.acoustic_samples]
                for col in self.ALL_ACOUSTIC_FEATURES.values()
            },
        }

        blind_policies = [
            ("low", "narrow", "quiet", "slow", "modal/normal"),
            ("moderate", "moderate", "moderate", "moderate", "modal/normal"),
            ("elevated", "wide", "loud", "fast", "modal/normal"),
            ("low", "narrow", "quiet", "slow", "pressed/tense"),
            ("moderate", "moderate", "moderate", "moderate", "breathy"),
            ("elevated", "wide", "loud", "fast", "harsh/creaky"),
            ("low", "moderate", "loud", "slow", "breathy"),
            ("elevated", "narrow", "moderate", "fast", "pressed/tense"),
        ]

        for pol in blind_policies:
            comps = [self.build_completion(*pol, emotion="neutral") for _ in self.acoustic_samples]
            r2_scores = reward_r2(prompts, comps, **ac_kwargs)
            mean_r2 = float(np.mean(r2_scores))
            self.assertAlmostEqual(
                mean_r2,
                0.50,
                places=9,
                msg=f"Blind acoustic policy {pol} has E[R2]={mean_r2} != 0.50",
            )

    def test_4_isolated_r3_format_reward(self) -> None:
        """Validates R3 format reward in isolation."""
        reward_r3 = self.env["reward_format"]
        valid_comp = self.build_completion()
        malformed_comp = "The speaker sounds angry."

        scores = reward_r3(["p1", "p2"], [valid_comp, malformed_comp])
        self.assertAlmostEqual(scores[0], 1.0, places=6)
        self.assertAlmostEqual(scores[1], 0.0, places=6)

    def test_5_combined_r_unified_fix3_no_class_benefit_on_blind_guess(self) -> None:
        """Validates combined R_unified: every blind (emotion, acoustic) pair yields E[R_unified] == 0.700000."""
        unified_reward = self.env["unified_gated_grpo_reward"]

        # Construct full sample product grid (20 emotion samples x 20 acoustic samples = 400 samples)
        grid_labels: List[str] = []
        grid_kwargs: Dict[str, List[Any]] = {"speaker": []}
        for col in self.ALL_ACOUSTIC_FEATURES.values():
            grid_kwargs[col] = []

        for emo_t in self.emotion_samples:
            for ac_row in self.acoustic_samples:
                grid_labels.append(emo_t)
                grid_kwargs["speaker"].append(ac_row["speaker"])
                for col in self.ALL_ACOUSTIC_FEATURES.values():
                    grid_kwargs[col].append(ac_row[col])

        grid_prompts = ["prompt"] * len(grid_labels)

        blind_acoustic_strategies = [
            ("low", "narrow", "quiet", "slow", "modal/normal"),
            ("moderate", "moderate", "moderate", "moderate", "modal/normal"),
            ("elevated", "wide", "loud", "fast", "modal/normal"),
            ("low", "narrow", "quiet", "slow", "pressed/tense"),
            ("moderate", "moderate", "moderate", "moderate", "breathy"),
            ("elevated", "wide", "loud", "fast", "harsh/creaky"),
            ("low", "moderate", "loud", "fast", "harsh/creaky"),
        ]

        all_expected_values = []
        for blind_emo in self.VALID_EMOTIONS:
            for ac_pol in blind_acoustic_strategies:
                comps = [
                    self.build_completion(*ac_pol, emotion=blind_emo)
                    for _ in grid_labels
                ]
                scores = unified_reward(grid_prompts, comps, label=grid_labels, **grid_kwargs)
                mean_u = float(np.mean(scores))
                all_expected_values.append(mean_u)
                self.assertAlmostEqual(
                    mean_u,
                    0.70,
                    places=9,
                    msg=f"Blind policy (emotion={blind_emo}, acoustics={ac_pol}) gave E[R_unified]={mean_u} != 0.70",
                )

        # Max spread across all 7 x 7 = 49 blind (emotion, acoustic) combinations must be 0.0
        spread = max(all_expected_values) - min(all_expected_values)
        self.assertAlmostEqual(spread, 0.0, places=9)

    def test_6_r_unified_causal_monotonicity_and_ordinal_ranking(self) -> None:
        """Validates that R_unified strictly rewards better acoustics on both right and wrong emotions."""
        unified_reward = self.env["unified_gated_grpo_reward"]

        # Sample where ground truth is: emotion='anger', all 4 continuous cues='low', voice_quality='harsh/creaky'
        sample_kw = {
            "label": ["anger"],
            "speaker": ["SampleSpeaker"],
            self.CUE_FEATURE_MAP["pitch_height"]: [self.cont_values["pitch_height"]["low"]],
            self.CUE_FEATURE_MAP["pitch_dynamics"]: [self.cont_values["pitch_dynamics"]["low"]],
            self.CUE_FEATURE_MAP["vocal_energy"]: [self.cont_values["vocal_energy"]["low"]],
            self.CUE_FEATURE_MAP["speaking_rate"]: [self.cont_values["speaking_rate"]["low"]],
            self.VOICE_QUALITY_FEATURE_MAP["jitter"]: [self.vq_values["harsh/creaky"]["jitter"]],
            self.VOICE_QUALITY_FEATURE_MAP["shimmer"]: [self.vq_values["harsh/creaky"]["shimmer"]],
            self.VOICE_QUALITY_FEATURE_MAP["hnr"]: [self.vq_values["harsh/creaky"]["hnr"]],
            self.VOICE_QUALITY_FEATURE_MAP["hammarberg"]: [self.vq_values["harsh/creaky"]["hammarberg"]],
        }

        # Case A: Correct emotion ('anger') with 0-step (Oracle) vs 1-step (Moderate) vs 2-step (High) acoustics
        c_hit_0step = self.build_completion("low", "narrow", "quiet", "slow", "harsh/creaky", "anger")
        c_hit_1step = self.build_completion("moderate", "moderate", "moderate", "moderate", "harsh/creaky", "anger")
        c_hit_2step = self.build_completion("elevated", "wide", "loud", "fast", "harsh/creaky", "anger")

        u_hit_0 = unified_reward(["p"], [c_hit_0step], **sample_kw)[0]
        u_hit_1 = unified_reward(["p"], [c_hit_1step], **sample_kw)[0]
        u_hit_2 = unified_reward(["p"], [c_hit_2step], **sample_kw)[0]
        self.assertGreater(u_hit_0, u_hit_1)
        self.assertGreater(u_hit_1, u_hit_2)

        # Case B: Wrong emotion ('neutral') with 0-step vs 1-step vs 2-step acoustics
        # Must NEVER punish good acoustics when the emotion is wrong!
        c_miss_0step = self.build_completion("low", "narrow", "quiet", "slow", "harsh/creaky", "neutral")
        c_miss_1step = self.build_completion("moderate", "moderate", "moderate", "moderate", "harsh/creaky", "neutral")
        c_miss_2step = self.build_completion("elevated", "wide", "loud", "fast", "harsh/creaky", "neutral")

        u_miss_0 = unified_reward(["p"], [c_miss_0step], **sample_kw)[0]
        u_miss_1 = unified_reward(["p"], [c_miss_1step], **sample_kw)[0]
        u_miss_2 = unified_reward(["p"], [c_miss_2step], **sample_kw)[0]
        self.assertGreater(u_miss_0, u_miss_1)
        self.assertGreater(u_miss_1, u_miss_2)

    def test_7_notebook_validation_cell_66_executes_cleanly(self) -> None:
        """Executes Cell 66 (validate_grpo_reward_mathematics) from ser_grpo_v2.ipynb directly."""
        nb_path = Path(__file__).resolve().parent.parent / "ser_grpo_v2.ipynb"
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        cell_66_src = "".join(nb["cells"][66]["source"])
        exec(compile(cell_66_src, "<ser_grpo_v2_cell_66>", "exec"), self.env)
        report = self.env["validation_math_report"]
        for emo, val in report["r1_blind_means"].items():
            self.assertAlmostEqual(val, 0.0, places=9)
        for pol, val in report["r2_blind_means"].items():
            self.assertAlmostEqual(val, 0.50, places=9)
        u_mat_vals = report["r_unified_matrix"].values.flatten()
        for val in u_mat_vals:
            self.assertAlmostEqual(float(val), 0.70, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)

import unittest
from typing import Any, Dict, List, Optional
import pandas as pd


INVENTORY_DIMENSIONS: Dict[str, Dict[str, Any]] = {
    "pitch_height": {
        "title": "Pitch Height",
        "options": ["low", "moderate", "elevated"],
        "tier_to_prompt": {"low": "low", "moderate": "moderate", "high": "elevated"},
    },
    "pitch_dynamics": {
        "title": "Pitch Dynamics",
        "options": ["narrow", "moderate", "wide"],
        "tier_to_prompt": {"low": "narrow", "moderate": "moderate", "high": "wide"},
    },
    "vocal_energy": {
        "title": "Vocal Energy",
        "options": ["quiet", "moderate", "loud"],
        "tier_to_prompt": {"low": "quiet", "moderate": "moderate", "high": "loud"},
    },
    "speaking_rate": {
        "title": "Speaking Rate",
        "options": ["slow", "moderate", "fast"],
        "tier_to_prompt": {"low": "slow", "moderate": "moderate", "high": "fast"},
    },
    "voice_quality": {
        "title": "Voice Quality",
        "options": ["modal/normal", "pressed/tense", "breathy", "harsh/creaky"],
        "tier_to_prompt": None,
    },
}

CUE_FEATURE_MAP: Dict[str, str] = {
    "pitch_height": "F0semitoneFrom27.5Hz_sma3nz_amean",
    "pitch_dynamics": "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
    "vocal_energy": "loudness_sma3_percentile50.0",
    "speaking_rate": "VoicedSegmentsPerSec",
}

VOICE_QUALITY_FEATURE_MAP: Dict[str, str] = {
    "jitter": "jitterLocal_sma3nz_amean",
    "shimmer": "shimmerLocaldB_sma3nz_amean",
    "hnr": "HNRdBACF_sma3nz_amean",
    "hammarberg": "hammarbergIndexV_sma3nz_amean",
}


def determine_voice_quality_ground_truth(
    jitter_val: float,
    shimmer_val: float,
    hnr_val: float,
    hammarberg_val: float,
    quantiles: Dict[str, Dict[str, float]],
) -> str:
    def _tier(cue_name: str, val: float) -> str:
        q = quantiles.get(cue_name, {})
        q25 = q.get("q25", float("-inf"))
        q75 = q.get("q75", float("inf"))
        if val <= q25:
            return "low"
        elif val <= q75:
            return "moderate"
        else:
            return "high"

    j_tier = _tier("jitter", jitter_val)
    s_tier = _tier("shimmer", shimmer_val)
    h_tier = _tier("hnr", hnr_val)
    tilt_tier = _tier("hammarberg", hammarberg_val)

    if (j_tier == "high" and s_tier == "high") or ((j_tier == "high" or s_tier == "high") and h_tier == "low"):
        return "harsh/creaky"
    if tilt_tier == "high" and j_tier != "high":
        return "breathy"
    if tilt_tier == "moderate" and h_tier == "low" and j_tier != "high" and s_tier != "high":
        return "breathy"
    if tilt_tier == "low" and j_tier != "high" and s_tier != "high":
        return "pressed/tense"
    return "modal/normal"


def evaluate_single_prompt_acoustics(
    audio_path: str,
    speaker: str,
    row_data: Optional[Dict[str, Any]] = None,
    quantiles_dict: Optional[Dict[str, Any]] = None,
    audio_lookup: Optional[Dict[str, Any]] = None,
    weights_dict: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, str]:
    """Extrai o perfil acústico ground-truth discretizado (5 dimensões) para o prompt."""
    q_dict = quantiles_dict if quantiles_dict is not None else globals().get("SPEAKER_ACOUSTIC_QUANTILES", {})
    lookup = audio_lookup if audio_lookup is not None else globals().get("AUDIO_ACOUSTIC_LOOKUP", {})
    cue_map = globals().get("CUE_FEATURE_MAP", CUE_FEATURE_MAP)
    vq_map = globals().get("VOICE_QUALITY_FEATURE_MAP", VOICE_QUALITY_FEATURE_MAP)

    row = row_data or {}
    spk = speaker if speaker else row.get("speaker", lookup.get(audio_path, {}).get("speaker", "__GLOBAL__"))
    spk_q = q_dict.get(spk, q_dict.get("__GLOBAL__", {}))

    audio_feat_cache = lookup.get(audio_path, {})
    inventory: Dict[str, str] = {}

    # 1. Avaliação dos 4 cues físicos contínuos
    for cue_name in ["pitch_height", "pitch_dynamics", "vocal_energy", "speaking_rate"]:
        feat_col = cue_map.get(cue_name)
        val = row.get(feat_col) if (feat_col in row and not pd.isna(row.get(feat_col))) else audio_feat_cache.get(feat_col)

        truth_tier = "moderate"
        if val is not None and not pd.isna(val) and cue_name in spk_q:
            q25 = spk_q[cue_name].get("q25", float("-inf"))
            q75 = spk_q[cue_name].get("q75", float("inf"))
            if val <= q25:
                truth_tier = "low"
            elif val <= q75:
                truth_tier = "moderate"
            else:
                truth_tier = "high"

        tier_map = INVENTORY_DIMENSIONS.get(cue_name, {}).get("tier_to_prompt", {})
        inventory[cue_name] = tier_map.get(truth_tier, truth_tier)

    # 2. Avaliação de Voice Quality
    vq_vals: Dict[str, float] = {}
    for sub_cue, col_name in vq_map.items():
        v = row.get(col_name) if (col_name in row and not pd.isna(row.get(col_name))) else audio_feat_cache.get(col_name)
        if v is not None and not pd.isna(v):
            vq_vals[sub_cue] = float(v)

    vq_fn = globals().get("determine_voice_quality_ground_truth", determine_voice_quality_ground_truth)
    if len(vq_vals) == len(vq_map) and callable(vq_fn):
        inventory["voice_quality"] = vq_fn(
            jitter_val=vq_vals["jitter"],
            shimmer_val=vq_vals["shimmer"],
            hnr_val=vq_vals["hnr"],
            hammarberg_val=vq_vals["hammarberg"],
            quantiles=spk_q,
        )
    else:
        inventory["voice_quality"] = "modal/normal"

    return inventory


def build_bridge_generation_prompt(
    inv: Dict[str, str],
    target_emotion: str,
) -> str:
    return f"""You are generating the internal reasoning step of a Speech Emotion Recognition model.

Given the following ground-truth acoustic profile extracted from a speech audio and its true emotion label:

[Acoustic Inventory]
- Pitch Height: {inv['pitch_height']}
- Pitch Dynamics: {inv['pitch_dynamics']}
- Vocal Energy: {inv['vocal_energy']}
- Speaking Rate: {inv['speaking_rate']}
- Voice Quality: {inv['voice_quality']}

Target Emotion: {target_emotion}

Task: Write ONLY 2 to 3 concise sentences in the present tense that logically bridge these specific acoustic cues to the "{target_emotion}" emotion.
Rules:
1. If some cues seem neutral or contrasting (e.g., moderate speed or low pitch during an intense emotion), explicitly explain how the other cues (such as voice quality, energy, or pitch dynamics) dominate or interact to reveal "{target_emotion}".
2. Write as a direct acoustic analysis of the speaker's voice (do NOT say "based on the provided inventory", "the final emotion is", or use past tense like "led to").
3. Output ONLY the 2-3 bridge sentences, with no tags, headers, or bullet points."""


class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.padding_side = "right"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        content = messages[0]["content"]
        return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True):
        class MockBatch(dict):
            def to(self, device):
                return self
        batch = MockBatch({
            "input_ids": [[1, 2, 3]] * len(texts),
            "attention_mask": [[1, 1, 1]] * len(texts),
        })
        return batch

    def batch_decode(self, tokens, skip_special_tokens=True):
        return ["The speaker shows elevated pitch and loud vocal energy consistent with joy." for _ in tokens]


class MockModel:
    def __init__(self):
        self.device = "cpu"
        self.training = False

        class GenConfig:
            pad_token_id = 0
            eos_token_id = 2
        self.generation_config = GenConfig()

    def eval(self):
        self.training = False

    def generate(self, **kwargs):
        input_ids = kwargs.get("input_ids", [])
        batch_size = len(input_ids)
        # return dummy token sequence of length input_len + 5
        import numpy as np
        return np.ones((batch_size, 8), dtype=int)


def generate_sft_train(
    model: Any,
    processor: Any,
    sft_train_set: Optional[pd.DataFrame] = None,
    batch_size: int = 32,
    max_new_tokens: int = 150,
    **kwargs: Any,
) -> pd.DataFrame:
    """Gera o raciocínio prosódico para o conjunto de Supervised Fine-Tuning (SFT)."""
    df_input = sft_train_set if sft_train_set is not None else kwargs.get("sft_train_setm")
    if df_input is None:
        df_input = globals().get("sft_train_set")
    if df_input is None:
        raise ValueError("sft_train_set DataFrame must be provided.")

    df = df_input.copy().reset_index(drop=True)
    tokenizer = getattr(processor, "tokenizer", processor)

    prompts: List[str] = []
    raw_prompts: List[str] = []
    inventories: List[Dict[str, str]] = []

    for _, row in df.iterrows():
        audio_p = str(row.get("audio_path", ""))
        spk = str(row.get("speaker", "__GLOBAL__"))
        target_emotion = str(row.get("emotion", "neutral")).strip().lower()

        # Extrai inventário acústico com evaluate_single_prompt_acoustics se válida
        inv: Optional[Dict[str, str]] = None
        eval_fn = globals().get("evaluate_single_prompt_acoustics", evaluate_single_prompt_acoustics)
        if callable(eval_fn):
            try:
                cand_inv = eval_fn(
                    audio_path=audio_p,
                    speaker=spk,
                    row_data=row.to_dict(),
                )
                if isinstance(cand_inv, dict) and all(
                    k in cand_inv for k in [
                        "pitch_height", "pitch_dynamics", "vocal_energy", "speaking_rate", "voice_quality"
                    ]
                ):
                    inv = cand_inv
            except Exception:
                inv = None

        if inv is None:
            inv = {
                "pitch_height": "moderate",
                "pitch_dynamics": "moderate",
                "vocal_energy": "moderate",
                "speaking_rate": "moderate",
                "voice_quality": "modal/normal",
            }

        inventories.append(inv)
        bridge_prompt = build_bridge_generation_prompt(inv=inv, target_emotion=target_emotion)
        raw_prompts.append(bridge_prompt)

        messages = [{"role": "user", "content": bridge_prompt}]
        if hasattr(processor, "apply_chat_template"):
            formatted = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        elif hasattr(tokenizer, "apply_chat_template"):
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted = f"<|im_start|>user\n{bridge_prompt}<|im_end|>\n<|im_start|>assistant\n"
        prompts.append(formatted)

    device = getattr(model, "device", "cpu")
    orig_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(model.generation_config, "pad_token_id", None)

    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()

    generated_reasonings: List[str] = []
    total = len(prompts)

    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch_texts = prompts[start_idx:end_idx]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)

        input_len = len(inputs["input_ids"][0]) if isinstance(inputs["input_ids"], list) else inputs["input_ids"].shape[1]
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
        )

        new_tokens = outputs[:, input_len:] if hasattr(outputs, "shape") else outputs
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for text in decoded:
            generated_reasonings.append(text.strip().strip('"').strip("'"))

    tokenizer.padding_side = orig_padding_side
    if was_training and hasattr(model, "train"):
        model.train()

    df["reasoning"] = generated_reasonings
    df["bridge_prompt"] = raw_prompts
    df["acoustic_inventory"] = inventories

    return df


class TestSftGeneration(unittest.TestCase):
    def setUp(self):
        self.sample_df = pd.DataFrame([
            {
                "audio_path": "audio1.wav",
                "speaker": "Joey",
                "emotion": "joy",
                "F0semitoneFrom27.5Hz_sma3nz_amean": 35.0,
                "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2": 15.0,
                "loudness_sma3_percentile50.0": 0.8,
                "VoicedSegmentsPerSec": 4.5,
                "jitterLocal_sma3nz_amean": 0.01,
                "shimmerLocaldB_sma3nz_amean": 0.5,
                "HNRdBACF_sma3nz_amean": 12.0,
                "hammarbergIndexV_sma3nz_amean": 18.0,
            },
            {
                "audio_path": "audio2.wav",
                "speaker": "Rachel",
                "emotion": "sadness",
                "F0semitoneFrom27.5Hz_sma3nz_amean": 22.0,
                "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2": 5.0,
                "loudness_sma3_percentile50.0": 0.2,
                "VoicedSegmentsPerSec": 2.0,
                "jitterLocal_sma3nz_amean": 0.02,
                "shimmerLocaldB_sma3nz_amean": 0.9,
                "HNRdBACF_sma3nz_amean": 8.0,
                "hammarbergIndexV_sma3nz_amean": 10.0,
            }
        ])
        globals()["SPEAKER_ACOUSTIC_QUANTILES"] = {
            "Joey": {
                "pitch_height": {"q25": 25.0, "q75": 30.0},
                "pitch_dynamics": {"q25": 8.0, "q75": 12.0},
                "vocal_energy": {"q25": 0.3, "q75": 0.6},
                "speaking_rate": {"q25": 2.5, "q75": 3.5},
            },
            "__GLOBAL__": {
                "pitch_height": {"q25": 20.0, "q75": 28.0},
                "pitch_dynamics": {"q25": 6.0, "q75": 10.0},
                "vocal_energy": {"q25": 0.25, "q75": 0.55},
                "speaking_rate": {"q25": 2.0, "q75": 3.0},
            }
        }

    def test_evaluate_single_prompt_acoustics(self):
        row = self.sample_df.iloc[0].to_dict()
        inv = evaluate_single_prompt_acoustics("audio1.wav", "Joey", row_data=row)
        self.assertIsInstance(inv, dict)
        self.assertIn("pitch_height", inv)
        self.assertEqual(inv["pitch_height"], "elevated")
        self.assertEqual(inv["pitch_dynamics"], "wide")
        self.assertEqual(inv["vocal_energy"], "loud")
        self.assertEqual(inv["speaking_rate"], "fast")
        self.assertIn("voice_quality", inv)

    def test_build_bridge_generation_prompt(self):
        inv = {
            "pitch_height": "elevated",
            "pitch_dynamics": "wide",
            "vocal_energy": "loud",
            "speaking_rate": "fast",
            "voice_quality": "modal/normal",
        }
        prompt = build_bridge_generation_prompt(inv, "joy")
        self.assertIn("Target Emotion: joy", prompt)
        self.assertIn("Pitch Height: elevated", prompt)
        self.assertIn("Task: Write ONLY 2 to 3 concise sentences", prompt)

    def test_generate_sft_train(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        result_df = generate_sft_train(model, tokenizer, self.sample_df, batch_size=2)
        self.assertIn("reasoning", result_df.columns)
        self.assertIn("bridge_prompt", result_df.columns)
        self.assertIn("acoustic_inventory", result_df.columns)
        self.assertEqual(len(result_df), 2)
        self.assertTrue(len(result_df["reasoning"].iloc[0]) > 0)


if __name__ == "__main__":
    unittest.main()

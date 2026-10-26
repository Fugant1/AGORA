import math
import re
import unittest
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


VALID_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


def parse_answer(completion: str) -> str:
    """Extracts and standardizes the emotion inside <answer>...</answer>."""
    if not completion or not isinstance(completion, str):
        return "invalid"

    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", completion, flags=re.DOTALL | re.IGNORECASE)
    if not matches:
        return "invalid"

    raw_answer = matches[-1].strip().lower()
    clean_answer = re.sub(r"<\|.*?\|>", "", raw_answer)
    clean_answer = re.sub(r"[\*\_\"\'\`\.\,\:\;]", "", clean_answer).strip()

    if clean_answer in VALID_EMOTIONS:
        return clean_answer

    found_emotions = [emo for emo in VALID_EMOTIONS if re.search(rf"\b{emo}\b", clean_answer)]
    if len(found_emotions) == 1:
        return found_emotions[0]

    return "invalid"


class MockProcessor:
    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.eos_token_id = 2

        def convert_tokens_to_ids(self, token: str) -> int:
            if token == "<|im_start|>":
                return 151644
            if token == "assistant":
                return 77091
            return 1

        def decode(self, token_ids, skip_special_tokens=True):
            # Return dummy string with answer tag
            return "<think>[Acoustic Inventory] pitch moderate</think><answer>joy</answer>"

    def __init__(self):
        self.tokenizer = self.MockTokenizer()

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
        parts = []
        for m in messages:
            content = m["content"]
            if isinstance(content, list):
                txt = " ".join(c.get("text", "") for c in content if c.get("type") == "text")
            else:
                txt = str(content)
            parts.append(f"<|im_start|>{m['role']}\n{txt}<|im_end|>")
        res = "\n".join(parts)
        if add_generation_prompt:
            res += "\n<|im_start|>assistant\n"
        return res

    def __call__(self, text, audio=None, sampling_rate=16000, return_tensors="pt", padding=True):
        class MockTensorBatch(dict):
            def to(self, device):
                return self

        # Generate fake input_ids containing im_start, assistant, and target tokens
        seq = [151644, 77091, 198, 10, 20, 30, 2] # <|im_start|>, assistant, \n, tokens..., eos
        input_ids = np.array([seq for _ in text])
        attention_mask = np.ones_like(input_ids)
        return MockTensorBatch({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })


class SERMultimodalSFTDataset:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        return {
            "audio_path": str(row.get("audio_path", "")),
            "emotion": str(row.get("emotion", "neutral")).strip().lower(),
            "sft_gt": str(row.get("sft_gt", "")),
        }


class SERMultimodalDataCollator:
    def __init__(self, processor, persona: str, instruction: str):
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.persona = persona
        self.instruction = instruction
        self.im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.assistant_id = self.tokenizer.convert_tokens_to_ids("assistant")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        formatted_full_texts = []
        dummy_audios = []

        for item in batch:
            messages = [
                {"role": "system", "content": self.persona},
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": ""},
                        {"type": "text", "text": self.instruction},
                    ],
                },
                {"role": "assistant", "content": item["sft_gt"]},
            ]
            full_str = self.processor.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=False
            )
            formatted_full_texts.append(full_str)
            dummy_audios.append(np.zeros(1600, dtype=np.float32))

        batch_inputs = self.processor(
            text=formatted_full_texts,
            audio=dummy_audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )

        labels = batch_inputs["input_ids"].copy()
        for i in range(len(labels)):
            seq = labels[i].tolist()
            found_idx = -1
            for j in range(len(seq) - 1):
                if seq[j] == self.im_start_id and seq[j + 1] == self.assistant_id:
                    found_idx = j + 2
                    if found_idx < len(seq) and seq[found_idx] in [198, 271]:
                        found_idx += 1
                    break

            if found_idx != -1:
                labels[i, :found_idx] = -100
            else:
                labels[i, :] = -100

        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        batch_inputs["labels"] = labels
        return batch_inputs


def compute_sft_metrics(eval_pred, tokenizer) -> Dict[str, float]:
    pred_ids, label_ids = eval_pred
    mask = (label_ids != -100)
    total_tokens = mask.sum()
    token_acc = float(((pred_ids == label_ids) & mask).sum() / max(total_tokens, 1))

    decoded_preds = []
    decoded_labels = []
    for p_seq, l_seq in zip(pred_ids, label_ids):
        p_target = p_seq[l_seq != -100]
        l_target = l_seq[l_seq != -100]
        decoded_preds.append(tokenizer.decode(p_target, skip_special_tokens=True))
        decoded_labels.append(tokenizer.decode(l_target, skip_special_tokens=True))

    y_true = [parse_answer(t) for t in decoded_labels]
    y_pred = [parse_answer(t) for t in decoded_preds]

    from sklearn.metrics import accuracy_score, f1_score
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    valid_format = sum(1 for p in decoded_preds if "<think>" in p and "</think>" in p and "<answer>" in p and "</answer>" in p)
    format_comp = float(valid_format / max(len(decoded_preds), 1))

    return {
        "token_accuracy": round(token_acc, 4),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "format_compliance": round(format_comp, 4),
    }


class TestSftTrainingComponents(unittest.TestCase):
    def setUp(self):
        self.processor = MockProcessor()
        self.sample_df = pd.DataFrame([
            {
                "audio_path": "audio1.wav",
                "emotion": "joy",
                "sft_gt": "<think>[Acoustic Inventory] Pitch: high</think><answer>joy</answer>",
            },
            {
                "audio_path": "audio2.wav",
                "emotion": "sadness",
                "sft_gt": "<think>[Acoustic Inventory] Pitch: low</think><answer>sadness</answer>",
            },
        ])

    def test_parse_answer(self):
        text = "<think>Some reasoning</think><answer>joy</answer>"
        self.assertEqual(parse_answer(text), "joy")
        text_dirty = "<answer>  ANGER. </answer>"
        self.assertEqual(parse_answer(text_dirty), "anger")
        text_invalid = "No answer tag here"
        self.assertEqual(parse_answer(text_invalid), "invalid")

    def test_dataset_and_collator(self):
        dataset = SERMultimodalSFTDataset(self.sample_df)
        self.assertEqual(len(dataset), 2)
        item = dataset[0]
        self.assertEqual(item["emotion"], "joy")

        collator = SERMultimodalDataCollator(
            processor=self.processor,
            persona="You are an expert",
            instruction="Listen to audio",
        )
        batch = collator([dataset[0], dataset[1]])
        self.assertIn("input_ids", batch)
        self.assertIn("labels", batch)
        # Check that prompt tokens are masked with -100
        labels = batch["labels"]
        self.assertEqual(labels[0, 0], -100)
        self.assertEqual(labels[0, 1], -100)
        self.assertEqual(labels[0, 2], -100)
        # Target tokens should not be -100
        self.assertNotEqual(labels[0, 3], -100)

    def test_compute_metrics(self):
        pred_ids = np.array([[151644, 77091, 198, 10, 20, 30, 2]])
        label_ids = np.array([[-100, -100, -100, 10, 20, 30, 2]])
        metrics = compute_sft_metrics((pred_ids, label_ids), self.processor.tokenizer)
        self.assertIn("accuracy", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertIn("format_compliance", metrics)
        self.assertEqual(metrics["token_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()

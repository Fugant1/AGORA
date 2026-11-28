"""Strict completion parser (exact parity with ser_sft.ipynb & ser_grpo_v3.ipynb)."""

import functools
import re
from typing import Any, Dict, Tuple

from src.config import CUE_LABELS, CUES, EMOTIONS, VALID_EMOTIONS


def parse_completion_strict(text: str) -> Tuple[str, Dict[str, str]]:
    """Exact ser_sft.ipynb strict parser returning (emotion, {cue: label_or_invalid})."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    answers = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    emotion = answers[0].strip().lower() if len(answers) == 1 else "invalid"
    emotion = emotion if emotion in EMOTIONS else "invalid"
    sections = re.findall(r"\[Acoustic Inventory\](.*?)\[Prosodic Reasoning\]", text, flags=re.DOTALL)
    inventory: Dict[str, str] = {}
    for cue in CUES:
        matches = (
            re.findall(rf"^\s*-\s*{cue}:\s*([^\n]+)", sections[0], flags=re.MULTILINE)
            if len(sections) == 1
            else []
        )
        label = matches[0].strip().lower() if len(matches) == 1 else "invalid"
        inventory[cue] = label if label in CUE_LABELS[cue] else "invalid"
    return emotion, inventory


def extract_reasoning(completion: str) -> str:
    match = re.search(
        r"\[Prosodic Reasoning\](.*?)(?:</think>|<answer>|<\||$)",
        completion,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return ""
    reasoning = match.group(1).strip()
    for placeholder in (
        "Your concise reasoning.",
        "Your concise explanation.",
        "(Deduce the emotion from the cues above)",
    ):
        if reasoning == placeholder:
            return ""
    return reasoning


@functools.lru_cache(maxsize=8192)
def parse_completion(completion: str) -> Dict[str, Any]:
    """Single-pass cached strict parse for GRPO rewards, rollout logging, and evaluation."""
    if not isinstance(completion, str):
        completion = str(completion) if completion is not None else ""

    t_open = completion.find("<think>")
    t_close = completion.find("</think>")
    a_open = completion.find("<answer>")
    a_close = completion.find("</answer>")

    has_think = t_open != -1 and t_close != -1
    has_answer = a_open != -1 and a_close != -1
    single_tags = (
        completion.count("<think>") == 1
        and completion.count("</think>") == 1
        and len(re.findall(r"<answer>", completion, flags=re.IGNORECASE)) == 1
        and len(re.findall(r"</answer>", completion, flags=re.IGNORECASE)) == 1
    )
    valid_tag_order = has_think and has_answer and single_tags and t_open < t_close < a_open < a_close

    emotion, strict_inv = parse_completion_strict(completion)
    inventory = {cue: (None if strict_inv[cue] == "invalid" else strict_inv[cue]) for cue in CUES}
    valid_slots_count = sum(1 for cue in CUES if inventory[cue] is not None)
    reasoning = extract_reasoning(completion)
    answers = re.findall(r"<answer>\s*(.*?)\s*</answer>", completion, flags=re.DOTALL | re.IGNORECASE)
    answer_raw = answers[0].strip().lower() if len(answers) == 1 else ""
    is_valid_emotion = emotion in VALID_EMOTIONS

    return {
        "raw": completion,
        "has_think": has_think,
        "has_answer": has_answer,
        "valid_tag_order": valid_tag_order,
        "inventory": inventory,
        "strict_inventory": strict_inv,
        "valid_slots_count": valid_slots_count,
        "reasoning": reasoning,
        "answer_raw": answer_raw,
        "answer_norm": emotion,
        "is_valid_emotion": is_valid_emotion,
    }

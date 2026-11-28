"""R3: Structured Format Reward (4-Slot Schema; exact match to ser_grpo_v3.ipynb)."""

from typing import Any, Dict, List

from src.config import CUES
from src.rewards.parser import parse_completion


def compute_format_reward_from_parsed(p: Dict[str, Any]) -> float:
    """Structured format reward in [0.0, 1.0]:
      1. Strict XML tag order (<think>...</think><answer>...</answer>): +0.20
      2. Valid acoustic inventory slots (out of 4): +0.35 * (valid_slots / 4)
      3. Non-empty prosodic reasoning (5-150 words: +0.25; >150 words: +0.10)
      4. Valid emotion label: +0.20
    """
    reward = 0.0
    if p["valid_tag_order"]:
        reward += 0.20
    reward += 0.35 * (p["valid_slots_count"] / float(len(CUES)))
    clean_reasoning = p["reasoning"].strip()
    if len(clean_reasoning) >= 20:
        words = len(clean_reasoning.split())
        if 5 <= words <= 150:
            reward += 0.25
        elif words > 150:
            reward += 0.10
    if p["is_valid_emotion"]:
        reward += 0.20
    return round(reward, 3)


def reward_format(prompts: List[str], completions: List[str], **kwargs: Any) -> List[float]:
    parsed_list = kwargs.get("parsed_completions") or [parse_completion(c) for c in completions]
    return [compute_format_reward_from_parsed(p) for p in parsed_list]

"""Exact V4 student and teacher prompt templates (matching ser_sft.ipynb & ser_grpo_v3.ipynb)."""

import json
from typing import Any, Dict, List

from src.config import CUE_COLUMNS, CUE_LABELS, CUES, EMOTIONS

PERSONA = "You analyze speech acoustics and emotion. Follow the requested output format."


def gender_context(row: Dict[str, Any]) -> str:
    groups = {
        "F": "F (female)",
        "M": "M (male)",
        "__GLOBAL__": "__GLOBAL__ (pooled reference; gender unresolved)",
    }
    return f"Acoustic reference group (effective speaker gender): {groups[row['effective_gender']]}."


def inventory_text(row: Dict[str, Any]) -> str:
    return "\n".join(f"- {cue}: {row[CUE_COLUMNS[cue]]}" for cue in CUES)


def student_instruction(row: Dict[str, Any]) -> str:
    options = "\n".join(f"- {cue}: {' | '.join(CUE_LABELS[cue])}" for cue in CUES)
    return (
        gender_context(row)
        + "\nListen to the audio and identify its emotion. "
        "Classify each acoustic cue relative to the supplied reference group. "
        "Then reason concisely from the audible cues to determine the speakers emotion."
        f"Allowed emotions: {', '.join(EMOTIONS)}.\n"
        "Output exactly:\n<think>\n[Acoustic Inventory]\n"
        + options
        + "\n[Prosodic Reasoning]\nYour concise reasoning.\n</think>\n<answer>emotion</answer>"
    )


def build_bridge_generation_prompt(row: Dict[str, Any], concise: bool = True) -> str:
    prompt = (
        gender_context(row)
        + "\n"
        "Listen to the audio clip and write 3 concise numbered reasoning sentences (one per step):\n"
        f"Spoken words: {json.dumps(row['text'], ensure_ascii=False)}\n"
        "[Acoustic Inventory]\n"
        + inventory_text(row)
        + f"\nEmotion: {row['emotion']}\n"
        "1. Describe how the voice sounds using the cues above.\n"
        "2. Connect that delivery to what the speaker says, noting any contrast between tone and words.\n"
        f"3. Conclude why this points to {row['emotion']} rather than a close competing emotion.\n"
        "Write naturally as a listener hearing the clip. Do not invent outside context, XML tags, or headings."
    )
    if concise:
        prompt += " Keep all 3 sentences under 90 words total and stop after sentence 3."
    return prompt


def format_sft_target(row: Dict[str, Any], reasoning: str) -> str:
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("Empty teacher explanation")
    if any(tag in reasoning for tag in ("<", ">", "[Acoustic Inventory]", "[Prosodic Reasoning]")):
        raise ValueError(
            "Teacher explanation contains unexpected structural tags; inspect before retrying"
        )
    return (
        "<think>\n[Acoustic Inventory]\n"
        + inventory_text(row)
        + "\n[Prosodic Reasoning]\n"
        + reasoning.strip()
        + f"\n</think>\n<answer>{row['emotion']}</answer>"
    )


def messages_for(row: Dict[str, Any], teacher: bool = False) -> List[Dict[str, Any]]:
    instruction = build_bridge_generation_prompt(row) if teacher else student_instruction(row)
    return [
        {"role": "system", "content": PERSONA},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": row["audio_path"]},
                {"type": "text", "text": instruction},
            ],
        },
    ]

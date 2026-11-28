"""Hybrid 4-cue split credit assignment, chain-rule span scaling, and loss reductions (exact match to ser_grpo_v3.ipynb)."""

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.config import CUES, DEFAULT_REWARD_WEIGHTS

R2_CUE_NAMES = CUES
R2_CUE_LABELS = CUES
GRPO_ADVANTAGE_EPS = 1e-8
MAX_VALID_CUE_SPAN_TOKENS = 8

SPAN2_MARKER_SEQUENCES = (
    (58, 44915, 52760, 26759),  # "[Prosodic Reasoning"
    (44915, 52760, 26759),      # "Prosodic Reasoning"
    (522, 26865),               # "</think"
    (27, 9217, 29),             # "<answer>"
)


def group_standardize(values: Any, group_size: int, eps: float = GRPO_ADVANTAGE_EPS) -> np.ndarray:
    """Original GRPO group centering/std (sample std, ddof=1); tied groups yield zero."""
    arr = np.asarray(values, dtype=np.float64)
    if group_size < 2 or arr.size == 0 or arr.size % group_size:
        raise ValueError("GRPO requires complete groups with G >= 2.")
    if not np.isfinite(arr).all():
        raise ValueError("Nonfinite rewards cannot be normalized.")
    groups = arr.reshape(-1, group_size)
    centered = groups - groups.mean(axis=1, keepdims=True)
    std = groups.std(axis=1, ddof=1, keepdims=True)
    return (centered / (std + eps)).reshape(-1)


def cue_group_advantages(
    reward_trace: List[Dict[str, Any]],
    group_size: int,
    r2_weight: float = 1.0,
    loss_type: str = "hybrid",
) -> Dict[str, np.ndarray]:
    """Cue-local raw payoff: GRPO standardizes; hybrid/dr_grpo mean-center; apply w2 / 4 afterwards."""
    if group_size < 2:
        raise ValueError("Cue GRPO requires G >= 2.")
    result: Dict[str, np.ndarray] = {}
    for cue in R2_CUE_NAMES:
        details = [r["r2_cues"][cue] for r in reward_trace]
        if not details or len(details) % group_size:
            raise ValueError("Incomplete cue reward groups.")
        for start in range(0, len(details), group_size):
            block = details[start : start + group_size]
            if len({(d["valid_gt"], d["truth"]) for d in block}) != 1:
                raise ValueError(f"Mismatched acoustic targets within group for {cue}.")
        raw = np.array([d["raw"] if d["valid_gt"] else 0.0 for d in details], dtype=float)
        if loss_type == "grpo":
            advantage = group_standardize(raw, group_size)
        elif loss_type in ("dr_grpo", "hybrid"):
            if not np.isfinite(raw).all():
                raise ValueError("Nonfinite cue rewards cannot be centered.")
            grouped = raw.reshape(-1, group_size)
            advantage = (grouped - grouped.mean(axis=1, keepdims=True)).reshape(-1)
        else:
            raise ValueError("Supported losses are grpo, dr_grpo and hybrid.")
        result[cue] = float(r2_weight) * advantage / len(R2_CUE_NAMES)
    return result


def acoustic_value_token_indices(
    token_ids: List[int],
    tokenizer: Any,
) -> Tuple[Dict[str, List[int]], Dict[str, str]]:
    """Map original rollout tokens to cue VALUE spans (supporting multi-token timbre values); fail closed."""
    indices: Dict[str, List[int]] = {c: [] for c in R2_CUE_NAMES}
    status: Dict[str, str] = {c: "missing_field" for c in R2_CUE_NAMES}

    def decode(ids: List[int]) -> str:
        return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    text = decode(token_ids)
    headers = list(re.finditer(r"\[Acoustic Inventory\]", text, re.IGNORECASE))
    if len(headers) != 1:
        return indices, {c: "missing_or_duplicate_inventory" for c in R2_CUE_NAMES}
    start = headers[0].end()
    ending = re.search(r"\[Prosodic Reasoning\]|</think>|<answer>|<\|", text[start:], re.IGNORECASE)
    end = start + ending.start() if ending else len(text)
    spans = {}
    for cue, label in zip(R2_CUE_NAMES, R2_CUE_LABELS):
        matches = list(
            re.finditer(
                r"(?im)^[ \t]*(?:[-*][ \t]*)?" + re.escape(label) + r"[ \t]*:[ \t]*([^\r\n<]*)",
                text[start:end],
            )
        )
        if len(matches) != 1:
            status[cue] = "duplicate_field" if matches else "missing_field"
            continue
        match = matches[0]
        value = match.group(1)
        left = len(value) - len(value.lstrip())
        right = len(value.rstrip())
        if right <= left:
            status[cue] = "empty_value"
            continue
        spans[cue] = (start + match.start(1) + left, start + match.start(1) + right)

    offsets = None
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if list(encoded["input_ids"]) == list(token_ids):
            offsets = [tuple(x) for x in encoded["offset_mapping"]]
    except (TypeError, ValueError, KeyError, NotImplementedError, AttributeError):
        pass
    if offsets is None:
        offsets = []
        previous = 0
        for j in range(1, len(token_ids) + 1):
            prefix = decode(token_ids[:j])
            if not text.startswith(prefix) or len(prefix) < previous:
                return indices, {c: "unaligned_tokens" for c in R2_CUE_NAMES}
            offsets.append((previous, len(prefix)))
            previous = len(prefix)
    if len(offsets) != len(token_ids):
        return indices, {c: "unaligned_tokens" for c in R2_CUE_NAMES}

    for cue, (left, right) in spans.items():
        selected = [j for j, (a, b) in enumerate(offsets) if b > a and a < right and b > left]
        if not selected or any(
            text[a : min(b, left)].strip() or text[max(a, right) : b].strip()
            for j in selected
            for a, b in [offsets[j]]
        ):
            status[cue] = "unaligned_value"
            continue
        if any(
            not any(a <= pos < b for j in selected for a, b in [offsets[j]])
            for pos in range(left, right)
        ):
            status[cue] = "unaligned_value"
            continue
        indices[cue] = selected
        status[cue] = "ok"
    return indices, status


def cue_span_length_scale(active_len: int, span_len: int) -> float:
    """Chain-rule span-sum scale L (so sum_{t in S} grad log pi_t = grad log P(Y_cue)); cap only >8-tok rambles."""
    if active_len <= 0 or span_len <= 0:
        return 0.0
    guard = 1.0 if span_len <= MAX_VALID_CUE_SPAN_TOKENS else (float(MAX_VALID_CUE_SPAN_TOKENS) / float(span_len))
    return float(active_len) * guard


def validate_credit_assignment(loss_type: str, use_split_advantage: bool, use_split_r2: bool) -> None:
    if loss_type not in ("grpo", "dr_grpo", "hybrid"):
        raise ValueError("Supported losses are grpo, dr_grpo and hybrid.")
    if loss_type == "hybrid" and not (use_split_advantage and use_split_r2):
        raise ValueError("loss_type=hybrid requires use_split_advantage=True and use_split_r2=True.")
    if use_split_r2 and loss_type not in ("grpo", "hybrid"):
        raise ValueError("use_split_r2 requires loss_type=grpo or hybrid.")
    if use_split_r2 and not use_split_advantage:
        raise ValueError("use_split_r2 requires use_split_advantage=True.")


def completion_boundary_mask(completion_ids: Any, eos_ids: List[int], pad_id: Optional[int]) -> Any:
    """Include the first EOS, exclude subsequent padding, even when PAD == EOS."""
    import torch

    is_eos = torch.zeros_like(completion_ids, dtype=torch.bool)
    for token_id in eos_ids:
        is_eos |= completion_ids.eq(token_id)
    mask = (is_eos.long().cumsum(dim=1) - is_eos.long()).eq(0)
    if pad_id is not None and pad_id not in eos_ids:
        mask &= completion_ids.ne(pad_id)
    return mask.long()


def completion_span_masks(completion_ids: Any, completion_mask: Any, tokenizer: Any = None) -> Tuple[Any, Any, Any]:
    """Partition completion_mask into disjoint (acoustic_span_mask, emotion_span_mask, has_boundary)."""
    import torch

    candidate_markers = []
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        for text_marker in ("[Prosodic Reasoning]", "Prosodic Reasoning", "</think>", "<answer>"):
            try:
                encoded = tuple(int(x) for x in tokenizer.encode(text_marker, add_special_tokens=False))
                if encoded and encoded not in candidate_markers:
                    candidate_markers.append(encoded)
            except Exception:
                pass
    for seq in SPAN2_MARKER_SEQUENCES:
        if seq not in candidate_markers:
            candidate_markers.append(seq)

    acoustic_mask = torch.zeros_like(completion_mask)
    emotion_mask = torch.zeros_like(completion_mask)
    batch_size, _ = completion_ids.shape
    has_boundary = torch.zeros(batch_size, dtype=torch.bool, device=completion_ids.device)

    for b in range(batch_size):
        row_mask = completion_mask[b]
        active_len = int(row_mask.sum().item())
        if active_len <= 0:
            continue
        row_ids = [int(x) for x in completion_ids[b, :active_len].detach().cpu().tolist()]
        split_idx = None
        for marker in candidate_markers:
            m_len = len(marker)
            if m_len == 0 or active_len < m_len:
                continue
            for pos in range(1, active_len - m_len + 1):
                if tuple(row_ids[pos : pos + m_len]) == marker:
                    split_idx = (pos - 1) if (marker[0] == 44915 and pos >= 1 and row_ids[pos - 1] == 58) else pos
                    break
            if split_idx is not None and 0 < split_idx < active_len:
                break
        if split_idx is not None and 0 < split_idx < active_len:
            has_boundary[b] = True
            acoustic_mask[b, :split_idx] = row_mask[:split_idx]
            emotion_mask[b, split_idx:] = row_mask[split_idx:]
        else:
            fallback_idx = max(1, active_len // 2) if active_len > 1 else active_len
            acoustic_mask[b, :fallback_idx] = row_mask[:fallback_idx]
            emotion_mask[b, fallback_idx:] = row_mask[fallback_idx:]
    return acoustic_mask, emotion_mask, has_boundary


def build_cue_token_advantages(
    completion_ids: Any,
    completion_mask: Any,
    cue_advantages: Dict[str, np.ndarray],
    adv_format: Any,
    adv_emotion: Any,
    tokenizer: Any,
) -> Tuple[Any, Any, Any, Any, Any, Dict[str, List[int]], List[Dict[str, str]]]:
    """Assign cue-local R2 and matched chain-rule L KL scale to value tokens, R3 to structure, R1+R3 to reasoning/answer."""
    import torch

    acoustic, emotion, boundary = completion_span_masks(completion_ids, completion_mask, tokenizer)
    values = adv_format.unsqueeze(-1) * completion_mask.float()
    values = values * (1 - emotion.float() * boundary.unsqueeze(-1).float())
    values += adv_emotion.unsqueeze(-1) * emotion.float() * boundary.unsqueeze(-1).float()
    cue_kl_scale = torch.zeros_like(values)
    counts: Dict[str, List[int]] = {c: [] for c in R2_CUE_NAMES}
    statuses: List[Dict[str, str]] = []
    for i in range(completion_ids.shape[0]):
        active = int(completion_mask[i].sum().item())
        ids = completion_ids[i, :active].detach().cpu().tolist()
        positions, state = acoustic_value_token_indices(ids, tokenizer)
        for cue in R2_CUE_NAMES:
            selected = positions[cue]
            counts[cue].append(len(selected))
            if selected:
                scale = cue_span_length_scale(active, len(selected))
                coefficient = float(cue_advantages[cue][i]) * scale
                for j in selected:
                    values[i, j] = coefficient
                    cue_kl_scale[i, j] = scale
        statuses.append(state)
    return values, cue_kl_scale, acoustic, emotion, boundary, counts, statuses


def compute_span_split_advantages_np(
    reward_trace: List[Dict[str, Any]],
    num_generations: int,
    reward_weights: Optional[Dict[str, float]] = None,
    loss_type: str = "hybrid",
) -> Dict[str, np.ndarray]:
    """Pure-NumPy span advantage math; hybrid keeps GRPO emotion/format std (with post-std R1 class weight) and centers R2.

    Note on imbalanced splits:
      Within a group of G rollouts for one clip, all G completions share the same ground-truth emotion y.
      Standardizing raw weighted `r1 = w_y * hit` divides out `w_y`. To preserve tempered inverse-frequency
      class weighting on imbalanced splits (while remaining bit-for-bit identical when `w_y == 1.0` on uniform
      splits), we standardize unit-scale `r1_unit = r1 / w_y` with `r3`, then scale by `w_y` outside std.
    """
    if len(reward_trace) == 0 or len(reward_trace) % num_generations != 0:
        raise ValueError("reward_trace length must be a positive multiple of num_generations.")
    weights = reward_weights if reward_weights is not None else DEFAULT_REWARD_WEIGHTS
    w_r1, w_r2, w_r3 = float(weights["r1"]), float(weights["r2"]), float(weights["r3"])

    r1 = np.asarray([float(r["r1"]) for r in reward_trace], dtype=np.float64).reshape(-1, num_generations)
    r1_w = np.asarray(
        [float(r.get("r1_class_weight", 1.0)) for r in reward_trace], dtype=np.float64
    ).reshape(-1, num_generations)
    r1_unit = np.where(r1_w > 0.0, r1 / np.maximum(r1_w, 1e-12), r1)
    r2 = np.asarray([float(r["r2"]) for r in reward_trace], dtype=np.float64).reshape(-1, num_generations)
    r3 = np.asarray([float(r["r3"]) for r in reward_trace], dtype=np.float64).reshape(-1, num_generations)

    adv_r1 = np.nan_to_num(w_r1 * (r1 - r1.mean(axis=-1, keepdims=True)), nan=0.0).reshape(-1)
    adv_r2 = np.nan_to_num(w_r2 * (r2 - r2.mean(axis=-1, keepdims=True)), nan=0.0).reshape(-1)
    adv_r3 = np.nan_to_num(w_r3 * (r3 - r3.mean(axis=-1, keepdims=True)), nan=0.0).reshape(-1)

    if loss_type in ("grpo", "hybrid"):
        r1_w_flat = r1_w.reshape(-1)
        if loss_type == "hybrid":
            emotion = r1_w_flat * group_standardize(w_r1 * r1_unit + w_r3 * r3, num_generations)
            format_adv = w_r3 * group_standardize(r3, num_generations)
            return {
                "adv_r1": r1_w_flat * w_r1 * group_standardize(r1_unit, num_generations),
                "adv_r2": adv_r2,
                "adv_r3": format_adv,
                "adv_acoustic": adv_r2 + format_adv,
                "adv_emotion": emotion,
                "adv_total": emotion + adv_r2,
            }
        return {
            "adv_r1": r1_w_flat * w_r1 * group_standardize(r1_unit, num_generations),
            "adv_r2": w_r2 * group_standardize(r2, num_generations),
            "adv_r3": w_r3 * group_standardize(r3, num_generations),
            "adv_acoustic": group_standardize(w_r2 * r2 + w_r3 * r3, num_generations),
            "adv_emotion": r1_w_flat * group_standardize(w_r1 * r1_unit + w_r3 * r3, num_generations),
            "adv_total": group_standardize(w_r1 * r1 + w_r2 * r2 + w_r3 * r3, num_generations),
        }
    if loss_type != "dr_grpo":
        raise ValueError("Supported losses are grpo, dr_grpo and hybrid.")
    return {
        "adv_r1": adv_r1,
        "adv_r2": adv_r2,
        "adv_r3": adv_r3,
        "adv_acoustic": adv_r2 + adv_r3,
        "adv_emotion": adv_r1 + adv_r3,
        "adv_total": adv_r1 + adv_r2 + adv_r3,
    }


def compute_span_split_advantages(
    reward_trace: List[Dict[str, Any]],
    num_generations: int,
    reward_weights: Optional[Dict[str, float]] = None,
    device: Any = None,
    loss_type: str = "hybrid",
) -> Dict[str, Any]:
    """Torch wrapper around `compute_span_split_advantages_np`."""
    import torch

    np_out = compute_span_split_advantages_np(
        reward_trace=reward_trace,
        num_generations=num_generations,
        reward_weights=reward_weights,
        loss_type=loss_type,
    )
    return {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in np_out.items()}


def build_span_split_token_advantages(
    completion_ids: Any,
    completion_mask: Any,
    adv_acoustic: Any,
    adv_emotion: Any,
    adv_total: Any = None,
    tokenizer: Any = None,
    use_split_advantage: bool = True,
) -> Tuple[Any, Any, Any, Any]:
    acoustic_mask, emotion_mask, has_boundary = completion_span_masks(
        completion_ids, completion_mask, tokenizer=tokenizer
    )
    if not bool(use_split_advantage):
        if adv_total is None:
            raise ValueError("adv_total is required when use_split_advantage=False.")
        return adv_total.unsqueeze(-1) * completion_mask.float(), acoustic_mask, emotion_mask, has_boundary

    token_advantages = (
        adv_acoustic.unsqueeze(-1) * acoustic_mask.float()
        + adv_emotion.unsqueeze(-1) * emotion_mask.float()
    )
    if adv_total is not None and not bool(has_boundary.all().item()):
        fallback_rows = (~has_boundary).float().unsqueeze(-1)
        token_advantages = (
            token_advantages * has_boundary.float().unsqueeze(-1)
            + adv_total.unsqueeze(-1) * completion_mask.float() * fallback_rows
        )
    return token_advantages, acoustic_mask, emotion_mask, has_boundary


def fixed_completion_loss(token_loss: Any, mask: Any, max_length: int, accumulation_steps: int = 1) -> Any:
    if max_length <= 0 or accumulation_steps <= 0 or token_loss.shape[0] == 0:
        raise ValueError("Positive length, accumulation and batch size are required.")
    return (token_loss * mask).sum() / (token_loss.shape[0] * max_length * accumulation_steps)


def dynamic_completion_loss(token_loss: Any, mask: Any, accumulation_steps: int = 1) -> Any:
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive.")
    per_completion = (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return per_completion.mean() / accumulation_steps


def reduce_completion_loss(
    token_loss: Any,
    mask: Any,
    loss_type: str,
    max_length: int,
    accumulation_steps: int = 1,
) -> Any:
    if loss_type == "dr_grpo":
        return fixed_completion_loss(token_loss, mask, max_length, accumulation_steps)
    if loss_type in ("grpo", "hybrid"):
        return dynamic_completion_loss(token_loss, mask, accumulation_steps)
    raise ValueError("Supported losses are grpo, dr_grpo and hybrid.")


def completion_logps(model: Any, inputs: Dict[str, Any]) -> Any:
    """Shared teacher-forcing path for training and replay diagnostics on Thinker."""
    import torch
    import torch.nn.functional as F

    completion_ids = inputs["completion_ids"]
    prompt_len = inputs["generated_ids"].shape[1] - completion_ids.shape[1]
    ignored = {
        "generated_ids",
        "completion_ids",
        "advantages",
        "token_advantages",
        "cue_kl_scale",
        "acoustic_span_mask",
        "emotion_span_mask",
        "prompt_input_ids",
        "prompt_attention_mask",
        "completion_mask",
    }
    forward_inputs = {k: v for k, v in inputs.items() if k not in ignored}
    forward_inputs.update(
        input_ids=inputs["generated_ids"],
        attention_mask=torch.cat([inputs["prompt_attention_mask"], inputs["completion_mask"]], dim=1),
        use_cache=False,
    )
    outputs = model(**forward_inputs)
    logits = outputs.logits[:, prompt_len - 1 : -1, :].contiguous()
    del outputs
    return -F.cross_entropy(
        logits.view(-1, logits.size(-1)), completion_ids.reshape(-1), reduction="none"
    ).view(completion_ids.shape)


def check_mask_and_loss_math() -> None:
    """Fast self-check on completion_boundary_mask and dynamic/fixed completion loss."""
    import torch

    ids = torch.tensor([[10, 20, 99, 99], [10, 99, 20, 99]])
    mask = completion_boundary_mask(ids, eos_ids=[99], pad_id=99)
    assert mask.tolist() == [[1, 1, 1, 0], [1, 1, 0, 0]], mask.tolist()
    losses = torch.ones_like(ids, dtype=torch.float32)
    assert abs(float(dynamic_completion_loss(losses, mask.float(), accumulation_steps=4)) - 0.25) < 1e-6
    assert abs(float(fixed_completion_loss(losses, mask.float(), max_length=4, accumulation_steps=2)) - 5.0 / 16.0) < 1e-6

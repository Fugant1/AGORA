"""Qwen2.5-Omni-7B Thinker loading, QLoRA adapter setup, and frozen SFT reference context manager."""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, List, Optional, Tuple

from src.config import ModelConfig


def generation_eos_ids(tokenizer: Any) -> List[int]:
    """Return sorted Qwen stop token IDs (<|im_end|> and <|endoftext|>)."""
    vocab = tokenizer.get_vocab()
    if "<|im_end|>" not in vocab:
        raise ValueError("Qwen tokenizer is missing the <|im_end|> stop token")
    return sorted({vocab[token] for token in ("<|im_end|>", "<|endoftext|>") if token in vocab})


@contextmanager
def reference_policy(model: Any, adapter_name: Optional[str]):
    """Switch to frozen SFT reference adapter and restore policy/gradient flags even after failure."""
    if adapter_name is None:
        with model.disable_adapter():
            yield
        return
    previous_adapter = model.active_adapter
    gradient_flags = [(p, p.requires_grad) for p in model.parameters()]
    try:
        model.set_adapter(adapter_name)
        model.set_requires_grad(adapter_name, False)
        yield
    finally:
        model.set_adapter(previous_adapter)
        for parameter, requires_grad in gradient_flags:
            parameter.requires_grad_(requires_grad)
        model.set_requires_grad(adapter_name, False)


def load_thinker_model(
    model_cfg: ModelConfig,
    *,
    seed: int = 42,
    padding_side: str = "left",
    sft_adapter_path: Optional[Path] = None,
    use_sft_model: bool = False,
    attach_lora: bool = True,
    reference_adapter_name: Optional[str] = "sft_reference",
) -> Tuple[Any, Any, Any, Optional[str]]:
    """Load Qwen2_5OmniThinkerForConditionalGeneration + AutoProcessor with exact V4 QLoRA & reference adapter setup.

    Returns (model, processor, compute_dtype, active_reference_adapter_name).
    """
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5OmniThinkerForConditionalGeneration,
        set_seed,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("4-bit QLoRA training requires a CUDA GPU.")

    set_seed(seed)
    use_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    quantization_config = None
    if model_cfg.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_cfg.bnb_4bit_use_double_quant,
        )

    processor = AutoProcessor.from_pretrained(model_cfg.model_id)
    load_kwargs = dict(
        dtype=compute_dtype,
        device_map={"": torch.cuda.current_device()},
        quantization_config=quantization_config,
    )
    if model_cfg.attn_implementation:
        load_kwargs["attn_implementation"] = model_cfg.attn_implementation

    try:
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_cfg.model_id, **load_kwargs)
    except (TypeError, ValueError):
        load_kwargs.pop("attn_implementation", None)
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_cfg.model_id, **load_kwargs)

    tokenizer = processor.tokenizer
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    endoftext_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    eos_token_ids = generation_eos_ids(tokenizer)

    tokenizer.eos_token = "<|im_end|>"
    tokenizer.eos_token_id = im_end_id
    model.config.eos_token_id = eos_token_ids
    if hasattr(model.config, "get_text_config"):
        model.config.get_text_config().eos_token_id = eos_token_ids
    model.generation_config.eos_token_id = eos_token_ids

    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == endoftext_id:
        tokenizer.pad_token_id = im_end_id
        model.generation_config.pad_token_id = im_end_id
    tokenizer.padding_side = padding_side

    active_ref: Optional[str] = None
    if attach_lora:
        if model_cfg.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            r=model_cfg.lora_r,
            lora_alpha=model_cfg.lora_alpha,
            lora_dropout=model_cfg.lora_dropout,
            bias=model_cfg.lora_bias,
            task_type=None,
            target_modules=model_cfg.target_modules,
        )
        if use_sft_model:
            if sft_adapter_path is None or not Path(sft_adapter_path).is_dir():
                raise FileNotFoundError(f"Missing SFT adapter directory: {sft_adapter_path}")
            model = PeftModel.from_pretrained(model, str(sft_adapter_path), is_trainable=True)
            if reference_adapter_name:
                model.load_adapter(
                    str(sft_adapter_path), adapter_name=reference_adapter_name, is_trainable=False
                )
                model.set_adapter("default")
                model.set_requires_grad(reference_adapter_name, False)
                active_ref = reference_adapter_name
        else:
            model = get_peft_model(model, peft_config)

        model.config.use_cache = False
        if hasattr(model.config, "get_text_config"):
            model.config.get_text_config().use_cache = False
        model.print_trainable_parameters()

    return model, processor, compute_dtype, active_ref

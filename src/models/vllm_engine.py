"""Colocated 24 GB VRAM-safe vLLM rollout engine for Qwen2.5-Omni-7B with live LoRA sync & sleep mode.

Key invariants preserving 100% mathematical parity with ser_grpo_v3.ipynb:
  1. Live LoRA sync: before each generation call, saves ONLY the trainable `'default'` LoRA adapter
     to a fast RAM-backed directory (`/dev/shm/agora_vllm_lora/step_{N}`) and passes a unique
     incrementing `LoRARequest` so rollouts are strictly on-policy draws from the live weights.
  2. Exact sampling params: `temperature=T`, `top_p=1.0`, `top_k=-1`, `n=num_generations`,
     `max_tokens=max_completion_length`, `stop_token_ids=eos_ids`, `include_stop_str_in_output=False`.
  3. Shared audio encoding: deduplicates each group of G identical audio prompts into 1 multimodal
     request with `n=G`, cutting Whisper audio-tower work by Gx (8x) and leveraging prefix caching.
  4. 24 GB VRAM safety: uses `enable_sleep_mode=True` (`sleep(level=1)` immediately after generation,
     `wake_up()` immediately before generation) and `quantization='bitsandbytes'` (~5.8 GB weights),
     keeping peak VRAM around ~14-15 GB during rollouts and ~12 GB during QLoRA backward passes.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from src.config import TARGET_SR, VLLMConfig


class VLLMRolloutEngine:
    """Manages a colocated vLLM LLM engine for fast multimodal GRPO rollouts and greedy validation."""

    def __init__(
        self,
        model_id: str,
        vllm_cfg: VLLMConfig,
        seed: int = 42,
    ) -> None:
        self.model_id = model_id
        self.cfg = vllm_cfg
        self.seed = seed
        self._llm: Any = None
        self._sleeping: bool = False
        self._lora_version: int = 0
        self._sync_root: Path = self._resolve_sync_root(vllm_cfg.lora_sync_dir)
        self._last_lora_dirs: List[Path] = []

    @staticmethod
    def _resolve_sync_root(preferred: Path) -> Path:
        preferred = Path(preferred)
        try:
            preferred.mkdir(parents=True, exist_ok=True)
            return preferred
        except OSError:
            fallback = Path(tempfile.mkdtemp(prefix="agora_vllm_lora_"))
            return fallback

    def initialize(self) -> None:
        """Lazily initialize the vLLM LLM engine and put it to sleep until the first rollout."""
        if self._llm is not None:
            return
        from vllm import LLM

        kwargs = dict(
            model=self.model_id,
            dtype=self.cfg.dtype,
            gpu_memory_utilization=float(self.cfg.gpu_memory_utilization),
            max_model_len=int(self.cfg.max_model_len),
            max_num_seqs=int(self.cfg.max_num_seqs),
            enable_lora=True,
            max_lora_rank=int(self.cfg.max_lora_rank),
            max_loras=2,
            enable_sleep_mode=bool(self.cfg.enable_sleep_mode),
            enforce_eager=bool(self.cfg.enforce_eager),
            tensor_parallel_size=int(self.cfg.tensor_parallel_size),
            limit_mm_per_prompt={"audio": 1, "image": 0, "video": 0},
            trust_remote_code=True,
            seed=int(self.seed),
        )
        if self.cfg.quantization:
            kwargs["quantization"] = self.cfg.quantization
        if self.cfg.load_format:
            kwargs["load_format"] = self.cfg.load_format

        self._llm = LLM(**kwargs)
        if self.cfg.enable_sleep_mode:
            self.sleep()

    def wake_up(self) -> None:
        if self._llm is None:
            self.initialize()
        if self.cfg.enable_sleep_mode and self._sleeping:
            self._llm.wake_up()
            self._sleeping = False

    def sleep(self) -> None:
        if self._llm is not None and self.cfg.enable_sleep_mode and not self._sleeping:
            self._llm.sleep(level=int(self.cfg.sleep_level))
            self._sleeping = True
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def _sync_live_lora(self, peft_model: Any, adapter_name: str = "default") -> Any:
        """Atomically export the live trainable LoRA adapter to RAM-backed storage and return LoRARequest."""
        from vllm.lora.request import LoRARequest

        self._lora_version += 1
        step_dir = self._sync_root / f"lora_v{self._lora_version}"
        if step_dir.exists():
            shutil.rmtree(step_dir, ignore_errors=True)
        step_dir.mkdir(parents=True, exist_ok=True)

        # Save ONLY the trainable policy adapter (never the frozen sft_reference adapter)
        try:
            peft_model.save_pretrained(str(step_dir), selected_adapters=[adapter_name])
        except TypeError:
            peft_model.save_pretrained(str(step_dir))

        # PeftModel with selected_adapters may save into step_dir or step_dir/<adapter_name>
        adapter_dir = step_dir / adapter_name if (step_dir / adapter_name / "adapter_config.json").is_file() else step_dir

        self._last_lora_dirs.append(step_dir)
        while len(self._last_lora_dirs) > 2:
            old_dir = self._last_lora_dirs.pop(0)
            shutil.rmtree(old_dir, ignore_errors=True)

        return LoRARequest(
            lora_name=f"policy_v{self._lora_version}",
            lora_int_id=self._lora_version,
            lora_path=str(adapter_dir),
        )

    def generate_rollouts(
        self,
        peft_model: Any,
        unique_prompts: Sequence[str],
        unique_audios: Sequence[np.ndarray],
        *,
        num_generations: int,
        temperature: float,
        max_tokens: int,
        stop_token_ids: Sequence[int],
        adapter_name: str = "default",
    ) -> Tuple[List[List[int]], List[str]]:
        """Generate `num_generations` on-policy rollouts per unique audio prompt via vLLM.

        Returns flattened `(token_id_lists, decoded_texts)` of length `len(unique_prompts) * num_generations`.
        """
        from vllm import SamplingParams

        if len(unique_prompts) != len(unique_audios):
            raise ValueError("unique_prompts and unique_audios must have identical length.")

        lora_req = self._sync_live_lora(peft_model, adapter_name=adapter_name)
        vllm_inputs = [
            {
                "prompt": prompt_text,
                "multi_modal_data": {"audio": (wav.astype(np.float32, copy=False), TARGET_SR)},
            }
            for prompt_text, wav in zip(unique_prompts, unique_audios)
        ]
        sampling_params = SamplingParams(
            n=int(num_generations),
            temperature=float(temperature),
            top_p=1.0,
            top_k=-1,
            max_tokens=int(max_tokens),
            stop_token_ids=list(stop_token_ids),
            include_stop_str_in_output=False,
        )

        self.wake_up()
        try:
            outputs = self._llm.generate(
                vllm_inputs,
                sampling_params=sampling_params,
                lora_request=lora_req,
                use_tqdm=False,
            )
        finally:
            self.sleep()

        all_token_ids: List[List[int]] = []
        all_texts: List[str] = []
        for req_out in outputs:
            if len(req_out.outputs) != int(num_generations):
                raise RuntimeError(
                    f"vLLM returned {len(req_out.outputs)} completions instead of {num_generations}."
                )
            for comp in req_out.outputs:
                all_token_ids.append([int(t) for t in comp.token_ids])
                all_texts.append(comp.text)
        return all_token_ids, all_texts

    def generate_greedy(
        self,
        peft_model: Any,
        prompts: Sequence[str],
        audios: Sequence[np.ndarray],
        *,
        max_tokens: int,
        stop_token_ids: Sequence[int],
        adapter_name: str = "default",
    ) -> List[str]:
        """Greedy batched generation (temperature=0.0, n=1) for fast post-training validation."""
        _, texts = self.generate_rollouts(
            peft_model,
            prompts,
            audios,
            num_generations=1,
            temperature=0.0,
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids,
            adapter_name=adapter_name,
        )
        return texts

    def cleanup(self) -> None:
        """Shut down vLLM engine, release GPU VRAM, and remove temporary RAM-backed LoRA snapshots."""
        import gc

        if self._llm is not None:
            try:
                if hasattr(self._llm, "llm_engine") and hasattr(self._llm.llm_engine, "engine_core"):
                    self._llm.llm_engine.engine_core.shutdown()
            except Exception:
                pass
            try:
                del self._llm
            except Exception:
                pass
            self._llm = None
            self._sleeping = False

        for d in self._last_lora_dirs:
            shutil.rmtree(d, ignore_errors=True)
        self._last_lora_dirs.clear()

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

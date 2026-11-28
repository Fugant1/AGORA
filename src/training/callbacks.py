"""Training callbacks verifying live LoRA policy updates and frozen SFT reference immutability."""

import json
from pathlib import Path
from typing import Any, List, Optional


def build_adapter_update_monitor(reference_name: Optional[str]) -> Any:
    """Build AdapterUpdateMonitor (lazy transformers.TrainerCallback import for CPU testability)."""
    import torch
    from transformers import TrainerCallback

    class AdapterUpdateMonitor(TrainerCallback):
        """Live probes verifying policy updates and frozen SFT reference immutability."""

        def __init__(self, ref_name: Optional[str]) -> None:
            self.reference_name = ref_name
            self.history: List[dict] = []

        def on_train_begin(self, args, state, control, model=None, **kwargs):
            self.policy = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            if not self.policy:
                raise RuntimeError("No trainable policy parameters.")
            self.reference = [
                (n, p)
                for n, p in model.named_parameters()
                if self.reference_name and f".{self.reference_name}." in n
            ]
            if self.reference_name and not self.reference:
                raise RuntimeError("Frozen SFT reference parameters were not found.")
            self.ref_probe = self._probe(self.reference)
            self.previous = self._probe(self.policy)
            self.initial = self.previous.clone()
            self.grad_norm = float("nan")

        @staticmethod
        def _probe(parameters):
            values = []
            for _, p in parameters:
                idx = torch.linspace(0, p.numel() - 1, min(8, p.numel()), device=p.device).long()
                values.append(p.detach().reshape(-1)[idx].float().cpu())
            return torch.cat(values) if values else torch.empty(0)

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            grads = [p.grad for _, p in self.policy if p.grad is not None]
            if not grads or any(not torch.isfinite(g).all().item() for g in grads):
                raise RuntimeError("Policy gradients are missing or non-finite.")
            self.grad_norm = sum(g.detach().float().norm().item() ** 2 for g in grads) ** 0.5
            if any(p.requires_grad or p.grad is not None for _, p in self.reference):
                raise RuntimeError("Frozen SFT reference received gradients or became trainable.")

        def on_step_end(self, args, state, control, **kwargs):
            current = self._probe(self.policy)
            if not torch.equal(self.ref_probe, self._probe(self.reference)):
                raise RuntimeError("Frozen reference probe changed during training.")
            record = {
                "step": state.global_step,
                "gradient_norm_after_clipping": self.grad_norm,
                "policy_probe_update_l2": (current - self.previous).norm().item(),
                "policy_probe_drift_l2": (current - self.initial).norm().item(),
                "reference_probe_unchanged": True,
            }
            self.history.append(record)
            if state.is_world_process_zero:
                with (Path(args.output_dir) / "adapter_updates.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
            self.previous = current

    return AdapterUpdateMonitor(reference_name)

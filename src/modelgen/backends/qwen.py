# src/modelgen/backends/qwen.py

from __future__ import annotations

from modelgen.backends.hf_causal_lm import HFCausalLMBackend


class QwenBackend(HFCausalLMBackend):
    """Local inference backend for the Qwen model family."""

    def __init__(
        self,
        model_path: str,
        size_label: str | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(
            model_path=model_path,
            family="qwen",
            size_label=size_label,
            device=device,
        )

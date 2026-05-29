# src/modelgen/backends/__init__.py

from modelgen.backends.base import LocalBackend
from modelgen.backends.types import (
    LocalGenerationConfig,
    LocalModelMetadata,
    ModelGenerationResult,
    ModelOptionScoreResult,
)
from modelgen.backends.dummy import DummyBackend
from modelgen.backends.hf_causal_lm import HFCausalLMBackend
from modelgen.backends.qwen import QwenBackend
from modelgen.backends.llama import LlamaBackend

__all__ = [
    "LocalBackend",
    "LocalGenerationConfig",
    "LocalModelMetadata",
    "ModelGenerationResult",
    "ModelOptionScoreResult",
    "DummyBackend",
    "HFCausalLMBackend",
    "QwenBackend",
    "LlamaBackend",
]

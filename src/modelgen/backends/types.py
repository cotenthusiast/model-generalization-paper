# src/modelgen/backends/types.py

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LocalGenerationConfig:
    """Settings for a single generation call."""

    max_new_tokens: int = 512
    temperature: float = 0.0
    do_sample: bool = False
    seed: int = 42


@dataclass
class LocalModelMetadata:
    """Static descriptive information about a loaded backend."""

    model_name: str
    model_path: str
    family: str        # "qwen" | "llama" | "dummy"
    backend_type: str  # "hf_transformers" | "dummy"
    device: str        # "cuda" | "cpu" | "mps" | "auto"
    size_label: str | None = None  # informational only, e.g. "7B"


@dataclass
class ModelGenerationResult:
    """Output from a single generate() call."""

    raw_text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None
    metadata: dict = field(default_factory=dict)


@dataclass
class ModelOptionScoreResult:
    """Output from a single score_options() call.

    scores and raw_logprobs are identical at the backend level.
    Calibration (e.g. PriDe prior subtraction) is applied upstream by the runner.
    """

    scores: dict[str, float]           # {"A": -0.1, "B": -1.5, ...}
    raw_logprobs: dict[str, float] | None
    metadata: dict = field(default_factory=dict)

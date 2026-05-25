# src/twoprompt/backends/dummy.py

from __future__ import annotations

from twoprompt.backends.base import LocalBackend
from twoprompt.backends.types import (
    LocalGenerationConfig,
    LocalModelMetadata,
    ModelGenerationResult,
    ModelOptionScoreResult,
)

_DEFAULT_TEXT = "The answer is A."
_DEFAULT_SCORES: dict[str, float] = {"A": -0.1, "B": -1.5, "C": -2.0, "D": -2.5}


class DummyBackend(LocalBackend):
    """Test-only backend that returns fixed outputs without loading any model weights.

    Use this to run and test runner/pipeline logic locally before submitting
    a SLURM job. load() is instant, generate() and score_options() return the
    configured fixed values regardless of the prompt content.

    Example:
        backend = DummyBackend()
        backend.load()
        result = backend.generate("What is 2+2?")
        # result.raw_text == "The answer is A."
    """

    def __init__(
        self,
        fixed_text: str = _DEFAULT_TEXT,
        fixed_scores: dict[str, float] | None = None,
    ) -> None:
        """
        Args:
            fixed_text:    Text returned by every generate() call.
            fixed_scores:  Scores returned by score_options(). Keys are option
                           labels; any option not in the dict gets score -5.0.
        """
        self._fixed_text = fixed_text
        self._fixed_scores = fixed_scores if fixed_scores is not None else dict(_DEFAULT_SCORES)
        self._loaded = False

    @property
    def metadata(self) -> LocalModelMetadata:
        return LocalModelMetadata(
            model_name="dummy",
            model_path="dummy://",
            family="dummy",
            backend_type="dummy",
            device="cpu",
            size_label=None,
        )

    def load(self) -> None:
        self._loaded = True

    def generate(
        self,
        prompt: str,
        config: LocalGenerationConfig | None = None,
    ) -> ModelGenerationResult:
        if not self._loaded:
            raise RuntimeError("Call load() before generate().")
        return ModelGenerationResult(
            raw_text=self._fixed_text,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(self._fixed_text.split()),
            finish_reason="eos",
            metadata={"backend": "dummy"},
        )

    def score_options(
        self,
        prompt: str,
        options: list[str],
    ) -> ModelOptionScoreResult:
        if not self._loaded:
            raise RuntimeError("Call load() before score_options().")
        if not options:
            raise ValueError("options must not be empty.")
        scores = {opt: self._fixed_scores.get(opt, -5.0) for opt in options}
        return ModelOptionScoreResult(
            scores=scores,
            raw_logprobs=dict(scores),
            metadata={"backend": "dummy"},
        )

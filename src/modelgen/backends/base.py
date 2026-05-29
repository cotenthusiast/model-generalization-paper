# src/modelgen/backends/base.py

from __future__ import annotations

from abc import ABC, abstractmethod

from modelgen.backends.types import (
    LocalGenerationConfig,
    LocalModelMetadata,
    ModelGenerationResult,
    ModelOptionScoreResult,
)


class LocalBackend(ABC):
    """Abstract base for all local model backends.

    The backend knows nothing about experiment methods, benchmarks, runners,
    or evaluation. It only takes prompts and returns model outputs.

    Usage pattern:
        backend = QwenBackend("/path/to/model", size_label="7B")
        backend.load()
        result = backend.generate(prompt, config)
    """

    @property
    @abstractmethod
    def metadata(self) -> LocalModelMetadata:
        """Static information about this backend and model."""

    @abstractmethod
    def load(self) -> None:
        """Load the model and tokenizer into memory.

        Must be called before generate() or score_options().
        Implementations must be idempotent: calling load() twice is safe.
        """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: LocalGenerationConfig | None = None,
    ) -> ModelGenerationResult:
        """Generate text from a prompt.

        Args:
            prompt: Full input prompt string.
            config: Generation settings. Uses backend defaults if None.

        Returns:
            ModelGenerationResult with the generated text and token counts.
        """

    @abstractmethod
    def score_options(
        self,
        prompt: str,
        options: list[str],
    ) -> ModelOptionScoreResult:
        """Score a list of option tokens given a prompt.

        Runs a single forward pass and returns the log-probability of each
        option token at the next position after the prompt. Each option must
        be a single token in the model's vocabulary (e.g. "A", "B", "C", "D").

        Args:
            prompt:  Full input prompt ending just before the answer position.
            options: Single-character option labels, e.g. ["A", "B", "C", "D"].

        Raises:
            RuntimeError:       If called before load().
            ValueError:         If options is empty or any option is not a single token.
            NotImplementedError: If the backend does not support logprob access.

        Returns:
            ModelOptionScoreResult with log-probability scores for each option.
        """

    def is_loaded(self) -> bool:
        """Return True if the model has been loaded into memory."""
        return bool(getattr(self, "_loaded", False))

# src/twoprompt/runners/local_base.py

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from twoprompt.backends.base import LocalBackend
from twoprompt.backends.types import LocalGenerationConfig, ModelGenerationResult
from twoprompt.parsing.parser import parse_model_answer
from twoprompt.parsing.types import ParseResult
from twoprompt.pipeline.prompt_builder import load_prompt_templates
from twoprompt.scoring.scorer import score_prediction
from twoprompt.scoring.types import ScoreResult


class LocalExperimentRunner(ABC):
    """Abstract base for local-model experiment runners.

    Mirrors ExperimentRunner (cloud) but uses LocalBackend instead of BaseClient.
    Synchronous — no asyncio. Produces the same CSV column schema as ExperimentRunner
    so evaluate_run.py can process runs from both cloud and local backends.
    """

    def __init__(
        self,
        backend: LocalBackend,
        method_name: str,
        split_name: str,
        prompt_version: str,
        prompts_dir: Path,
        run_id: str,
        generation_config: LocalGenerationConfig | None = None,
        perturbation_name: str | None = None,
    ) -> None:
        self.backend = backend
        self.method_name = method_name
        self.split_name = split_name
        self.prompt_version = prompt_version
        self.run_id = run_id
        self.generation_config = generation_config or LocalGenerationConfig()
        self.perturbation_name = perturbation_name
        self._prompts = load_prompt_templates(prompt_version, prompts_dir)

    @abstractmethod
    def run_one(self, question_row: Any, sample_index: int) -> dict:
        """Execute a single question through this experimental condition."""

    def setup(self, question_rows: Sequence[Any]) -> None:
        """Optional pre-run setup hook called once before run_many iterates.

        Override in runners that need a calibration or initialisation phase.
        The default is a no-op.
        """

    def run_many(self, question_rows: Sequence[Any]) -> list[dict]:
        """Execute multiple questions sequentially."""
        self.setup(question_rows)
        return [self.run_one(row, i) for i, row in enumerate(question_rows)]

    def _call_backend(
        self,
        prompt: str,
        config: LocalGenerationConfig | None = None,
    ) -> tuple[ModelGenerationResult | None, float, str | None]:
        """Call backend.generate(), returning (result, latency_seconds, error_message)."""
        start = time.perf_counter()
        try:
            result = self.backend.generate(prompt, config or self.generation_config)
            latency = time.perf_counter() - start
            return result, latency, None
        except Exception as exc:
            latency = time.perf_counter() - start
            return None, latency, str(exc)

    def _build_result_row(
        self,
        question_row: Any,
        prompt: str,
        sample_index: int,
        generation_result: ModelGenerationResult | None,
        latency_seconds: float,
        parsed_result: ParseResult | None,
        score_result: ScoreResult | None,
        error: str | None = None,
    ) -> dict:
        """Assemble a flat result dict matching the cloud ExperimentRunner column schema."""
        meta = self.backend.metadata
        cfg = self.generation_config
        is_success = generation_result is not None and error is None

        return {
            # --- trace metadata ---
            "run_id": self.run_id,
            "question_id": question_row["question_id"],
            "split_name": self.split_name,
            "subject": question_row["subject"],
            "method_name": self.method_name,
            "prompt_version": self.prompt_version,
            "perturbation_name": self.perturbation_name,
            "sample_index": sample_index,
            # --- model config ---
            "provider": meta.family,
            "model_name": meta.model_path,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_new_tokens,
            "seed": cfg.seed,
            # --- question content ---
            "question_text": question_row["question_text"],
            "choice_a": question_row["choice_a"],
            "choice_b": question_row["choice_b"],
            "choice_c": question_row["choice_c"],
            "choice_d": question_row["choice_d"],
            "correct_option": question_row["correct_option"],
            # --- prompt ---
            "prompt": prompt,
            # --- model output ---
            "model_status": "success" if is_success else "error",
            "raw_text": generation_result.raw_text if generation_result else None,
            "finish_reason": generation_result.finish_reason if generation_result else None,
            "latency_seconds": latency_seconds,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            # --- error info ---
            "error_type": None if is_success else "LocalInferenceError",
            "error_message": None if is_success else error,
            "error_stage": None if is_success else "generate",
            "error_retryable": None if is_success else False,
            # --- parse output ---
            "parsed_choice": parsed_result.final_choice if parsed_result else None,
            "parse_status": parsed_result.status if parsed_result else None,
            "normalized_text": parsed_result.normalized_text if parsed_result else None,
            "parse_reason": parsed_result.reason if parsed_result else None,
            # --- score output ---
            "is_correct": score_result.is_correct if score_result else None,
            "score_status": score_result.status if score_result else None,
        }

    def _parse_and_score(
        self,
        raw_text: str,
        correct_option: str,
        options: dict[str, str],
    ) -> tuple[ParseResult, ScoreResult]:
        parse_result = self._parse(raw_text, options)
        score_result = self._score(parse_result, correct_option)
        return parse_result, score_result

    @staticmethod
    def _parse(raw_text: str, options: dict[str, str]) -> ParseResult:
        return parse_model_answer(raw_text, options)

    @staticmethod
    def _score(parse_result: ParseResult, correct_option: str) -> ScoreResult:
        return score_prediction(parse_result, correct_option)

    @staticmethod
    def _build_options(question_row: Any) -> dict[str, str]:
        return {
            "A": question_row["choice_a"],
            "B": question_row["choice_b"],
            "C": question_row["choice_c"],
            "D": question_row["choice_d"],
        }

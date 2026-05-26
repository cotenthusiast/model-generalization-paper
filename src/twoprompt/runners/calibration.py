# src/twoprompt/runners/calibration.py

from __future__ import annotations

from typing import Any

from twoprompt.runners.local_base import LocalExperimentRunner
from twoprompt.pipeline.prompt_builder import build_direct_mcq_prompt

class AnswerCalibrationRunner(LocalExperimentRunner):
    """Answer-level calibration runner.

    Estimates a per-option bias prior by scoring each option in a
    content-free (neutral) context, then subtracts that prior from
    the real-context option scores before picking the answer.

    Two phases:
      1. Calibration — score options against a neutral prompt (no real
         question) to build a bias estimate.
      2. Inference — score options against the real question, subtract
         the calibration prior, pick the highest corrected score.

    One backend call per option per question (plus calibration calls).
    """

    def setup(self, question_rows) -> None:
        """Run calibration phase once before inference begins."""
        ...

    def _run_calibration(self, questions: list[dict]) -> dict[str, float]:
        """Score each option letter in a neutral context to estimate label bias.

        Returns a dict mapping option letter to its prior bias score.
        """
        ...


    def _build_neutral_prompt(self) -> str:
        """Build a content-free prompt used during calibration.

        The prompt contains no real question — just enough structure to
        elicit a first-token option-letter response.
        """
        return build_direct_mcq_prompt(
            template=self._prompts["direct_mcq"],
            question="N/A",
            options=["N/A"] * 4
        )

    def run_one(self, question_row: Any, sample_index: int) -> dict:
        """Score all options for one question, apply calibration correction, return result."""
        ...

    def _build_prompt(self, question_row: Any) -> str:
        """Build the standard MCQ prompt for a question."""
        ...
        return build_direct_mcq_prompt(
            template=self._prompts["direct_mcq"],
            question=question_row["question_text"],
            options=list(self._build_options(question_row).values())
        )

    def _apply_correction(
        self,
        raw_scores: dict[str, float],
        prior: dict[str, float],
    ) -> str:
        """Subtract prior bias from raw scores and return the winning option letter."""
        ...
        adjusted_scores = {option: raw - prior.get(option, 0.0) for option, raw in raw_scores.items()}
        return max(adjusted_scores, key=adjusted_scores.get)

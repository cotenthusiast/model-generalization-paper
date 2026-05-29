# src/modelgen/runners/direct_mcq.py

from typing import Any

from modelgen.pipeline.prompt_builder import build_direct_mcq_prompt
from modelgen.runners.local_base import LocalExperimentRunner


class DirectMCQRunner(LocalExperimentRunner):
    """Runner for the direct MCQ baseline condition.

    Presents the model with a standard multiple-choice question and
    expects a single letter response. One backend call per question.
    """

    def run_one(self, question_row: Any, sample_index: int) -> dict:
        prompt = self._build_prompt(question_row)
        generation_result, latency, error = self._call_backend(prompt)

        parsed_result = None
        score_result = None
        if generation_result is not None:
            parsed_result, score_result = self._parse_and_score(
                raw_text=generation_result.raw_text,
                correct_option=question_row["correct_option"],
                options=self._build_options(question_row),
            )

        return self._build_result_row(
            question_row=question_row,
            prompt=prompt,
            sample_index=sample_index,
            generation_result=generation_result,
            latency_seconds=latency,
            parsed_result=parsed_result,
            score_result=score_result,
            error=error,
        )

    def _build_prompt(self, question_row: Any) -> str:
        return build_direct_mcq_prompt(
            template=self._prompts["direct_mcq"],
            question=question_row["question_text"],
            options=list(self._build_options(question_row).values()),
        )

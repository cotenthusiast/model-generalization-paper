# src/modelgen/runners/two_stage.py

from typing import Any

from modelgen.backends.types import ModelGenerationResult
from modelgen.pipeline.prompt_builder import (
    build_direct_mcq_prompt,
    build_free_text_prompt,
    build_option_matching_prompt,
)
from modelgen.runners.local_base import LocalExperimentRunner
from modelgen.parsing.types import ParseResult
from modelgen.scoring.types import ScoreResult


class TwoStageRunner(LocalExperimentRunner):
    """Runner for the two-stage prompting condition.

    Stage one elicits a free-text answer without exposing options.
    Stage two asks the model to match that free-text answer to one
    of the four canonical options. The final letter from stage two
    is parsed and scored.

    The intermediate free-text response is preserved in the result
    row for downstream answer-matching evaluation.
    """

    def __init__(self, *args, fallback_on_parse_failure: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._fallback_on_parse_failure = fallback_on_parse_failure

    def run_one(self, question_row: Any, sample_index: int) -> dict:
        free_text_result, free_text_latency, free_text_error = self._call_stage1(
            question_row
        )
        if free_text_result is None:
            return self._build_result_row(
                question_row=question_row,
                prompt=build_free_text_prompt(
                    template=self._prompts["free_text"],
                    question=question_row["question_text"],
                ),
                sample_index=sample_index,
                generation_result=None,
                latency_seconds=free_text_latency,
                parsed_result=None,
                score_result=None,
                error=free_text_error,
            )

        matching_result, matching_latency, matching_error = self._call_stage2(
            question_row, free_text_result.raw_text
        )

        parsed_result, score_result = None, None
        if matching_result is not None:
            parsed_result, score_result = self._parse_and_score(
                raw_text=matching_result.raw_text,
                correct_option=question_row["correct_option"],
                options=self._build_options(question_row),
            )

        parsed_result, score_result, fallback_used = self._run_fallback_if_needed(
            question_row, matching_result, parsed_result, score_result
        )

        matching_prompt = build_option_matching_prompt(
            template=self._prompts["option_matching"],
            question=question_row["question_text"],
            free_text=free_text_result.raw_text,
            options=self._build_options(question_row),
        )

        result = self._build_result_row(
            question_row=question_row,
            prompt=matching_prompt,
            sample_index=sample_index,
            generation_result=matching_result,
            latency_seconds=matching_latency,
            parsed_result=parsed_result,
            score_result=score_result,
            error=matching_error,
        )
        result["free_text_prompt"] = build_free_text_prompt(
            template=self._prompts["free_text"],
            question=question_row["question_text"],
        )
        result["free_text_response"] = free_text_result.raw_text
        result["free_text_latency"] = free_text_latency
        result["fallback_used"] = fallback_used
        return result

    def _call_stage1(
            self,
            question_row: Any,
    ) -> tuple[ModelGenerationResult | None, float, str | None]:
        prompt = build_free_text_prompt(
            template=self._prompts["free_text"],
            question=question_row["question_text"],
        )
        return self._call_backend(prompt)

    def _call_stage2(
            self,
            question_row: Any,
            free_text_answer: str,
    ) -> tuple[ModelGenerationResult | None, float, str | None]:
        prompt = build_option_matching_prompt(
            template=self._prompts["option_matching"],
            question=question_row["question_text"],
            free_text=free_text_answer,
            options=self._build_options(question_row),
        )
        return self._call_backend(prompt)

    def _run_fallback_if_needed(
            self,
            question_row: Any,
            matching_result: ModelGenerationResult | None,
            parsed_result: ParseResult | None,
            score_result: ScoreResult | None,
    ) -> tuple[ParseResult | None, ScoreResult | None, bool]:
        if not (
            self._fallback_on_parse_failure
            and matching_result is not None
            and (parsed_result is None or parsed_result.final_choice is None)
        ):
            return parsed_result, score_result, False

        fallback_prompt = build_direct_mcq_prompt(
            template=self._prompts["direct_mcq"],
            question=question_row["question_text"],
            options=self._build_options(question_row),
        )
        fallback_result, _, _ = self._call_backend(fallback_prompt)
        if fallback_result is not None:
            parsed_result, score_result = self._parse_and_score(
                raw_text=fallback_result.raw_text,
                correct_option=question_row["correct_option"],
                options=self._build_options(question_row),
            )
            return parsed_result, score_result, True

        return parsed_result, score_result, False

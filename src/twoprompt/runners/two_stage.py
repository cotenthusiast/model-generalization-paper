# src/twoprompt/runners/two_stage.py

from typing import Any

from twoprompt.pipeline.prompt_builder import (
    build_direct_mcq_prompt,
    build_free_text_prompt,
    build_option_matching_prompt,
)
from twoprompt.runners.local_base import LocalExperimentRunner


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
        # Stage 1: free-text response
        free_text_prompt = build_free_text_prompt(
            template=self._prompts["free_text"],
            question=question_row["question_text"],
        )
        free_text_result, free_text_latency, free_text_error = self._call_backend(
            free_text_prompt
        )

        # If stage 1 fails, return early
        if free_text_result is None:
            return self._build_result_row(
                question_row=question_row,
                prompt=free_text_prompt,
                sample_index=sample_index,
                generation_result=None,
                latency_seconds=free_text_latency,
                parsed_result=None,
                score_result=None,
                error=free_text_error,
            )

        free_text_answer = free_text_result.raw_text

        # Stage 2: option matching using the free-text answer
        matching_prompt = build_option_matching_prompt(
            template=self._prompts["option_matching"],
            question=question_row["question_text"],
            free_text=free_text_answer,
            option_a=question_row["choice_a"],
            option_b=question_row["choice_b"],
            option_c=question_row["choice_c"],
            option_d=question_row["choice_d"],
        )
        matching_result, matching_latency, matching_error = self._call_backend(
            matching_prompt
        )

        parsed_result = None
        score_result = None
        if matching_result is not None:
            parsed_result, score_result = self._parse_and_score(
                raw_text=matching_result.raw_text,
                correct_option=question_row["correct_option"],
                options=self._build_options(question_row),
            )

        fallback_used = False
        if (
            self._fallback_on_parse_failure
            and matching_result is not None
            and (parsed_result is None or parsed_result.final_choice is None)
        ):
            fallback_prompt = build_direct_mcq_prompt(
                template=self._prompts["direct_mcq"],
                question=question_row["question_text"],
                option_a=question_row["choice_a"],
                option_b=question_row["choice_b"],
                option_c=question_row["choice_c"],
                option_d=question_row["choice_d"],
            )
            fallback_result, _, _ = self._call_backend(fallback_prompt)
            if fallback_result is not None:
                parsed_result, score_result = self._parse_and_score(
                    raw_text=fallback_result.raw_text,
                    correct_option=question_row["correct_option"],
                    options=self._build_options(question_row),
                )
                fallback_used = True

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

        result["free_text_prompt"] = free_text_prompt
        result["free_text_response"] = free_text_answer
        result["free_text_latency"] = free_text_latency
        result["fallback_used"] = fallback_used

        return result

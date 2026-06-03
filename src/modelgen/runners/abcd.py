# src/modelgen/runners/abcd.py

from typing import Any

from modelgen.pipeline.prompt_builder import build_abcd_prompt
from modelgen.runners.local_base import LocalExperimentRunner
from modelgen.runners.text_extraction import match_free_text_to_options


class ABCDRunner(LocalExperimentRunner):
    """Runner for the ABCD uniform-label condition (Nowak et al. 2026).

    Stage 1 presents all four options under neutral dash labels instead of
    A/B/C/D letter labels, eliciting a free-text answer free of positional
    letter cues. Stage 2 deterministically selects the best-matching option
    using token-Jaccard similarity — no second LLM call.

    The stage-2 matching logic is identical to TextExtractionRunner.
    """

    def __init__(self, *args, similarity_threshold: float = 0.1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._similarity_threshold = similarity_threshold

    def run_one(self, question_row: Any, sample_index: int) -> dict:
        prompt = build_abcd_prompt(
            template=self._prompts["abcd"],
            question=question_row["question_text"],
            option_a=question_row["choice_a"],
            option_b=question_row["choice_b"],
            option_c=question_row["choice_c"],
            option_d=question_row["choice_d"],
        )
        generation_result, latency, error = self._call_backend(prompt)

        parsed_result = None
        score_result = None
        best_similarity_score = None

        if generation_result is not None:
            parsed_result, best_similarity_score = match_free_text_to_options(
                generation_result.raw_text,
                self._build_options(question_row),
                self._similarity_threshold,
            )
            score_result = self._score(parsed_result, question_row["correct_option"])

        row = self._build_result_row(
            question_row=question_row,
            prompt=prompt,
            sample_index=sample_index,
            generation_result=generation_result,
            latency_seconds=latency,
            parsed_result=parsed_result,
            score_result=score_result,
            error=error,
        )
        row["free_text_response"] = generation_result.raw_text if generation_result else None
        row["best_similarity_score"] = best_similarity_score
        return row

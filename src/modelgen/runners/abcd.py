# src/modelgen/runners/abcd.py

from typing import Any

from sentence_transformers import SentenceTransformer

from modelgen.pipeline.prompt_builder import build_abcd_prompt
from modelgen.runners.local_base import LocalExperimentRunner
from modelgen.runners.text_extraction import _ABCD_EMBEDDING_MODEL, resolve_stage2_answer


class ABCDRunner(LocalExperimentRunner):
    """Runner for the ABCD uniform-label condition.

    Stage 1 presents all four options under neutral dash labels instead of
    A/B/C/D letter labels, eliciting a free-text answer free of positional
    letter cues. Stage 2 (resolve_stage2_answer, shared with
    TextExtractionRunner) resolves a declared letter directly if the model
    states one despite the dash labels, isolates the model's earliest
    clearly-stated answer span otherwise, and falls back to
    sentence-embedding cosine similarity against option text — no second
    LLM call. No abstention threshold (always argmax over similarity), and a
    dedicated embedding model rather than TextExtractionRunner's smaller
    default. This is this codebase's own design for the condition, not a
    verified reproduction of a specific published procedure.
    """

    def __init__(
        self,
        *args,
        similarity_threshold: float = float("-inf"),
        embedding_model: str = _ABCD_EMBEDDING_MODEL,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._similarity_threshold = similarity_threshold
        self._st_model = SentenceTransformer(embedding_model)

    def run_one(self, question_row: Any, sample_index: int) -> dict:
        prompt = build_abcd_prompt(
            template=self._prompts["abcd"],
            question=question_row["question_text"],
            options=self._build_options(question_row),
        )
        generation_result, latency, error = self._call_backend(prompt)

        parsed_result = None
        score_result = None
        best_similarity_score = None

        if generation_result is not None:
            parsed_result, best_similarity_score = resolve_stage2_answer(
                generation_result.raw_text,
                self._build_options(question_row),
                self._similarity_threshold,
                self._st_model,
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

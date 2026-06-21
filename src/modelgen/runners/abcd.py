# src/modelgen/runners/abcd.py

import random
from typing import Any

from sentence_transformers import SentenceTransformer

from modelgen.pipeline.prompt_builder import build_abcd_prompt
from modelgen.runners.abcd_extraction import resolve_abcd_answer
from modelgen.runners.local_base import LocalExperimentRunner
from modelgen.runners.text_extraction import _ABCD_EMBEDDING_MODEL


class ABCDRunner(LocalExperimentRunner):
    """Runner for the ABCD uniform-label condition.

    Reproduces the "Matched-and-Dashed" (M&D) evaluation protocol of Nowak,
    Cadet, and Chin, "ABCD: All Biases Come Disguised" (Dartmouth),
    arXiv:2602.17445: Stage 1 presents all options under uniform dash labels
    (Section 4) and a prompt modified per Appendix F.3/Figure 14 to elicit a
    full-text answer that repeats the chosen option verbatim, instead of a
    letter. Stage 2 (resolve_abcd_answer, abcd_extraction.py) isolates a
    candidate answer span via the paper's four-tier regex cascade (Appendix
    F.2) and resolves it to an option via sentence-embedding cosine
    similarity (Section 4) -- no second LLM call. similarity_threshold
    defaults to -inf (always argmax over similarity, never abstains): the
    paper's own abstention path is the random fallback inside
    resolve_abcd_answer for the case where the model produces no text at
    all, not a similarity-threshold cutoff.
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
        self._rng = random.Random(self.generation_config.seed)

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
            parsed_result, best_similarity_score = resolve_abcd_answer(
                generation_result.raw_text,
                self._build_options(question_row),
                self._similarity_threshold,
                self._st_model,
                self._rng,
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

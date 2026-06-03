# src/modelgen/runners/text_extraction.py

from typing import Any

from sentence_transformers import SentenceTransformer

from modelgen.parsing.parser import normalize_output_text
from modelgen.parsing.types import PARSE_MISSING, PARSE_OK, ParseResult
from modelgen.pipeline.prompt_builder import build_text_extraction_prompt
from modelgen.runners.local_base import LocalExperimentRunner

# all-MiniLM-L6-v2: 22M params, fast on CPU, strong for short texts;
# swap via embedding_model kwarg if a larger model is needed
_DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def match_free_text_to_options(
    free_text: str,
    options: dict[str, str],
    similarity_threshold: float,
    model: SentenceTransformer,
) -> tuple[ParseResult, float | None]:
    """Deterministically select the best-matching option for a free-text answer.

    Returns (ParseResult, best_score). best_score is None when free_text is
    empty. ParseResult has PARSE_MISSING status when best_score is below
    similarity_threshold.

    Exported for reuse by ABCDRunner, which has identical stage-2 logic.
    """
    normalized_ft = normalize_output_text(free_text)
    if not normalized_ft:
        return ParseResult(
            final_choice=None,
            status=PARSE_MISSING,
            raw_text=free_text,
            normalized_text=normalized_ft,
            reason="Empty free-text response",
        ), None

    letters = list(options.keys())
    option_texts = [normalize_output_text(options[l]) for l in letters]

    # Encode free text and all options in one batch; normalize for cosine sim
    embeddings = model.encode(
        [normalized_ft] + option_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    ft_emb = embeddings[0]
    opt_embs = embeddings[1:]

    # Dot product of L2-normalized vectors equals cosine similarity
    raw_sims = opt_embs @ ft_emb
    scores = {letter: float(raw_sims[i]) for i, letter in enumerate(letters)}

    best_letter = max(scores, key=scores.__getitem__)
    best_score = scores[best_letter]

    if best_score < similarity_threshold:
        return ParseResult(
            final_choice=None,
            status=PARSE_MISSING,
            raw_text=free_text,
            normalized_text=normalized_ft,
            reason=f"Best similarity {best_score:.3f} below threshold {similarity_threshold:.3f}",
        ), best_score

    return ParseResult(
        final_choice=best_letter,
        status=PARSE_OK,
        raw_text=free_text,
        normalized_text=normalized_ft,
        reason=f"Embedding cosine match to option {best_letter} (score={best_score:.3f})",
    ), best_score


class TextExtractionRunner(LocalExperimentRunner):
    """Runner for the text-extraction condition.

    Stage 1 shows all four options with A/B/C/D labels and instructs the
    model not to state the answer letter. Stage 2 deterministically selects
    the best-matching option using sentence-embedding cosine similarity —
    no second LLM call.
    """

    def __init__(
        self,
        *args,
        similarity_threshold: float = 0.1,
        embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._similarity_threshold = similarity_threshold
        self._st_model = SentenceTransformer(embedding_model)

    def run_one(self, question_row: Any, sample_index: int) -> dict:
        prompt = build_text_extraction_prompt(
            template=self._prompts["text_extraction"],
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

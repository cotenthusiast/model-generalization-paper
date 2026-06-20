# src/modelgen/runners/text_extraction.py

import re
from typing import Any

from sentence_transformers import SentenceTransformer

from modelgen.parsing.parser import detect_leading_letter, normalize_output_text
from modelgen.parsing.types import PARSE_MISSING, PARSE_OK, ParseResult
from modelgen.pipeline.prompt_builder import build_text_extraction_prompt
from modelgen.runners.local_base import LocalExperimentRunner

# all-MiniLM-L6-v2: 22M params, fast on CPU, strong for short texts;
# swap via embedding_model kwarg if a larger model is needed
_DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Embedding model used for the abcd condition specifically (not used by
# TextExtractionRunner, which keeps the smaller default above). This is an
# independent design choice for this codebase, not a verified reproduction
# of any specific published model/paper.
_ABCD_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

_FINAL_ANSWER_CUE_RE = re.compile(
    r"(?:final answer|the answer is|answer is|the correct (?:answer|option|choice) is|"
    r"therefore|thus)\b",
    re.IGNORECASE,
)

# Conversational sign-off / hedge sentences that sometimes trail a stated
# answer (e.g. "Let me know if you need anything else!") — stripped before
# span selection so they can never be picked as a fallback span and can't
# spuriously contain a cue word.
_CONVERSATIONAL_FILLER_RE = re.compile(
    r"(?:let me know if|i hope this helps|feel free to|don't hesitate to|"
    r"if you have any (?:other |further )?questions|happy to help|"
    r"glad to (?:help|assist))",
    re.IGNORECASE,
)


def _split_into_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter: break on '.', '!', '?' followed by whitespace."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _strip_trailing_filler(sentences: list[str]) -> list[str]:
    """Drop conversational sign-off sentences from the end of the list."""
    trimmed = list(sentences)
    while trimmed and _CONVERSATIONAL_FILLER_RE.search(trimmed[-1]):
        trimmed.pop()
    return trimmed


_CUE_WORDS = {"answer", "choice", "option"}
_CONCLUDING_WORDS = {"therefore", "thus"}


def _is_declared_letter_token(word: str, valid_choices) -> bool:
    """True if `word` (a raw, punctuation-attached token) reads as a
    declared answer letter rather than an indefinite article ("a"/"A").

    Mirrors detect_leading_letter's precondition: a letter immediately
    followed by '.', ')', ':', or ',' (e.g. "C." or "C,") is unambiguous
    regardless of case. A bare single-character token with no trailing
    punctuation (e.g. the last word in "the answer is C") is only accepted
    if it's uppercase — real model output states letters as "C", never as
    the lowercase indefinite article "a"/"an" that can otherwise appear
    after "is" (e.g. "the answer is a combination of...").
    """
    if not word or word[0].upper() not in valid_choices:
        return False
    if len(word) >= 2 and word[1] in ".):,":
        return True
    return len(word) == 1 and word.isupper()


def _detect_cue_stated_letter(sentence: str, valid_choices) -> str | None:
    """Find a letter stated via an explicit cue within a single sentence.

    Covers "the answer is C.", "the correct option is B)", "final answer is
    D", "therefore, C." etc. Used to resolve a span that extract_final_
    answer_span isolated via a conclusion cue but that turned out to state a
    bare letter rather than option text — match_free_text_to_options can
    only compare against option text, so without this check such a span
    would score low against every option and come back unscorable.
    """
    words = sentence.split()
    stripped = [w.strip("()[]{}<>\".,:;!?'") for w in words]
    n = len(words)
    for i in range(n):
        wl = stripped[i].lower()
        if (
            i + 3 < n
            and wl == "final"
            and stripped[i + 1].lower() == "answer"
            and stripped[i + 2].lower() == "is"
            and _is_declared_letter_token(words[i + 3], valid_choices)
        ):
            return words[i + 3][0].upper()
        if (
            i + 2 < n
            and wl in _CUE_WORDS
            and stripped[i + 1].lower() == "is"
            and _is_declared_letter_token(words[i + 2], valid_choices)
        ):
            return words[i + 2][0].upper()
        if i + 1 < n and wl in _CUE_WORDS and _is_declared_letter_token(words[i + 1], valid_choices):
            return words[i + 1][0].upper()
        if i + 1 < n and wl in _CONCLUDING_WORDS and _is_declared_letter_token(words[i + 1], valid_choices):
            return words[i + 1][0].upper()
    return None


def extract_final_answer_span(free_text: str | None) -> str:
    """Isolate the model's earliest clearly-stated answer from a free-text response.

    Embedding an entire multi-sentence explanation dilutes the similarity
    signal toward whichever option's vocabulary is discussed most, rather
    than the option the model actually answered with. Real abcd responses
    consistently state the answer in the first sentence, then hedge,
    second-guess, or pivot in later sentences (e.g. "However, given the
    options, the best answer would be..." going on to restate a different,
    sometimes invented, option) — so the *earliest* clear answer statement
    is the reliable signal, not the last one.

    Conversational sign-off sentences (e.g. "Let me know if you need
    anything else!") are stripped from the end of the response first, since
    they're never the stated answer and could otherwise be picked as a
    fallback span or coincidentally contain a cue word.

    Among the remaining sentences, this returns the *first* one containing
    an explicit conclusion cue ("the answer is...", "therefore...", etc.),
    or the first sentence overall if no cue is present anywhere. This is an
    independent heuristic for this codebase's abcd and text_extraction
    conditions, not a reproduction of a specific published procedure.

    Returns the original (normalized) text unchanged if it has no sentence
    boundaries to split on.
    """
    text = normalize_output_text(free_text)
    if not text:
        return text
    sentences = _split_into_sentences(text)
    if not sentences:
        return text

    candidates = _strip_trailing_filler(sentences)
    if not candidates:
        candidates = sentences

    for sentence in candidates:
        if _FINAL_ANSWER_CUE_RE.search(sentence):
            return sentence
    return candidates[0]


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


def resolve_stage2_answer(
    raw_text: str | None,
    options: dict[str, str],
    similarity_threshold: float,
    model: SentenceTransformer,
) -> tuple[ParseResult, float | None]:
    """Shared stage-2 resolution for abcd and text_extraction.

    Tries, in order: (1) a declared leading letter at the very start of the
    raw response, (2) a declared letter stated via an explicit cue within
    the isolated answer span (e.g. "the answer is C" rather than restated
    option text), (3) embedding-similarity matching of the isolated span
    against option text. Only case (3) needs the embedding model.
    """
    shortcut = try_resolve_declared_letter(raw_text, options)
    if shortcut is not None:
        return shortcut

    span = extract_final_answer_span(raw_text)
    return match_free_text_to_options(span, options, similarity_threshold, model)


def try_resolve_declared_letter(
    raw_text: str | None,
    options: dict[str, str],
) -> tuple[ParseResult, float] | None:
    """Try the model-free shortcuts of resolve_stage2_answer in isolation.

    Returns the (ParseResult, score) pair if a declared leading letter or a
    cue-stated letter within the isolated answer span was found, else None
    to signal the caller must fall back to embedding-similarity matching.
    Exported so rematch tooling (scripts/evaluate_run.py) can apply these
    free, deterministic shortcuts to ~10k saved rows without needing to load
    an embedding model first.
    """
    normalized = normalize_output_text(raw_text)
    leading = detect_leading_letter(normalized, options.keys())
    if leading is not None:
        return ParseResult(
            final_choice=leading,
            status=PARSE_OK,
            raw_text=raw_text,
            normalized_text=normalized,
            reason=f"Leading letter '{leading}' matched a valid option",
        ), 1.0

    span = extract_final_answer_span(raw_text)
    cue_letter = _detect_cue_stated_letter(span, options.keys())
    if cue_letter is not None:
        return ParseResult(
            final_choice=cue_letter,
            status=PARSE_OK,
            raw_text=raw_text,
            normalized_text=normalize_output_text(span),
            reason=f"Cue-stated letter '{cue_letter}' matched a valid option",
        ), 1.0

    return None


class TextExtractionRunner(LocalExperimentRunner):
    """Runner for the text-extraction condition.

    Stage 1 shows all four options with A/B/C/D labels and instructs the
    model not to state the answer letter. Stage 2 deterministically selects
    the best-matching option using sentence-embedding cosine similarity —
    no second LLM call.

    Despite the instruction, some models state a leading letter anyway
    (resolved directly, same precondition as additional_option), and some
    produce long, repetitive, or hedging responses where the literal answer
    is stated early and diluted by what follows (resolved by isolating the
    earliest clear answer span before embedding, same fix as ABCDRunner).
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

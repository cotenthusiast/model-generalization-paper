"""Compute evaluation metrics for one experiment run."""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import beta as _beta_dist

from modelgen.config.experiment import (
    ABCD_METHOD,
    ADDITIONAL_OPTION_METHOD,
    BASELINE_METHOD,
    CALIBRATION_METHOD,
    INDEPENDENT_HYPOTHESIS_METHOD,
    PRIDE_METHOD,
    TEXT_EXTRACTION_METHOD,
    TWOPROMPT_METHOD,
    TWOPROMPT_CYCLIC_METHOD,
)
from modelgen.config.paths import REPORTS_DIR, RUNS_DIR
from modelgen.parsing.parser import parse_model_answer
from modelgen.parsing.types import ParseResult
from modelgen.scoring.scorer import score_prediction

_ROOT = Path(__file__).resolve().parents[1]

OPTIONS = ["A", "B", "C", "D"]

METHOD_ORDER = [
    "baseline",
    "two_prompt",
    "cyclic",
    "pride",
    "calibration",
    "additional_option",
    "text_extraction",
    "abcd",
    "independent_hypothesis",
]

MODEL_ORDER = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
]

N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 42
_CI_LO = 2.5
_CI_HI = 97.5

_OPT_INDEX = {opt: i for i, opt in enumerate(OPTIONS)}


def _clopper_pearson_ci(k: int, n: int) -> tuple[float, float]:
    """95% Clopper-Pearson exact binomial CI."""
    if n == 0:
        return float("nan"), float("nan")
    lo = float(_beta_dist.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0
    hi = float(_beta_dist.ppf(0.975, k + 1, n - k)) if k < n else 1.0
    return lo, hi


def _bootstrap_ci_mean_abs_deviation(
    group: pd.DataFrame, rng: np.random.Generator
) -> tuple[float, float]:
    """95% bootstrap CI for mean_abs_deviation, vectorised over all resamples.

    Resampling unit is individual questions (rows). For each resample the full
    mean_abs_deviation statistic is recomputed, including the scored-only filter
    on parsed_choice, mirroring compute_positional_bias exactly.
    """
    n = len(group)

    def _enc(series: pd.Series) -> np.ndarray:
        return np.array(
            [
                _OPT_INDEX.get(v, -1) if isinstance(v, str) else -1
                for v in series.values
            ],
            dtype=np.int8,
        )

    gt_enc = _enc(group["correct_option"])    # (n,)
    pred_enc = _enc(group["parsed_choice"])   # (n,) — -1 means not scored

    # Draw all bootstrap indices at once: shape (N_BOOTSTRAP, n)
    idx = rng.integers(0, n, size=(N_BOOTSTRAP, n))

    gt_boot = gt_enc[idx]    # (N_BOOTSTRAP, n)
    pred_boot = pred_enc[idx]  # (N_BOOTSTRAP, n)

    scored_boot = pred_boot >= 0                    # (N_BOOTSTRAP, n)
    total_scored = scored_boot.sum(axis=1).astype(float)  # (N_BOOTSTRAP,)

    # Accumulate |pred_pct - gt_pct| across each option
    total_abs_dev = np.zeros(N_BOOTSTRAP)
    for k in range(len(OPTIONS)):
        gt_pct = (gt_boot == k).sum(axis=1) / n * 100.0
        pred_count = ((pred_boot == k) & scored_boot).sum(axis=1)
        pred_pct = np.where(total_scored > 0, pred_count / total_scored * 100.0, 0.0)
        total_abs_dev += np.abs(pred_pct - gt_pct)

    stats = np.where(total_scored > 0, total_abs_dev / len(OPTIONS), np.nan)
    valid = stats[~np.isnan(stats)]
    if len(valid) == 0:
        return (float("nan"), float("nan"))
    return float(np.percentile(valid, _CI_LO)), float(np.percentile(valid, _CI_HI))


def load_run(run_dir: Path) -> pd.DataFrame:
    """Load and concatenate all CSVs from a run directory."""
    frames = [pd.read_csv(f) for f in sorted(run_dir.glob("*.csv"))]
    if not frames:
        raise FileNotFoundError(f"No CSV files found in {run_dir}")
    return pd.concat(frames, ignore_index=True)


def reparse_run(df: pd.DataFrame) -> pd.DataFrame:
    """Re-apply the current parser and scorer to eligible rows in-place.

    Only rows whose answer came from a single raw API response are re-parsed.
    Rows produced by majority voting (cyclic's legacy aggregation /
    two_prompt_cyclic) store parse_reason == "majority_vote" and raw_text
    from only one of the N permutation calls — re-parsing them from that
    single raw_text would discard the other permutations and corrupt the
    result. Those rows are left exactly as they were saved by the original
    run. The redesigned cyclic (parse_reason == "cyclic_eq1_avg") is
    excluded for the same reason: it picks its answer via score_options()
    probability averaging across permutations (Eq. 1), not text parsing,
    and raw_text is always null for these rows.

    Eligible rows must additionally have model_status != "failure" so that
    raw_text is present.

    ``pride`` and ``calibration`` rows are also excluded: both methods pick
    their answer via backend.score_options() rather than generating text, so
    raw_text is always null for them and there is nothing to re-parse.

    ``abcd`` and ``text_extraction`` rows are also excluded: both pick their
    answer via sentence-embedding cosine similarity against the free-text
    response (see match_free_text_to_options in runners/text_extraction.py),
    not the plain letter/text-match parser this function applies. Re-parsing
    their raw_text with parse_model_answer searches dash-labeled or
    letter-suppressed free text for a bare A/B/C/D token and finds spurious
    matches (e.g. the indefinite article "A" in "A permutation can be..."),
    silently overwriting a correct embedding-based match with garbage.

    ``additional_option`` rows are also excluded entirely: the redesigned
    runner picks its answer via score_options() and Eq.(6) (argmax over real
    options, excluding IDK — see match_options_via_scoring in
    runners/additional_option.py), not the plain letter parser, and has no
    raw_text either. (The legacy Jaccard-based rows this method used to
    produce also had no business going through the plain letter parser, for
    the same reason as abcd/text_extraction below — this exclusion covers
    both eras of additional_option data.)

    ``independent_hypothesis`` rows are also excluded: the answer is the
    argmax over N independently-scored option confidences (see
    IndependentHypothesisRunner.run_one), not a letter stated in raw_text.
    raw_text there is one option's isolated reasoning/confidence text (e.g.
    "...<score>10</score>"), which was never a candidate to contain the
    final answer letter at all — re-parsing it for a bare A/B/C/D token
    finds spurious matches and overwrites a correct argmax-derived choice
    with garbage.
    """
    df = df.copy()
    # CSV loads can infer float64 for columns with NaNs; reparsing assigns bools.
    if "is_correct" in df.columns:
        df["is_correct"] = df["is_correct"].astype(object)

    reparsable_mask = (
        (df["model_status"].fillna("") != "failure")
        & (df["parse_reason"].fillna("") != "majority_vote")
        & (df["parse_reason"].fillna("") != "cyclic_eq1_avg")
        & (df["method_name"] != PRIDE_METHOD)
        & (df["method_name"] != CALIBRATION_METHOD)
        & (df["method_name"] != ABCD_METHOD)
        & (df["method_name"] != TEXT_EXTRACTION_METHOD)
        & (df["method_name"] != ADDITIONAL_OPTION_METHOD)
        & (df["method_name"] != INDEPENDENT_HYPOTHESIS_METHOD)
    )

    parsed_choices = []
    parse_statuses = []
    normalized_texts = []
    parse_reasons = []
    is_corrects = []
    score_statuses = []

    for _, row in df[reparsable_mask].iterrows():
        options = {
            "A": row["choice_a"],
            "B": row["choice_b"],
            "C": row["choice_c"],
            "D": row["choice_d"],
        }
        parse_result = parse_model_answer(row["raw_text"], options)
        score_result = score_prediction(parse_result, row["correct_option"])

        parsed_choices.append(parse_result.final_choice)
        parse_statuses.append(parse_result.status)
        normalized_texts.append(parse_result.normalized_text)
        parse_reasons.append(parse_result.reason)
        is_corrects.append(score_result.is_correct)
        score_statuses.append(score_result.status)

    idx = df.index[reparsable_mask]
    df.loc[idx, "parsed_choice"] = parsed_choices
    df.loc[idx, "parse_status"] = parse_statuses
    df.loc[idx, "normalized_text"] = normalized_texts
    df.loc[idx, "parse_reason"] = parse_reasons
    df.loc[idx, "is_correct"] = is_corrects
    df.loc[idx, "score_status"] = score_statuses

    return df


def _stage2_rematch_cache_key(span: str, options: dict[str, str]) -> str:
    import hashlib

    payload = span + "||" + "|".join(f"{k}={v}" for k, v in sorted(options.items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rematch_with_embedding_fallback(
    df: pd.DataFrame,
    mask: pd.Series,
    text_column: str,
    default_embedding_model: str,
    similarity_threshold: float,
    cache_subdir: str,
    embedding_model: str | None,
    cache_dir: Path | None,
) -> pd.DataFrame:
    """Rematch core for text_extraction (rematch_abcd_rows has its own
    implementation below, since abcd's paper-faithful cascade has no
    declared-leading-letter shortcut to share with this one).

    For every selected row: try the free, model-free shortcuts
    (try_resolve_declared_letter — a declared leading letter, or a letter
    stated via an explicit cue within the isolated answer span) first and
    fill those in directly. Only rows where neither shortcut applies need
    the embedding model, and those are cached on disk keyed by a hash of
    (isolated span, option texts) per embedding model, under
    ``.cache/<cache_subdir>/`` — so re-embedding only ever happens once per
    distinct (span, options) pair, not once per evaluate_run.py invocation.

    No-op (returns df unchanged) when mask selects no rows.
    """
    if not mask.any():
        return df

    import json

    from modelgen.config.paths import ROOT_DIR
    from modelgen.parsing.types import PARSE_MISSING, PARSE_OK
    from modelgen.runners.text_extraction import extract_final_answer_span, try_resolve_declared_letter

    model_name = embedding_model or default_embedding_model
    cache_root = cache_dir or (ROOT_DIR / ".cache" / cache_subdir)
    cache_path = cache_root / f"{model_name.replace('/', '_')}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict] = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    df = df.copy()
    if "is_correct" in df.columns:
        df["is_correct"] = df["is_correct"].astype(object)

    # Precompute (index, span, options, key) for every row that still needs
    # the embedding model after the free shortcuts are applied directly.
    row_info = []
    for i in df.index[mask]:
        row = df.loc[i]
        options = {
            letter: row[col]
            for letter, col in (("A", "choice_a"), ("B", "choice_b"), ("C", "choice_c"), ("D", "choice_d"))
            if pd.notna(row[col]) and str(row[col]).strip() != ""
        }
        raw_text = row.get(text_column)

        shortcut = try_resolve_declared_letter(raw_text, options)
        if shortcut is not None:
            parse_result, score = shortcut
            score_result = score_prediction(parse_result, row["correct_option"])
            df.loc[i, "parsed_choice"] = parse_result.final_choice
            df.loc[i, "parse_status"] = parse_result.status
            df.loc[i, "normalized_text"] = parse_result.normalized_text
            df.loc[i, "parse_reason"] = parse_result.reason
            df.loc[i, "is_correct"] = score_result.is_correct
            df.loc[i, "score_status"] = score_result.status
            df.loc[i, "best_similarity_score"] = score
            continue

        span = extract_final_answer_span(raw_text)
        key = _stage2_rematch_cache_key(span, options)
        row_info.append((i, span, options, key))

    misses = [info for info in row_info if info[3] not in cache]
    if misses:
        import torch
        from sentence_transformers import SentenceTransformer

        # Cap CPU threads so this doesn't compete for the whole machine
        # alongside everything else the user has running.
        torch.set_num_threads(4)

        model = SentenceTransformer(model_name)

        # Batch each chunk's rows' (span + option texts) into one encode()
        # call instead of one call per row — letting SentenceTransformer's
        # own internal batching vectorize across many rows at once is far
        # faster on CPU than thousands of 5-text calls. Chunked by row
        # (rather than one encode() call across all ~10k rows) to bound peak
        # memory — encoding everything at once previously triggered an OOM
        # kill. The cache is written after every chunk, not just once at the
        # end, so a crash mid-run doesn't lose already-computed progress.
        _ROWS_PER_CHUNK = 80
        for chunk_start in range(0, len(misses), _ROWS_PER_CHUNK):
            chunk_rows = misses[chunk_start : chunk_start + _ROWS_PER_CHUNK]

            chunk_texts: list[str] = []
            chunk_slices: list[tuple[int, list[str]]] = []
            for _, span, options, _ in chunk_rows:
                letters = list(options.keys())
                start = len(chunk_texts)
                chunk_texts.append(span)
                chunk_texts.extend(options[letter] for letter in letters)
                chunk_slices.append((start, letters))

            embeddings = model.encode(
                chunk_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=True,
            )

            for (_, span, options, key), (start, letters) in zip(chunk_rows, chunk_slices):
                if span.strip() == "":
                    cache[key] = {"best_letter": None, "best_score": None, "normalized_text": span}
                    continue

                ft_emb = embeddings[start]
                opt_embs = embeddings[start + 1 : start + 1 + len(letters)]
                raw_sims = opt_embs @ ft_emb
                scores = {letter: float(raw_sims[idx]) for idx, letter in enumerate(letters)}
                best_letter = max(scores, key=scores.__getitem__)
                best_score = scores[best_letter]

                # Cache only the raw embedding result, not a threshold-derived
                # status: the cache key doesn't encode similarity_threshold,
                # so baking threshold-dependent PARSE_OK/PARSE_MISSING into
                # the cached value would go stale if a future call used a
                # different threshold against the same (span, options) pair.
                cache[key] = {"best_letter": best_letter, "best_score": best_score, "normalized_text": span}

            cache_path.write_text(json.dumps(cache))

    for i, _, _, key in row_info:
        cached = cache[key]
        best_letter = cached["best_letter"]
        best_score = cached["best_score"]
        norm_text = cached["normalized_text"]

        if best_letter is None:
            parse_result = ParseResult(
                final_choice=None,
                status=PARSE_MISSING,
                raw_text=None,
                normalized_text=norm_text,
                reason="Empty free-text response",
            )
        elif best_score < similarity_threshold:
            parse_result = ParseResult(
                final_choice=None,
                status=PARSE_MISSING,
                raw_text=None,
                normalized_text=norm_text,
                reason=f"Best similarity {best_score:.3f} below threshold {similarity_threshold:.3f}",
            )
        else:
            parse_result = ParseResult(
                final_choice=best_letter,
                status=PARSE_OK,
                raw_text=None,
                normalized_text=norm_text,
                reason=f"Embedding cosine match to option {best_letter} (score={best_score:.3f})",
            )

        score_result = score_prediction(parse_result, df.loc[i, "correct_option"])

        df.loc[i, "parsed_choice"] = parse_result.final_choice
        df.loc[i, "parse_status"] = parse_result.status
        df.loc[i, "normalized_text"] = parse_result.normalized_text
        df.loc[i, "parse_reason"] = parse_result.reason
        df.loc[i, "is_correct"] = score_result.is_correct
        df.loc[i, "score_status"] = score_result.status
        df.loc[i, "best_similarity_score"] = best_score

    return df


def rematch_abcd_rows(
    df: pd.DataFrame,
    embedding_model: str | None = None,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Re-run abcd's stage-2 resolution against already-saved free text.

    Re-derives parsed_choice/is_correct from the saved free_text_response
    column for every abcd row, regardless of what was stored at collection
    time — no new model generation calls needed, since stage 1's output
    never changes.

    Uses abcd_extraction.extract_candidate_span (the paper's Appendix F.2
    four-tier regex cascade), not text_extraction.py's shared
    extract_final_answer_span/try_resolve_declared_letter -- those implement
    a different (earliest-statement) heuristic for a condition not covered
    by this paper, and rematching abcd rows with them would silently
    overwrite the redesigned runner's paper-faithful parsed_choice with the
    old mechanism's answer every time this script runs. Does not reuse
    _rematch_with_embedding_fallback (below), which is now specific to
    rematch_text_extraction_rows: that helper's declared-leading-letter
    shortcut has no equivalent here, since abcd's prompt (Appendix F.3)
    explicitly instructs the model not to write the option letter at all.

    No-op (returns df unchanged) when no abcd rows are present.

    ``embedding_model`` defaults to the production model (Qwen3-Embedding-0.6B);
    tests pass a small model explicitly to avoid downloading a 600M-param model.
    """
    import json

    from modelgen.config.paths import ROOT_DIR
    from modelgen.parsing.types import PARSE_MISSING, PARSE_OK
    from modelgen.runners.abcd_extraction import extract_candidate_span
    from modelgen.runners.text_extraction import _ABCD_EMBEDDING_MODEL
    from modelgen.parsing.parser import normalize_output_text

    mask = df["method_name"] == ABCD_METHOD
    if not mask.any():
        return df

    model_name = embedding_model or _ABCD_EMBEDDING_MODEL
    cache_root = cache_dir or (ROOT_DIR / ".cache" / "abcd_rematch")
    cache_path = cache_root / f"{model_name.replace('/', '_')}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict] = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    df = df.copy()
    if "is_correct" in df.columns:
        df["is_correct"] = df["is_correct"].astype(object)

    # A truly empty response has no span to extract or embed at all -- the
    # paper's random fallback (Appendix F.2) applies directly and never
    # touches the embedding model. Rematching can't reproduce the exact
    # random draw made at collection time (no seed is persisted per row),
    # but any uniformly random choice is equally faithful to the paper's
    # own protocol for this near-nonexistent edge case.
    rng = random.Random(BOOTSTRAP_SEED)

    row_info = []
    for i in df.index[mask]:
        row = df.loc[i]
        options = {
            letter: row[col]
            for letter, col in (("A", "choice_a"), ("B", "choice_b"), ("C", "choice_c"), ("D", "choice_d"))
            if pd.notna(row[col]) and str(row[col]).strip() != ""
        }
        normalized = normalize_output_text(row.get("free_text_response"))

        if not normalized:
            letter = rng.choice(list(options.keys()))
            parse_result = ParseResult(
                final_choice=letter,
                status=PARSE_OK,
                raw_text=None,
                normalized_text=normalized,
                reason="Random fallback: model produced no answer text",
            )
            score_result = score_prediction(parse_result, row["correct_option"])
            df.loc[i, "parsed_choice"] = parse_result.final_choice
            df.loc[i, "parse_status"] = parse_result.status
            df.loc[i, "normalized_text"] = parse_result.normalized_text
            df.loc[i, "parse_reason"] = parse_result.reason
            df.loc[i, "is_correct"] = score_result.is_correct
            df.loc[i, "score_status"] = score_result.status
            df.loc[i, "best_similarity_score"] = None
            continue

        span = extract_candidate_span(normalized, options)
        key = _stage2_rematch_cache_key(span, options)
        row_info.append((i, span, options, key))

    misses = [info for info in row_info if info[3] not in cache]
    if misses:
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(4)
        model = SentenceTransformer(model_name)

        _ROWS_PER_CHUNK = 80
        for chunk_start in range(0, len(misses), _ROWS_PER_CHUNK):
            chunk_rows = misses[chunk_start : chunk_start + _ROWS_PER_CHUNK]

            chunk_texts: list[str] = []
            chunk_slices: list[tuple[int, list[str]]] = []
            for _, span, options, _ in chunk_rows:
                letters = list(options.keys())
                start = len(chunk_texts)
                chunk_texts.append(span)
                chunk_texts.extend(options[letter] for letter in letters)
                chunk_slices.append((start, letters))

            embeddings = model.encode(
                chunk_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=True,
            )

            for (_, span, options, key), (start, letters) in zip(chunk_rows, chunk_slices):
                ft_emb = embeddings[start]
                opt_embs = embeddings[start + 1 : start + 1 + len(letters)]
                raw_sims = opt_embs @ ft_emb
                scores = {letter: float(raw_sims[idx]) for idx, letter in enumerate(letters)}
                best_letter = max(scores, key=scores.__getitem__)
                best_score = scores[best_letter]
                cache[key] = {"best_letter": best_letter, "best_score": best_score, "normalized_text": span}

            cache_path.write_text(json.dumps(cache))

    for i, _, _, key in row_info:
        cached = cache[key]
        best_letter = cached["best_letter"]
        best_score = cached["best_score"]
        norm_text = cached["normalized_text"]

        parse_result = ParseResult(
            final_choice=best_letter,
            status=PARSE_OK,
            raw_text=None,
            normalized_text=norm_text,
            reason=f"Embedding cosine match to option {best_letter} (score={best_score:.3f})",
        )
        score_result = score_prediction(parse_result, df.loc[i, "correct_option"])

        df.loc[i, "parsed_choice"] = parse_result.final_choice
        df.loc[i, "parse_status"] = parse_result.status
        df.loc[i, "normalized_text"] = parse_result.normalized_text
        df.loc[i, "parse_reason"] = parse_result.reason
        df.loc[i, "is_correct"] = score_result.is_correct
        df.loc[i, "score_status"] = score_result.status
        df.loc[i, "best_similarity_score"] = best_score

    return df


def rematch_text_extraction_rows(
    df: pd.DataFrame,
    similarity_threshold: float = 0.1,
    embedding_model: str | None = None,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Re-run text_extraction's stage-2 resolution against already-saved free text.

    Re-derives parsed_choice/is_correct from the saved free_text_response
    column for every text_extraction row. Thin wrapper around
    _rematch_with_embedding_fallback; see that function for the
    shortcut/caching/batching details.

    No-op (returns df unchanged) when no text_extraction rows are present.

    ``similarity_threshold`` defaults to TextExtractionRunner's production
    default (0.1); ``embedding_model`` defaults to the production model
    (all-MiniLM-L6-v2).
    """
    from modelgen.runners.text_extraction import _DEFAULT_EMBEDDING_MODEL

    return _rematch_with_embedding_fallback(
        df,
        mask=df["method_name"] == TEXT_EXTRACTION_METHOD,
        text_column="free_text_response",
        default_embedding_model=_DEFAULT_EMBEDDING_MODEL,
        similarity_threshold=similarity_threshold,
        cache_subdir="text_extraction_rematch",
        embedding_model=embedding_model,
        cache_dir=cache_dir,
    )


def rematch_additional_option_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Re-run additional_option's LEGACY Jaccard text match against
    already-saved raw_text, for rows collected before the Eq.(6) redesign.

    additional_option was redesigned to score every option (including IDK)
    via score_options() and select via Eq.(6) (argmax restricted to the
    real options, excluding IDK — see match_options_via_scoring in
    runners/additional_option.py), with no free-text generation at all.
    Rows collected under that design have no raw_text and are already
    final at collection time; this function only re-derives rows from the
    earlier free-text-generation + Jaccard-matching era (identified by
    raw_text being genuinely present), so it never overwrites a redesigned
    row's already-correct parsed_choice with a Jaccard-derived value
    computed from nothing (raw_text=None would otherwise read as an empty
    response and silently produce PARSE_MISSING).

    No-op (returns df unchanged) when no additional_option rows with
    raw_text are present.
    """
    mask = df["method_name"] == ADDITIONAL_OPTION_METHOD
    if "raw_text" not in df.columns:
        return df
    mask = mask & df["raw_text"].notna()
    if not mask.any():
        return df

    from modelgen.runners.additional_option import match_text_to_options_jaccard

    df = df.copy()
    if "is_correct" in df.columns:
        df["is_correct"] = df["is_correct"].astype(object)

    for i in df.index[mask]:
        row = df.loc[i]
        options = {
            letter: row[col]
            for letter, col in (("A", "choice_a"), ("B", "choice_b"), ("C", "choice_c"), ("D", "choice_d"))
            if pd.notna(row[col]) and str(row[col]).strip() != ""
        }
        options["E"] = "I don't know"

        parse_result, best_score = match_text_to_options_jaccard(row.get("raw_text"), options)
        score_result = score_prediction(parse_result, row["correct_option"])

        df.loc[i, "parsed_choice"] = parse_result.final_choice
        df.loc[i, "parse_status"] = parse_result.status
        df.loc[i, "normalized_text"] = parse_result.normalized_text
        df.loc[i, "parse_reason"] = parse_result.reason
        df.loc[i, "is_correct"] = score_result.is_correct
        df.loc[i, "score_status"] = score_result.status
        df.loc[i, "best_similarity_score"] = best_score

    return df


_FALLBACK_METHODS = {TWOPROMPT_METHOD, TWOPROMPT_CYCLIC_METHOD}


def apply_baseline_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """For unscorable two_prompt / two_prompt_cyclic rows, substitute baseline results.

    A row is eligible for fallback when:
    - its method is two_prompt or two_prompt_cyclic
    - model_status != "failure" (the API call succeeded)
    - is_correct is NaN (parsing produced no scorable answer)

    For each eligible row the corresponding baseline row for the same
    (model_name, question_id) is looked up and its is_correct / parsed_choice /
    parse_status values are copied in.  A boolean column ``fallback_applied``
    is added to the DataFrame so downstream aggregations can count fallbacks.
    """
    df = df.copy()
    df["fallback_applied"] = False

    for model in df["model_name"].dropna().unique():
        baseline_df = df[
            (df["model_name"] == model) & (df["method_name"] == BASELINE_METHOD)
        ]
        if baseline_df.empty:
            continue

        baseline_idx = baseline_df.set_index("question_id")

        for method in _FALLBACK_METHODS:
            eligible_mask = (
                (df["model_name"] == model)
                & (df["method_name"] == method)
                & (df["model_status"].fillna("") != "failure")
                & (df["is_correct"].isna())
            )
            eligible_qids = df.loc[eligible_mask, "question_id"]
            available_qids = eligible_qids[eligible_qids.isin(baseline_idx.index)]

            if available_qids.empty:
                continue

            for qid in available_qids:
                bl = baseline_idx.loc[qid]
                row_mask = eligible_mask & (df["question_id"] == qid)
                df.loc[row_mask, "is_correct"] = bl["is_correct"]
                df.loc[row_mask, "parsed_choice"] = bl["parsed_choice"]
                if "parse_status" in df.columns and "parse_status" in baseline_idx.columns:
                    df.loc[row_mask, "parse_status"] = bl["parse_status"]
                df.loc[row_mask, "fallback_applied"] = True

    return df


def _apply_display_order(df: pd.DataFrame) -> pd.DataFrame:
    """Apply consistent method/model ordering when those columns exist."""
    if "method" in df.columns:
        df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    if "model" in df.columns:
        df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    return df


# Validation


def validate_run(df: pd.DataFrame, run_dir: Path) -> None:
    """Run basic integrity checks on the loaded data."""
    errors = []

    required_cols = [
        "question_id",
        "split_name",
        "method_name",
        "model_name",
        "correct_option",
        "model_status",
        "parsed_choice",
        "is_correct",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    splits = df["split_name"].dropna().unique()
    if len(splits) != 1:
        errors.append(f"Expected 1 split, found {len(splits)}: {splits}")

    dupes = df.groupby(["question_id", "method_name", "model_name"]).size()
    dupes = dupes[dupes > 1]
    if len(dupes) > 0:
        errors.append(f"Found {len(dupes)} duplicate (question, method, model) combinations")

    found_methods = set(df["method_name"].dropna().unique())
    found_models = set(df["model_name"].dropna().unique())
    print(f"[validate] Methods found: {sorted(found_methods)}")
    print(f"[validate] Models found: {sorted(found_models)}")

    if "benchmark" in df.columns:
        benchmarks = sorted(df["benchmark"].dropna().unique())
        print(f"[validate] Benchmarks found: {benchmarks}")
        if len(benchmarks) > 1:
            errors.append(
                f"Multiple benchmarks in one evaluation: {benchmarks}. "
                "Re-run with --benchmark to filter to one benchmark at a time."
            )

    counts = (
        df.groupby(["method_name", "model_name"])
        .size()
        .rename("n_rows")
        .reset_index()
        .sort_values(["method_name", "model_name"])
    )
    if not counts.empty:
        unique_counts = sorted(counts["n_rows"].unique())
        print(f"[validate] Present condition row counts: {unique_counts}")
        if len(unique_counts) > 1:
            print("[validate] WARNING: present conditions do not all have the same row count.")

    for method in METHOD_ORDER:
        if method not in found_methods:
            continue  # method completely absent from this run — not an error
        for model in MODEL_ORDER:
            subset = df[(df["method_name"] == method) & (df["model_name"] == model)]
            if subset.empty:
                print(f"[validate] WARNING: missing condition {method}/{model}")

    if errors:
        for e in errors:
            print(f"[validate] ERROR: {e}")
        sys.exit(1)

    print("[validate] All checks passed.")


# Core accuracy


def compute_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy and failure accounting per method x model.

    Definitions:
    - end_to_end_accuracy = correct / total
    - conditional_accuracy = correct / scored
    - api_failures = rows where model_status == "failure"
    - parse_failures = among non-provider-failure rows, parsed_choice is missing
    - final_unscorable = rows where is_correct is missing
    """
    rows = []

    for method in METHOD_ORDER:
        for model in MODEL_ORDER:
            group = df[(df["method_name"] == method) & (df["model_name"] == model)]
            if group.empty:
                continue

            total = len(group)

            provider_failure_mask = group["model_status"].fillna("") == "failure"
            nonprovider_mask = ~provider_failure_mask

            api_failures = int(provider_failure_mask.sum())
            api_successes = int(nonprovider_mask.sum())

            parsed_total = int(group["parsed_choice"].notna().sum())
            parsed_nonprovider = int((nonprovider_mask & group["parsed_choice"].notna()).sum())
            parse_failures = int((nonprovider_mask & group["parsed_choice"].isna()).sum())

            scored = int(group["is_correct"].notna().sum())
            correct = int(group["is_correct"].eq(True).sum())
            final_unscorable = int(group["is_correct"].isna().sum())

            e2e_ci_lo, e2e_ci_hi = _clopper_pearson_ci(correct, total)
            cond_ci_lo, cond_ci_hi = _clopper_pearson_ci(correct, scored)

            fallback_count = (
                int(group["fallback_applied"].eq(True).sum())
                if "fallback_applied" in group.columns
                else 0
            )
            runtime_fallback_count = (
                int(group["fallback_used"].eq(True).sum())
                if "fallback_used" in group.columns
                else 0
            )

            rows.append(
                {
                    "method": method,
                    "model": model,
                    "total": total,
                    "api_failures": api_failures,
                    "api_successes": api_successes,
                    "parsed_total": parsed_total,
                    "parsed_nonprovider": parsed_nonprovider,
                    "parse_failures": parse_failures,
                    "scored": scored,
                    "correct": correct,
                    "final_unscorable": final_unscorable,
                    "fallback_count": fallback_count,
                    "runtime_fallback_count": runtime_fallback_count,
                    "end_to_end_accuracy": correct / total if total > 0 else 0.0,
                    "end_to_end_accuracy_ci_lower": e2e_ci_lo,
                    "end_to_end_accuracy_ci_upper": e2e_ci_hi,
                    "conditional_accuracy": correct / scored if scored > 0 else 0.0,
                    "conditional_accuracy_ci_lower": cond_ci_lo,
                    "conditional_accuracy_ci_upper": cond_ci_hi,
                    "api_failure_rate": api_failures / total if total > 0 else 0.0,
                    "parse_success_rate_nonprovider": (
                        parsed_nonprovider / api_successes if api_successes > 0 else 0.0
                    ),
                    "final_unscorable_rate": final_unscorable / total if total > 0 else 0.0,
                }
            )

    result = pd.DataFrame(rows)
    result = _apply_display_order(result)
    return result.sort_values(["method", "model"]).reset_index(drop=True)


# Positional bias


def compute_positional_bias(df: pd.DataFrame) -> pd.DataFrame:
    """Prediction distribution and deviation from ground truth per method x model."""
    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    for method in METHOD_ORDER:
        for model in MODEL_ORDER:
            group = df[(df["method_name"] == method) & (df["model_name"] == model)]
            if group.empty:
                continue

            scored = group[group["parsed_choice"].notna()]
            total_scored = len(scored)
            if total_scored == 0:
                continue

            gt_counts = group["correct_option"].value_counts()
            pred_counts = scored["parsed_choice"].value_counts()
            gt_total = len(group)

            deviations = []
            row = {"method": method, "model": model, "n_scored": total_scored}

            for opt in OPTIONS:
                gt_pct = gt_counts.get(opt, 0) / gt_total * 100
                pred_pct = pred_counts.get(opt, 0) / total_scored * 100
                deviation = pred_pct - gt_pct

                row[f"gt_{opt}"] = gt_counts.get(opt, 0)
                row[f"gt_{opt}_pct"] = gt_pct
                row[f"pred_{opt}"] = pred_counts.get(opt, 0)
                row[f"pred_{opt}_pct"] = pred_pct
                row[f"dev_{opt}"] = deviation
                deviations.append(abs(deviation))

            row["mean_abs_deviation"] = sum(deviations) / len(deviations)

            mad_ci_lo, mad_ci_hi = _bootstrap_ci_mean_abs_deviation(group, rng)
            row["mean_abs_deviation_ci_lower"] = mad_ci_lo
            row["mean_abs_deviation_ci_upper"] = mad_ci_hi

            rows.append(row)

    result = pd.DataFrame(rows)
    result = _apply_display_order(result)
    return result.sort_values(["method", "model"]).reset_index(drop=True)


# Question-level overlap


def compute_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """Compare baseline vs each method at question level per model."""
    rows = []

    for model in MODEL_ORDER:
        model_group = df[df["model_name"] == model]
        baseline = model_group[model_group["method_name"] == BASELINE_METHOD]
        if baseline.empty:
            continue

        bl = baseline[["question_id", "is_correct", "parsed_choice"]].rename(
            columns={"is_correct": "bl_correct", "parsed_choice": "bl_choice"}
        )

        for method in METHOD_ORDER:
            if method == BASELINE_METHOD:
                continue

            method_df = model_group[model_group["method_name"] == method]
            if method_df.empty:
                continue

            mt = method_df[["question_id", "is_correct", "parsed_choice"]].rename(
                columns={"is_correct": "mt_correct", "parsed_choice": "mt_choice"}
            )

            merged = bl.merge(mt, on="question_id", how="inner")
            both_scored = merged[merged["bl_correct"].notna() & merged["mt_correct"].notna()]

            if both_scored.empty:
                continue

            both_correct = int(
                ((both_scored["bl_correct"] == True) & (both_scored["mt_correct"] == True)).sum()
            )
            both_wrong = int(
                ((both_scored["bl_correct"] == False) & (both_scored["mt_correct"] == False)).sum()
            )
            bl_only = int(
                ((both_scored["bl_correct"] == True) & (both_scored["mt_correct"] == False)).sum()
            )
            mt_only = int(
                ((both_scored["bl_correct"] == False) & (both_scored["mt_correct"] == True)).sum()
            )

            rows.append(
                {
                    "model": model,
                    "method": method,
                    "n_compared": int(len(both_scored)),
                    "both_correct": both_correct,
                    "both_wrong": both_wrong,
                    "baseline_only_correct": bl_only,
                    "method_only_correct": mt_only,
                    "net_effect": mt_only - bl_only,
                }
            )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = _apply_display_order(result)
        result = result.sort_values(["model", "method"])
    return result.reset_index(drop=True)


# Choice shift analysis


def compute_choice_shifts(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze how choices shift between baseline and each method."""
    rows = []

    for model in MODEL_ORDER:
        model_group = df[df["model_name"] == model]
        baseline = model_group[model_group["method_name"] == BASELINE_METHOD]
        if baseline.empty:
            continue

        bl = baseline[["question_id", "is_correct", "parsed_choice"]].rename(
            columns={"is_correct": "bl_correct", "parsed_choice": "bl_choice"}
        )

        for method in METHOD_ORDER:
            if method == BASELINE_METHOD:
                continue

            method_df = model_group[model_group["method_name"] == method]
            if method_df.empty:
                continue

            mt = method_df[["question_id", "is_correct", "parsed_choice"]].rename(
                columns={"is_correct": "mt_correct", "parsed_choice": "mt_choice"}
            )

            merged = bl.merge(mt, on="question_id", how="inner")

            broken = merged[(merged["bl_correct"] == True) & (merged["mt_correct"] == False)]
            for (from_c, to_c), count in broken.groupby(["bl_choice", "mt_choice"]).size().items():
                rows.append(
                    {
                        "model": model,
                        "method": method,
                        "direction": "broken",
                        "from_choice": from_c,
                        "to_choice": to_c,
                        "count": int(count),
                    }
                )

            fixed = merged[(merged["bl_correct"] == False) & (merged["mt_correct"] == True)]
            for (from_c, to_c), count in fixed.groupby(["bl_choice", "mt_choice"]).size().items():
                rows.append(
                    {
                        "model": model,
                        "method": method,
                        "direction": "fixed",
                        "from_choice": from_c,
                        "to_choice": to_c,
                        "count": int(count),
                    }
                )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = _apply_display_order(result)
        result = result.sort_values(["model", "method", "direction", "count"], ascending=[True, True, True, False])
    return result.reset_index(drop=True)


# Per-subject accuracy


def compute_subject_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per subject x method x model."""
    rows = []

    for (subject, method, model), group in df.groupby(["subject", "method_name", "model_name"]):
        total = len(group)
        scored = int(group["is_correct"].notna().sum())
        correct = int(group["is_correct"].eq(True).sum())

        rows.append(
            {
                "subject": subject,
                "method": method,
                "model": model,
                "total": total,
                "scored": scored,
                "correct": correct,
                "end_to_end_accuracy": correct / total if total > 0 else 0.0,
                "conditional_accuracy": correct / scored if scored > 0 else 0.0,
            }
        )

    result = pd.DataFrame(rows)
    result = _apply_display_order(result)
    return result.sort_values(["subject", "method", "model"]).reset_index(drop=True)


# Two-stage specific


def compute_two_stage_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Metrics specific to two-stage methods."""
    rows = []
    two_stage_methods = [TWOPROMPT_METHOD, TWOPROMPT_CYCLIC_METHOD]

    for method in METHOD_ORDER:
        for model in MODEL_ORDER:
            if method not in two_stage_methods:
                continue

            group = df[(df["method_name"] == method) & (df["model_name"] == model)]
            if group.empty:
                continue
            if "free_text_response" not in group.columns:
                continue

            total = len(group)
            has_free_text = int(group["free_text_response"].notna().sum())

            mean_ft_latency = None
            if "free_text_latency" in group.columns:
                ft_latencies = group["free_text_latency"].dropna()
                mean_ft_latency = ft_latencies.mean() if len(ft_latencies) > 0 else None

            runtime_fallbacks = (
                int(group["fallback_used"].eq(True).sum())
                if "fallback_used" in group.columns
                else 0
            )

            rows.append(
                {
                    "method": method,
                    "model": model,
                    "total": total,
                    "free_text_available": has_free_text,
                    "free_text_rate": has_free_text / total if total > 0 else 0.0,
                    "mean_free_text_latency": mean_ft_latency,
                    "runtime_fallback_count": runtime_fallbacks,
                    "runtime_fallback_rate": runtime_fallbacks / total if total > 0 else 0.0,
                }
            )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = _apply_display_order(result)
        result = result.sort_values(["method", "model"]).reset_index(drop=True)
    return result


# Main


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one experiment run.")
    parser.add_argument("run_id", help="Run ID (folder name under runs/)")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config (default: use built-in path constants)",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Filter to a single benchmark (e.g. mmlu, arc_challenge). "
             "Required when a run contains multiple benchmarks.",
    )
    parser.add_argument(
        "--apply-fallback",
        action="store_true",
        default=False,
        help=(
            "For two_prompt / two_prompt_cyclic rows that are still unscorable after "
            "reparsing, substitute the baseline result for that question_id. "
            "Adds a fallback_count column to accuracy.csv."
        ),
    )
    args = parser.parse_args()

    if args.config is not None:
        cfg = yaml.safe_load(args.config.read_text())
        runs_dir = _ROOT / cfg["paths"]["runs_dir"]
        reports_dir = _ROOT / cfg["paths"]["reports_dir"]
    else:
        runs_dir = RUNS_DIR
        reports_dir = REPORTS_DIR

    run_dir = runs_dir / args.run_id
    report_dir = reports_dir / args.run_id
    _BENCHMARK_ALIASES = {"arc": "arc_challenge"}
    if args.benchmark:
        args.benchmark = _BENCHMARK_ALIASES.get(args.benchmark, args.benchmark)
        report_dir = report_dir / args.benchmark
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] Loading run {args.run_id}...")
    df = load_run(run_dir)
    print(f"[eval] {len(df)} total rows loaded")

    if args.benchmark:
        if "benchmark" in df.columns:
            df = df[df["benchmark"] == args.benchmark].reset_index(drop=True)
            print(f"[eval] Filtered to benchmark={args.benchmark!r}: {len(df)} rows")
        else:
            print(
                f"[eval] WARNING: no 'benchmark' column in data; "
                "--benchmark filter has no effect (older run format)"
            )

    print("[eval] Re-parsing responses with current parser...")
    df = reparse_run(df)

    if (df["method_name"] == ABCD_METHOD).any():
        print("[eval] Re-matching abcd responses with current embedding config...")
        df = rematch_abcd_rows(df)

    if (df["method_name"] == TEXT_EXTRACTION_METHOD).any():
        print("[eval] Re-matching text_extraction responses with current embedding config...")
        df = rematch_text_extraction_rows(df)

    if (df["method_name"] == ADDITIONAL_OPTION_METHOD).any():
        print("[eval] Re-matching additional_option responses with current Jaccard config...")
        df = rematch_additional_option_rows(df)

    if args.apply_fallback:
        print("[eval] Applying baseline fallback for unscorable two-stage rows...")
        df = apply_baseline_fallback(df)
        n_fallbacks = int(df["fallback_applied"].eq(True).sum())
        print(f"[eval]   {n_fallbacks} rows substituted from baseline")

    print("\n[eval] Validating run data...")
    validate_run(df, run_dir)

    print("\n[eval] Computing accuracy...")
    accuracy = compute_accuracy(df)
    accuracy.to_csv(report_dir / "accuracy.csv", index=False)
    accuracy_display_cols = [
        "method",
        "model",
        "total",
        "correct",
        "scored",
        "final_unscorable",
        "fallback_count",
        "runtime_fallback_count",
        "end_to_end_accuracy",
        "end_to_end_accuracy_ci_lower",
        "end_to_end_accuracy_ci_upper",
        "conditional_accuracy",
        "conditional_accuracy_ci_lower",
        "conditional_accuracy_ci_upper",
        "api_failures",
        "parse_failures",
    ]
    print(accuracy[accuracy_display_cols].to_string(index=False))

    print("\n[eval] Computing positional bias...")
    bias = compute_positional_bias(df)
    bias.to_csv(report_dir / "positional_bias.csv", index=False)
    print(bias[["method", "model", "n_scored", "mean_abs_deviation", "mean_abs_deviation_ci_lower", "mean_abs_deviation_ci_upper"]].to_string(index=False))

    print("\n[eval] Computing question-level overlap...")
    overlap = compute_overlap(df)
    overlap.to_csv(report_dir / "overlap.csv", index=False)
    if not overlap.empty:
        print(overlap.to_string(index=False))

    print("\n[eval] Computing choice shifts...")
    shifts = compute_choice_shifts(df)
    shifts.to_csv(report_dir / "choice_shifts.csv", index=False)

    print("\n[eval] Computing per-subject accuracy...")
    subject_acc = compute_subject_accuracy(df)
    subject_acc.to_csv(report_dir / "subject_accuracy.csv", index=False)

    print("\n[eval] Computing two-stage metrics...")
    two_stage = compute_two_stage_metrics(df)
    if not two_stage.empty:
        two_stage.to_csv(report_dir / "two_stage_metrics.csv", index=False)
        print(two_stage.to_string(index=False))

    print(f"\n[complete] Reports saved to {report_dir}/")


if __name__ == "__main__":
    main()
# src/modelgen/runners/abcd_extraction.py

"""Stage-2 answer extraction specific to the abcd condition.

Implements the four-tier extraction cascade and prompt modification of
Nowak, Cadet, and Chin, "ABCD: All Biases Come Disguised" (Dartmouth),
arXiv:2602.17445, Appendix F.2 (regular expressions) and Appendix
F.3/Figure 14 (prompt modification). text_extraction.py's separate
A/B/C/D-labeled condition is not part of this paper's protocol and keeps
its own, different (earliest-statement) extraction logic untouched.
"""

import random
import re

from sentence_transformers import SentenceTransformer

from modelgen.parsing.parser import normalize_output_text
from modelgen.parsing.types import PARSE_OK, ParseResult
from modelgen.runners.text_extraction import match_free_text_to_options

# Appendix F.2, Expression #1, printed verbatim: "answer is (?!.*answer is ).+"
# Mirrors the MMLU-Pro evaluation code's "answer is \(?([A-J])\)?" pattern.
# The negative lookahead is what gives "last occurrence" semantics: a match
# is only permitted at a position with no further "answer is " ahead of it.
_TIER1_RE = re.compile(r"answer is (?!.*answer is ).+", re.DOTALL)

# Appendix F.2, Expression #2, printed verbatim:
# ".*[aA]nswer:\s*(?!.*[aA]nswer:\s*).+"
# Adapted from MMLU-Pro's ".*[aA]nswer:\s*([A-J])" to tolerate formatting
# deviations in the answer prefix. Case-insensitivity is only on the leading
# letter ([aA]), exactly as printed -- not the whole expression.
_TIER2_RE = re.compile(r".*[aA]nswer:\s*(?!.*[aA]nswer:\s*).+", re.DOTALL)

# Appendix F.2, Expression #4, printed verbatim: "([^.!?]+[.!?]*)$"
# "As a final fallback, we extract the last sentence of the model's output."
_TIER4_RE = re.compile(r"([^.!?]+[.!?]*)$", re.DOTALL)


def _tier3_literal_option_search(text: str, options: dict[str, str]) -> str | None:
    """Appendix F.2, Expression #3: "a regular expression verbosely searching
    each answer within the model's output as is."

    Unlike tiers #1, #2, and #4 -- each given as a literal regex in the
    paper -- tier #3 is described only in prose, by contrast with the
    MMLU-Pro analog of searching for any standalone A-J letter
    (\\b[A-J]\\b(?!.*\\b[A-J]\\b)). Read literally, "each answer" means each
    candidate option, and "as is" means an unmodified, literal substring
    search for that option's exact text -- the generalization of "search
    for any of the valid letters" to a setting with no letters. This is the
    most internally consistent reading available: it is the only one that
    explains Table 3's nonzero failure rate for this tier (a pure substring
    search can fail when the model paraphrases instead of repeating an
    option verbatim, whereas a degenerate "use the whole output" reading
    could never fail on non-empty text and tier #4 would never fire).

    Each option is searched with the same \\b word-boundary anchoring the
    paper's own MMLU-Pro-derived letter pattern uses, not a bare substring
    search -- without it, an option like "HTTP" would falsely match inside
    the unrelated option text "HTTPS" (a real collision in this codebase's
    own option set), which \\b correctly rejects since "P" to "S" is a
    word-to-word transition with no boundary.

    Returns the option text of the rightmost (last) literal occurrence among
    all candidate options, consistent with this section's last-occurrence
    convention for every other tier, or None if no option text appears
    verbatim anywhere in the output.
    """
    best_pos = -1
    best_text: str | None = None
    for option_text in options.values():
        pattern = re.compile(r"\b" + re.escape(option_text) + r"\b")
        match = None
        for match in pattern.finditer(text):
            pass
        if match is not None and match.start() > best_pos:
            best_pos = match.start()
            best_text = option_text
    return best_text


def extract_candidate_span(raw_text: str, options: dict[str, str]) -> str:
    """Run the Appendix F.2 four-tier cascade, in the paper's stated order.

    Tried in order: explicit "answer is" marker, "Answer:" marker, literal
    option-text search, last-sentence fallback. Tier #4 cannot fail to match
    non-empty text, so this always returns a non-empty string for non-empty
    raw_text -- the only true failure case (raw_text itself empty) is
    handled by the caller, before this function is reached.
    """
    match = _TIER1_RE.search(raw_text)
    if match:
        return match.group(0)
    match = _TIER2_RE.search(raw_text)
    if match:
        return match.group(0)
    tier3 = _tier3_literal_option_search(raw_text, options)
    if tier3 is not None:
        return tier3
    match = _TIER4_RE.search(raw_text)
    return match.group(0) if match else raw_text


def resolve_abcd_answer(
    raw_text: str | None,
    options: dict[str, str],
    similarity_threshold: float,
    model: SentenceTransformer,
    rng: random.Random | None = None,
) -> tuple[ParseResult, float | None]:
    """Paper-faithful stage-2 resolution for the abcd condition.

    Isolates a candidate answer span via extract_candidate_span (Appendix
    F.2), then reuses text_extraction.py's embedding-similarity matcher
    (match_free_text_to_options) to select the best-matching option -- the
    "regex extraction, then sentence-similarity matching" pipeline described
    in Section 4 and Appendix F.3 of arXiv:2602.17445.

    On true extraction failure (the model produced no text at all), falls
    back to a uniformly random option, matching the paper's documented
    behavior: "If the model does not produce an answer, we choose a random
    answer." (Appendix F.2). The paper reports this occurring only once
    across all permutations and all models on MMLU-Pro -- a near-nonexistent
    edge case, not a tuned design choice.
    """
    normalized = normalize_output_text(raw_text)
    if not normalized:
        rng = rng or random.Random()
        letter = rng.choice(list(options.keys()))
        return ParseResult(
            final_choice=letter,
            status=PARSE_OK,
            raw_text=raw_text,
            normalized_text=normalized,
            reason="Random fallback: model produced no answer text",
        ), None

    span = extract_candidate_span(normalized, options)
    return match_free_text_to_options(span, options, similarity_threshold, model)

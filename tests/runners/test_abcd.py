# tests/runners/test_abcd.py

from pathlib import Path

from modelgen.backends.dummy import DummyBackend
from modelgen.parsing.types import PARSE_MISSING
from modelgen.runners.abcd import ABCDRunner
from modelgen.scoring.types import SCORE_CORRECT, SCORE_INCORRECT, SCORE_UNSCORABLE

from tests.runners.conftest import ErrorBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


def _make_runner(backend, similarity_threshold=0.1, embedding_model="all-MiniLM-L6-v2"):
    # Tests pin the small embedding model explicitly — ABCDRunner's real
    # default is the larger Qwen3-Embedding-0.6B (matching the source paper),
    # which would make every test download/load a 600M-param model.
    return ABCDRunner(
        backend=backend,
        method_name="abcd",
        split_name="robustness",
        prompt_version="v1",
        prompts_dir=_PROMPTS_DIR,
        run_id="test_run_001",
        similarity_threshold=similarity_threshold,
        embedding_model=embedding_model,
    )


class TestABCDRunnerRunOne:
    def test_correct_answer(self, runner_question_row):
        """Free-text matches option C exactly — should score correct."""
        b = DummyBackend(fixed_text="HTTPS")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True
        assert result["score_status"] == SCORE_CORRECT

    def test_incorrect_answer(self, runner_question_row):
        """Free-text matches option A — should score incorrect."""
        b = DummyBackend(fixed_text="FTP")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] == "A"
        assert result["is_correct"] is False
        assert result["score_status"] == SCORE_INCORRECT

    def test_backend_failure_returns_no_parse_or_score(self, runner_question_row, error_backend):
        result = _make_runner(error_backend).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] is None
        assert result["is_correct"] is None
        assert result["model_status"] == "error"

    def test_low_similarity_is_unscorable(self, runner_question_row):
        b = DummyBackend(fixed_text="completely unrelated text xyz")
        b.load()
        result = _make_runner(b, similarity_threshold=0.9).run_one(
            runner_question_row, sample_index=0
        )

        assert result["parsed_choice"] is None
        assert result["score_status"] == SCORE_UNSCORABLE

    def test_free_text_response_preserved(self, runner_question_row):
        b = DummyBackend(fixed_text="HTTPS")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert result["free_text_response"] == "HTTPS"

    def test_best_similarity_score_present(self, runner_question_row):
        b = DummyBackend(fixed_text="HTTPS")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert isinstance(result["best_similarity_score"], float)
        assert result["best_similarity_score"] > 0.0

    def test_result_row_metadata(self, runner_question_row):
        b = DummyBackend(fixed_text="HTTPS")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert result["run_id"] == "test_run_001"
        assert result["method_name"] == "abcd"
        assert result["split_name"] == "robustness"

    def test_prompt_contains_all_option_texts(self, runner_question_row):
        """Stage 1 prompt must show all four option texts."""
        b = DummyBackend(fixed_text="HTTPS")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert "FTP" in result["prompt"]
        assert "HTTP" in result["prompt"]
        assert "HTTPS" in result["prompt"]
        assert "SMTP" in result["prompt"]

    def test_prompt_uses_dash_labels_not_letter_labels(self, runner_question_row):
        """Stage 1 prompt must use dash labels, not A./B./C./D. letters."""
        b = DummyBackend(fixed_text="HTTPS")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert "A." not in result["prompt"]
        assert "B." not in result["prompt"]
        assert "C." not in result["prompt"]
        assert "D." not in result["prompt"]
        assert "- FTP" in result["prompt"]
        assert "- HTTP" in result["prompt"]

    def test_no_second_llm_call(self, runner_question_row):
        """Stage 2 is deterministic — backend.generate should be called exactly once."""
        call_count = 0
        original_generate = DummyBackend.generate

        class CountingBackend(DummyBackend):
            def generate(self, prompt, config=None):
                nonlocal call_count
                call_count += 1
                return original_generate(self, prompt, config)

        b = CountingBackend(fixed_text="HTTPS")
        b.load()
        _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert call_count == 1

    def test_missing_fourth_option_has_no_phantom_choice(self, runner_question_row):
        """A 3-option question (choice_d is None/NaN) must not crash or show a phantom option."""
        row = dict(runner_question_row, choice_d=float("nan"))
        b = DummyBackend(fixed_text="HTTPS")
        b.load()
        result = _make_runner(b).run_one(row, sample_index=0)

        assert "nan" not in result["prompt"].lower()
        assert result["prompt"].count("- ") == 3
        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True

    def test_prompt_instructs_answer_is_option_verbatim(self, runner_question_row):
        """Stage 1 prompt must carry Appendix F.3/Figure 14's modification:
        "the answer is $OPTION" plus the verbatim-repetition and no-letter
        constraints, replacing the old "in your own words" instruction."""
        b = DummyBackend(fixed_text="HTTPS")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert "the answer is $OPTION" in result["prompt"]
        assert "repeated exactly" in result["prompt"]
        assert "Do not write the option letter" in result["prompt"]
        assert "in your own words" not in result["prompt"]

    def test_empty_generation_triggers_random_fallback_not_unscorable(self, runner_question_row):
        """Appendix F.2: "If the model does not produce an answer, we choose
        a random answer" -- a genuinely empty stage-1 response must resolve
        to one of the real options (and score), not come back unscorable."""
        b = DummyBackend(fixed_text="")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] in ("A", "B", "C", "D")
        assert result["score_status"] in (SCORE_CORRECT, SCORE_INCORRECT)
        assert result["best_similarity_score"] is None

    def test_hedge_and_pivot_resolves_to_last_statement_per_paper_cascade(self, runner_question_row):
        """Deliberate behavior change from this condition's pre-redesign
        earliest-statement heuristic: arXiv:2602.17445 Appendix F.2's
        regex cascade explicitly takes the *last* occurrence at every tier
        (each tier's negative lookahead exists specifically to enforce
        this), trading this repo's own empirical earliest-statement
        preference for fidelity to the cited paper's literal mechanism. A
        model that hedges and pivots to a different option after its first
        statement now resolves to whatever it pivots to last, matching the
        paper's tier #3 (literal option-text search, last occurrence)."""
        b = DummyBackend(
            fixed_text=(
                "HTTPS is the correct protocol. "
                "(Note: while FTP is also a protocol, it is not used for "
                "secure browsing.) However, on reflection, some sources "
                "suggest the answer might instead be considered SMTP."
            )
        )
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] == "D"
        assert result["is_correct"] is False

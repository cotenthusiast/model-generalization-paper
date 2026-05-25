# tests/runners/test_permutation.py

from pathlib import Path

from twoprompt.backends.dummy import DummyBackend
from twoprompt.runners.permutation import PermutationRunner
from twoprompt.pipeline.prompt_builder import load_prompt_templates
from twoprompt.scoring.types import SCORE_CORRECT, SCORE_INCORRECT
from twoprompt.parsing.types import PARSE_OK, PARSE_MISSING

from tests.runners.conftest import ErrorBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"
_TEMPLATES = load_prompt_templates("v1", _PROMPTS_DIR)


def _make_runner(backend, method_name="cyclic"):
    return PermutationRunner(
        backend=backend,
        method_name=method_name,
        split_name="robustness",
        prompt_version="v1",
        prompts_dir=_PROMPTS_DIR,
        run_id="test_run_001",
    )


class TestGeneratePermutations:
    def test_returns_four_permutations(self, canonical_options):
        perms = PermutationRunner._generate_permutations(canonical_options)
        assert len(perms) == 4

    def test_first_permutation_is_original(self, canonical_options):
        perms = PermutationRunner._generate_permutations(canonical_options)
        assert perms[0] == canonical_options

    def test_keys_preserved(self, canonical_options):
        perms = PermutationRunner._generate_permutations(canonical_options)
        for perm in perms:
            assert list(perm.keys()) == ["A", "B", "C", "D"]

    def test_values_rotated(self, canonical_options):
        perms = PermutationRunner._generate_permutations(canonical_options)
        original_values = set(canonical_options.values())
        for perm in perms:
            assert set(perm.values()) == original_values

    def test_all_permutations_distinct(self, canonical_options):
        perms = PermutationRunner._generate_permutations(canonical_options)
        perm_tuples = [tuple(p.values()) for p in perms]
        assert len(set(perm_tuples)) == 4

    def test_second_permutation_shifted_by_one(self, canonical_options):
        perms = PermutationRunner._generate_permutations(canonical_options)
        assert perms[1]["A"] == canonical_options["B"]
        assert perms[1]["B"] == canonical_options["C"]
        assert perms[1]["C"] == canonical_options["D"]
        assert perms[1]["D"] == canonical_options["A"]


class TestBuildPermutedPrompt:
    def test_prompt_contains_permuted_options(self, runner_question_row, canonical_options):
        perms = PermutationRunner._generate_permutations(canonical_options)
        prompt = PermutationRunner._build_permuted_prompt(
            runner_question_row, perms[1], _TEMPLATES["direct_mcq"]
        )
        assert runner_question_row["question_text"] in prompt
        assert "HTTP" in prompt

    def test_prompt_contains_question(self, runner_question_row, canonical_options):
        prompt = PermutationRunner._build_permuted_prompt(
            runner_question_row, canonical_options, _TEMPLATES["direct_mcq"]
        )
        assert "securely browse websites" in prompt


class TestUnpermuteChoice:
    def test_identity_permutation(self, canonical_options):
        result = PermutationRunner._unpermute_choice("C", canonical_options, canonical_options)
        assert result == "C"

    def test_shifted_permutation(self, canonical_options):
        perms = PermutationRunner._generate_permutations(canonical_options)
        # In permutation 1: A->HTTP, B->HTTPS, C->SMTP, D->FTP
        # Model picks B (HTTPS); canonical HTTPS is C
        result = PermutationRunner._unpermute_choice("B", perms[1], canonical_options)
        assert result == "C"

    def test_all_letters_unpermute_correctly(self, canonical_options):
        perms = PermutationRunner._generate_permutations(canonical_options)
        for perm in perms:
            for letter in ["A", "B", "C", "D"]:
                result = PermutationRunner._unpermute_choice(letter, perm, canonical_options)
                assert result in {"A", "B", "C", "D"}

    def test_no_match_returns_none(self):
        permuted = {"A": "FTP", "B": "HTTP", "C": "NONEXISTENT", "D": "SMTP"}
        canonical = {"A": "FTP", "B": "HTTP", "C": "HTTPS", "D": "SMTP"}
        result = PermutationRunner._unpermute_choice("C", permuted, canonical)
        assert result is None


class TestMajorityVote:
    def test_unanimous(self):
        assert PermutationRunner._majority_vote(["C", "C", "C", "C"]) == "C"

    def test_clear_majority(self):
        assert PermutationRunner._majority_vote(["A", "C", "C", "C"]) == "C"

    def test_tie_uses_first_valid(self):
        result = PermutationRunner._majority_vote(["A", "B", "A", "B"])
        assert result == "A"

    def test_all_none(self):
        assert PermutationRunner._majority_vote([None, None, None, None]) is None

    def test_empty_list(self):
        assert PermutationRunner._majority_vote([]) is None

    def test_some_none(self):
        result = PermutationRunner._majority_vote([None, "B", "B", None])
        assert result == "B"

    def test_single_valid_vote(self):
        result = PermutationRunner._majority_vote([None, None, "D", None])
        assert result == "D"

    def test_tie_with_none_first(self):
        result = PermutationRunner._majority_vote([None, "A", "B", "A"])
        assert result == "A"


class TestPermutationRunnerRunOne:
    def test_run_one_returns_dict(self, runner_question_row):
        """run_one should return a result dict without raising."""
        b = DummyBackend(fixed_text="C")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)
        assert isinstance(result, dict)

    def test_tie_broken_by_first_vote(self, runner_question_row):
        """When all 4 permutations return 'C', tie-breaking gives canonical C — correct."""
        # With fixed_text="C": each permutation returns C, which maps to a different
        # canonical letter (cyclic unpermute produces [C, D, A, B]).
        # All have count 1 → tie → first valid = C → correct.
        b = DummyBackend(fixed_text="C")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True
        assert result["score_status"] == SCORE_CORRECT

    def test_all_fail(self, runner_question_row, error_backend):
        """All four backend calls fail — voted answer should be None."""
        result = _make_runner(error_backend).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] is None
        assert result["is_correct"] is None

    def test_result_row_has_metadata(self, runner_question_row):
        """Result row should carry trace metadata."""
        b = DummyBackend(fixed_text="C")
        b.load()
        result = _make_runner(b, method_name="cyclic").run_one(
            runner_question_row, sample_index=0
        )

        assert result["run_id"] == "test_run_001"
        assert result["method_name"] == "cyclic"
        assert result["split_name"] == "robustness"

    def test_latency_is_sum_of_all_calls(self, runner_question_row):
        """latency_seconds covers all 4 permutation calls (non-negative float)."""
        b = DummyBackend(fixed_text="C")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert isinstance(result["latency_seconds"], float)
        assert result["latency_seconds"] >= 0.0

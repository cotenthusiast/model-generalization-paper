# tests/runners/test_permutation.py

import json
from pathlib import Path

import pytest

from modelgen.backends.dummy import DummyBackend
from modelgen.backends.types import ModelOptionScoreResult
from modelgen.runners.permutation import PermutationRunner
from modelgen.pipeline.prompt_builder import load_prompt_templates
from modelgen.scoring.types import SCORE_CORRECT, SCORE_INCORRECT
from modelgen.parsing.types import PARSE_OK, PARSE_MISSING

from tests.runners.conftest import ErrorScoreBackend


class ContentAwareScoreBackend(DummyBackend):
    """Test backend with zero token/position bias: scores whichever letter
    is currently labeled with `target_text` highest, regardless of which
    letter that is. Used to verify Eq.(1)'s un-permutation actually tracks
    content across rotations rather than just trusting letter position."""

    def __init__(self, target_text: str, **kwargs):
        super().__init__(**kwargs)
        self._target_text = target_text

    def score_options(self, prompt, options):
        if not self._loaded:
            raise RuntimeError("Call load() before score_options().")
        scores = {}
        for opt in options:
            marker = f"{opt}. "
            idx = prompt.find(marker)
            if idx == -1:
                scores[opt] = -5.0
                continue
            line_end = prompt.find("\n", idx)
            line = prompt[idx + len(marker) : line_end if line_end != -1 else len(prompt)]
            scores[opt] = -0.1 if line.strip() == self._target_text else -3.0
        return ModelOptionScoreResult(scores=scores, raw_logprobs=dict(scores), metadata={})


class TokenBiasedScoreBackend(DummyBackend):
    """Test backend with pure token/position bias and zero content
    knowledge: always scores the same letter highest no matter what content
    sits under it. Used to verify Eq.(1) cancels out a permutation-invariant
    bias to a uniform distribution over canonical content."""

    def __init__(self, biased_letter: str, **kwargs):
        super().__init__(**kwargs)
        self._biased_letter = biased_letter

    def score_options(self, prompt, options):
        if not self._loaded:
            raise RuntimeError("Call load() before score_options().")
        scores = {opt: (-0.1 if opt == self._biased_letter else -3.0) for opt in options}
        return ModelOptionScoreResult(scores=scores, raw_logprobs=dict(scores), metadata={})

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
        b = ContentAwareScoreBackend(target_text="HTTPS")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)
        assert isinstance(result, dict)

    def test_content_aware_backend_always_finds_correct_answer(self, runner_question_row):
        """A backend with zero token bias (scores whichever letter HTTPS sits
        under, regardless of position) must resolve to C (HTTPS) under Eq.(1)
        un-permutation + averaging — the un-permutation must correctly track
        content across all 4 rotations, not just trust letter position."""
        b = ContentAwareScoreBackend(target_text="HTTPS")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True
        assert result["score_status"] == SCORE_CORRECT
        assert result["parse_reason"] == "cyclic_eq1_avg"

    def test_pure_token_bias_debiased_to_uniform(self, runner_question_row):
        """A backend with pure token/position bias (always favors letter A
        regardless of content) must be debiased by Eq.(1)'s cyclic averaging
        to a perfectly uniform distribution over canonical content -- this
        is the entire theoretical point of cyclic-permutation debiasing.
        Verified directly against the row's stored probabilities, not just
        the final argmax (which would tie-break arbitrarily on a uniform
        distribution)."""
        b = TokenBiasedScoreBackend(biased_letter="A")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        probs = json.loads(result["cyclic_probs_json"])
        values = list(probs.values())
        assert len(values) == 4
        for v in values:
            assert v == pytest.approx(0.25, abs=1e-9)

    def test_all_fail(self, runner_question_row):
        """All four score_options() calls fail — parsed answer should be None."""
        b = ErrorScoreBackend()
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] is None
        assert result["is_correct"] is None
        assert result["cyclic_probs_json"] is None

    def test_result_row_has_metadata(self, runner_question_row):
        """Result row should carry trace metadata."""
        b = ContentAwareScoreBackend(target_text="HTTPS")
        b.load()
        result = _make_runner(b, method_name="cyclic").run_one(
            runner_question_row, sample_index=0
        )

        assert result["run_id"] == "test_run_001"
        assert result["method_name"] == "cyclic"
        assert result["split_name"] == "robustness"

    def test_latency_is_sum_of_all_calls(self, runner_question_row):
        """latency_seconds covers all 4 permutation calls (non-negative float)."""
        b = ContentAwareScoreBackend(target_text="HTTPS")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert isinstance(result["latency_seconds"], float)
        assert result["latency_seconds"] >= 0.0

    def test_missing_fourth_option_has_no_phantom_choice(self, runner_question_row):
        """A 3-option question must rotate only 3 real options, never a phantom D."""
        row = dict(runner_question_row, choice_d=float("nan"))
        b = ContentAwareScoreBackend(target_text="HTTPS")
        b.load()
        result = _make_runner(b).run_one(row, sample_index=0)

        assert "nan" not in result["prompt"].lower()
        assert "D." not in result["prompt"]
        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True

    def test_partial_permutation_failure_still_answers(self, runner_question_row):
        """If only some of the 4 score_options() calls fail, the question
        should still be answered (uniform-row fallback for the failed
        permutations, same precedent as PriDeRunner._cyclic_rollout_prob_matrix),
        not treated as a total failure."""

        class FlakyBackend(ContentAwareScoreBackend):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._call_count = 0

            def score_options(self, prompt, options):
                self._call_count += 1
                if self._call_count == 1:
                    raise RuntimeError("simulated transient failure")
                return super().score_options(prompt, options)

        b = FlakyBackend(target_text="HTTPS")
        b.load()
        result = _make_runner(b).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True
        assert result["cyclic_permutation_failures"] == 1

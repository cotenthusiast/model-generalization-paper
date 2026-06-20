# src/modelgen/runners/permutation.py

import collections
import json
import time
from typing import Any

import numpy as np

from modelgen.parsing.types import ParseResult, PARSE_OK, PARSE_MISSING
from modelgen.pipeline.prompt_builder import build_direct_mcq_prompt
from modelgen.runners.local_base import LocalExperimentRunner
from modelgen.runners.pride_debias import (
    equation1_cyclic_debiased_content_probs,
    logprob_map_to_label_distribution,
)


class PermutationRunner(LocalExperimentRunner):
    """Runner for the cyclic permutation condition (Zheng et al. 2024, Eq. 1).

    Generates N cyclic permutations of the option content under fixed
    A-D labels (unchanged from before), scores each permutation's full
    option set with one score_options() call (not generate()+parse), and
    combines the N resulting probability distributions via Eq.(1):
    un-permute each row back to canonical content identity and average,
    then take the argmax of the averaged distribution. N score_options()
    calls per question, no text generation.

    equation1_cyclic_debiased_content_probs (pride_debias.py) already
    implements Eq.(1)'s un-permutation + averaging math faithfully; this
    runner is responsible for producing its expected input shape: an
    (N, N) matrix where row k is the probability distribution over LETTERS
    (in canonical A/B/C/.. order) observed under cyclic permutation k. That
    row/column convention matches _generate_permutations' rotation exactly
    (verified): permutation k maps canonical content index i to letter
    index (i - k) mod N, which is the same indexing
    equation1_cyclic_debiased_content_probs un-permutes by.
    """

    def run_one(self, question_row: Any, sample_index: int) -> dict:
        canonical_options = self._build_options(question_row)
        real_letters = list(canonical_options.keys())
        n = len(real_letters)
        permutations = self._generate_permutations(canonical_options)

        prompts = [
            self._build_permuted_prompt(question_row, perm, self._prompts["direct_mcq"])
            for perm in permutations
        ]

        uniform_row = np.ones(n, dtype=np.float64) / n
        rows: list[np.ndarray] = []
        n_failures = 0
        last_error: str | None = None

        start = time.perf_counter()
        for prompt in prompts:
            try:
                score_result_obj = self.backend.score_options(prompt, real_letters)
                rows.append(
                    logprob_map_to_label_distribution(score_result_obj.scores, letters=real_letters)
                )
            except Exception as exc:
                # Mirrors PriDeRunner._cyclic_rollout_prob_matrix's established
                # fallback: a uniform row dilutes but doesn't corrupt Eq.(1)'s
                # average, and lets the question still be answered if at
                # least one permutation succeeds.
                rows.append(uniform_row.copy())
                n_failures += 1
                last_error = str(exc)
        latency = time.perf_counter() - start

        if n_failures == n:
            row = self._build_result_row(
                question_row=question_row,
                prompt=prompts[0],
                sample_index=sample_index,
                generation_result=None,
                latency_seconds=latency,
                parsed_result=None,
                score_result=None,
                error=last_error,
            )
            row["cyclic_probs_json"] = None
            row["cyclic_permutation_failures"] = n_failures
            return row

        per_perm_label_probs = np.stack(rows, axis=0)
        debiased = equation1_cyclic_debiased_content_probs(per_perm_label_probs)
        best_letter = real_letters[int(np.argmax(debiased))]

        parse_result = ParseResult(
            final_choice=best_letter,
            status=PARSE_OK,
            raw_text=None,
            normalized_text=best_letter,
            reason="cyclic_eq1_avg",
        )
        score_result = self._score(parse_result, question_row["correct_option"])

        row = self._build_result_row(
            question_row=question_row,
            prompt=prompts[0],
            sample_index=sample_index,
            generation_result=None,
            latency_seconds=latency,
            parsed_result=parse_result,
            score_result=score_result,
            error=None,
        )
        row["model_status"] = "success"
        row["error_type"] = None
        row["error_message"] = None
        row["error_stage"] = None
        row["error_retryable"] = None
        row["cyclic_probs_json"] = json.dumps(
            {letter: float(p) for letter, p in zip(real_letters, debiased)}
        )
        row["cyclic_permutation_failures"] = n_failures
        return row

    @staticmethod
    def _generate_permutations(
            options: dict[str, str],
    ) -> list[dict[str, str]]:
        """Generate cyclic permutations of the canonical option ordering."""
        keys = list(options.keys())
        values = list(options.values())
        return [
            dict(zip(keys, values[i:] + values[:i]))
            for i in range(len(options))
        ]

    @staticmethod
    def _build_permuted_prompt(
            question_row: Any,
            permuted_options: dict[str, str],
            template: str,
    ) -> str:
        """Build a direct MCQ prompt using a permuted option ordering."""
        return build_direct_mcq_prompt(
            template=template,
            question=question_row["question_text"],
            options=permuted_options,
        )

    @staticmethod
    def _unpermute_choice(
            parsed_letter: str,
            permuted_options: dict[str, str],
            canonical_options: dict[str, str],
    ) -> str | None:
        """Map a parsed letter from permuted ordering back to canonical.

        LEGACY — part of the old generate()+parse+majority-vote aggregation
        (superseded by Eq.(1) probability averaging in run_one above). Not
        called by the active run_one; kept because the old majority-vote
        run data in runs/ may need to be reproduced or cross-checked later.
        """
        selected_text = permuted_options[parsed_letter]
        for key, value in canonical_options.items():
            if value == selected_text:
                return key
        return None

    @staticmethod
    def _majority_vote(choices: list[str | None]) -> str | None:
        """Determine the final answer by majority vote.

        LEGACY — see _unpermute_choice docstring above; not called by the
        active run_one.

        Ties are broken by the first valid vote (canonical ordering tiebreaker).
        """
        cleaned = [x for x in choices if x is not None]
        if not cleaned:
            return None

        top = collections.Counter(cleaned).most_common(2)
        if len(top) == 1 or top[0][1] != top[1][1]:
            return top[0][0]
        return cleaned[0]

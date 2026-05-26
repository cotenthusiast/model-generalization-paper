# src/twoprompt/runners/permutation.py

import collections
from typing import Any

from twoprompt.parsing.types import ParseResult, PARSE_OK, PARSE_MISSING
from twoprompt.pipeline.prompt_builder import build_direct_mcq_prompt
from twoprompt.runners.local_base import LocalExperimentRunner


class PermutationRunner(LocalExperimentRunner):
    """Runner for the cyclic permutation condition.

    Generates N cyclic permutations of the option order for each question,
    makes N sequential backend calls, un-permutes each parsed answer back to
    canonical ordering, and determines the final answer by majority vote.
    """

    def run_one(self, question_row: Any, sample_index: int) -> dict:
        canonical_options = self._build_options(question_row)
        permutations = self._generate_permutations(canonical_options)

        prompts = [
            self._build_permuted_prompt(question_row, perm, self._prompts["direct_mcq"])
            for perm in permutations
        ]

        # Sequential calls — local model is synchronous
        canonical_choices: list[str | None] = []
        generation_results = []
        latencies = []
        errors = []

        for prompt, permutation in zip(prompts, permutations):
            result, latency, error = self._call_backend(prompt)
            generation_results.append(result)
            latencies.append(latency)
            errors.append(error)

            if result is not None:
                parsed = self._parse(result.raw_text, permutation)
                if parsed.final_choice is not None:
                    canonical_choices.append(
                        self._unpermute_choice(
                            parsed.final_choice, permutation, canonical_options
                        )
                    )
                else:
                    canonical_choices.append(None)
            else:
                canonical_choices.append(None)

        voted_letter = self._majority_vote(canonical_choices)

        voted_parse = ParseResult(
            final_choice=voted_letter,
            status=PARSE_OK if voted_letter else PARSE_MISSING,
            raw_text=None,
            normalized_text="",
            reason="majority_vote",
        )

        score_result = None
        if voted_letter:
            score_result = self._score(voted_parse, question_row["correct_option"])

        # Use first permutation's trace for result row; sum latencies across all calls
        return self._build_result_row(
            question_row=question_row,
            prompt=prompts[0],
            sample_index=sample_index,
            generation_result=generation_results[0],
            latency_seconds=sum(latencies),
            parsed_result=voted_parse,
            score_result=score_result,
            error=errors[0],
        )

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
            options=list(permuted_options.values()),
        )

    @staticmethod
    def _unpermute_choice(
            parsed_letter: str,
            permuted_options: dict[str, str],
            canonical_options: dict[str, str],
    ) -> str | None:
        """Map a parsed letter from permuted ordering back to canonical."""
        selected_text = permuted_options[parsed_letter]
        for key, value in canonical_options.items():
            if value == selected_text:
                return key
        return None

    @staticmethod
    def _majority_vote(choices: list[str | None]) -> str | None:
        """Determine the final answer by majority vote.

        Ties are broken by the first valid vote (canonical ordering tiebreaker).
        """
        cleaned = [x for x in choices if x is not None]
        if not cleaned:
            return None

        top = collections.Counter(cleaned).most_common(2)
        if len(top) == 1 or top[0][1] != top[1][1]:
            return top[0][0]
        return cleaned[0]

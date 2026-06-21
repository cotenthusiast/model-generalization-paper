import numpy as np

from modelgen.runners.pride_debias import (
    CalibrationState,
    apply_debiased_choice_from_defaults,
    average_prior_probability_dicts,
    equation1_cyclic_debiased_content_probs,
    equation7_prior_from_rollouts,
    equation8_debiased_content_probs,
    logprob_map_to_label_distribution,
)


class TestPriDeEquations:
    def test_eq7_uniform_rollout_is_uniform_prior(self):
        n = 4
        u = np.ones((n, n), dtype=np.float64) / n
        prior = equation7_prior_from_rollouts(u)
        assert prior.shape == (n,)
        np.testing.assert_allclose(prior, np.ones(n) / n, rtol=0, atol=1e-6)

    def test_eq1_balanced_matrix_is_uniform_over_content(self):
        n = 4
        u = np.ones((n, n), dtype=np.float64) / n
        ped = equation1_cyclic_debiased_content_probs(u)
        np.testing.assert_allclose(ped, np.ones(n) / n, rtol=0, atol=1e-6)

    def test_eq8_uniform_prior_preserves_argmax(self):
        obs = np.array([0.50, 0.30, 0.15, 0.05], dtype=np.float64)
        prior = np.ones(4, dtype=np.float64) / 4.0
        deb = equation8_debiased_content_probs(obs, prior)
        assert int(np.argmax(deb)) == int(np.argmax(obs))

    def test_logprob_map_preferences(self):
        p = logprob_map_to_label_distribution({"A": -5.0, "B": -0.5, "C": -10.0, "D": -10.0})
        assert np.argmax(p) == 1  # B


class TestAveragePriorProbabilityDicts:
    """Regression coverage for the calibration prior's masked averaging —
    a 3-option ARC-Challenge calibration question's per-question prior dict
    has no "D" key, and that absence must not pull the global D average
    toward zero/floor; D's average should be computed only from the
    calibration questions that actually have a real D option."""

    def test_empty_list_returns_uniform(self):
        result = average_prior_probability_dicts([])
        assert result == {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}

    def test_letter_missing_from_one_dict_excluded_from_its_average(self):
        four_opt = {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4}
        three_opt = {"A": 0.5, "B": 0.3, "C": 0.2}  # no D — 3-option question

        result = average_prior_probability_dicts([four_opt, three_opt])

        # D's masked mean has exactly one contributor (four_opt); A/B/C are
        # averaged over both. The whole dict is then renormalized to sum 1.
        unnormalized = {
            "A": (0.1 + 0.5) / 2,
            "B": (0.2 + 0.3) / 2,
            "C": (0.3 + 0.2) / 2,
            "D": 0.4,
        }
        total = sum(unnormalized.values())
        for letter, raw in unnormalized.items():
            assert abs(result[letter] - raw / total) < 1e-9

    def test_sums_to_one(self):
        result = average_prior_probability_dicts(
            [{"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4}, {"A": 0.5, "B": 0.3, "C": 0.2}]
        )
        assert abs(sum(result.values()) - 1.0) < 1e-9


class TestApplyDebiasedChoiceFromDefaults:
    """Regression coverage for restricting Eq.(8)'s argmax to a question's
    real options — the hardcoded-OPTION_LETTERS bug let a never-shown "D"
    win when its raw logprob happened to be high."""

    def test_restricted_to_three_letters_never_returns_d(self):
        state = CalibrationState(peprior_probs={"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
        # D would win an unrestricted argmax by a wide margin.
        logp_map = {"A": -2.0, "B": -2.0, "C": -2.0, "D": -0.1}

        choice = apply_debiased_choice_from_defaults(
            state, logp_map, letters=("A", "B", "C"),
        )

        assert choice != "D"
        assert choice in {"A", "B", "C"}

    def test_default_letters_still_full_abcd(self):
        state = CalibrationState(peprior_probs={"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
        logp_map = {"A": -2.0, "B": -2.0, "C": -2.0, "D": -0.1}

        choice = apply_debiased_choice_from_defaults(state, logp_map)

        assert choice == "D"

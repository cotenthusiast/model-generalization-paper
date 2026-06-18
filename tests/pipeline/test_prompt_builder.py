# tests/pipeline/test_prompt_builder.py

from pathlib import Path

from modelgen.pipeline.prompt_builder import (
    build_abcd_prompt,
    build_direct_mcq_prompt,
    build_free_text_prompt,
    build_option_matching_prompt,
    build_text_extraction_prompt,
    load_prompt_templates,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = _REPO_ROOT / "prompts"
_TEMPLATES = load_prompt_templates("v1", _PROMPTS_DIR)


class TestBuildDirectMcqPrompt:
    """Tests for build_direct_mcq_prompt."""

    def test_includes_question_options_and_letter_instruction(self):
        question = "Which number has one factor?"
        options = {"A": "one", "B": "two", "C": "three", "D": "four"}

        prompt = build_direct_mcq_prompt(_TEMPLATES["direct_mcq"], question, options)

        assert question in prompt
        assert "Respond with only the letter." in prompt
        assert prompt.index("A. one") < prompt.index("B. two")
        assert prompt.index("B. two") < prompt.index("C. three")
        assert prompt.index("C. three") < prompt.index("D. four")


class TestBuildFreeTextPrompt:
    """Tests for build_free_text_prompt."""

    def test_includes_question_and_excludes_options(self):
        question = "Which number has one factor?"

        actual = build_free_text_prompt(_TEMPLATES["free_text"], question)

        assert question in actual
        assert "Options:" not in actual
        assert "A." not in actual
        assert "B." not in actual
        assert "C." not in actual
        assert "D." not in actual


class TestBuildOptionMatchingPrompt:
    """Tests for build_option_matching_prompt."""

    def test_includes_question_free_text_options_and_letter_instruction(self):
        question = "Which number has one factor?"
        options = {"A": "one", "B": "two", "C": "three", "D": "four"}
        free_response = "one"

        prompt = build_option_matching_prompt(
            _TEMPLATES["option_matching"],
            question,
            free_response,
            options,
        )

        assert "Select the option that best matches the reference answer in the context of the question.".lower() in prompt.lower()
        assert question in prompt
        assert "Respond with only the letter." in prompt
        assert prompt.index("A. one") < prompt.index("B. two")
        assert prompt.index("B. two") < prompt.index("C. three")
        assert prompt.index("C. three") < prompt.index("D. four")
        assert free_response in prompt


class TestMissingOptionOmitted:
    """A question with fewer than 4 real options must never render a phantom option."""

    _three_options = {"A": "one", "B": "two", "C": "three"}

    def test_direct_mcq_prompt_omits_missing_fourth_option(self):
        prompt = build_direct_mcq_prompt(_TEMPLATES["direct_mcq"], "Q?", self._three_options)

        assert "nan" not in prompt.lower()
        assert "D." not in prompt
        assert "C. three" in prompt

    def test_text_extraction_prompt_omits_missing_fourth_option(self):
        prompt = build_text_extraction_prompt(
            _TEMPLATES["text_extraction"], "Q?", self._three_options
        )

        assert "nan" not in prompt.lower()
        assert "D." not in prompt
        assert "C. three" in prompt

    def test_abcd_prompt_omits_missing_fourth_option(self):
        prompt = build_abcd_prompt(_TEMPLATES["abcd"], "Q?", self._three_options)

        assert "nan" not in prompt.lower()
        assert prompt.count("- ") == 3

    def test_option_matching_prompt_omits_missing_fourth_option(self):
        prompt = build_option_matching_prompt(
            _TEMPLATES["option_matching"], "Q?", "one", self._three_options
        )

        assert "nan" not in prompt.lower()
        assert "D." not in prompt
        assert "C. three" in prompt

    def test_labels_use_actual_dict_keys_not_position(self):
        """A label must never shift into a gap left by a dropped option.

        E.g. additional_option's "E" (I don't know) must stay "E" when "D"
        is missing, not silently become "D" because it's now in 4th position.
        """
        options = {"A": "one", "B": "two", "C": "three", "E": "I don't know"}
        prompt = build_direct_mcq_prompt(_TEMPLATES["direct_mcq"], "Q?", options)

        assert "E. I don't know" in prompt
        assert "D." not in prompt

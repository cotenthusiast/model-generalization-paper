# tests/runners/test_additional_option.py

from pathlib import Path

from twoprompt.backends.dummy import DummyBackend
from twoprompt.parsing.parser import parse_model_answer
from twoprompt.parsing.types import PARSE_OK
from twoprompt.runners.additional_option import AdditionalOptionRunner

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


def _make_runner(backend):
    return AdditionalOptionRunner(
        backend=backend,
        method_name="additional_option",
        split_name="robustness",
        prompt_version="v1",
        prompts_dir=_PROMPTS_DIR,
        run_id="additional_option_test",
    )


class TestAdditionalOptionPrompt:
    def test_prompt_contains_option_e(self, runner_question_row):
        b = DummyBackend()
        b.load()
        runner = _make_runner(b)
        prompt = runner._build_prompt(runner_question_row)

        assert "E. I don't know" in prompt

    def test_prompt_does_not_contain_option_f(self, runner_question_row):
        b = DummyBackend()
        b.load()
        runner = _make_runner(b)
        prompt = runner._build_prompt(runner_question_row)

        assert "F. I don't know" not in prompt
        assert "F." not in prompt

    def test_options_dict_contains_e(self, runner_question_row):
        b = DummyBackend()
        b.load()
        runner = _make_runner(b)
        options = runner._build_options(runner_question_row)

        assert "E" in options
        assert options["E"] == "I don't know"

    def test_options_dict_has_exactly_five_keys(self, runner_question_row):
        b = DummyBackend()
        b.load()
        runner = _make_runner(b)
        options = runner._build_options(runner_question_row)

        assert set(options.keys()) == {"A", "B", "C", "D", "E"}


class TestAdditionalOptionEParsing:
    def test_parse_model_answer_e_with_e_in_options(self, runner_question_row):
        b = DummyBackend()
        b.load()
        runner = _make_runner(b)
        options = runner._build_options(runner_question_row)

        result = parse_model_answer("E", options)

        assert result.final_choice == "E"
        assert result.status == PARSE_OK

    def test_run_many_produces_result_rows(self, runner_question_row):
        b = DummyBackend()
        b.load()
        rows = _make_runner(b).run_many([runner_question_row])

        assert len(rows) == 1
        assert rows[0]["model_status"] == "success"

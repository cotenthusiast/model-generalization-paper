# tests/runners/conftest.py

import pytest

from modelgen.backends.dummy import DummyBackend
from modelgen.backends.base import LocalBackend
from modelgen.backends.types import LocalGenerationConfig, ModelGenerationResult


class ErrorBackend(DummyBackend):
    """DummyBackend that raises RuntimeError on generate()."""

    def generate(self, prompt: str, config: LocalGenerationConfig | None = None):
        if not self._loaded:
            raise RuntimeError("Call load() before generate().")
        raise RuntimeError("Simulated inference failure.")


class ErrorScoreBackend(DummyBackend):
    """DummyBackend that raises RuntimeError on score_options()."""

    def score_options(self, prompt: str, options: list[str]):
        if not self._loaded:
            raise RuntimeError("Call load() before score_options().")
        raise RuntimeError("Simulated score_options failure.")


@pytest.fixture
def dummy_backend():
    b = DummyBackend()
    b.load()
    return b


@pytest.fixture
def error_backend():
    b = ErrorBackend()
    b.load()
    return b


@pytest.fixture
def runner_question_row() -> dict:
    return {
        "question_id": "4865890d7f0efae8",
        "subject": "computer_security",
        "question_text": "Which protocol is primarily used to securely browse websites?",
        "choice_a": "FTP",
        "choice_b": "HTTP",
        "choice_c": "HTTPS",
        "choice_d": "SMTP",
        "correct_option": "C",
        "correct_answer_text": "HTTPS",
    }


@pytest.fixture
def canonical_options() -> dict[str, str]:
    return {
        "A": "FTP",
        "B": "HTTP",
        "C": "HTTPS",
        "D": "SMTP",
    }

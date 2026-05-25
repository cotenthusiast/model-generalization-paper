# tests/backends/test_dummy.py

import pytest

from twoprompt.backends.dummy import DummyBackend
from twoprompt.backends.types import (
    LocalGenerationConfig,
    ModelGenerationResult,
    ModelOptionScoreResult,
)


@pytest.fixture
def backend():
    return DummyBackend()


@pytest.fixture
def loaded_backend():
    b = DummyBackend()
    b.load()
    return b


class TestDummyBackendLoad:
    def test_not_loaded_initially(self, backend):
        assert not backend.is_loaded()

    def test_load_sets_loaded(self, backend):
        backend.load()
        assert backend.is_loaded()

    def test_load_is_idempotent(self, backend):
        backend.load()
        backend.load()
        assert backend.is_loaded()


class TestDummyBackendMetadata:
    def test_family(self, backend):
        assert backend.metadata.family == "dummy"

    def test_backend_type(self, backend):
        assert backend.metadata.backend_type == "dummy"

    def test_device(self, backend):
        assert backend.metadata.device == "cpu"


class TestDummyBackendGenerate:
    def test_raises_if_not_loaded(self, backend):
        with pytest.raises(RuntimeError, match="load()"):
            backend.generate("hello")

    def test_returns_model_generation_result(self, loaded_backend):
        assert isinstance(loaded_backend.generate("What is 2+2?"), ModelGenerationResult)

    def test_returns_default_fixed_text(self, loaded_backend):
        assert loaded_backend.generate("anything").raw_text == "The answer is A."

    def test_custom_fixed_text(self):
        b = DummyBackend(fixed_text="B")
        b.load()
        assert b.generate("q").raw_text == "B"

    def test_finish_reason_is_eos(self, loaded_backend):
        assert loaded_backend.generate("q").finish_reason == "eos"

    def test_accepts_generation_config(self, loaded_backend):
        cfg = LocalGenerationConfig(max_new_tokens=10)
        result = loaded_backend.generate("q", config=cfg)
        assert isinstance(result, ModelGenerationResult)

    def test_metadata_backend_field(self, loaded_backend):
        assert loaded_backend.generate("q").metadata["backend"] == "dummy"


class TestDummyBackendScoreOptions:
    def test_raises_if_not_loaded(self, backend):
        with pytest.raises(RuntimeError, match="load()"):
            backend.score_options("hello", ["A", "B"])

    def test_raises_on_empty_options(self, loaded_backend):
        with pytest.raises(ValueError):
            loaded_backend.score_options("hello", [])

    def test_returns_model_option_score_result(self, loaded_backend):
        result = loaded_backend.score_options("q", ["A", "B", "C", "D"])
        assert isinstance(result, ModelOptionScoreResult)

    def test_scores_keys_match_options(self, loaded_backend):
        result = loaded_backend.score_options("q", ["A", "B"])
        assert set(result.scores.keys()) == {"A", "B"}

    def test_default_scores_ordering(self, loaded_backend):
        result = loaded_backend.score_options("q", ["A", "B", "C", "D"])
        assert result.scores["A"] > result.scores["B"] > result.scores["C"] > result.scores["D"]

    def test_unknown_option_gets_floor_score(self, loaded_backend):
        result = loaded_backend.score_options("q", ["Z"])
        assert result.scores["Z"] == -5.0

    def test_custom_fixed_scores(self):
        b = DummyBackend(fixed_scores={"A": -0.5, "B": -3.0})
        b.load()
        result = b.score_options("q", ["A", "B"])
        assert result.scores["A"] == -0.5
        assert result.scores["B"] == -3.0

    def test_raw_logprobs_matches_scores(self, loaded_backend):
        result = loaded_backend.score_options("q", ["A", "B"])
        assert result.raw_logprobs == result.scores

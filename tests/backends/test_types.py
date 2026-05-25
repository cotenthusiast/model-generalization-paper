# tests/backends/test_types.py

import pytest

from twoprompt.backends.types import (
    LocalGenerationConfig,
    LocalModelMetadata,
    ModelGenerationResult,
    ModelOptionScoreResult,
)


class TestLocalGenerationConfig:
    def test_defaults(self):
        cfg = LocalGenerationConfig()
        assert cfg.max_new_tokens == 512
        assert cfg.temperature == 0.0
        assert cfg.do_sample is False
        assert cfg.seed == 42

    def test_custom_values(self):
        cfg = LocalGenerationConfig(max_new_tokens=128, temperature=0.7, do_sample=True, seed=0)
        assert cfg.max_new_tokens == 128
        assert cfg.temperature == 0.7
        assert cfg.do_sample is True
        assert cfg.seed == 0


class TestLocalModelMetadata:
    def test_required_fields(self):
        m = LocalModelMetadata(
            model_name="qwen-7b",
            model_path="/models/qwen",
            family="qwen",
            backend_type="hf_transformers",
            device="cuda",
        )
        assert m.model_name == "qwen-7b"
        assert m.family == "qwen"
        assert m.size_label is None

    def test_size_label(self):
        m = LocalModelMetadata("n", "p", "qwen", "hf_transformers", "cpu", "7B")
        assert m.size_label == "7B"


class TestModelGenerationResult:
    def test_fields(self):
        r = ModelGenerationResult(
            raw_text="The answer is A.",
            prompt_tokens=10,
            completion_tokens=5,
            finish_reason="eos",
        )
        assert r.raw_text == "The answer is A."
        assert r.prompt_tokens == 10
        assert r.metadata == {}

    def test_optional_fields_can_be_none(self):
        r = ModelGenerationResult(
            raw_text="x",
            prompt_tokens=None,
            completion_tokens=None,
            finish_reason=None,
        )
        assert r.prompt_tokens is None
        assert r.finish_reason is None


class TestModelOptionScoreResult:
    def test_fields(self):
        r = ModelOptionScoreResult(
            scores={"A": -0.1, "B": -1.5},
            raw_logprobs={"A": -0.1, "B": -1.5},
        )
        assert r.scores["A"] == -0.1
        assert r.metadata == {}

    def test_raw_logprobs_can_be_none(self):
        r = ModelOptionScoreResult(scores={"A": -0.1}, raw_logprobs=None)
        assert r.raw_logprobs is None

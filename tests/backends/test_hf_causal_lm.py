# tests/backends/test_hf_causal_lm.py
#
# These tests require torch but NOT a real GPU or transformers installation.
# torch is used to build realistic fake tensors; transformers is mocked via
# sys.modules injection so no model weights are downloaded or loaded.
#
# Install requirements: pip install -e '.[local,dev]'

import sys

import pytest

torch = pytest.importorskip(
    "torch",
    reason="torch is required for HF backend tests; install with: pip install -e '.[local]'",
)

from unittest.mock import MagicMock, patch

from modelgen.backends.hf_causal_lm import HFCausalLMBackend
from modelgen.backends.types import (
    LocalGenerationConfig,
    ModelGenerationResult,
    ModelOptionScoreResult,
)


# ---------------------------------------------------------------------------
# Helpers: build lightweight mock tokenizer and model objects
# ---------------------------------------------------------------------------

def _make_mock_tokenizer(
    prompt_token_ids: list[int] | None = None,
    encode_map: dict[str, list[int]] | None = None,
    eos_token_id: int = 2,
):
    """Mock tokenizer returning controlled token IDs and decode strings."""
    tok = MagicMock()
    tok.eos_token_id = eos_token_id

    ids = prompt_token_ids or [1, 2, 3]
    tok.return_value = {"input_ids": torch.tensor([ids])}

    _encode_map = encode_map or {"A": [100], "B": [101], "C": [102], "D": [103]}
    tok.encode.side_effect = lambda opt, add_special_tokens=True: _encode_map.get(opt, [999])
    tok.decode.return_value = "The answer is A."

    return tok


def _make_mock_model(vocab_size: int = 200, eos_token_id: int = 2):
    """Mock model returning controlled generate output and logits."""
    model = MagicMock()
    model.eval.return_value = None

    # generate() returns (1, prompt_len + completion_len); last token is eos.
    model.generate.return_value = torch.tensor([[1, 2, 3, 100, eos_token_id]])

    # forward pass: logits shape (1, seq_len, vocab_size).
    # Token 100 (A) is given the highest logit so it wins after log_softmax.
    logits = torch.zeros(1, 3, vocab_size)
    logits[0, -1, 100] = 5.0
    logits[0, -1, 101] = 1.0
    logits[0, -1, 102] = 0.5
    logits[0, -1, 103] = 0.3
    fwd_output = MagicMock()
    fwd_output.logits = logits
    model.return_value = fwd_output

    return model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backend():
    return HFCausalLMBackend(
        model_path="test/qwen-7b",
        family="qwen",
        size_label="7B",
        device="cpu",
    )


@pytest.fixture
def mock_transformers():
    """Inject a fake transformers module so tests don't need it installed."""
    mock_module = MagicMock()
    with patch.dict(sys.modules, {"transformers": mock_module}):
        yield mock_module


@pytest.fixture
def loaded_backend(backend, mock_transformers):
    mock_transformers.AutoTokenizer.from_pretrained.return_value = _make_mock_tokenizer()
    mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = _make_mock_model()
    backend.load()
    return backend


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

class TestHFCausalLMBackendLoad:
    def test_not_loaded_initially(self, backend):
        assert not backend.is_loaded()

    def test_load_sets_loaded(self, backend, mock_transformers):
        mock_transformers.AutoTokenizer.from_pretrained.return_value = _make_mock_tokenizer()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = _make_mock_model()
        backend.load()
        assert backend.is_loaded()

    def test_load_is_idempotent(self, backend, mock_transformers):
        mock_transformers.AutoTokenizer.from_pretrained.return_value = _make_mock_tokenizer()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = _make_mock_model()
        backend.load()
        backend.load()
        assert mock_transformers.AutoTokenizer.from_pretrained.call_count == 1

    def test_load_calls_model_eval(self, backend, mock_transformers):
        mock_model = _make_mock_model()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = _make_mock_tokenizer()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        backend.load()
        mock_model.eval.assert_called_once()


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

class TestHFCausalLMBackendMetadata:
    def test_family(self, backend):
        assert backend.metadata.family == "qwen"

    def test_backend_type(self, backend):
        assert backend.metadata.backend_type == "hf_transformers"

    def test_size_label(self, backend):
        assert backend.metadata.size_label == "7B"

    def test_model_path(self, backend):
        assert backend.metadata.model_path == "test/qwen-7b"

    def test_model_name_derived_from_path(self, backend):
        # model_name is the last segment of model_path
        assert backend.metadata.model_name == "qwen-7b"


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

class TestHFCausalLMBackendGenerate:
    def test_raises_if_not_loaded(self, backend):
        with pytest.raises(RuntimeError, match="load()"):
            backend.generate("hello")

    def test_returns_model_generation_result(self, loaded_backend):
        assert isinstance(loaded_backend.generate("What is 2+2?"), ModelGenerationResult)

    def test_raw_text_comes_from_tokenizer_decode(self, loaded_backend):
        result = loaded_backend.generate("What is 2+2?")
        assert result.raw_text == "The answer is A."

    def test_prompt_tokens_is_input_length(self, loaded_backend):
        # mock tokenizer returns [1, 2, 3] → 3 prompt tokens
        assert loaded_backend.generate("q").prompt_tokens == 3

    def test_completion_tokens_is_generated_length(self, loaded_backend):
        # generate() returned [1,2,3,100,2]; prompt was 3 → 2 new tokens
        assert loaded_backend.generate("q").completion_tokens == 2

    def test_finish_reason_eos_when_last_token_is_eos(self, loaded_backend):
        # _make_mock_model ends the sequence with eos_token_id=2
        assert loaded_backend.generate("q").finish_reason == "eos"

    def test_finish_reason_length_when_last_token_is_not_eos(self, backend, mock_transformers):
        mock_tok = _make_mock_tokenizer(eos_token_id=2)
        mock_model = _make_mock_model(eos_token_id=2)
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 100, 99]])  # 99 != eos
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tok
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        backend.load()
        assert backend.generate("q").finish_reason == "length"

    def test_decodes_only_generated_tokens_not_prompt(self, loaded_backend):
        loaded_backend.generate("What is 2+2?")
        decoded_ids = loaded_backend._tokenizer.decode.call_args[0][0]
        # full output [1,2,3,100,2], prompt was 3 → decoded portion is [100,2]
        assert len(decoded_ids) == 2

    def test_accepts_generation_config(self, loaded_backend):
        cfg = LocalGenerationConfig(max_new_tokens=16, seed=0)
        assert isinstance(loaded_backend.generate("q", config=cfg), ModelGenerationResult)

    def test_metadata_contains_model_path(self, loaded_backend):
        result = loaded_backend.generate("q")
        assert result.metadata["model_path"] == "test/qwen-7b"


# ---------------------------------------------------------------------------
# Score options
# ---------------------------------------------------------------------------

class TestHFCausalLMBackendScoreOptions:
    def test_raises_if_not_loaded(self, backend):
        with pytest.raises(RuntimeError, match="load()"):
            backend.score_options("hello", ["A", "B"])

    def test_raises_on_empty_options(self, loaded_backend):
        with pytest.raises(ValueError):
            loaded_backend.score_options("q", [])

    def test_returns_model_option_score_result(self, loaded_backend):
        result = loaded_backend.score_options("q", ["A", "B", "C", "D"])
        assert isinstance(result, ModelOptionScoreResult)

    def test_scores_keys_match_options(self, loaded_backend):
        result = loaded_backend.score_options("q", ["A", "B", "C", "D"])
        assert set(result.scores.keys()) == {"A", "B", "C", "D"}

    def test_highest_logit_option_has_highest_score(self, loaded_backend):
        result = loaded_backend.score_options("q", ["A", "B", "C", "D"])
        # logits: A=5.0, B=1.0, C=0.5, D=0.3 → A wins after log_softmax
        assert result.scores["A"] == max(result.scores.values())

    def test_scores_are_log_probabilities(self, loaded_backend):
        result = loaded_backend.score_options("q", ["A", "B", "C", "D"])
        for v in result.scores.values():
            assert v <= 0.0

    def test_raw_logprobs_matches_scores(self, loaded_backend):
        result = loaded_backend.score_options("q", ["A", "B", "C", "D"])
        assert result.raw_logprobs == result.scores

    def test_raises_on_multi_token_option(self, loaded_backend):
        # Override encode to return two tokens for every option
        loaded_backend._tokenizer.encode.side_effect = (
            lambda opt, add_special_tokens=True: [100, 101]
        )
        with pytest.raises(ValueError, match="2 token"):
            loaded_backend.score_options("q", ["AB"])

    def test_metadata_contains_model_path(self, loaded_backend):
        result = loaded_backend.score_options("q", ["A", "B"])
        assert result.metadata["model_path"] == "test/qwen-7b"

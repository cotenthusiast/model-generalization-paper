# tests/clients/conftest.py

import pytest

from twoprompt.clients.types import (
    SUCCESS_STATUS,
    FAILURE_STATUS,
    ErrorInfo,
    ModelRequest,
    ModelResponse,
    RequestMetadata,
    UsageInfo,
)


@pytest.fixture
def valid_metadata() -> RequestMetadata:
    return RequestMetadata(
        question_id="q_001",
        split_name="robustness",
        method_name="baseline",
        subject="anatomy",
        run_id="run_001",
        prompt_version="v1",
        perturbation_name="original",
        sample_index=0,
    )


@pytest.fixture
def valid_request(valid_metadata: RequestMetadata) -> ModelRequest:
    return ModelRequest(
        provider="openai",
        model_name="gpt-4.1-mini",
        payload="Question: What is 2 + 2?\nA. 3\nB. 4\nC. 5\nD. 6",
        metadata=valid_metadata,
    )


@pytest.fixture
def successful_response(valid_metadata: RequestMetadata) -> ModelResponse:
    return ModelResponse(
        provider="openai",
        model_name="gpt-4.1-mini",
        status=SUCCESS_STATUS,
        latency_seconds=0.25,
        metadata=valid_metadata,
        raw_text="B",
        finish_reason="stop",
        usage=UsageInfo(
            prompt_tokens=25,
            completion_tokens=3,
            total_tokens=28,
        ),
        error=None,
        timestamp_utc="2026-03-13T21:00:00Z",
    )


@pytest.fixture
def failed_response(valid_metadata: RequestMetadata) -> ModelResponse:
    return ModelResponse(
        provider="openai",
        model_name="gpt-4.1-mini",
        status=FAILURE_STATUS,
        latency_seconds=0.40,
        metadata=valid_metadata,
        raw_text=None,
        finish_reason=None,
        usage=None,
        error=ErrorInfo(
            error_type="ProviderTimeoutError",
            message="Request timed out.",
            retryable=True,
            stage="provider_call",
        ),
        timestamp_utc=None,
    )

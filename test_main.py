"""Focused tests for the FastAPI AI service with Groq mocked out."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import main  # noqa: E402
from fastapi import HTTPException  # noqa: E402

VALID_REVIEW = {
    "bug_analysis": "No obvious bugs.",
    "explanation": "The program prints numbers.",
    "suggestions": "Add input validation.",
    "optimized_code": "print('hello')",
    "quality_score": 80,
}

VALID_TESTS = {
    "test_cases": [
        {
            "name": f"Case {i}",
            "category": "Normal",
            "input": "1",
            "expected_output": "1",
            "explanation": "Basic case.",
        }
        for i in range(1, 5)
    ],
    "notes": "Not executed.",
}


def _fake(exc_type, message="boom"):
    """Build an exception without requiring SDK request/response objects."""
    exc = exc_type.__new__(exc_type)
    exc.args = (message,)
    return exc


def test_app_imports_and_routes_exist():
    paths = {route.path for route in main.app.routes}
    assert "/" in paths
    assert "/health" in paths
    assert "/review" in paths
    assert "/generate-tests" in paths


def test_review_returns_structured_response(monkeypatch):
    monkeypatch.setattr(
        main, "ask_groq", lambda prompt, num_predict=1800: _as_json(VALID_REVIEW)
    )

    response = main.review_code(
        main.CodeReviewRequest(code="print('hi')", language="python")
    )

    assert response.quality_score == 80
    assert response.optimized_code == "print('hello')"


def test_review_clamps_out_of_range_score(monkeypatch):
    payload = dict(VALID_REVIEW, quality_score=250)
    monkeypatch.setattr(
        main, "ask_groq", lambda prompt, num_predict=1800: _as_json(payload)
    )

    response = main.review_code(
        main.CodeReviewRequest(code="print('hi')", language="python")
    )

    assert response.quality_score == 100


def test_review_rejects_incomplete_ai_response(monkeypatch):
    monkeypatch.setattr(
        main, "ask_groq", lambda prompt, num_predict=1800: '{"bug_analysis": "x"}'
    )

    try:
        main.review_code(
            main.CodeReviewRequest(code="print('hi')", language="python")
        )
    except HTTPException as exc:
        assert exc.status_code == 502
    else:
        raise AssertionError("Expected an HTTPException for incomplete responses")


def test_review_handles_groq_connection_error(monkeypatch):
    def boom(prompt, num_predict=1800):
        raise _fake(main.groq.APIConnectionError)

    monkeypatch.setattr(main, "ask_groq", boom)

    try:
        main.review_code(
            main.CodeReviewRequest(code="print('hi')", language="python")
        )
    except HTTPException as exc:
        assert exc.status_code == 503
    else:
        raise AssertionError("Expected a 503 when Groq is unreachable")


def test_review_handles_missing_api_key(monkeypatch):
    def boom():
        raise RuntimeError("GROQ_API_KEY is not set.")

    monkeypatch.setattr(main, "get_groq_client", boom)

    try:
        main.review_code(
            main.CodeReviewRequest(code="print('hi')", language="python")
        )
    except HTTPException as exc:
        assert exc.status_code == 500
        assert "API key" in str(exc.detail)
    else:
        raise AssertionError("Expected a 500 when the API key is missing")


def test_review_handles_invalid_api_key_without_leaking_it(monkeypatch):
    def boom():
        raise _fake(main.groq.AuthenticationError)

    monkeypatch.setattr(main, "get_groq_client", boom)

    try:
        main.review_code(
            main.CodeReviewRequest(code="print('hi')", language="python")
        )
    except HTTPException as exc:
        assert exc.status_code == 500
        assert "gsk_" not in str(exc.detail)
    else:
        raise AssertionError("Expected a 500 for an invalid API key")


def test_review_handles_rate_limit(monkeypatch):
    def boom(prompt, num_predict=1800):
        raise _fake(main.groq.RateLimitError)

    monkeypatch.setattr(main, "ask_groq", boom)

    try:
        main.review_code(
            main.CodeReviewRequest(code="print('hi')", language="python")
        )
    except HTTPException as exc:
        assert exc.status_code == 503
    else:
        raise AssertionError("Expected a 503 on a Groq rate limit")


def test_review_handles_groq_timeout(monkeypatch):
    def boom(prompt, num_predict=1800):
        raise _fake(main.groq.APITimeoutError)

    monkeypatch.setattr(main, "ask_groq", boom)

    try:
        main.review_code(
            main.CodeReviewRequest(code="print('hi')", language="python")
        )
    except HTTPException as exc:
        assert exc.status_code == 504
    else:
        raise AssertionError("Expected a 504 on a Groq timeout")


def test_review_handles_empty_model_response(monkeypatch):
    monkeypatch.setattr(
        main, "ask_groq", lambda prompt, num_predict=1800: ""
    )

    try:
        main.review_code(
            main.CodeReviewRequest(code="print('hi')", language="python")
        )
    except HTTPException as exc:
        assert exc.status_code == 502
    else:
        raise AssertionError("Expected a 502 for an empty model response")


def test_review_rejects_blank_code():
    import pytest
    from pydantic import ValidationError

    # Blank code is rejected by request validation before any AI call happens.
    with pytest.raises((ValidationError, HTTPException)):
        main.review_code(main.CodeReviewRequest(code=" ", language="python"))


def test_generate_tests_returns_structured_cases(monkeypatch):
    monkeypatch.setattr(
        main, "ask_groq", lambda prompt, num_predict=1800: _as_json(VALID_TESTS)
    )

    response = main.generate_test_cases(
        main.TestCaseRequest(code="def f(): return 1", language="python")
    )

    assert len(response.test_cases) == 4
    assert response.test_cases[0].name == "Case 1"
    assert response.notes


def test_generate_tests_rejects_wrong_case_count(monkeypatch):
    payload = dict(VALID_TESTS, test_cases=VALID_TESTS["test_cases"][:2])
    monkeypatch.setattr(
        main, "ask_groq", lambda prompt, num_predict=1800: _as_json(payload)
    )

    try:
        main.generate_test_cases(
            main.TestCaseRequest(code="def f(): return 1", language="python")
        )
    except HTTPException as exc:
        assert exc.status_code == 502
    else:
        raise AssertionError("Expected rejection of an invalid test case count")


def test_extract_json_handles_markdown_fences():
    payload = '{"a": 1}'
    assert main.extract_json(f"```json\n{payload}\n```") == {"a": 1}
    assert main.extract_json(f"noise before {payload} noise after") == {"a": 1}


def test_get_groq_client_missing_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")

    try:
        main.get_groq_client()
    except RuntimeError as exc:
        assert "GROQ_API_KEY" in str(exc)
    else:
        raise AssertionError("Expected a RuntimeError when the key is missing")


def test_get_groq_model_uses_env_when_set(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "my-test-model")

    try:
        assert main.get_groq_model() == "my-test-model"
    finally:
        main._groq_model = None


def _as_json(payload):
    import json

    return json.dumps(payload)
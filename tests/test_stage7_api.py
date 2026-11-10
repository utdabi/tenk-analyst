"""Stage 7: the back end. No LLM calls. Only the last test connects to Neon."""
import logging

import pytest
from fastapi.testclient import TestClient

from backend import api, pipelines

CODE = {"X-Access-Code": "test-code"}
BODY = {"question": "What was Ford's R&D expense in 2025?", "chat_id": "chat-1", "pipeline": "agent",
        "companies": [], "years": []}


@pytest.fixture
def client(monkeypatch):
    """A test client without the start-up step (so it does not touch Neon), with a known access code."""
    monkeypatch.setenv("ACCESS_CODE", "test-code")
    return TestClient(api.app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize("headers", [{}, {"X-Access-Code": "wrong"}])
def test_missing_or_wrong_access_code(client, headers):
    response = client.post("/ask", json=BODY, headers=headers)
    assert response.status_code == 401
    assert response.json() == {"error": "Wrong or missing access code."}


@pytest.mark.parametrize("question, message", [
    ("", "The question is empty."),
    ("   ", "The question is empty."),
    ("x" * 1001, "The question is too long. Use at most 1,000 characters."),
])
def test_bad_questions(client, question, message):
    response = client.post("/ask", json={**BODY, "question": question}, headers=CODE)
    assert response.status_code == 400
    assert response.json() == {"error": message}


def test_a_question_of_exactly_1000_characters_is_allowed(client, monkeypatch):
    monkeypatch.setattr(pipelines, "answer", lambda *args: {"answer": "ok"})
    assert client.post("/ask", json={**BODY, "question": "x" * 1000}, headers=CODE).status_code == 200


def test_unknown_pipeline(client):
    response = client.post("/ask", json={**BODY, "pipeline": "magic"}, headers=CODE)
    assert response.status_code == 400
    assert response.json() == {"error": "Unknown pipeline. Use simple, filters, rerank, decompose, or agent."}


def test_the_access_code_is_checked_before_the_question(client):
    response = client.post("/ask", json={**BODY, "question": ""}, headers={"X-Access-Code": "wrong"})
    assert response.status_code == 401


def test_a_good_request_calls_answer_with_the_request(client, monkeypatch):
    calls = []

    def fake_answer(*args):
        calls.append(args)
        return {"answer": "fine", "pipeline": "filters", "chat_id": "chat-1", "tools_used": [], "sources": []}

    monkeypatch.setattr(pipelines, "answer", fake_answer)
    body = {**BODY, "pipeline": "filters", "companies": ["F"], "years": [2025]}
    response = client.post("/ask", json=body, headers=CODE)
    assert response.status_code == 200
    assert response.json()["answer"] == "fine"
    assert calls == [("What was Ford's R&D expense in 2025?", "filters", ["F"], [2025], "chat-1")]


def test_a_failure_gives_503_and_logs_only_the_error_type(client, monkeypatch, caplog):
    def broken(*args):
        raise RuntimeError("sk-or-secret-key-123")  # an error message that contains a key

    monkeypatch.setattr(pipelines, "answer", broken)
    with caplog.at_level(logging.ERROR, logger="tenk"):
        response = client.post("/ask", json=BODY, headers=CODE)

    assert response.status_code == 503
    assert response.json() == {"error": "The answer service is not available right now. Try again in a minute."}
    assert "RuntimeError" in caplog.text
    assert "sk-or-secret-key-123" not in caplog.text
    assert "Traceback" not in caplog.text and "secret" not in response.text


# ----- the start-up check -----

def set_all_variables(monkeypatch):
    for name in api.REQUIRED:
        monkeypatch.setenv(name, "value")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)


def test_start_up_check_passes_when_everything_is_set(monkeypatch):
    set_all_variables(monkeypatch)
    api.check_environment()  # LangSmith is optional, so it may be missing


@pytest.mark.parametrize("name", api.REQUIRED)
def test_start_up_check_names_the_missing_variable(monkeypatch, name):
    set_all_variables(monkeypatch)
    monkeypatch.delenv(name)
    with pytest.raises(SystemExit) as stop:
        api.check_environment()
    assert name in str(stop.value)


@pytest.mark.parametrize("name", api.REQUIRED + ["LANGSMITH_API_KEY"])
def test_start_up_check_catches_a_line_break_without_printing_the_value(monkeypatch, name):
    set_all_variables(monkeypatch)
    monkeypatch.setenv(name, "secret-part-one\nsecret-part-two")
    with pytest.raises(SystemExit) as stop:
        api.check_environment()
    assert name in str(stop.value)
    assert "secret" not in str(stop.value)


def test_the_server_starts_and_stops():
    """The real start-up (needs the real .env): opens the Neon connections, then closes them."""
    with TestClient(api.app) as started:
        assert started.get("/health").json() == {"status": "ok"}

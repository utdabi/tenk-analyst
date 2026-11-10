"""Stage 1: the scope check and the simple pipeline. These tests call OpenRouter (a few cents)."""
import uuid

import pytest
from langchain_core.documents import Document

from backend import pipelines


def question_by_id(test_data, number: int) -> dict:
    return next(q for q in test_data["questions"] if q["id"] == number)


def test_q1_simple_pipeline(test_data):
    q = question_by_id(test_data, 1)
    result = pipelines.answer(q["question"], q["pipeline"], [], [], str(uuid.uuid4()))

    assert result["answer"].strip()
    assert result["tools_used"] == []
    assert len(result["sources"]) == q["expect"]["source_count"] == 5
    for source in result["sources"]:
        for field in q["expect"]["source_fields"]:
            assert source[field] not in (None, "")
    print("\nANSWER:", result["answer"])
    for s in result["sources"]:
        print("SOURCE:", s["company"], s["year"], "Item", s["item"])


def test_prompt_injection_is_ignored(test_data):
    t = test_data["injection_test"]
    chunk = Document(page_content=t["chunk_text"], metadata={"company": "TSLA", "year": 2025, "item": "1A"})
    answer = pipelines.generate_answer(t["question"], [chunk])
    print("\nANSWER:", answer)
    assert t["expect"]["answer_not_contains"] not in answer


@pytest.mark.parametrize("question, expected", [
    ("What risks does Toyota report?", pipelines.COMPANY_MESSAGE),
    ("What does Panasonic supply to Tesla?", pipelines.COMPANY_MESSAGE),  # a supplier is an outside company
    ("What did Ford say about tariffs in 2023?", pipelines.YEAR_MESSAGE),
    ("Compare Toyota and Ford in 2022.", pipelines.COMPANY_MESSAGE),  # company message comes first
])
def test_out_of_scope_questions(question, expected):
    result = pipelines.answer(question, "simple", [], [], str(uuid.uuid4()))
    assert result["answer"] == expected
    assert result["sources"] == []
    assert result["tools_used"] == []


def test_scope_check_finds_companies_and_years():
    scope = pipelines.run_scope_check("What does GM say about cybersecurity in 2024?")
    assert scope.companies == ["GM"]
    assert scope.years == [2024]
    assert scope.other_companies == []


def test_no_year_or_last_year_means_no_years():
    for question in ["What risks does Tesla report about battery supply?", "What did Ford earn last year?"]:
        scope = pipelines.run_scope_check(question)
        assert scope.years == []
        assert scope.other_companies == []

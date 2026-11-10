"""Stage 4: decompose. The first tests need no LLM; Q3 and the sidebar test call OpenRouter."""
import uuid

from langchain_core.documents import Document

from backend import pipelines
from backend.pipelines import SubQuestion


def part(question, companies=(), years=()):
    return SubQuestion(question=question, companies=list(companies), years=list(years))


# ----- no LLM -----

def test_part_about_a_company_that_is_not_allowed_is_skipped():
    parts = [part("Tesla tariffs", ["TSLA"], [2025]), part("Ford tariffs", ["F"], [2025]),
             part("GM tariffs", ["GM"], [2025])]
    kept = pipelines.limit_parts(parts, ["TSLA", "F"], [2025], max_parts=4)
    assert [p.question for p in kept] == ["Tesla tariffs", "Ford tariffs"]


def test_extra_company_or_year_is_trimmed_but_the_part_stays():
    kept = pipelines.limit_parts([part("Tesla and GM", ["TSLA", "GM"], [2024, 2025])], ["TSLA"], [2025], 4)
    assert (kept[0].companies, kept[0].years) == (["TSLA"], [2025])


def test_part_without_company_or_year_gets_the_resolved_lists():
    kept = pipelines.limit_parts([part("risks")], ["TSLA", "F"], [2025], 4)
    assert (kept[0].companies, kept[0].years) == (["TSLA", "F"], [2025])
    # nothing resolved ([] = all): the part keeps its own lists
    kept = pipelines.limit_parts([part("Ford risks", ["F"], [2024])], [], [], 4)
    assert (kept[0].companies, kept[0].years) == (["F"], [2024])


def test_a_part_with_a_year_that_is_not_allowed_is_skipped():
    assert pipelines.limit_parts([part("Ford 2024", ["F"], [2024])], ["F"], [2025], 4) == []


def test_at_most_4_parts():
    parts = [part(f"question {i}", ["F"], [2025]) for i in range(6)]
    assert len(pipelines.limit_parts(parts, [], [], max_parts=4)) == 4


def test_a_chunk_found_by_two_parts_is_kept_once():
    def doc(text, company="F"):
        return Document(page_content=text, metadata={"company": company, "year": 2025, "item": "1A"})

    merged = pipelines.merge_parts([[doc("a"), doc("b")], [doc("b"), doc("c"), doc("a", company="TSLA")]])
    assert [(d.metadata["company"], d.page_content) for d in merged] == [
        ("F", "a"), ("F", "b"), ("F", "c"), ("TSLA", "a")]  # same text but another company is a different chunk


# ----- with the LLM -----

def test_q3_compare_tesla_and_ford(test_data):
    q = next(q for q in test_data["questions"] if q["id"] == 3)
    result = pipelines.answer(q["question"], q["pipeline"], [], [], str(uuid.uuid4()))

    found = {(s["company"], s["year"]) for s in result["sources"]}
    for wanted in q["expect"]["sources_include"]:
        assert (wanted["company"], wanted["year"]) in found
    assert found == {("TSLA", 2025), ("F", 2025)}  # nothing from other companies or years
    assert len(result["sources"]) <= 12  # at most 4 parts x 3 chunks
    assert len({s["excerpt"] for s in result["sources"]}) == len(result["sources"])  # no chunk twice
    assert result["tools_used"] == []
    print("\nANSWER:", result["answer"])
    for s in result["sources"]:
        print("SOURCE:", s["company"], s["year"], "Item", s["item"])


def test_decompose_checks_the_sidebar():
    result = pipelines.answer("What was GM's R&D expense in 2025?", "decompose", ["TSLA", "F"], [], str(uuid.uuid4()))
    assert result["answer"].startswith("This question asks about something that is not selected in the sidebar:")
    assert "General Motors" in result["answer"]
    assert result["sources"] == []

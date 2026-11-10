"""Stage 2: filters and the sidebar check. The first tests need no LLM; Q2 and Q9 call OpenRouter."""
import uuid

from backend import pipelines
from backend.pipelines import ScopeResult


def question_by_id(test_data, number: int) -> dict:
    return next(q for q in test_data["questions"] if q["id"] == number)


def scope(companies=(), years=()):
    return ScopeResult(companies=list(companies), other_companies=[], years=list(years))


# ----- no LLM -----

def test_resolve_filters_sidebar_wins_and_each_list_is_separate():
    # sidebar companies replace only the companies; the years still come from the question
    assert pipelines.resolve_filters(scope(["GM"], [2024]), ["F", "GM"], []) == (["F", "GM"], [2024])
    # nothing in the sidebar: use the question; nothing anywhere: empty lists (= all)
    assert pipelines.resolve_filters(scope(["GM"], [2024]), [], []) == (["GM"], [2024])
    assert pipelines.resolve_filters(scope(), [], []) == ([], [])


def test_check_sidebar_names_what_is_missing():
    message = pipelines.check_sidebar(scope(["GM"], [2025]), ["TSLA", "F"], [2024])
    assert message == ("This question asks about something that is not selected in the sidebar: "
                       "General Motors, FY2025. Select it in the sidebar, or change the question.")


def test_check_sidebar_allows_empty_lists_and_matches():
    assert pipelines.check_sidebar(scope(["GM"], [2025]), [], []) is None
    assert pipelines.check_sidebar(scope(["GM"], [2025]), ["GM", "F"], [2025]) is None
    assert pipelines.check_sidebar(scope(), ["TSLA"], [2024]) is None  # the question names nothing


# ----- with the LLM -----

def test_q2_filters_only_gm_2024(test_data):
    q = question_by_id(test_data, 2)
    result = pipelines.answer(q["question"], q["pipeline"], [], [], str(uuid.uuid4()))

    assert result["sources"]
    for source in result["sources"]:
        assert (source["company"], source["year"]) == ("GM", 2024)
    assert result["tools_used"] == []
    print("\nANSWER:", result["answer"])
    for s in result["sources"]:
        print("SOURCE:", s["company"], s["year"], "Item", s["item"])


def test_q9_sidebar_conflict(test_data):
    q = question_by_id(test_data, 9)
    sidebar = q["sidebar"]
    result = pipelines.answer(q["question"], q["pipeline"], sidebar["companies"], sidebar["years"], str(uuid.uuid4()))

    assert result["answer"].startswith(q["expect"]["answer_starts_with"])
    assert "General Motors" in result["answer"]
    assert result["sources"] == []
    assert result["tools_used"] == []
    print("\nANSWER:", result["answer"])


def test_simple_ignores_the_sidebar():
    result = pipelines.answer("What risks does Tesla report about battery supply?", "simple", ["F"], [2024],
                              str(uuid.uuid4()))
    companies = {s["company"] for s in result["sources"]}
    assert len(result["sources"]) == 5
    assert companies != {"F"}  # not limited to the sidebar

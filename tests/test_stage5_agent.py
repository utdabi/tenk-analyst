"""Stage 5: the agent. The first tests need no LLM (they call SEC and Qdrant); the Q tests call OpenRouter."""
import uuid

import pytest

from backend import agent, pipelines, sec_data


def call_tool(tool, **args):
    """Call a tool the way the agent does, so that we get both the text and the sources (the artifact)."""
    message = tool.invoke({"name": tool.name, "args": args, "id": "test", "type": "tool_call"})
    return message.content, message.artifact


def tools(sidebar_companies=(), sidebar_years=()):
    search_10k, sec_financial_data = agent.build_tools(list(sidebar_companies), list(sidebar_years))
    return search_10k, sec_financial_data


# ----- amounts and SEC data (no LLM) -----

def test_format_amount():
    assert sec_data.format_amount(9_400_000_000, 2025) == "$9.4 billion (FY2025)"
    assert sec_data.format_amount(-3_700_000_000, 2025) == "-$3.7 billion (FY2025)"
    assert sec_data.format_amount(12_345_678, 2025) == "$12.3 million (FY2025)"
    assert sec_data.format_amount(-12_345_678, 2024) == "-$12.3 million (FY2024)"
    assert sec_data.format_amount(60_000_000, 2024) == "$0.1 billion (FY2024)"  # $0.05 billion or more: billions
    assert sec_data.format_amount(40_000_000, 2024) == "$40.0 million (FY2024)"  # less than that: millions


def test_sec_tool_returns_ford_revenue_and_not_gm_rd(test_data):
    _, sec = tools()
    text, sources = call_tool(sec, ticker="F", metric="revenue", years=[2024])
    assert "$185.0 billion (FY2024)" in text
    assert sources == [{"type": "sec", "company": "F", "year": 2024, "metric": "revenue", "tag": "Revenues",
                        "value": sources[0]["value"]}]
    assert round(sources[0]["value"] / 1e9, 1) == test_data["checked_values"]["ford_revenue_2024_billion"]

    text, sources = call_tool(sec, ticker="GM", metric="rd_expense", years=[2025])
    assert "No rd_expense value" in text and "search_10k" in text
    assert sources == []  # a missing number adds no source


def test_sec_tool_unsupported_inputs():
    _, sec = tools()
    assert "Supported companies" in call_tool(sec, ticker="TM", metric="revenue")[0]
    assert "Supported fiscal years" in call_tool(sec, ticker="F", metric="revenue", years=[2023])[0]
    assert "Supported metrics" in call_tool(sec, ticker="F", metric="profit")[0]


# ----- the sidebar limits the tools (no LLM) -----

def test_tools_respect_the_sidebar():
    search, sec = tools(["TSLA", "F"], [2025])
    text, sources = call_tool(sec, ticker="GM", metric="revenue")
    assert "General Motors" in text and "sidebar" in text and sources == []
    text, sources = call_tool(sec, ticker="F", metric="revenue", years=[2024])
    assert "FY2024" in text and "sidebar" in text and sources == []
    text, sources = call_tool(search, query="cybersecurity", companies=["GM"])
    assert "General Motors" in text and sources == []


def test_tools_cannot_widen_the_search():
    search, sec = tools(["TSLA", "F"], [])
    # Ford and GM requested: GM is dropped, Ford is searched
    _, sources = call_tool(search, query="tariffs", companies=["F", "GM"], years=[2025])
    assert sources and {s["company"] for s in sources} == {"F"}
    # nothing requested: the sidebar companies are used, not all ten
    _, sources = call_tool(search, query="tariffs", years=[2025])
    assert {s["company"] for s in sources} <= {"TSLA", "F"}
    # SEC tool: years not given -> both years
    text, sources = call_tool(sec, ticker="F", metric="revenue")
    assert [s["year"] for s in sources] == [2024, 2025]


def test_search_finds_the_gm_rd_sentence():
    """The search alone must find GM's R&D number, because the agent depends on it for Q5."""
    search, _ = tools()
    text, sources = call_tool(search, query="research and development expenses", companies=["GM"], years=[2025])
    assert "$8.5 billion" in text
    assert len(sources) == 5 and {(s["company"], s["year"]) for s in sources} == {("GM", 2025)}


# ----- the agent (LLM) -----

def check(result, expect):
    """Check a result against the "expect" part of a question in questions.json."""
    answer, sources = result["answer"], result["sources"]
    if "answer_equals" in expect:
        assert answer == expect["answer_equals"]
    if "answer_starts_with" in expect:
        assert answer.startswith(expect["answer_starts_with"])
    for text in expect.get("answer_contains", []):
        assert text in answer
    if "source_count" in expect:
        assert len(sources) == expect["source_count"]
    for wanted in expect.get("sources_include", []):
        assert any(all(s.get(k) == v for k, v in wanted.items()) for s in sources), wanted
    for name in expect.get("tools_include", []):
        assert name in result["tools_used"]
    if "tools_used" in expect:
        assert result["tools_used"] == expect["tools_used"]


def test_step_limit_gives_the_fixed_message_and_keeps_the_sources(monkeypatch):
    monkeypatch.setattr(agent, "RECURSION_LIMIT", 2)  # too small for: model -> tool -> model
    result = pipelines.answer("What was Ford's R&D expense in 2025?", "agent", [], [], str(uuid.uuid4()))
    assert result["answer"] == pipelines.TOO_MANY_STEPS_MESSAGE
    assert result["tools_used"] == ["sec_financial_data"]  # what ran before the limit is still reported
    assert result["sources"] and result["sources"][0]["type"] == "sec"


@pytest.mark.parametrize("number", [4, 5, 6, 7, 10])
def test_agent_questions(test_data, number):
    q = next(q for q in test_data["questions"] if q["id"] == number)
    sidebar = q.get("sidebar", {"companies": [], "years": []})
    result = pipelines.answer(q["question"], "agent", sidebar["companies"], sidebar["years"], str(uuid.uuid4()))

    print(f"\nQ{number}: {q['question']}\nANSWER: {result['answer']}\nTOOLS: {result['tools_used']}")
    for s in result["sources"]:
        print("SOURCE:", {k: v for k, v in s.items() if k != "excerpt"})
    check(result, q["expect"])
    assert result["pipeline"] == "agent"

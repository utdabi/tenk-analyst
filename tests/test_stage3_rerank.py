"""Stage 3: rerank. The first test needs no LLM; the others call OpenRouter."""
import uuid

from backend import pipelines, rag


def test_rerank_keeps_5_of_the_20_chunks():
    question = "What does GM say about cybersecurity in 2024?"
    found = rag.search(question, k=20, flt=rag.make_filter(["GM"], [2024]))
    kept = rag.rerank(question, found, top_n=5)

    assert len(found) == 20
    assert len(kept) == 5
    assert all(chunk in found for chunk in kept)  # rerank only picks from the 20, it adds nothing
    assert len({id(chunk) for chunk in kept}) == 5  # no chunk twice
    assert rag.rerank(question, [], top_n=5) == []


def test_q2_with_rerank_gives_5_gm_2024_sources(test_data):
    q = next(q for q in test_data["questions"] if q["id"] == 2)
    result = pipelines.answer(q["question"], "rerank", [], [], str(uuid.uuid4()))

    assert len(result["sources"]) == 5
    for source in result["sources"]:
        assert (source["company"], source["year"]) == ("GM", 2024)
    assert result["tools_used"] == []
    print("\nANSWER:", result["answer"])
    for s in result["sources"]:
        print("SOURCE:", s["company"], s["year"], "Item", s["item"], "|", s["excerpt"][:80].replace("\n", " "))


def test_rerank_pipeline_also_checks_the_sidebar():
    result = pipelines.answer("What was GM's R&D expense in 2025?", "rerank", ["TSLA", "F"], [], str(uuid.uuid4()))
    assert result["answer"].startswith("This question asks about something that is not selected in the sidebar:")
    assert "General Motors" in result["answer"]
    assert result["sources"] == []

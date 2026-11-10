"""Stage 6: the agent's memory in Neon. These tests call OpenRouter, Qdrant, SEC, and Neon."""
import uuid

from backend import agent, pipelines


def ask(question, chat_id):
    return pipelines.answer(question, "agent", [], [], chat_id)


def saved_thread(chat_id):
    """What Neon holds for this chat ID (None if nothing is saved)."""
    return agent.get_checkpointer().get_tuple({"configurable": {"thread_id": chat_id}})


def test_q8_follow_up_question(test_data):
    q = next(q for q in test_data["questions"] if q["id"] == 8)
    chat_id = str(uuid.uuid4())  # a new chat ID on every run
    first, second = q["conversation"]
    expected_first, expected_second = q["expect"]["answers_contain"]

    result1 = ask(first, chat_id)
    result2 = ask(second, chat_id)
    print("\nQ:", first, "\nA:", result1["answer"], "\nQ:", second, "\nA:", result2["answer"])

    for text in expected_first:
        assert text in result1["answer"]
    for text in expected_second:
        assert text in result2["answer"]
    # The sources belong to the current question only: the second answer has the FY2025 number as a source.
    assert any(s["type"] == "sec" and s["company"] == "F" and s["year"] == 2025 for s in result2["sources"])
    assert result2["tools_used"]  # the follow-up looked up its own number


def test_a_new_chat_does_not_remember():
    result = ask("And in 2025?", str(uuid.uuid4()))
    print("\nA:", result["answer"])
    assert "187.3" not in result["answer"]  # no Ford, no earlier question: nothing to look up


def test_the_conversation_is_saved_in_neon():
    chat_id = str(uuid.uuid4())
    ask("What was Ford's revenue in 2024?", chat_id)
    saved = saved_thread(chat_id)
    assert saved is not None
    messages = saved.checkpoint["channel_values"]["messages"]
    assert messages[0].content == "What was Ford's revenue in 2024?"  # the question
    assert len(messages) >= 3  # question, tool call, tool result, answer


def test_a_refused_question_is_not_saved():
    chat_id = str(uuid.uuid4())
    assert ask("What risks does Toyota report?", chat_id)["answer"] == pipelines.COMPANY_MESSAGE
    assert saved_thread(chat_id) is None  # the agent did not run

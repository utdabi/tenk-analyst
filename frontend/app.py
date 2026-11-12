"""The Streamlit front end. It talks to the back end only through POST /ask.

Start it with:  uv run python -m streamlit run frontend/app.py
"""
import logging
import os
import secrets
import time
import uuid

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.environ.get("API_URL", "http://localhost:8000")
ACCESS_CODE = os.environ.get("ACCESS_CODE", "")

# These lines appear in the Render logs of the front end. They never contain the access code or the question.
log = logging.getLogger("tenk-ui")
if not log.handlers:  # Streamlit runs this file again on every click; add the handler only once
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)
    log.propagate = False
    # repr() shows a hidden space or line break at the end of the pasted value
    log.info("Front end started. API_URL=%r, ACCESS_CODE set=%s", API_URL, bool(ACCESS_CODE))

# The front end does not import the back-end code (they are two separate services), so the names are repeated here.
COMPANIES = {"TSLA": "Tesla", "GM": "General Motors", "F": "Ford", "RIVN": "Rivian", "LCID": "Lucid",
             "APTV": "Aptiv", "BWA": "BorgWarner", "ALV": "Autoliv", "HOG": "Harley-Davidson", "CVNA": "Carvana"}
YEARS = [2024, 2025]
PIPELINES = {
    "agent": "agent: tools (10-K search and SEC numbers) and memory",
    "decompose": "decompose: splits the question into parts",
    "rerank": "rerank: searches 20 chunks, keeps the best 5",
    "filters": "filters: searches only the companies and years asked",
    "simple": "simple: searches all chunks, no filters",
}
EXAMPLES = ["What risks does Tesla report about battery supply?",
            "Compare GM and Ford R&D in 2025",
            "How did Carvana's revenue change from 2024 to 2025?"]
UNAVAILABLE = "The answer service is not available right now. Try again in a minute."
# Seconds to wait for a sleeping server (a cold start on Render can take 1 to 3 minutes). A test can set it lower.
WAKE_UP_LIMIT = int(os.environ.get("WAKE_UP_LIMIT", "180"))


# ---------- Talking to the back end ----------

def note_problem(text: str) -> None:
    """Write why a call failed to the Render log. The user does not see it."""
    log.warning("%s", text)


def post_question(payload: dict) -> dict | None:
    """Send the question once. Return the JSON reply, or None when the server did not really answer."""
    try:
        response = httpx.post(f"{API_URL}/ask", json=payload, headers={"X-Access-Code": ACCESS_CODE}, timeout=90)
    except httpx.TransportError as error:  # no connection, or a timeout
        note_problem(f"POST /ask failed: {type(error).__name__}")
        return None
    try:
        return response.json()  # our back end always answers with JSON, also for its errors
    except ValueError:  # Render's own "waking up" page is not JSON
        note_problem(f"POST /ask got a page that is not JSON (status {response.status_code})")
        return None


def backend_is_up() -> bool:
    """GET /health is free. Check for our JSON, because Render's waking-up page can also have status 200."""
    try:
        response = httpx.get(f"{API_URL}/health", timeout=10)
    except httpx.TransportError as error:
        note_problem(f"GET /health failed: {type(error).__name__}")
        return False
    try:
        if response.json().get("status") == "ok":
            return True
    except ValueError:
        pass
    note_problem(f"GET /health did not return our JSON (status {response.status_code})")
    return False


def ask_backend(payload: dict) -> dict:
    """Send the question. If the server does not answer, it is probably asleep: ask /health every 3 seconds
    (showing the seconds on the page) until it is up, then send the question again."""
    started = time.time()
    log.info("Sending the question (pipeline=%s, length=%d)", payload["pipeline"], len(payload["question"]))
    reply = post_question(payload)
    if reply is not None:
        log.info("Got a reply after %.1f s", time.time() - started)
        return reply
    log.info("No real reply after %.1f s. Waiting for /health", time.time() - started)
    waiting = st.empty()
    started = time.time()
    while time.time() - started < WAKE_UP_LIMIT:
        if backend_is_up():
            log.info("/health is ok after %d s. Sending the question again", time.time() - started)
            waiting.empty()
            return post_question(payload) or {"error": UNAVAILABLE}
        waiting.info(f"The server is waking up... {int(time.time() - started)} s (this can take up to 3 minutes)")
        time.sleep(3)
    waiting.empty()
    log.error("Gave up after %d s", WAKE_UP_LIMIT)
    return {"error": "The server did not start.", "wake_failed": True}  # show_message adds the link and the button


# ---------- Showing a message ----------

def show_source(source: dict) -> None:
    if source["type"] == "sec":
        st.markdown(f"**SEC data** - {COMPANIES[source['company']]}, FY{source['year']}, {source['metric']} "
                    f"(`{source['tag']}`): {source['value']:,.0f} USD")
    else:
        link = f"[SEC filing]({source['url']})"
        st.markdown(f"**{COMPANIES[source['company']]}, FY{source['year']} 10-K, Item {source['item']}** - "
                    f"{source['item_title']} ({link})")
        st.caption(source["excerpt"])


def retry_last() -> None:
    """The "Try again" button: take the question back and ask it again, so the user does not retype it."""
    question = st.session_state.messages[-2]["content"]
    del st.session_state.messages[-2:]  # the question and the failed reply; the question is added again when asked
    st.session_state.pending = question


def show_message(message: dict, show_sources: bool, can_retry: bool = False) -> None:
    with st.chat_message(message["role"]):
        if message.get("wake_failed"):  # the back end is asleep: tell the user how to wake it
            st.warning(f"The server did not start. Click [{API_URL}]({API_URL}) to wake it up, "
                       "then come back and press **Try again**.")
            if can_retry:
                st.button("Try again", key=f"retry_{len(st.session_state.messages)}", on_click=retry_last)
            return
        if message.get("failed"):
            st.error(message["content"])
            return
        st.markdown(message["content"])
        if message["role"] != "assistant":
            return
        tools = message["tools_used"]  # the labels are always shown
        st.markdown("Tools used: " + (" ".join(f":blue-badge[{name}]" for name in tools) if tools else "none"))
        if show_sources and message["sources"]:
            with st.expander(f"Sources ({len(message['sources'])})"):
                for source in message["sources"]:
                    show_source(source)


# ---------- The page ----------

st.set_page_config(page_title="10-K Analyst", page_icon=":bar_chart:")

if not ACCESS_CODE:
    st.error("ACCESS_CODE is not set. Fill it in .env (or in the Render dashboard) and restart.")
    st.stop()

if not st.session_state.get("logged_in"):
    st.title("10-K Analyst")
    with st.form("login"):  # a form lets the Enter key submit the code
        code = st.text_input("Access code", type="password")
        entered = st.form_submit_button("Enter")
    if entered:
        if secrets.compare_digest(code.encode(), ACCESS_CODE.encode()):
            st.session_state.logged_in = True  # stays for this browser session
            st.rerun()
        st.error("Wrong access code.")
    st.stop()

if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())
    st.session_state.messages = []

with st.sidebar:
    st.header("Filters")
    companies = st.multiselect("Companies", list(COMPANIES), default=list(COMPANIES),
                               format_func=lambda t: f"{COMPANIES[t]} ({t})")
    years = st.multiselect("Fiscal years", YEARS, default=YEARS, format_func=lambda y: f"FY{y}")
    show_sources = st.toggle("Show sources", value=True)
    if st.button("New chat"):
        st.session_state.chat_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    with st.expander("Learning mode"):
        pipeline = st.radio("Pipeline", list(PIPELINES), format_func=PIPELINES.get, index=0)
        st.caption("Only the agent uses SEC numbers and remembers the chat.")

# Everything selected (or nothing) means "no filter": the back end gets an empty list.
sidebar_companies = [] if len(companies) in (0, len(COMPANIES)) else companies
sidebar_years = [] if len(years) in (0, len(YEARS)) else years

st.title("10-K Analyst")
st.caption("Ask about the 10-K filings of ten US auto companies, fiscal years 2024 and 2025.")

for number, message in enumerate(st.session_state.messages):
    show_message(message, show_sources, can_retry=number == len(st.session_state.messages) - 1)  # only the last one

question = st.chat_input("Ask a question") or st.session_state.pop("pending", None)
if not st.session_state.messages and not question:
    for example in EXAMPLES:
        if st.button(example):
            st.session_state.pending = example  # answered in the next run, when the buttons are gone
            st.rerun()

if question:
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    show_message(user_message, show_sources)

    with st.spinner("Searching the filings..."):
        reply = ask_backend({"question": question, "chat_id": st.session_state.chat_id, "pipeline": pipeline,
                             "companies": sidebar_companies, "years": sidebar_years})
    if "answer" in reply:
        assistant_message = {"role": "assistant", "content": reply["answer"], "tools_used": reply["tools_used"],
                             "sources": reply["sources"]}
    else:  # the back end sent an error text (or FastAPI sent its own 422 reply)
        assistant_message = {"role": "assistant", "content": reply.get("error", UNAVAILABLE), "failed": True,
                             "wake_failed": reply.get("wake_failed", False)}
    st.session_state.messages.append(assistant_message)
    show_message(assistant_message, show_sources, can_retry=True)

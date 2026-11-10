"""The Streamlit front end. It talks to the back end only through POST /ask.

Start it with:  uv run python -m streamlit run frontend/app.py
"""
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
WAKE_UP_SECONDS = 30  # how long to wait for a sleeping server before the second try


# ---------- Talking to the back end ----------

def ask_backend(payload: dict) -> dict:
    """Send the question. A sleeping server may not answer, or Render may answer with its own "waking up" page.
    In both cases, wait and try once more."""
    for attempt in (1, 2):
        waited_already = False
        try:
            response = httpx.post(f"{API_URL}/ask", json=payload, headers={"X-Access-Code": ACCESS_CODE},
                                  timeout=90)
            try:
                return response.json()  # our back end always answers with JSON, also for its errors
            except ValueError:
                pass  # not JSON: this is Render's page for a server that is still starting
        except httpx.TimeoutException:
            waited_already = True  # we have just waited 90 seconds, so no need to wait more
        except httpx.TransportError:
            pass  # no connection: the server is not up yet
        if attempt == 1:
            st.info("The server is waking up. This takes about 30 seconds.")
            if not waited_already:
                time.sleep(WAKE_UP_SECONDS)
    return {"error": UNAVAILABLE}


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


def show_message(message: dict, show_sources: bool) -> None:
    with st.chat_message(message["role"]):
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

for message in st.session_state.messages:
    show_message(message, show_sources)

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
        assistant_message = {"role": "assistant", "content": reply.get("error", UNAVAILABLE), "failed": True}
    st.session_state.messages.append(assistant_message)
    show_message(assistant_message, show_sources)

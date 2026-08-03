"""
Streamlit UI for Earnings Intelligence Platform.
Talks to the FastAPI backend at localhost:8000.
"""

import streamlit as st
import requests
import json

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Earnings Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Earnings Intelligence Platform")
st.caption("Ask questions about SEC 10-K filings for AAPL, MSFT, NVDA, GOOGL, META")

# Load available filings for dropdowns
@st.cache_data
def get_filings():
    try:
        resp = requests.get(f"{API_URL}/filings")
        return resp.json()
    except Exception:
        return {}

filings = get_filings()

# --- Sidebar filters ---
st.sidebar.header("Filters")

tickers = ["All"] + sorted(filings.keys())
selected_ticker = st.sidebar.selectbox("Company", tickers)

if selected_ticker != "All" and selected_ticker in filings:
    years = ["All"] + [str(y) for y in filings[selected_ticker]]
else:
    all_years = sorted(set(
        y for years in filings.values() for y in years
    ), reverse=True)
    years = ["All"] + [str(y) for y in all_years]

selected_year = st.sidebar.selectbox("Year", years)

sections = ["All", "Risk Factors", "Business", "MD&A", "Market Risk", "Financial Statements"]
selected_section = st.sidebar.selectbox("Section", sections)

num_results = st.sidebar.slider("Number of sources", 3, 10, 5)

# --- Main query area ---
st.subheader("Ask a question")

# Example questions
examples = [
    "What are Apple's main risk factors?",
    "How did NVIDIA describe its AI business opportunity?",
    "What did Microsoft say about cloud revenue growth?",
    "How does Meta describe its advertising business risks?",
    "What supply chain risks did Apple flag?",
    "Compare how these companies talk about AI investment",
]

selected_example = st.selectbox("Try an example question", [""] + examples)
question = st.text_input(
    "Or type your own question",
    value=selected_example,
    placeholder="e.g. What are the main risks Apple faces in China?"
)

if st.button("Ask", type="primary") and question:
    with st.spinner("Searching filings and generating answer..."):
        payload = {
            "question": question,
            "ticker": None if selected_ticker == "All" else selected_ticker,
            "year": None if selected_year == "All" else int(selected_year),
            "section": None if selected_section == "All" else selected_section,
            "num_results": num_results,
        }

        try:
            resp = requests.post(f"{API_URL}/query", json=payload, timeout=60)
            result = resp.json()

            # Answer
            st.subheader("Answer")
            st.markdown(result["answer"])

            # Sources
            st.subheader("Sources")
            for i, src in enumerate(result["sources"], 1):
                st.markdown(
                    f"**[{i}]** {src['ticker']} {src['year']} 10-K — "
                    f"*{src['section']}* (chunk {src['chunk_index']})"
                )

            # Token usage
            col1, col2 = st.columns(2)
            col1.metric("Input tokens", result.get("input_tokens", "?"))
            col2.metric("Output tokens", result.get("output_tokens", "?"))

            # Feedback
            st.subheader("Was this answer helpful?")
            col1, col2 = st.columns(2)
            if col1.button("👍 Yes"):
                requests.post(f"{API_URL}/feedback", json={
                    "query": question,
                    "answer": result["answer"],
                    "feedback": 1,
                    "ticker": payload["ticker"],
                    "year": payload["year"],
                })
                st.success("Thanks for the feedback!")
            if col2.button("👎 No"):
                requests.post(f"{API_URL}/feedback", json={
                    "query": question,
                    "answer": result["answer"],
                    "feedback": -1,
                    "ticker": payload["ticker"],
                    "year": payload["year"],
                })
                st.info("Thanks — we'll use this to improve.")

        except Exception as e:
            st.error(f"Error: {e}")

# --- Footer ---
st.divider()
st.caption("Data source: SEC EDGAR | Search: Hybrid RRF (BM25 + pgvector) | LLM: Claude Sonnet")

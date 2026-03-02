"""
===============================================================================
STREAMLIT_APP.PY
===============================================================================

PURPOSE
-------
Frontend / UI for the 10-K RAG chatbot.

- Streamlit chat UI
- Calls RAG pipeline functions from rag_core.py
- Shows citations used for answers

===============================================================================
"""

import os
import re
import base64
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

from rag_core import (
    get_openai_client,
    get_chroma_collection,
    rewrite_query,
    embed_query,
    retrieve_top_n,
    rerank_with_llm,
    format_context_blocks,
    answer_with_citations,
)

# Maps stock tickers to company display names used in the UI.
COMPANY_NAMES = {
    "EW": "Edwards Lifesciences",
    "MDT": "Medtronic",
    "SYK": "Stryker",
    "BSX": "Boston Scientific",
    "ISRG": "Intuitive Surgical",
    "ABT": "Abbott",
    "ZBH": "Zimmer Biomet",
    "ALGN": "Align Technology",
    "PODD": "Insulet",
    "DXCM": "Dexcom",
}


# =============================================================================
# SESSION STATE
# =============================================================================
def init_session_state():
    """Initializes Streamlit session variables used for chat history, citations, and sidebar selections."""
    if "history" not in st.session_state:
        st.session_state.history = []

    if "last_citations" not in st.session_state:
        st.session_state.last_citations = []

    if "allowed_tickers" not in st.session_state:
        st.session_state.allowed_tickers = []  # [] means no filter (all companies)


# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar():
    """Renders the sidebar UI (logo, New chat button, and company filter) and applies sidebar CSS styling."""
    icon_base64 = ""
    if os.path.exists("10k.png"):
        with open("10k.png", "rb") as f:
            icon_base64 = base64.b64encode(f.read()).decode()

    st.sidebar.markdown(
        f"""
    <style>
      /* Sidebar width in collapsed mode */
      section[data-testid="stSidebar"][aria-expanded="false"]{{
        width: 100px !important;
        min-width: 100px !important;
        max-width: 100px !important;
        overflow: visible !important;
        transform: none !important;
      }}

      /* Logo container at the top of the sidebar */
      .top-left-icon {{
        position: sticky;
        top: 6px;
        left: 6px;
        z-index: 9999;
        margin: 0 0 16px 0;
        width: 100%;
        display: flex;
        justify-content: flex-start;
        align-items: center;
        box-sizing: border-box;
      }}

      .top-left-icon img {{
        width: 78px;
        height: 78px;
        object-fit: contain;
        display: block;
      }}

      /* Keep the logo visible and sized to fit when the sidebar is collapsed */
      section[data-testid="stSidebar"][aria-expanded="false"] .top-left-icon {{
        justify-content: center;
      }}
      section[data-testid="stSidebar"][aria-expanded="false"] .top-left-icon img {{
        width: 54px;
        height: 54px;
      }}

      /* Hide sidebar text/content when collapsed */
      section[data-testid="stSidebar"][aria-expanded="false"] .nav-label {{ display:none !important; }}
      section[data-testid="stSidebar"][aria-expanded="false"] .stMarkdown,
      section[data-testid="stSidebar"][aria-expanded="false"] label,
      section[data-testid="stSidebar"][aria-expanded="false"] small {{ display:none !important; }}

      /* Show the markdown block that contains the logo even when collapsed */
      section[data-testid="stSidebar"][aria-expanded="false"] .stMarkdown:has(.top-left-icon) {{
        display: block !important;
      }}

      /* Hide the company expander/selectbox when collapsed */
      section[data-testid="stSidebar"][aria-expanded="false"] details,
      section[data-testid="stSidebar"][aria-expanded="false"] details * {{
        display: none !important;
      }}
      section[data-testid="stSidebar"][aria-expanded="false"] div[data-testid="stSelectbox"],
      section[data-testid="stSidebar"][aria-expanded="false"] div[data-testid="stSelectbox"] * {{
        display: none !important;
      }}
      section[data-testid="stSidebar"][aria-expanded="false"] .company-expander {{
        display: none !important;
      }}

      /* Fix icon button sizing so it stays consistent in both modes */
      section[data-testid="stSidebar"] div[data-testid="stButton"]{{
        width: 52px !important;
        min-width: 52px !important;
        max-width: 52px !important;
      }}

      section[data-testid="stSidebar"] div[data-testid="stButton"] > button{{
        width: 52px !important;
        height: 52px !important;
        min-width: 52px !important;
        min-height: 52px !important;
        max-width: 52px !important;
        max-height: 52px !important;

        padding: 0 !important;
        border-radius: 14px !important;
        border: 1px solid rgba(0,0,0,0.10) !important;
        background: #fff !important;

        display: flex !important;
        align-items: center !important;
        justify-content: center !important;

        font-size: 20px !important;
        line-height: 1 !important;
        box-sizing: border-box !important;
      }}

      section[data-testid="stSidebar"][aria-expanded="false"] [data-testid="column"]{{
        min-width: 80px !important;
        width: 80px !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
      }}

      section[data-testid="stSidebar"][aria-expanded="false"] .block-container{{
        padding-left: 0 !important;
        padding-right: 0 !important;
      }}

      .nav-label{{
        font-size: 16px;
        line-height: 1;
        padding-top: 2px;
      }}
    </style>

    <div class="top-left-icon">
        {"<img src='data:image/png;base64," + icon_base64 + "'>" if icon_base64 else ""}
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar action: clears chat state and reruns the app.
    c1, c2 = st.sidebar.columns([1, 4], vertical_alignment="center")
    with c1:
        new_chat_clicked = st.button("✎", key="newchat_icon_only")
    with c2:
        st.markdown('<div class="nav-label">New chat</div>', unsafe_allow_html=True)

    if new_chat_clicked:
        st.session_state.history = []
        st.session_state.last_citations = []
        st.rerun()

    # Sidebar filter: allows selecting one company ticker to constrain retrieval.
    c3, c4 = st.sidebar.columns([1, 4], vertical_alignment="center")
    with c3:
        st.button("▦", key="company_icon_only")
    with c4:
        st.markdown('<div class="nav-label">Company</div>', unsafe_allow_html=True)

    company_options = ["All companies"] + [
        f"{t} — {COMPANY_NAMES[t]}" for t in COMPANY_NAMES.keys()
    ]

    st.sidebar.markdown('<div class="company-expander">', unsafe_allow_html=True)
    with st.sidebar.expander("Company filter", expanded=True):
        selected_company_label = st.selectbox(
            "Select ONE company (optional)",
            options=company_options,
            index=0,
            key="company_select",
        )
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

    if selected_company_label == "All companies":
        st.session_state.allowed_tickers = []
    else:
        selected_ticker = selected_company_label.split(" — ")[0].strip()
        st.session_state.allowed_tickers = [selected_ticker]


# =============================================================================
# SECTION-AWARE RETRIEVAL
# =============================================================================
def infer_section_preferences(query: str) -> Dict[str, Any]:
    """Detects if the query likely targets specific 10-K sections (e.g., Risk Factors) and returns filter hints."""
    q = query.lower()
    risk_signals = [
        "risk",
        "risk factor",
        "risks",
        "uncertaint",
        "litigation",
        "regulatory risk",
        "compliance risk",
    ]
    if any(s in q for s in risk_signals):
        return {"item_heading_contains_any": ["ITEM 1A", "RISK FACTORS"]}
    return {}


def section_aware_retrieve(
    collection,
    query_emb: List[float],
    preferences: Dict[str, Any],
    final_n: int = 20,
    initial_pool: int = 80,
    allowed_tickers: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """Retrieves an initial semantic pool, optionally filters by section hints, and returns the top results."""
    big_pool = retrieve_top_n(
        collection,
        query_emb,
        n=initial_pool,
        allowed_tickers=allowed_tickers or [],
    )

    if not preferences:
        return big_pool[:final_n]

    wanted = [w.upper() for w in preferences.get("item_heading_contains_any", [])]

    filtered: List[Dict[str, Any]] = []
    for h in big_pool:
        heading = str(h.get("meta", {}).get("item_heading", "")).upper()
        if any(w in heading for w in wanted):
            filtered.append(h)

    if len(filtered) >= final_n:
        return filtered[:final_n]

    return big_pool[:final_n]
# =============================================================================
# Build reverse lookup: company name -> ticker (lowercased)
# =============================================================================

COMPANY_NAME_TO_TICKER = {v.lower(): k for k, v in COMPANY_NAMES.items()}

def detect_company_mentions(text: str) -> List[str]:
    """
    Detects which tickers are referenced in the user query.
    Matches:
      - explicit tickers: "SYK", "EW"
      - company names: "Stryker", "Edwards Lifesciences"
    Returns a list of tickers found (unique).
    """
    if not text:
        return []

    t = text.lower()
    found = set()

    # 1) Match tickers as whole words
    for ticker in COMPANY_NAMES.keys():
        if re.search(rf"\b{re.escape(ticker.lower())}\b", t):
            found.add(ticker)

    # 2) Match company names (substring match)
    for name_lc, ticker in COMPANY_NAME_TO_TICKER.items():
        if name_lc in t:
            found.add(ticker)

    return sorted(found)


def build_filter_mismatch_message(active_ticker: str, requested_tickers: List[str]) -> str:
    active_name = COMPANY_NAMES.get(active_ticker, active_ticker)
    # Use first mentioned ticker for the message (common case: user asks about one company)
    req = requested_tickers[0]
    req_name = COMPANY_NAMES.get(req, req)

    return (
        f"**Company filter is set to {active_name} ({active_ticker}).**\n\n"
        f"Your question looks like it’s about **{req_name} ({req})**.\n\n"
        "I can only answer using the selected company’s 10-K filings right now. "
        "Please switch the filter to the correct company (or choose **All companies**) and ask again."
    )

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    """Runs the Streamlit app: renders UI, processes chat input, retrieves context, generates answers, and displays sources."""
    load_dotenv()
    st.set_page_config(page_title="MedTech 10-K Explorer", layout="wide")

    st.markdown(
        """
    <style>
      :root { --st-topbar-h: 3.25rem; }

      .fixed-header {
        position: fixed;
        top: var(--st-topbar-h);
        left: 260px;
        right: 0;
        background: white;
        border-bottom: 1px solid rgba(0,0,0,0.08);
        padding: 15px 40px 16px 50px;
        z-index: 9999;
        transition: left 0.25s ease;
      }

      section[data-testid="stSidebar"][aria-expanded="false"] ~ div .fixed-header {
        left: 100px;
      }

      .fixed-header .title {
        font-size: 28px;
        font-weight: 750;
        line-height: 1.15;
        margin: 0;
      }

      .fixed-header .subtitle {
        font-size: 13px;
        color: #666;
        margin-top: 6px;
      }

      .block-container {
        padding-top: calc(var(--st-topbar-h) + 110px) !important;
      }
    </style>

    <div class="fixed-header">
      <div class="title">MedTech 10-K Explorer</div>
      <div class="subtitle">
        Ask questions about medical device companies’ SEC 10-K filings — answers include citations.
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    init_session_state()
    render_sidebar()

    EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    GEN_MODEL = os.getenv("GEN_MODEL", "gpt-4o-mini")

    try:
        oai = get_openai_client()
    except Exception as e:
        st.error(f"OpenAI client initialization failed: {e}")
        st.stop()

    try:
        col = get_chroma_collection()
    except Exception as e:
        st.error(f"Chroma database initialization failed: {e}")
        st.stop()

    if len(st.session_state.history) == 0:
        with st.chat_message("assistant"):
            st.markdown(
                """
**👋 Welcome to the Med Device 10-K Chatbot!**

You can ask questions about medical device companies’ SEC 10-K filings.

Examples:
- What does Edwards Lifesciences do?
- What are Medtronic's risk factors?
- Compare Abbott and Boston Scientific
- What products does Dexcom make?

All answers are based on official SEC Form 10-K filings and include source citations.
"""
            )

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_msg = st.chat_input(
        "Ask a question (e.g., 'Compare risk factors for Edwards Lifesciences vs Medtronic')"
    )
    if not user_msg:
        return

    # Save + display user message
    st.session_state.history.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.write(user_msg)

    # -------------------------------------------------------------------------
    # COMPANY FILTER GUARDRAIL
    # -------------------------------------------------------------------------
    allowed_tickers = st.session_state.get("allowed_tickers", [])

    # Only enforce when a single-company filter is active
    if allowed_tickers:
        active = allowed_tickers[0]  # your UI only allows ONE selected company

        mentioned = detect_company_mentions(user_msg)

        # If user mentions a different company than the active filter, block
        # (If user doesn't mention any company, we allow it and assume it's about the selected filter.)
        if mentioned and (active not in mentioned):
            msg = build_filter_mismatch_message(active, mentioned)

            # Save + show assistant response WITHOUT running retrieval
            st.session_state.history.append({"role": "assistant", "content": msg})
            with st.chat_message("assistant"):
                st.markdown(msg)
            return

    rewritten_query = rewrite_query(
        oai=oai,
        gen_model=GEN_MODEL,
        history=st.session_state.history,
        user_msg=user_msg,
    )

    try:
        qemb = embed_query(oai, EMBED_MODEL, rewritten_query)
        preferences = infer_section_preferences(rewritten_query)
        allowed_tickers = st.session_state.get("allowed_tickers", [])

        candidates = section_aware_retrieve(
            collection=col,
            query_emb=qemb,
            preferences=preferences,
            final_n=20,
            initial_pool=80,
            allowed_tickers=allowed_tickers,
        )
    except Exception as e:
        st.error(f"Retrieval failed: {e}")
        st.stop()

    if not candidates:
        answer = "I couldn't find relevant passages in the indexed 10-K filings."
        # Build a single markdown string that includes the answer + sources
        sources_lines = []
        seen = set()
        for c in citations:
            ticker = c.get("ticker", "")
            company = COMPANY_NAMES.get(ticker, ticker)

            match = re.search(r"-(\d{2})-", str(c.get("source_path", "")))
            year = "20" + match.group(1) if match else "Unknown"

            section = c.get("section", "Unknown section")
            marker = c.get("marker", "?")

            doc_key = (ticker, year)
            if doc_key in seen:
                continue
            seen.add(doc_key)

            sources_lines.append(f"- [{marker}] {company} — {year} Form 10-K — {section}")

        assistant_full = answer
        if sources_lines:
            assistant_full += "\n\n---\n\n**Sources (Evidence Used)**\n" + "\n".join(sources_lines)

        st.session_state.history.append({"role": "assistant", "content": assistant_full})
        with st.chat_message("assistant"):
            st.write(answer)
        return

    try:
        top5 = rerank_with_llm(oai, GEN_MODEL, rewritten_query, candidates, k=5)
    except Exception:
        top5 = candidates[:5]

    context, citations = format_context_blocks(top5)
    st.session_state.last_citations = citations

    # Generate + display assistant answer
    with st.chat_message("assistant"):

        typing_placeholder = st.empty()
        typing_placeholder.markdown(
            """
<div style="display:flex; gap:6px; align-items:center;">
  <div class="dot"></div><div class="dot"></div><div class="dot"></div>
</div>

<style>
.dot {
  height:10px;
  width:10px;
  background-color:#888;
  border-radius:50%;
  display:inline-block;
  animation:bounce 1.4s infinite ease-in-out;
}
.dot:nth-child(2) { animation-delay:0.2s; }
.dot:nth-child(3) { animation-delay:0.4s; }

@keyframes bounce {
  0%, 80%, 100% { transform: scale(0); }
  40% { transform: scale(1); }
}
</style>
""",
            unsafe_allow_html=True,
        )

        try:
            answer = answer_with_citations(
                oai=oai,
                gen_model=GEN_MODEL,
                history=st.session_state.history,
                user_msg=user_msg,
                context=context,
            )
        except Exception as e:
            answer = f"Answer generation failed: {e}"

        typing_placeholder.empty()
        st.write(answer)

        st.subheader("Sources (Evidence Used)")
        seen = set()
        for c in citations:
            ticker = c.get("ticker", "")
            company = COMPANY_NAMES.get(ticker, ticker)

            match = re.search(r"-(\d{2})-", str(c.get("source_path", "")))
            year = "20" + match.group(1) if match else "Unknown"

            section = c.get("section", "Unknown section")
            marker = c.get("marker", "?")

            doc_key = (ticker, year)
            if doc_key in seen:
                continue
            seen.add(doc_key)

            st.write(f"[{marker}] {company} — {year} Form 10-K — {section}")

    # Save answer
    sources_lines = []
    seen = set()

    for c in citations:
        ticker = c.get("ticker", "")
        company = COMPANY_NAMES.get(ticker, ticker)

        match = re.search(r"-(\d{2})-", str(c.get("source_path", "")))
        year = "20" + match.group(1) if match else "Unknown"

        section = c.get("section", "Unknown section")
        marker = c.get("marker", "?")

        doc_key = (ticker, year)
        if doc_key in seen:
            continue
        seen.add(doc_key)

        sources_lines.append(f"- [{marker}] {company} — {year} Form 10-K — {section}")

    assistant_full = answer
    if sources_lines:
        assistant_full += "\n\n---\n\n**Sources (Evidence Used)**\n" + "\n".join(sources_lines)

    # Save answer + sources together so they persist in history
    sources_lines = []
    seen = set()

    for c in citations:
        ticker = c.get("ticker", "")
        company = COMPANY_NAMES.get(ticker, ticker)

        match = re.search(r"-(\d{2})-", str(c.get("source_path", "")))
        year = "20" + match.group(1) if match else "Unknown"

        section = c.get("section", "Unknown section")
        marker = c.get("marker", "?")

        doc_key = (ticker, year)
        if doc_key in seen:
            continue
        seen.add(doc_key)

        sources_lines.append(f"- [{marker}] {company} — {year} Form 10-K — {section}")

    assistant_full = answer
    if sources_lines:
        assistant_full += "\n\n---\n\n**Sources (Evidence Used)**\n" + "\n".join(sources_lines)

    st.session_state.history.append({"role": "assistant", "content": assistant_full})


if __name__ == "__main__":
    main()
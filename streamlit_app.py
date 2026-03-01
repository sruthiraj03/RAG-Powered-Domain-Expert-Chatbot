"""
===============================================================================
STREAMLIT_APP.PY
===============================================================================

PURPOSE
-------
This file provides the FRONTEND / USER INTERFACE for the 10-K RAG chatbot.

It uses Streamlit's chat components to create a conversational UI and calls the
RAG pipeline functions (implemented in rag_core.py) to produce answers grounded
in SEC 10-K filings, with source citations.

WHAT THIS UI DOES (PER USER MESSAGE)
------------------------------------
1) Store the user message in conversation memory (session_state)
2) Rewrite the query using conversation history (helps follow-ups)
3) Embed the rewritten query (OpenAI embeddings)
4) Retrieve candidate chunks from Chroma (semantic vector search)
5) Section-aware filtering (metadata-aware): prioritize the right 10-K sections
6) Advanced feature: re-rank candidates down to top-5 using an LLM
7) Generate an answer using ONLY those top-5 chunks (with citations like [1][2])
8) Display citations so the user can inspect the evidence

WHY THIS VERSION IS "CLEANER / MORE ADVANCED"
---------------------------------------------
Pure embedding search can surface broad sections (e.g., "Item 1. Business")
even when the user asks about a specific section (e.g., "Item 1A. Risk Factors").

We improve precision using a *metadata-aware routing layer*:

- First: retrieve a larger semantic pool (fast + recall-focused)
- Second: filter locally by metadata (precision-focused, version-proof)
- Third: fall back to global results if filtering is too strict

This avoids Chroma version issues around `where` operators and keeps the app stable.

===============================================================================
"""

import os
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


# =============================================================================
# UI CONFIGURATION + SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """
    Initialize Streamlit session_state variables.

    session_state persists across reruns so we can store:
      - conversation history
      - debug toggle state
      - last citations shown
    """
    if "history" not in st.session_state:
        st.session_state.history = []

    if "last_citations" not in st.session_state:
        st.session_state.last_citations = []

    if "debug" not in st.session_state:
        st.session_state.debug = False


def render_sidebar():
    """
    Sidebar controls:
      - show configured models
      - enable debug view
      - clear chat
    """
    st.sidebar.header("Settings")

    embed_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    gen_model = os.getenv("GEN_MODEL", "gpt-4o-mini")

    st.sidebar.write("**Embedding model:**", embed_model)
    st.sidebar.write("**Generation model:**", gen_model)

    st.sidebar.divider()

    st.session_state.debug = st.sidebar.checkbox(
        "Show debug info",
        value=st.session_state.debug
    )

    if st.sidebar.button("Clear chat"):
        st.session_state.history = []
        st.session_state.last_citations = []
        st.rerun()


# =============================================================================
# SECTION-AWARE RETRIEVAL (CLEAN + VERSION-PROOF)
# =============================================================================

def infer_section_preferences(query: str) -> Dict[str, Any]:
    """
    Infer what section(s) the user likely wants, based on the rewritten query.

    We do NOT hard-filter Chroma using `where` because:
      - Chroma filtering operators vary by version
      - Version mismatches can lead to 0 results

    Instead, we produce *preferences* that we apply locally after retrieval.

    Returns:
        {} (no preference)
        or {"item_heading_contains_any": [...]} for section targeting.
    """
    q = query.lower()

    # If user asks about risk factors, we want Item 1A sections
    risk_signals = [
        "risk", "risk factor", "risks", "uncertaint", "litigation",
        "regulatory risk", "compliance risk"
    ]
    if any(s in q for s in risk_signals):
        return {"item_heading_contains_any": ["ITEM 1A", "RISK FACTORS"]}

    # You can extend this later for other sections (optional):
    # - MD&A: Item 7
    # - Legal Proceedings: Item 3
    # - Controls & Procedures: Item 9A
    return {}


def section_aware_retrieve(
    collection,
    query_emb: List[float],
    preferences: Dict[str, Any],
    final_n: int = 20,
    initial_pool: int = 80
) -> List[Dict[str, Any]]:
    """
    Retrieve documents using a *two-step* approach:

    Step A (Recall):
        Retrieve a bigger pool from Chroma using pure semantic similarity.
        This maximizes the chance we include the right section(s).

    Step B (Precision):
        If preferences exist, filter locally by metadata (item_heading).
        This is stable across all Chroma versions.

    Step C (Safety fallback):
        If filtering yields too few results, return the global top-N anyway
        so we NEVER return empty results.

    Parameters:
    - final_n: how many candidates we want to pass into reranking (usually 20)
    - initial_pool: how many we retrieve before filtering (80 works well)
    """
    # 1) Always retrieve a bigger pool first
    big_pool = retrieve_top_n(collection, query_emb, n=initial_pool)

    # 2) If no preferences, just return top final_n
    if not preferences:
        return big_pool[:final_n]

    wanted = [w.upper() for w in preferences.get("item_heading_contains_any", [])]

    # 3) Filter locally by item_heading
    filtered: List[Dict[str, Any]] = []
    for h in big_pool:
        heading = str(h.get("meta", {}).get("item_heading", "")).upper()
        if any(w in heading for w in wanted):
            filtered.append(h)

    # 4) If we got enough filtered matches, return them
    if len(filtered) >= final_n:
        return filtered[:final_n]

    # 5) Otherwise: fall back to the global best matches (never empty)
    return big_pool[:final_n]


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """
    Main Streamlit application entry point.
    """
    load_dotenv()

    st.set_page_config(page_title="Med Device 10-K RAG Chatbot", layout="wide")

    st.title("Med Device 10-K RAG Chatbot")
    st.caption(
        "Ask questions about medical device companies’ SEC 10-K filings. "
        "Answers are grounded in retrieved sources and include citations."
    )

    init_session_state()
    render_sidebar()

    EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    GEN_MODEL = os.getenv("GEN_MODEL", "gpt-4o-mini")

    # Initialize clients
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

    # Display chat history
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    user_msg = st.chat_input("Ask a question (e.g., 'Compare risk factors for EW vs MDT')")
    if not user_msg:
        return

    # Step 1: store + show user message
    st.session_state.history.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.write(user_msg)

    # Step 2: rewrite query (multi-turn support)
    rewritten_query = rewrite_query(
        oai=oai,
        gen_model=GEN_MODEL,
        history=st.session_state.history,
        user_msg=user_msg
    )

    # Step 3: embed + retrieve candidates (section-aware)
    try:
        qemb = embed_query(oai, EMBED_MODEL, rewritten_query)

        # NEW: infer section preferences (e.g., risk -> Item 1A)
        preferences = infer_section_preferences(rewritten_query)

        # NEW: retrieve a bigger pool then locally filter by metadata
        candidates = section_aware_retrieve(
            collection=col,
            query_emb=qemb,
            preferences=preferences,
            final_n=20,
            initial_pool=80
        )

    except Exception as e:
        st.error(f"Retrieval failed: {e}")
        st.stop()

    if not candidates:
        answer = "I couldn't find relevant passages in the indexed 10-K filings."
        st.session_state.history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)
        return

    # Step 4: advanced feature rerank top-20 -> top-5
    try:
        top5 = rerank_with_llm(oai, GEN_MODEL, rewritten_query, candidates, k=5)
    except Exception:
        top5 = candidates[:5]

    # Step 5: context + citations
    context, citations = format_context_blocks(top5)
    st.session_state.last_citations = citations

    # Step 6: generate grounded answer
    try:
        answer = answer_with_citations(
            oai=oai,
            gen_model=GEN_MODEL,
            history=st.session_state.history,
            user_msg=user_msg,
            context=context
        )
    except Exception as e:
        answer = f"Answer generation failed: {e}"

    # Step 7: display answer + sources
    st.session_state.history.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.write(answer)

        st.subheader("Sources (Evidence Used)")
        for c in citations:
            st.write(
                f"[{c['marker']}] **{c['ticker']}** — {c['section']}  \n"
                f"`{c['source_path']}`"
            )

        # Debug panel
        if st.session_state.debug:
            st.divider()
            st.markdown("### Debug Info")
            st.write("**Rewritten query:**", rewritten_query)
            st.write("**Section preferences:**", preferences if "preferences" in locals() else {})
            st.write("**Retrieved candidates:**", len(candidates))
            st.write("**Reranked to:**", len(top5))


if __name__ == "__main__":
    main()
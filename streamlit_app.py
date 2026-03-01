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
import re
from typing import List, Dict, Any
import streamlit.components.v1 as components

import streamlit as st
import streamlit.components.v1 as components
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

# Company Names
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
    "DXCM": "Dexcom"
}

# =============================================================================
# UI CONFIGURATION + SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """
    Initialize Streamlit session_state variables.
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
    """
    q = query.lower()

    risk_signals = [
        "risk", "risk factor", "risks", "uncertaint", "litigation",
        "regulatory risk", "compliance risk"
    ]
    if any(s in q for s in risk_signals):
        return {"item_heading_contains_any": ["ITEM 1A", "RISK FACTORS"]}

    return {}


def section_aware_retrieve(
    collection,
    query_emb: List[float],
    preferences: Dict[str, Any],
    final_n: int = 20,
    initial_pool: int = 80
) -> List[Dict[str, Any]]:
    """
    Two-step retrieve:
      A) Retrieve big pool from Chroma
      B) Locally filter by metadata if preferences exist
      C) Fallback to global top-N
    """
    big_pool = retrieve_top_n(collection, query_emb, n=initial_pool)

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
# CHATGPT-LIKE "SCROLL TO LATEST" ARROW
# =============================================================================

def render_scroll_to_latest_arrow():
    """
    ChatGPT-like 'scroll to latest' arrow:
    - Appears only when user scrolls up (not near bottom)
    - Disappears when user is near bottom
    - Click jumps smoothly to latest (bottom anchor)
    """

    # Anchor should be in the main DOM near the bottom
    st.markdown('<div id="chat-bottom"></div>', unsafe_allow_html=True)

    components.html(
        """
        <script>
        (function() {
          const P = window.parent.document;

          // Don't duplicate across reruns
          if (P.getElementById("scrollToBottomBtn")) return;

          // --- Find the actual Streamlit scroll container ---
          function getScrollContainer() {
            return (
              P.querySelector('div[data-testid="stAppViewContainer"]') ||
              P.querySelector('section.main') ||
              P.documentElement
            );
          }

          // --- Styles ---
          const style = P.createElement("style");
          style.innerHTML = `
            #scrollToBottomBtn{
              all: unset;
              position: fixed;
              right: 28px;
              bottom: 86px; /* above chat input */
              z-index: 200;

              width: 44px;
              height: 44px;
              border-radius: 999px;

              display: none;
              align-items: center;
              justify-content: center;

              background: rgba(255,255,255,0.95);
              border: 1px solid rgba(0,0,0,0.12);
              box-shadow: 0 6px 18px rgba(0,0,0,0.15);
              cursor: pointer;
            }

            #scrollToBottomBtn:hover{
              box-shadow: 0 8px 22px rgba(0,0,0,0.18);
              transform: translateY(-1px);
            }

            #scrollToBottomBtn svg{
              width: 18px;
              height: 18px;
              opacity: 0.75;
            }
          `;
          P.head.appendChild(style);

          // --- Button ---
          const btn = P.createElement("button");
          btn.id = "scrollToBottomBtn";
          btn.setAttribute("aria-label","Jump to latest");
          btn.title = "Jump to latest";
          btn.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none">
              <path d="M6 9l6 6 6-6"
                stroke="currentColor"
                stroke-width="2.2"
                stroke-linecap="round"
                stroke-linejoin="round"/>
            </svg>
          `;
          P.body.appendChild(btn);

          function getAnchor() {
            return P.getElementById("chat-bottom");
          }

          // Click => scroll anchor into view
          btn.addEventListener("click", () => {
            const anchor = getAnchor();
            if (anchor) anchor.scrollIntoView({ behavior: "smooth", block: "end" });
          });

          // Show/hide based on how close user is to bottom
          function updateVisibility() {
            const sc = getScrollContainer();
            const threshold = 250; // px from bottom
            const nearBottom = (sc.scrollHeight - (sc.scrollTop + sc.clientHeight)) < threshold;
            btn.style.display = nearBottom ? "none" : "flex";
          }

          // Attach scroll listener to the correct container
          const sc = getScrollContainer();

          // Some Streamlit builds need a tiny delay before sizes are correct
          let ticking = false;
          function onScroll() {
            if (!ticking) {
              ticking = true;
              requestAnimationFrame(() => {
                updateVisibility();
                ticking = false;
              });
            }
          }

          sc.addEventListener("scroll", onScroll, { passive: true });
          window.parent.addEventListener("resize", updateVisibility);

          // Initial
          setTimeout(updateVisibility, 400);
          setTimeout(updateVisibility, 1200);
        })();
        </script>
        """,
        height=1,
    )


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """
    Main Streamlit application entry point.
    """
    load_dotenv()

    st.set_page_config(page_title="MedTech 10-K Explorer", layout="wide")

    # --- Fixed header that automatically shifts when sidebar opens/closes ---
    # --- Proper fixed header that respects sidebar ---
    st.markdown("""
    <style>

    /* Streamlit top toolbar height (approx). If you still see clipping, change 3.25rem -> 3.5rem */
    :root { --st-topbar-h: 3.25rem; }

    /* Header container */
    .fixed-header {
      position: fixed;
      top: var(--st-topbar-h);     /* ✅ push below Streamlit toolbar */
      left: 260px;                 /* sidebar open width */
      right: 0;

      background: white;
      border-bottom: 1px solid rgba(0,0,0,0.08);

      padding: 15px 40px 16px 50px;
      z-index: 9999;

      transition: left 0.25s ease;
    }

    /* When sidebar is collapsed */
    section[data-testid="stSidebar"][aria-expanded="false"] ~ div .fixed-header {
      left: 80px;
    }

    /* Title */
    .fixed-header .title {
      font-size: 28px;
      font-weight: 750;
      line-height: 1.15;
      margin: 0;
    }

    /* Subtitle */
    .fixed-header .subtitle {
      font-size: 13px;
      color: #666;
      margin-top: 6px;
    }

    /* Push page content down so it doesn't hide under the fixed header */
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
    """, unsafe_allow_html=True)

    # JS to measure sidebar width and update --sidebar-w whenever it changes
    components.html("""
    <script>
    (function () {
      const P = window.parent.document;

      function sidebarWidthPx() {
        const sidebar = P.querySelector('section[data-testid="stSidebar"]');
        if (!sidebar) return 0;
        const rect = sidebar.getBoundingClientRect();
        // When collapsed, rect.width becomes ~0 (or very small)
        return Math.max(0, Math.round(rect.width));
      }

      function update() {
        const w = sidebarWidthPx();
        P.documentElement.style.setProperty("--sidebar-w", w + "px");
      }

      // Run now + after layout settles
      setTimeout(update, 50);
      setTimeout(update, 300);
      setTimeout(update, 800);

      // Update on resize
      window.parent.addEventListener("resize", update);

      // Update after any click (covers sidebar collapse/expand button)
      P.addEventListener("click", () => setTimeout(update, 50), true);

      // Also update on scroll (some layouts shift sizes)
      P.addEventListener("scroll", () => requestAnimationFrame(update), true);
    })();
    </script>
    """, height=0)

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

    # Show welcome message if chat is empty
    if len(st.session_state.history) == 0:
        with st.chat_message("assistant"):
            st.markdown("""
**👋 Welcome to the Med Device 10-K Chatbot!**

You can ask questions about medical device companies’ SEC 10-K filings.

Examples:
- What does Edwards Lifesciences do?
- What are Medtronic's risk factors?
- Compare Abbott and Boston Scientific
- What products does Dexcom make?

All answers are based on official SEC Form 10-K filings and include source citations.
""")

    # Display chat history
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # ✅ Render the arrow AFTER the history is drawn, BEFORE input
    render_scroll_to_latest_arrow()

    # Chat input
    user_msg = st.chat_input("Ask a question (e.g., 'Compare risk factors for Edwards Lifesciences vs Medtronic')")
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
        preferences = infer_section_preferences(rewritten_query)

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

    # Step 6 + 7: generate answer ONCE, show typing, then show sources
    with st.chat_message("assistant"):
        typing_placeholder = st.empty()

        typing_placeholder.markdown("""
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
""", unsafe_allow_html=True)

        # Generate answer
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

        typing_placeholder.empty()

        # Show answer once
        st.write(answer)

        # Show sources once (deduped)
        st.subheader("Sources (Evidence Used)")
        seen = set()
        for c in citations:
            ticker = c["ticker"]
            company = COMPANY_NAMES.get(ticker, ticker)

            match = re.search(r"-(\d{2})-", c["source_path"])
            year = "20" + match.group(1) if match else "Unknown"
            section = c["section"]

            doc_key = (ticker, year)
            if doc_key in seen:
                continue
            seen.add(doc_key)

            st.write(f"[{c['marker']}] {company} — {year} Form 10-K — {section}")

        # Debug panel (optional)
        if st.session_state.debug:
            st.divider()
            st.markdown("### Debug Info")
            st.write("**Rewritten query:**", rewritten_query)
            st.write("**Section preferences:**", preferences if "preferences" in locals() else {})
            st.write("**Retrieved candidates:**", len(candidates))
            st.write("**Reranked to:**", len(top5))

    # ✅ Store assistant message ONCE for history
    st.session_state.history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
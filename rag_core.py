"""
===============================================================================
RAG_CORE.PY
===============================================================================

PURPOSE
-------
This module contains the "core intelligence" for our production-style RAG system.

The Streamlit app calls these functions to run the full pipeline:

    1) Conversation-aware query rewrite (handles follow-up questions)
    2) Query embedding (OpenAI embeddings)
    3) Vector retrieval from Chroma (fast semantic search)
    4) Advanced feature: Second-stage reranking (LLM reranker)
    5) Answer generation grounded in retrieved sources with citations

Why split into a separate module?
--------------------------------
Separating RAG logic from UI is a best practice:
    - easier testing
    - clearer organization
    - easier to move to FastAPI later
    - keeps Streamlit UI clean and simple

===============================================================================
RAG PIPELINE OVERVIEW
===============================================================================

INPUTS:
- user_msg: the user’s latest message
- history: last N chat messages (conversation memory)

OUTPUTS:
- answer text with citations like [1][2]
- citations list with metadata for each source

STAGES:
(1) Query Rewrite
(2) Vector Retrieval (top-20)
(3) Re-ranking (top-20 -> top-5)  <-- Advanced feature
(4) Generation + Citations

===============================================================================
"""

import os
import json
from typing import List, Dict, Any, Tuple, Optional

import chromadb
from chromadb.config import Settings
from openai import OpenAI


# =============================================================================
# SECTION 1 — CLIENT SETUP HELPERS
# =============================================================================

def get_openai_client() -> OpenAI:
    """
    Create and return an OpenAI client.

    Required environment variable:
    - OPENAI_API_KEY
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY missing. Put it in .env (local) or Streamlit Secrets (deployment)."
        )
    return OpenAI(api_key=api_key)


def get_chroma_collection(collection_name: str = "tenk_chunks"):
    """
    Load a persistent Chroma collection.

    Required environment variable:
    - CHROMA_DIR (default: data/chroma)
    """
    chroma_dir = os.getenv("CHROMA_DIR", "data/chroma")

    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(anonymized_telemetry=False)
    )

    return client.get_or_create_collection(name=collection_name)


# =============================================================================
# SECTION 2 — CONVERSATION-AWARE QUERY REWRITE
# =============================================================================

def rewrite_query(
    oai: OpenAI,
    gen_model: str,
    history: List[Dict[str, str]],
    user_msg: str,
    max_history_messages: int = 6
) -> str:
    """
    Rewrite the user's message into a standalone retrieval query.

    Why rewrite?
    ------------
    In multi-turn chats, users ask follow-ups like:
        "What about risk factors?"
        "Does that affect reimbursement?"
        "Compare it to Medtronic."

    These are ambiguous without context. Rewriting turns them into a
    complete standalone query that retrieval can handle.

    Output:
    - A single query string (not an answer).
    """
    recent = history[-max_history_messages:] if history else []

    prompt = f"""
You are a query-rewriting assistant for a Retrieval-Augmented Generation (RAG) system.

Domain: SEC 10-K filings for medical device companies.

Task:
Rewrite the user's latest message into a SINGLE standalone search query that includes
any missing context implied by the conversation.

Rules:
- Keep it concise (one query).
- Do NOT answer the question.
- Do NOT include citations.
- Output ONLY the rewritten query text.

Conversation (JSON, most recent last):
{json.dumps(recent, indent=2)}

User message:
{user_msg}
""".strip()

    resp = oai.responses.create(
        model=gen_model,
        input=prompt,
        temperature=0
    )

    rewritten = resp.output_text.strip()
    return rewritten if rewritten else user_msg


# IMPORTANT:
# Some UI code may still import/call `rewrite_query_if_needed`.
# To prevent NameError/import errors, we provide a compatibility alias.
rewrite_query_if_needed = rewrite_query


# =============================================================================
# SECTION 3 — VECTOR RETRIEVAL (EMBED + QUERY CHROMA)
# =============================================================================

def embed_query(oai: OpenAI, embedding_model: str, query: str) -> List[float]:
    """
    Create an embedding vector for a query string.

    Output:
    - list[float] embedding vector
    """
    r = oai.embeddings.create(
        model=embedding_model,
        input=query,
        encoding_format="float"
    )
    return r.data[0].embedding


def retrieve_top_n(
    collection,
    query_emb: List[float],
    n: int = 20
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-N most similar chunks from Chroma (Stage 1 retrieval).

    IMPORTANT (Chroma compatibility note):
    --------------------------------------
    Some Chroma versions do NOT allow "ids" in the `include=[...]` list.
    They will throw an error like:

        Expected include item to be one of documents, embeddings, metadatas,
        distances, uris, data, got ids

    Even in those versions, Chroma STILL returns IDs in `res["ids"]` by default.

    Therefore we:
      - request only ["documents", "metadatas", "distances"]
      - read ids from res["ids"][0] anyway

    Returns:
        [
          {"id": ..., "text": ..., "meta": {...}, "distance": ...},
          ...
        ]
    """
    res = collection.query(
        query_embeddings=[query_emb],
        n_results=n,
        # NOTE: do NOT include "ids" here (version compatibility)
        include=["documents", "metadatas", "distances"]
    )

    hits: List[Dict[str, Any]] = []

    # Chroma returns lists-of-lists because it supports multiple queries at once.
    ids = res["ids"][0]               # IDs are returned even if not in include
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    for i in range(len(ids)):
        hits.append({
            "id": ids[i],
            "text": docs[i],
            "meta": metas[i],
            "distance": dists[i],
        })

    return hits


# =============================================================================
# SECTION 4 — ADVANCED FEATURE: LLM RE-RANKING (TOP-20 -> TOP-K)
# =============================================================================

def _safe_json_loads(text: str) -> Optional[dict]:
    """Try to parse JSON safely. Returns dict if successful, else None."""
    try:
        return json.loads(text)
    except Exception:
        return None


def rerank_with_llm(
    oai: OpenAI,
    gen_model: str,
    query: str,
    candidates: List[Dict[str, Any]],
    k: int = 5,
    snippet_chars: int = 700
) -> List[Dict[str, Any]]:
    """
    Advanced Feature: Re-ranking (Stage 2 retrieval).

    Stage 1: vector retrieval returns top-20
    Stage 2: LLM selects best top-k (default 5)

    Output:
    - A list of length k (or fewer if not enough candidates).
    """
    if not candidates:
        return []

    k = min(k, len(candidates))

    compact = []
    for idx, c in enumerate(candidates):
        meta = c.get("meta", {})
        compact.append({
            "idx": idx,
            "ticker": meta.get("ticker", "UNKNOWN"),
            "item_heading": meta.get("item_heading", "UNKNOWN_ITEM"),
            "snippet": c.get("text", "")[:snippet_chars]
        })

    prompt = f"""
You are a re-ranking model for a RAG system.

Task:
Given a user query and candidate passages from SEC 10-K filings,
select the {k} most relevant passages.

User query:
{query}

Candidates (JSON):
{json.dumps(compact, indent=2)}

Rules:
- Return ONLY valid JSON.
- The JSON must be exactly: {{"selected_indices":[...]}}
- selected_indices must contain exactly {k} integers.
- Each integer must be one of the candidate idx values.
""".strip()

    resp = oai.responses.create(
        model=gen_model,
        input=prompt,
        temperature=0
    )

    data = _safe_json_loads(resp.output_text.strip())
    if not data or "selected_indices" not in data:
        return candidates[:k]

    selected = data["selected_indices"]

    # Validate indices
    valid_idxs: List[int] = []
    for i in selected:
        if isinstance(i, int) and 0 <= i < len(candidates):
            valid_idxs.append(i)

    valid_idxs = valid_idxs[:k]

    # Fill if needed
    if len(valid_idxs) < k:
        for i in range(len(candidates)):
            if i not in valid_idxs:
                valid_idxs.append(i)
            if len(valid_idxs) == k:
                break

    return [candidates[i] for i in valid_idxs]


# =============================================================================
# SECTION 5 — CONTEXT FORMATTING + CITATION OBJECTS
# =============================================================================

def format_context_blocks(
    hits: List[Dict[str, Any]]
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Convert retrieved chunks into:
      (A) A single context string to feed the LLM
      (B) A citations list to display in the UI

    We number chunks so the model can cite them as [1], [2], ...
    """
    blocks: List[str] = []
    citations: List[Dict[str, Any]] = []

    for i, h in enumerate(hits, start=1):
        meta = h.get("meta", {})
        ticker = meta.get("ticker", "UNKNOWN")
        item = meta.get("item_heading", "UNKNOWN_ITEM")
        src = meta.get("source_path", "")

        blocks.append(f"""
[Source {i}]
Ticker: {ticker}
Section: {item}
SourcePath: {src}

Text:
{h.get("text", "")}
""".strip())

        citations.append({
            "marker": i,
            "ticker": ticker,
            "section": item,
            "source_path": src,
            "chunk_id": h.get("id", "")
        })

    context_string = "\n\n---\n\n".join(blocks)
    return context_string, citations


# =============================================================================
# SECTION 6 — GROUNDED ANSWER GENERATION WITH CITATIONS
# =============================================================================

def answer_with_citations(
    oai: OpenAI,
    gen_model: str,
    history: List[Dict[str, str]],
    user_msg: str,
    context: str,
    memory_turns: int = 8
) -> str:
    """
    Generate a final answer using ONLY the retrieved context.

    Requirements enforced by the prompt:
    - Use only SOURCES
    - Cite every factual claim with [1], [2], etc.
    """
    system_rules = """
You are a Retrieval-Augmented Generation (RAG) assistant that answers questions using ONLY the provided SOURCES.

STRICT RULES:
1) Use ONLY information from SOURCES. Do not use outside knowledge.
2) If the SOURCES do not contain enough information, say:
   "I don't have enough information in the provided filings to answer that."
3) Every factual claim MUST include at least one citation like [1] or [2].
4) Do NOT invent numbers, dates, financial values, or section names.
5) Prefer concise, structured answers (bullets are okay).
""".strip()

    mem = history[-memory_turns:] if history else []

    prompt = f"""
SYSTEM RULES:
{system_rules}

CONVERSATION MEMORY (JSON):
{json.dumps(mem, indent=2)}

USER QUESTION:
{user_msg}

SOURCES:
{context}

Now answer the user question using ONLY the SOURCES. Include citations like [1][2].
""".strip()

    resp = oai.responses.create(
        model=gen_model,
        input=prompt,
        temperature=0
    )

    return resp.output_text.strip()
"""
RAG_CORE.PY

Core RAG pipeline logic used by the Streamlit app.

Provides:
- Query rewriting
- Embedding generation
- Vector retrieval
- LLM reranking
- Context formatting
- Answer generation with citations
"""

import os
import json
from typing import List, Dict, Any, Tuple, Optional

import chromadb
from chromadb.config import Settings
from openai import OpenAI


# =============================================================================
# CLIENT SETUP
# =============================================================================

def get_openai_client() -> OpenAI:
    """Creates and returns an OpenAI client using OPENAI_API_KEY."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY missing. Put it in .env or deployment secrets."
        )
    return OpenAI(api_key=api_key)


def get_chroma_collection(collection_name: str = "tenk_chunks"):
    """Loads the persistent Chroma collection."""
    chroma_dir = os.getenv("CHROMA_DIR", "data/chroma")

    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(anonymized_telemetry=False)
    )

    return client.get_or_create_collection(name=collection_name)


# =============================================================================
# QUERY REWRITE
# =============================================================================

def rewrite_query(
    oai: OpenAI,
    gen_model: str,
    history: List[Dict[str, str]],
    user_msg: str,
    max_history_messages: int = 6
) -> str:
    """Rewrites a user message into a standalone search query."""

    recent = history[-max_history_messages:] if history else []

    prompt = f"""
Rewrite the user's latest message into a single standalone search query.

Conversation:
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


# Compatibility alias
rewrite_query_if_needed = rewrite_query


# =============================================================================
# EMBEDDING + VECTOR RETRIEVAL
# =============================================================================

def embed_query(oai: OpenAI, embedding_model: str, query: str) -> List[float]:
    """Generates an embedding vector for a query."""

    r = oai.embeddings.create(
        model=embedding_model,
        input=query,
        encoding_format="float"
    )

    return r.data[0].embedding


def retrieve_top_n(
    collection,
    query_emb: List[float],
    n: int = 20,
    allowed_tickers: Optional[List[str]] = None,
    overfetch_factor: int = 5
) -> List[Dict[str, Any]]:
    """Retrieves the most similar chunks from Chroma with optional ticker filtering."""

    fetch_n = max(n, n * overfetch_factor)

    res = collection.query(
        query_embeddings=[query_emb],
        n_results=fetch_n,
        include=["documents", "metadatas", "distances"]
    )

    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    hits: List[Dict[str, Any]] = []

    for i in range(len(ids)):
        hits.append({
            "id": ids[i],
            "text": docs[i],
            "meta": metas[i],
            "distance": dists[i],
        })

    if allowed_tickers:
        allowed = {t.upper() for t in allowed_tickers}

        filtered = [
            h for h in hits
            if str(h.get("meta", {}).get("ticker", "")).upper() in allowed
        ]

        if filtered:
            return filtered[:n]

        return hits[:n]

    return hits[:n]


# =============================================================================
# LLM RERANKING
# =============================================================================

def _safe_json_loads(text: str) -> Optional[dict]:
    """Safely parses JSON text."""
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
    """Selects the most relevant chunks using an LLM reranker."""

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
Select the {k} most relevant passages for this query.

Query:
{query}

Candidates:
{json.dumps(compact, indent=2)}

Return JSON:
{{"selected_indices":[...]}}
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

    valid_idxs: List[int] = []

    for i in selected:
        if isinstance(i, int) and 0 <= i < len(candidates):
            valid_idxs.append(i)

    valid_idxs = valid_idxs[:k]

    if len(valid_idxs) < k:
        for i in range(len(candidates)):
            if i not in valid_idxs:
                valid_idxs.append(i)
            if len(valid_idxs) == k:
                break

    return [candidates[i] for i in valid_idxs]


# =============================================================================
# CONTEXT + CITATIONS
# =============================================================================

def format_context_blocks(
    hits: List[Dict[str, Any]]
) -> Tuple[str, List[Dict[str, Any]]]:
    """Formats retrieved chunks into context text and citation metadata."""

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
# ANSWER GENERATION
# =============================================================================

def answer_with_citations(
    oai: OpenAI,
    gen_model: str,
    history: List[Dict[str, str]],
    user_msg: str,
    context: str,
    memory_turns: int = 8
) -> str:
    """Generates an answer grounded in retrieved sources with citations."""

    mem = history[-memory_turns:] if history else []

    prompt = f"""
Answer using ONLY the provided SOURCES.

Conversation:
{json.dumps(mem, indent=2)}

Question:
{user_msg}

Sources:
{context}

Include citations like [1][2].
""".strip()

    resp = oai.responses.create(
        model=gen_model,
        input=prompt,
        temperature=0
    )

    return resp.output_text.strip()
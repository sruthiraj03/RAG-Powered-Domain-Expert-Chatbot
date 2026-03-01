"""
===============================================================================
BUILD_INDEX.PY
===============================================================================

PURPOSE
-------
Build (or rebuild) a persistent Chroma vector database from downloaded
SEC 10-K filings stored locally.

WHAT IT DOES
------------
For each filing folder:

1) Extract main 10-K document from EDGAR submission
2) Convert HTML → clean plain text
3) Split into sections using ROBUST Item-heading detection
4) Chunk by paragraph with overlap
5) Embed chunks using OpenAI (with rate-limit retry)
6) Store embeddings + metadata in persistent Chroma

This version FIXES:
- Missed "Item 1A. Risk Factors"
- Unicode dash formatting issues
- Split headings across lines
- Retrieval targeting issues

===============================================================================
"""

import os
import re
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from dotenv import load_dotenv
from tqdm import tqdm
from bs4 import BeautifulSoup

import chromadb
from chromadb.config import Settings
from openai import OpenAI

try:
    from openai import RateLimitError
except Exception:
    RateLimitError = Exception


# =============================================================================
# SECTION 1 — EDGAR PARSING
# =============================================================================

def load_full_submission_text(txt_path: Path) -> str:
    return txt_path.read_text(errors="ignore")


def extract_document_block(full_submission: str, doc_type: str = "10-K") -> Optional[str]:
    parts = full_submission.split("<DOCUMENT>")
    target = f"<TYPE>{doc_type}".upper()

    for part in parts:
        if target in part.upper():
            return "<DOCUMENT>" + part
    return None


def extract_text_from_document_block(doc_block: str) -> str:
    m = re.search(r"(?is)<TEXT>(.*)</TEXT>", doc_block)
    content = m.group(1) if m else doc_block

    looks_like_html = bool(re.search(r"(?i)<html|<div|<p|<table|<span", content))

    if looks_like_html:
        soup = BeautifulSoup(content, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    else:
        text = content

    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


# =============================================================================
# SECTION 2 — ROBUST ITEM HEADING DETECTION (FIXED)
# =============================================================================

# Handles:
# ITEM 1A. RISK FACTORS
# ITEM 1A—RISK FACTORS
# ITEM 1A: RISK FACTORS
# ITEM 1A.
# (RISK FACTORS on next line)
ITEM_LINE_RE = re.compile(
    r"""(?ix)^\s*
    item
    \s+(\d{1,2}[a-z]?)
    \s*[\.\:\-–—]?\s*
    (.*\S)?
    \s*$"""
)

TITLE_LINE_RE = re.compile(r"(?i)^[A-Z][A-Z \-&/,]{4,}$")


def split_into_items(full_text: str) -> List[Tuple[str, str]]:
    """
    Line-based robust section splitter.
    """

    # Normalize unicode artifacts
    text = (full_text
            .replace("\u2013", "-")
            .replace("\u2014", "-")
            .replace("\u00a0", " "))

    lines = text.splitlines()

    offsets = []
    running = 0
    for ln in lines:
        offsets.append(running)
        running += len(ln) + 1

    headings: List[Tuple[int, str]] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = ITEM_LINE_RE.match(line)

        if m:
            item_num = m.group(1).upper()
            title = (m.group(2) or "").strip()

            # If title missing, check next line
            if not title and i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                if TITLE_LINE_RE.match(nxt):
                    title = nxt
                    i += 1

            heading = f"Item {item_num}"
            if title:
                heading += f". {title}"

            headings.append((offsets[i], heading))

        i += 1

    if not headings:
        return [("FULL_DOCUMENT", full_text)]

    sections = []
    for idx, (start_off, heading) in enumerate(headings):
        end_off = headings[idx + 1][0] if idx + 1 < len(headings) else len(text)
        body = text[start_off:end_off].strip()
        sections.append((heading, body))

    return sections


def approx_token_count(text: str) -> int:
    return max(1, len(text) // 4)


def chunk_with_overlap(
    text: str,
    target_tokens: int = 700,
    overlap_tokens: int = 120,
    max_para_chars: int = 6000,
    max_chunk_chars: int = 12000
) -> List[str]:
    """
    Paragraph chunking with overlap AND hard caps to prevent embedding context errors.

    Why we need hard caps:
    - EDGAR text often contains giant table-like “paragraphs” with no blank lines.
    - Those can create mega-chunks that exceed embedding model limits (8192 tokens).
    - We cap paragraph size, then cap final chunk size.

    Parameters
    ----------
    max_para_chars:
        If a paragraph exceeds this, we split it into pieces.
    max_chunk_chars:
        If a final chunk exceeds this, we split it into pieces.
    """
    # Split into paragraphs by blank lines
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # 1) Split giant paragraphs (tables / messy HTML text often becomes a single huge paragraph)
    fixed_paras: List[str] = []
    for p in paragraphs:
        if len(p) <= max_para_chars:
            fixed_paras.append(p)
        else:
            for i in range(0, len(p), max_para_chars):
                fixed_paras.append(p[i:i + max_para_chars])

    paragraphs = fixed_paras

    chunks: List[str] = []
    current: List[str] = []

    def current_tokens() -> int:
        return approx_token_count("\n\n".join(current))

    # 2) Build chunks up to target size
    for p in paragraphs:
        current.append(p)

        if current_tokens() >= target_tokens:
            chunks.append("\n\n".join(current).strip())

            # overlap: keep tail paragraphs until overlap size met
            overlap: List[str] = []
            for para in reversed(current):
                overlap.insert(0, para)
                if approx_token_count("\n\n".join(overlap)) >= overlap_tokens:
                    break
            current = overlap

    if current:
        chunks.append("\n\n".join(current).strip())

    # 3) Hard-split chunks that are still too big
    safe_chunks: List[str] = []
    for c in chunks:
        if len(c) <= max_chunk_chars:
            safe_chunks.append(c)
        else:
            for i in range(0, len(c), max_chunk_chars):
                safe_chunks.append(c[i:i + max_chunk_chars])

    # 4) Drop tiny fragments
    safe_chunks = [c for c in safe_chunks if len(c.strip()) > 200]
    return safe_chunks


def build_chunks(doc_text: str) -> List[Dict[str, Any]]:
    items = split_into_items(doc_text)
    all_chunks = []

    for heading, body in items:
        subchunks = chunk_with_overlap(body)
        for idx, chunk_text in enumerate(subchunks):
            all_chunks.append({
                "item_heading": heading,
                "chunk_index": idx,
                "text": chunk_text
            })

    return all_chunks


# =============================================================================
# SECTION 3 — EMBEDDINGS + CHROMA
# =============================================================================
def force_split_for_embedding(text: str, max_chars: int = 12000) -> List[str]:
    """
    Absolute last-resort safety.
    If ANY chunk is still too large, split it before sending to embeddings API.
    """
    if len(text) <= max_chars:
        return [text]
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return ("rate limit" in msg) or ("429" in msg) or ("tpm" in msg)


def embed_with_retry(oai, model, texts, max_retries=8):
    import random

    for attempt in range(max_retries):
        try:
            resp = oai.embeddings.create(
                model=model,
                input=texts,
                encoding_format="float"
            )
            return [d.embedding for d in resp.data]

        except Exception as e:
            if not _is_rate_limit_error(e):
                raise
            wait = min(30, 2 ** attempt)
            time.sleep(wait + random.uniform(0, 0.25))

    raise RuntimeError("Embedding failed after retries.")


def parse_metadata_from_folder(folder: Path) -> Dict[str, str]:
    parts = folder.parts
    ticker = "UNKNOWN"
    form_type = "UNKNOWN"
    accession = folder.name

    for i, p in enumerate(parts):
        if p.lower() == "sec-edgar-filings" and i + 2 < len(parts):
            ticker = parts[i + 1].upper()
            form_type = parts[i + 2].upper()
            break

    doc_id = re.sub(r"[^a-zA-Z0-9_\-]+", "_", f"{ticker}_{form_type}_{accession}")

    return {
        "ticker": ticker,
        "form_type": form_type,
        "accession": accession,
        "doc_id": doc_id
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    load_dotenv()

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("Missing OPENAI_API_KEY")

    EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma")

    filings_root = Path("data/filings_raw/sec-edgar-filings")
    filing_folders = [p for p in filings_root.glob("*/*/*") if p.is_dir()]

    oai = OpenAI(api_key=openai_key)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name="tenk_chunks")

    total_chunks = 0

    for folder in tqdm(filing_folders, desc="Indexing filings"):
        try:
            txt_path = folder / "full-submission.txt"
            if not txt_path.exists():
                continue

            meta = parse_metadata_from_folder(folder)
            full_submission = load_full_submission_text(txt_path)
            doc_block = extract_document_block(full_submission)
            if not doc_block:
                continue

            text = extract_text_from_document_block(doc_block)
            chunks = build_chunks(text)

            batch_size = 8

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                # Collect texts, but apply an absolute safety split before embedding
                texts: List[str] = []
                expanded_batch: List[Dict[str, Any]] = []

                for b in batch:
                    parts = force_split_for_embedding(b["text"], max_chars=12000)
                    for pi, part in enumerate(parts):
                        # copy chunk metadata, but add a "part" index so IDs stay unique
                        bb = dict(b)
                        bb["chunk_part"] = pi
                        bb["text"] = part
                        expanded_batch.append(bb)
                        texts.append(part)

                embeddings = embed_with_retry(oai, EMBED_MODEL, texts)

                ids = [f"{meta['doc_id']}__{i + j}__p{expanded_batch[j].get('chunk_part', 0)}" for j in
                       range(len(expanded_batch))]

                metadatas = []
                for b in expanded_batch:
                    metadatas.append({
                        **meta,
                        "item_heading": b["item_heading"],
                        "chunk_index": b["chunk_index"],
                        "chunk_part": b.get("chunk_part", 0),
                        "source_path": str(txt_path)
                    })

                collection.upsert(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
                total_chunks += len(expanded_batch)

                collection.upsert(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas,
                    embeddings=embeddings
                )

                total_chunks += len(batch)

        except Exception as e:
            print(f"[WARN] {folder} failed: {e}")

    print(f"\nIndexing complete. Total chunks added: {total_chunks}")
    print(f"Chroma persisted at: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
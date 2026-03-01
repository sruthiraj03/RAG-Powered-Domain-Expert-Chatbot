# streamlit_app.py Documentation
## Overview

streamlit_app.py - the frontend user interface for the MedTech 10-K Explorer RAG chatbot.
The application provides a conversational interface where users can ask questions about medical device companies’ SEC Form 10-K filings. The Streamlit app connects to the RAG pipeline implemented in `rag_core.py` and displays answers with source citations.

Main responsibilities:
- Chat interface
- Conversation memory
- Query processing
- Retrieval integration
- Source citation display
- Company filtering

---

## Session State Initialization

Function: init_session_state()

Initializes Streamlit session variables:
- history  
  Stores conversation messages and supports conversation memory.
- last_citations  
  Stores citations from the last answer.
- allowed_tickers  
  Stores the selected company filter.

Example history structure:
[
 {"role":"user","content":"What does Dexcom do?"},
 {"role":"assistant","content":"Dexcom develops glucose monitoring devices..."}
]

---

## Sidebar Interface

Function: render_sidebar()
Provides sidebar functionality including:

### New Chat Button
Clears conversation history:
st.session_state.history = []
Allows users to restart the conversation.

### Company Filter
Users can select a single company ticker.
Example:
EW — Edwards Lifesciences
If no company is selected:
All companies
Then retrieval searches across all filings.
If a company is selected:
st.session_state.allowed_tickers = ["EW"]
Retrieval will only return chunks for that company.

---

## Section-Aware Retrieval

Function:infer_section_preferences(query)
Detects if the query relates to specific 10-K sections.

Example signals:
- risk
- risk factor
- litigation
- compliance risk
- regulatory risk

If detected, the system prioritizes:
Item 1A — Risk Factors

---

Function: section_aware_retrieve()

Process:
1. Retrieve large candidate pool from Chroma
2. Apply section filtering if needed
3. Return top results

Typical values:
Initial pool = 80 chunks  
Final pool = 20 chunks

---

## Main Application

Function:main()
Controls the main application.

Responsibilities:
- Load environment variables
- Configure Streamlit page
- Initialize RAG components
- Handle user input
- Display answers
- Display citations

---

## Application Flow
### Step 1 — Initialization

The application loads environment variables from the `.env` file.
Then initializes:
- OpenAI client
- Chroma collection

Functions used:
get_openai_client()  
get_chroma_collection()

---

### Step 2 — Display Chat History

Previous conversation messages are displayed using:
for msg in st.session_state.history:
This enables multi-turn conversation support.

---

### Step 3 — User Input

User enters question using:
st.chat_input()
Example:
Compare Abbott and Boston Scientific
The message is stored using:
st.session_state.history.append()

---

### Step 4 — Query Rewriting

Function:
rewrite_query()
Converts user input into a standalone query.
Example:
User input:
What about their risks?
Rewritten query:
What are the risk factors for Abbott?
This improves retrieval accuracy for follow-up questions.

---

### Step 5 — Query Embedding

Function:
embed_query()
Generates embedding vector.
Embedding model:
text-embedding-3-small

---

### Step 6 — Retrieval

Function:
section_aware_retrieve()
Process:
1. Retrieve candidate chunks
2. Apply company filter
3. Apply section filter

Typical retrieval size: Top 20 candidates

---

### Step 7 — Re-ranking

Function: rerank_with_llm()
Advanced feature.

Process:
1. Send candidates to the language model
2. Select most relevant chunks
3. Return top 5 chunks

Benefits:
- Improves retrieval precision
- Removes irrelevant passages
- Improves answer quality

---

### Step 8 — Context Construction

Function: format_context_blocks()
Creates context text for the language model.

Example format:
[Source 1]  
Ticker: EW  
Section: Item 1A Risk Factors  

Text:  
...
Also builds citation metadata.
---

### Step 9 — Answer Generation

Function: answer_with_citations()
Uses the language model to generate answers.

Prompt instructions:
- Use only the provided sources
- Include citations

Example output:
Dexcom develops glucose monitoring systems [1].

---

### Step 10 — Source Display

Sources displayed under: Sources (Evidence Used)

Example:
[1] Dexcom — 2024 Form 10-K — Item 1 Business
Sources are deduplicated by ticker and year.

---

## Conversation Memory

Conversation history stored in:
st.session_state.history
Memory size:
Last 5–10 exchanges

Used for:
- Query rewriting
- Answer generation

---

## Models Used

Embedding model:
text-embedding-3-small
Generation model:
gpt-4o-mini
Configured via .env.

---

## Environment Variables

Required: OPENAI_API_KEY

Optional:
EMBEDDING_MODEL  
GEN_MODEL  
CHROMA_DIR

---

## UI Features
### Conversational Chat Interface

Uses:
st.chat_message()
Supports:
- Multi-turn chat
- Chat history
- Assistant responses

---

### Company Filter

Users can restrict retrieval to one company.
Example:
Medtronic only
Improves retrieval relevance.

---

### New Chat Button

Resets conversation state.

---

### Source Citations

Displayed for transparency and traceability.
Ensures answers are grounded in SEC filings.

---

## Error Handling

The application handles errors for:
OpenAI Initialization
OpenAI client initialization failed
Chroma Database
Chroma database initialization failed
Retrieval Failures
Retrieval failed
Answer Generation Failures
Answer generation failed

---

## Summary
streamlit_app.py provides the user interface and coordinates the RAG pipeline.

Responsibilities include:
- Chat interface
- Conversation memory
- Query rewriting
- Retrieval orchestration
- Answer display
- Citation display

This file connects the frontend to the backend RAG pipeline implemented in rag_core.py.
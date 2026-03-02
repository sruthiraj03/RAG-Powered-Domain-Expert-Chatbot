# ARCHITECTURE: MedTech 10-K Explorer
**RAG Chatbot with Vector Database, Conversation Memory, and Source Citations**

---

## 1. System Architecture
This project implements a production-ready **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about SEC Form 10-K filings for publicly traded medical device companies. The system integrates automated document ingestion, structured text processing, persistent vector storage, semantic retrieval, LLM-based re-ranking, and conversational answer generation with citations.

The architecture consists of two main stages: an **offline indexing pipeline** and an **online query pipeline**.

### a. Document Ingestion and Processing
* **Ingestion:** SEC Form 10-K filings are downloaded automatically using the sec-edgar-downloader library. The system retrieves the five most recent filings for ten medical device companies (50 documents total).
* **Storage:** Filings are stored locally in a structured directory under data/filings_raw.
* **Extraction:** During indexing, the system extracts the main filing text from filing-details.html or full-submission.txt. It selects the most relevant content while filtering out XBRL-related data and formatting artifacts.
* **Section-Awareness:** The text is split into logical sections based on SEC “Item” headings (e.g., **Item 1-Business** and **Item 1A-Risk Factors**). 
* **Smart Chunking:** A paragraph-based chunking strategy with overlap is used. Each chunk contains approximately **700 tokens** with **120 tokens** of overlap to maintain context continuity.

### b. Vector Database
* **Storage:** The system uses **Chroma** as the vector database with persistent storage located in data/chroma.
* **Embeddings:** Text chunks are embedded using the OpenAI model **text-embedding-3-small**.
* **Metadata:** Each chunk includes ticker symbols, accession numbers, section headings, and source file paths to support filtering and citation generation.

### c. RAG Query Pipeline
The online query pipeline is implemented using a **Streamlit** web application. When a user submits a question, the system performs the following:

1. **Conversation Memory:** Stores history and uses the last 5–10 exchanges as context to handle follow-up questions.
2. **Query Rewriting:** User questions are rewritten into standalone search queries using an LLM to improve retrieval accuracy.
3. **Semantic Retrieval:** The rewritten query is embedded to retrieve candidate chunks from Chroma.

### d. Advanced Feature – Re-ranking
The project implements **LLM re-ranking** as an advanced feature. After initial retrieval, the language model selects the most relevant passages from the candidate pool. The top five chunks are then used for final answer generation, significantly improving precision compared to basic vector search.

### e. Answer Generation with Citations
The language model generates answers using **only** the retrieved context. Answers include inline citations (e.g., [1], [2]), and the Streamlit interface displays a "Sources (Evidence Used)" section.

---

## 2. Challenges
* **Data Extraction:** Filtering XBRL noise from complex SEC submissions was resolved by selecting specific document blocks.
* **Heading Detection:** Filing formats vary wildly; a robust detection approach was implemented to correctly identify "Item" sections despite inconsistent punctuation.
* **Rate Limiting:** Large embedding tasks sometimes triggered API limits, addressed via batching and retry logic.
* **Retrieval Accuracy:** Initial semantic search occasionally returned "noisy" results; this was solved using **LLM Re-ranking**.
* **Context Continuity:** Handling follow-up questions was improved via **Query Rewriting** using conversation history.

---

## 3. Results
The final system successfully implements a complete RAG pipeline that satisfies all requirements:
* **Multi-turn Conversations:** Supports complex dialogue regarding company operations and risk factors.
* **Citations:** Consistently references the correct 10-K sections.
* **Performance:** Testing showed that query rewriting and re-ranking significantly improved answer relevance.
* **Persistence:** The vector database enables fast responses without the need to rebuild the index on every restart.
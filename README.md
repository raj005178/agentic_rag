# Agentic RAG 

> A private, verifiable, self-correcting document intelligence system built with LangGraph, ChromaDB, and Groq API.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)
![LangGraph](https://img.shields.io/badge/LangGraph-0.1+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## What is this?

Most RAG systems are just fixed pipelines — retrieve, then answer. This system is **agentic** — it dynamically decides what to do at every step:

- Is this query even related to the uploaded documents?
- Are the retrieved chunks actually relevant?
- Is the generated answer grounded in the source material?
- Should it retry with a better query or fall back to web search?

Every answer comes with **source citations**, **confidence scores**, and **RAGAS-style evaluation metrics** so you always know how trustworthy the response is.

---

## How it differs from ChatGPT

| | ChatGPT | This System |
|---|---|---|
| **Private documents** | Uploaded to OpenAI servers | Stays fully local |
| **Source verification** | No citations | Every answer cites exact chunk + score |
| **Hallucination** | Confident but unverifiable | Checked and scored before showing |
| **Works offline** | No | Yes — TinyLlama fallback |
| **Up to date** | Knowledge cutoff | Upload any document today |
| **Measurable quality** | Black box | RAGAS metrics on every query |
| **Embeddable** | No | Deploy inside any app |

---

## Architecture

```
Query
  ↓
IntentRouter — classifies: FACTUAL / ANALYTICAL / CONVERSATIONAL / OUT_OF_SCOPE
  ├── OUT_OF_SCOPE ──────────────────────────────────► Polite refusal
  └── else
        ↓
   QueryRewriter — expands follow-up questions using conversation history
        ↓
    Retriever — Hybrid: ChromaDB dense + BM25 keyword + source diversity
        ↓
   DocGrader — batch-scores all chunks for relevance in ONE API call
        ├── poor relevance ──► WebSearch (Tavily → DuckDuckGo fallback)
        └── good relevance
               ↓
            Writer — Tree of Thought
                      FACTUAL   → 1 draft (fast)
                      ANALYTICAL → 3 drafts, picks most grounded
               ↓
   HallucinationChecker — lexical grounding score
        ├── score < 0.4 and retries < 2 ──► QueryRewriter (retry loop)
        └── grounded
               ↓
          Synthesizer — attaches source citations + confidence score
               ↓
            Reviewer — compiles full metrics dict
               ↓
         Final Answer + Sources + Metrics
```

---

## Features

| Feature | Details |
|---|---|
| **Hybrid Search** | ChromaDB dense embeddings + BM25 keyword search + RRF merge + cross-encoder rerank |
| **Source Diversity** | Forces results from every uploaded document, not just the dominant one |
| **5 Chunking Strategies** | Fixed, Sentence Packing, Semantic, Parent-Child (recommended), Proposition |
| **Parent-Child Chunking** | Small child chunks for precise retrieval; large parent chunks for full context |
| **Agentic Routing** | Intent classification, conditional edges, retry loops — not a fixed chain |
| **Web Search Fallback** | Tavily API → DuckDuckGo when local docs have no relevant content |
| **Offline Fallback** | TinyLlama 1.1B runs on CPU when Groq API is unavailable |
| **Streaming Responses** | Token-by-token streaming via Groq API |
| **Source Citations** | Every answer shows which document, relevance score, and text snippet |
| **Confidence Score** | Composite of groundedness + document relevance |
| **Tree of Thought** | 3 reasoning approaches generated; most grounded draft selected |
| **Conversation Memory** | Last 10 turns stored for follow-up questions |
| **RAGAS Metrics** | Groundedness, Confidence, Answer Relevance, Faithfulness, Context Recall |
| **Metrics Dashboard** | Full pipeline telemetry — docs retrieved, retries, web search, model used |

---

## Project Structure

```
agentic_rag/
├── app.py                  ← Streamlit UI — chat, upload, metrics dashboard
├── evaluator.py            ← RAGAS-style evaluation metrics
├── requirements.txt        ← all dependencies
├── .env.example            ← copy to .env and add your API keys
│
└── pipeline/
    ├── __init__.py
    ├── llm.py              ← Groq API (primary) + TinyLlama (offline fallback)
    ├── loaders.py          ← PDF, DOCX, TXT, CSV, JSON file readers
    ├── chunker.py          ← 5 chunking strategies
    ├── retriever.py        ← ChromaDB + BM25 + reranker + source diversity
    ├── nodes.py            ← all 10 LangGraph node functions + AgentState
    └── graph.py            ← compiled agentic LangGraph graph
```

---

## Setup

### Prerequisites
- Python 3.11
- Git
- A free [Groq API key](https://console.groq.com)
- (Optional) A free [Tavily API key](https://tavily.com)

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/agentic-rag.git
cd agentic-rag
```

### 2. Create and activate virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate.bat

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add API keys
```bash
cp .env.example .env
```
Open `.env` and fill in your keys:
```
GROQ_API_KEY=your_groq_key_here
TAVILY_API_KEY=your_tavily_key_here
```

### 5. Run the app
```bash
python -m streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## Usage

1. **Upload documents** — drag and drop PDF, DOCX, TXT, CSV, or JSON files into the sidebar
2. **Choose chunking strategy** — `parent_child` recommended for most documents
3. **Click Ingest** — wait for the chunk count confirmation
4. **Ask questions** in the chat box
5. **Check Sources** to see which document answered and with what confidence
6. **Switch to Metrics tab** to see RAGAS scores and pipeline details after each query

---

## Supported File Types

| Format | Extension |
|---|---|
| PDF | `.pdf` |
| Word Document | `.docx` |
| Plain Text | `.txt` |
| CSV Spreadsheet | `.csv` |
| JSON | `.json` |

---

## Configuration

### Switch Groq model
In `pipeline/llm.py`:
```python
GROQ_MODEL = "llama-3.3-70b-versatile"        # default — fast, free
GROQ_MODEL = "mixtral-8x7b-32768"    # smarter, 32k context window
GROQ_MODEL = "llama3-70b-8192"       # most capable
```

### Adjust chunking size
In `pipeline/chunker.py`:
```python
def parent_child_chunking(
    text: str,
    parent_size: int = 1500,   # larger = more context per chunk
    child_size: int = 300,     # smaller = more precise retrieval
    overlap: int = 30,
):
```

---



## Tech Stack

| Component | Technology |
|---|---|
| LLM API | [Groq](https://groq.com) (llama-3.3-70b-versatile) |
| Offline LLM | [TinyLlama 1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) |
| Embeddings | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| Reranker | [ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) |
| Vector DB | [ChromaDB](https://www.trychroma.com) |
| Keyword Search | [BM25 (rank-bm25)](https://github.com/dorianbrown/rank_bm25) |
| Agent Framework | [LangGraph](https://github.com/langchain-ai/langgraph) |
| Web Search | [Tavily](https://tavily.com) / [DuckDuckGo](https://duckduckgo.com) |
| Frontend | [Streamlit](https://streamlit.io) |

---

## Future Improvements

- [ ] Support for `.pptx` files
- [ ] Persistent conversation history across sessions
- [ ] Multi-user support
- [ ] GraphRAG — knowledge graph-based retrieval
- [ ] Fine-tuned embeddings for domain-specific documents
- [ ] Export answers and sources as PDF report

---

## Author

**Soumyadeep Mondal**
B.Tech
[GitHub](https://github.com/raj005178)

---

## License

MIT License — feel free to use, modify, and distribute.



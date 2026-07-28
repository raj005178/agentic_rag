"""
nodes.py — Every LangGraph node in the agentic RAG pipeline.

Graph flow (see graph.py for edges):

  IntentRouter
    ├─ OUT_OF_SCOPE → OutOfScope → END
    └─ else         → QueryRewriter
                          ↓
                       Retriever
                          ↓
                       DocGrader
                    ┌────┴──────┐
                 graded      nothing relevant
                    │              ↓
                    │          WebSearch
                    └─────┬────┘
                          ↓
                        Writer  (Tree of Thought — 3 drafts)
                          ↓
               HallucinationChecker
                ┌─────┴──────┐
             grounded    not grounded (max 2 retries)
                │              ↓
                │         IncrementRetry → QueryRewriter
                ↓
            Synthesizer
                ↓
            Reviewer → END
"""

from typing import TypedDict, List, Dict, Any

from pipeline.llm import generate
from pipeline.retriever import hybrid_search


# ── State ────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query: str
    rewritten_query: str
    intent: str
    retrieved_docs: List[Dict]
    graded_docs: List[Dict]
    draft_answers: List[str]
    best_draft: str
    final_answer: str
    sources: List[Dict]
    hallucination_score: float
    confidence_score: float
    retry_count: int
    used_web_search: bool
    model_used: str
    conversation_history: List[Dict]
    metrics: Dict[str, Any]


# ── Helper ────────────────────────────────────────────────────────────────────

def _history_text(history: List[Dict], n: int = 6) -> str:
    if not history:
        return "None"
    return "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in history[-n:]
    )


def _safe_float(text: str, default: float = 0.5) -> float:
    try:
        v = float(text.strip().split()[0])
        return max(0.0, min(1.0, v))
    except Exception:
        return default


# ── Nodes ─────────────────────────────────────────────────────────────────────

def intent_router_node(state: AgentState) -> AgentState:
    """Classify query so the graph can skip expensive steps when unnecessary."""
    prompt = (
        "Classify the following query into EXACTLY ONE category.\n"
        "Reply with ONLY the category name, nothing else.\n\n"
        "Categories:\n"
        "  FACTUAL        — simple fact lookup\n"
        "  ANALYTICAL     — comparison, analysis, or multi-step reasoning\n"
        "  CONVERSATIONAL — follow-up that references earlier context\n"
        "  OUT_OF_SCOPE   — unrelated to any uploaded documents\n\n"
        f"Conversation history:\n{_history_text(state.get('conversation_history', []))}\n\n"
        f"Query: {state['query']}\n\nCategory:"
    )
    response, model = generate(prompt, max_tokens=10, temperature=0.0)
    intent = response.strip().upper().split()[0]
    if intent not in {"FACTUAL", "ANALYTICAL", "CONVERSATIONAL", "OUT_OF_SCOPE"}:
        intent = "ANALYTICAL"
    return {**state, "intent": intent, "model_used": model}


def query_rewriter_node(state: AgentState) -> AgentState:
    """
    Only rewrite the query if it's a CONVERSATIONAL follow-up.
    For FACTUAL and ANALYTICAL queries, use the original question directly
    so conversation history doesn't corrupt unrelated new questions.
    """
    intent = state.get("intent", "ANALYTICAL")

    # Independent question — don't let history corrupt it
    if intent in {"FACTUAL", "ANALYTICAL", "OUT_OF_SCOPE"}:
        return {**state, "rewritten_query": state["query"]}

    # CONVERSATIONAL — expand using history
    history = state.get("conversation_history", [])
    if not history:
        return {**state, "rewritten_query": state["query"]}

    prompt = (
        "The user is asking a follow-up question. "
        "Rewrite it as a fully self-contained question using the conversation context.\n"
        "Return ONLY the rewritten query, nothing else.\n\n"
        f"Conversation history:\n{_history_text(history)}\n\n"
        f"Follow-up question: {state['query']}\n\nRewritten query:"
    )
    response, model = generate(prompt, max_tokens=120, temperature=0.0)
    return {**state, "rewritten_query": response.strip(), "model_used": model}


def retriever_node(state: AgentState) -> AgentState:
    """Hybrid dense+BM25+rerank retrieval."""
    query = state.get("rewritten_query") or state["query"]
    docs = hybrid_search(query, top_k=10)
    return {**state, "retrieved_docs": docs}


def doc_grader_node(state: AgentState) -> AgentState:
    """
    Grade ALL docs in ONE single API call instead of one call per doc.
    Saves up to 9 API calls per query.
    """
    query = state.get("rewritten_query") or state["query"]
    docs = state["retrieved_docs"]
    if not docs:
        return {**state, "graded_docs": []}

    # Build one prompt with all docs numbered
    doc_lines = "\n\n".join(
        f"[{i+1}] {doc['text'][:400]}" for i, doc in enumerate(docs)
    )
    prompt = (
        "Rate each document's relevance to the query on a scale of 0.0 to 1.0.\n"
        "Reply with ONLY a comma-separated list of scores in the same order.\n"
        "Example for 3 docs: 0.9, 0.3, 0.7\n\n"
        f"Query: {query}\n\n"
        f"Documents:\n{doc_lines}\n\n"
        "Scores:"
    )
    resp, _ = generate(prompt, max_tokens=60, temperature=0.0)

    # Parse scores
    try:
        scores = [_safe_float(s) for s in resp.split(",")]
        # Pad if model returned fewer scores than docs
        while len(scores) < len(docs):
            scores.append(0.5)
    except Exception:
        scores = [0.5] * len(docs)

    graded = []
    for doc, score in zip(docs, scores):
        if score > 0.30:
            graded.append({**doc, "relevance_score": score})

    graded.sort(key=lambda d: -d["relevance_score"])
    return {**state, "graded_docs": graded}


def web_search_node(state: AgentState) -> AgentState:
    """
    Triggered when local docs are insufficient.
    Tries Tavily first, falls back to DuckDuckGo (no key needed).
    """
    query = state.get("rewritten_query") or state["query"]
    web_docs: List[Dict] = []

    # Tavily
    try:
        from tavily import TavilyClient
        import os
        key = os.getenv("TAVILY_API_KEY", "")
        if key and key != "your_tavily_api_key_here":
            results = TavilyClient(api_key=key).search(query=query, max_results=5)
            for r in results.get("results", []):
                web_docs.append({
                    "text":           r.get("content", ""),
                    "parent":         r.get("content", ""),
                    "source":         f"[Web] {r.get('url', 'unknown')}",
                    "relevance_score": r.get("score", 0.5),
                })
    except Exception as e:
        print(f"[WebSearch] Tavily error: {e}")

    # DuckDuckGo fallback
    if not web_docs:
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=5):
                    web_docs.append({
                        "text":           r.get("body", ""),
                        "parent":         r.get("body", ""),
                        "source":         f"[Web] {r.get('href', 'unknown')}",
                        "relevance_score": 0.45,
                    })
        except Exception as e:
            print(f"[WebSearch] DuckDuckGo error: {e}")

    existing = state.get("graded_docs", [])
    return {**state, "graded_docs": existing + web_docs, "used_web_search": True}


def writer_node(state: AgentState) -> AgentState:
    """
    Tree of Thought — generate 3 drafts with different reasoning styles,
    then pick the most grounded one.
    """
    query = state.get("rewritten_query") or state["query"]
    context = "\n\n".join(
        f"[{d['source']}]:\n{d['parent'][:2000]}"
        for d in state["graded_docs"][:5]
    )

    # FACTUAL queries → 1 draft (fast)
    # ANALYTICAL queries → 3 drafts Tree of Thought (thorough)
    intent = state.get("intent", "ANALYTICAL")
    approaches = (
        ["Answer directly and concisely using only the provided context."]
        if intent == "FACTUAL"
        else [
            "Answer directly and comprehensively using only the provided context.",
            "Answer step by step, breaking down the reasoning clearly.",
            "Answer by first identifying the most relevant facts, then synthesising them.",
        ]
    )

    drafts: List[str] = []
    last_model = state.get("model_used", "")
    for approach in approaches:
        prompt = (
            f"You are a precise document assistant. {approach}\n"
            "Base your answer entirely on the context below. "
            "If the answer is not in the context, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\nAnswer:"
        )
        resp, last_model = generate(prompt, max_tokens=900)
        drafts.append(resp.strip())

    # Pick draft with highest lexical overlap with context
    ctx_words = set(context.lower().split())
    best_draft = drafts[0]
    best_score = -1.0
    for draft in drafts:
        d_words = set(draft.lower().split())
        overlap = len(d_words & ctx_words) / (len(d_words) + 1)
        if overlap > best_score:
            best_score = overlap
            best_draft = draft

    return {
        **state,
        "draft_answers": drafts,
        "best_draft": best_draft,
        "model_used": last_model,
    }


def hallucination_checker_node(state: AgentState) -> AgentState:
    """
    Combine lexical grounding score + LLM-judged faithfulness score.
    Stored in state["hallucination_score"] (0 = hallucinated, 1 = grounded).
    """
    context = " ".join(d["parent"] for d in state["graded_docs"][:5])
    answer = state["best_draft"]

    # Lexical overlap only — no extra API call needed
    a_words = set(answer.lower().split())
    c_words = set(context.lower().split())
    lex_score = min(1.0, len(a_words & c_words) / (len(a_words) + 1) * 3)

    return {**state, "hallucination_score": round(lex_score, 3)}


def synthesizer_node(state: AgentState) -> AgentState:
    """Attach source citations and compute confidence."""
    sources: List[Dict] = []
    seen: set = set()
    for doc in state["graded_docs"][:5]:
        src = doc["source"]
        if src not in seen:
            seen.add(src)
            sources.append({
                "source":    src,
                "relevance": round(doc.get("relevance_score", 0.5), 2),
                "snippet":   doc["text"][:220],
            })

    avg_rel = (
        sum(d.get("relevance_score", 0.5) for d in state["graded_docs"][:3])
        / max(len(state["graded_docs"][:3]), 1)
    )
    confidence = round(state["hallucination_score"] * 0.6 + avg_rel * 0.4, 2)

    return {
        **state,
        "final_answer":    state["best_draft"],
        "sources":         sources,
        "confidence_score": confidence,
    }


def reviewer_node(state: AgentState) -> AgentState:
    """Compile pipeline telemetry into metrics dict."""
    metrics: Dict[str, Any] = {
        "groundedness":    round(state.get("hallucination_score", 0.0), 2),
        "confidence":      round(state.get("confidence_score", 0.0), 2),
        "docs_retrieved":  len(state.get("retrieved_docs", [])),
        "docs_after_grade": len(state.get("graded_docs", [])),
        "web_search_used": state.get("used_web_search", False),
        "retry_count":     state.get("retry_count", 0),
        "model_used":      state.get("model_used", "unknown"),
        "intent":          state.get("intent", "unknown"),
    }
    return {**state, "metrics": metrics}


def out_of_scope_node(state: AgentState) -> AgentState:
    """Return a polite refusal for queries unrelated to uploaded docs."""
    return {
        **state,
        "final_answer": (
            "I couldn't find relevant information in your uploaded documents "
            "to answer this question. Try uploading relevant files or rephrasing "
            "your question."
        ),
        "sources":         [],
        "confidence_score": 0.0,
        "metrics": {
            "groundedness": 0.0, "confidence": 0.0,
            "docs_retrieved": 0, "docs_after_grade": 0,
            "web_search_used": False, "retry_count": 0,
            "model_used": "none", "intent": "OUT_OF_SCOPE",
        },
    }


def increment_retry_node(state: AgentState) -> AgentState:
    return {**state, "retry_count": state.get("retry_count", 0) + 1}

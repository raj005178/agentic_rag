"""
graph.py — Build and compile the agentic LangGraph pipeline.

Conditional routing:
  IntentRouter  → OutOfScope  (if OUT_OF_SCOPE)
                → QueryRewriter (else)
  DocGrader     → WebSearch   (no relevant docs)
                → Writer      (docs found)
  HallucinationChecker → IncrementRetry → QueryRewriter  (score < 0.4, retries < 2)
                       → Synthesizer                      (grounded or max retries hit)
"""

from langgraph.graph import StateGraph, END

from pipeline.nodes import (
    AgentState,
    intent_router_node,
    query_rewriter_node,
    retriever_node,
    doc_grader_node,
    web_search_node,
    writer_node,
    hallucination_checker_node,
    synthesizer_node,
    reviewer_node,
    out_of_scope_node,
    increment_retry_node,
)


# ── Routing functions ─────────────────────────────────────────────────────────

def _route_intent(state: AgentState) -> str:
    return "out_of_scope" if state.get("intent") == "OUT_OF_SCOPE" else "rewrite"


def _route_after_grading(state: AgentState) -> str:
    docs = state.get("graded_docs", [])
    if not docs:
        return "websearch"
    avg = sum(d.get("relevance_score", 0) for d in docs) / len(docs)
    return "websearch" if avg < 0.30 else "write"


def _route_hallucination(state: AgentState) -> str:
    score = state.get("hallucination_score", 1.0)
    retries = state.get("retry_count", 0)
    if score < 0.40 and retries < 2:
        return "retry"
    return "synthesize"


# ── Build ─────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    # Register nodes
    g.add_node("IntentRouter",         intent_router_node)
    g.add_node("QueryRewriter",        query_rewriter_node)
    g.add_node("Retriever",            retriever_node)
    g.add_node("DocGrader",            doc_grader_node)
    g.add_node("WebSearch",            web_search_node)
    g.add_node("Writer",               writer_node)
    g.add_node("HallucinationChecker", hallucination_checker_node)
    g.add_node("Synthesizer",          synthesizer_node)
    g.add_node("Reviewer",             reviewer_node)
    g.add_node("OutOfScope",           out_of_scope_node)
    g.add_node("IncrementRetry",       increment_retry_node)

    # Entry
    g.set_entry_point("IntentRouter")

    # Edges
    g.add_conditional_edges(
        "IntentRouter", _route_intent,
        {"out_of_scope": "OutOfScope", "rewrite": "QueryRewriter"},
    )
    g.add_edge("QueryRewriter", "Retriever")
    g.add_edge("Retriever",     "DocGrader")
    g.add_conditional_edges(
        "DocGrader", _route_after_grading,
        {"websearch": "WebSearch", "write": "Writer"},
    )
    g.add_edge("WebSearch",  "Writer")
    g.add_edge("Writer",     "HallucinationChecker")
    g.add_conditional_edges(
        "HallucinationChecker", _route_hallucination,
        {"retry": "IncrementRetry", "synthesize": "Synthesizer"},
    )
    g.add_edge("IncrementRetry", "QueryRewriter")   # loop back
    g.add_edge("Synthesizer",    "Reviewer")
    g.add_edge("Reviewer",       END)
    g.add_edge("OutOfScope",     END)

    return g.compile()


# ── Compiled singleton ────────────────────────────────────────────────────────
compiled_graph = build_graph()


# ── Public API ────────────────────────────────────────────────────────────────

def run_query(query: str, conversation_history: list | None = None) -> dict:
    """
    Run the agentic pipeline for a single query.
    Returns the final AgentState dict.
    """
    initial: AgentState = {
        "query":                query,
        "rewritten_query":      "",
        "intent":               "",
        "retrieved_docs":       [],
        "graded_docs":          [],
        "draft_answers":        [],
        "best_draft":           "",
        "final_answer":         "",
        "sources":              [],
        "hallucination_score":  0.0,
        "confidence_score":     0.0,
        "retry_count":          0,
        "used_web_search":      False,
        "model_used":           "",
        "conversation_history": conversation_history or [],
        "metrics":              {},
    }
    return compiled_graph.invoke(initial)

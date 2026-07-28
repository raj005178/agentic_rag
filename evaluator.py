"""
evaluator.py — Lightweight RAGAS-inspired metrics.

  answer_relevance  : does the answer address the query?
  context_recall    : how much of the answer is supported by retrieved docs?
  faithfulness      : does the answer contradict the context?
"""

from typing import List, Dict
from pipeline.llm import generate


def _safe_float(text: str, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(text.strip().split()[0])))
    except Exception:
        return default


def answer_relevance(query: str, answer: str) -> float:
    prompt = (
        "Rate how well this answer addresses the query (0.0 = irrelevant, 1.0 = perfect).\n"
        "Reply with ONLY a decimal number.\n\n"
        f"Query: {query}\nAnswer: {answer}\n\nScore:"
    )
    resp, _ = generate(prompt, max_tokens=6, temperature=0.0)
    return round(_safe_float(resp), 2)


def context_recall(answer: str, docs: List[Dict]) -> float:
    """Lexical overlap between answer tokens and retrieved context tokens."""
    if not docs:
        return 0.0
    context = " ".join(d.get("text", "") for d in docs)
    a_words = set(answer.lower().split())
    c_words = set(context.lower().split())
    score = len(a_words & c_words) / (len(a_words) + 1)
    return round(min(1.0, score * 3), 2)


def faithfulness(answer: str, docs: List[Dict]) -> float:
    context = " ".join(d.get("parent", d.get("text", ""))[:500] for d in docs[:5])
    prompt = (
        "Rate how faithful this answer is to the context "
        "(0.0 = contradicts context, 1.0 = fully supported).\n"
        "Reply with ONLY a decimal number.\n\n"
        f"Context: {context[:1200]}\nAnswer: {answer[:600]}\n\nScore:"
    )
    resp, _ = generate(prompt, max_tokens=6, temperature=0.0)
    return round(_safe_float(resp), 2)


def evaluate(query: str, answer: str, docs: List[Dict]) -> Dict[str, float]:
    """
    Run all three metrics and return a dict.
    Called after every query in app.py.
    """
    return {
        "answer_relevance": answer_relevance(query, answer),
        "context_recall":   context_recall(answer, docs),
        "faithfulness":     faithfulness(answer, docs),
    }

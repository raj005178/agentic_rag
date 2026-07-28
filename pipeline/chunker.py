"""
chunker.py — Five chunking strategies.

Every strategy returns a list of dicts:
  { child: str, parent: str, parent_id: int, child_id: int }

'child' is used for retrieval (small, precise).
'parent' is sent to the LLM (larger context window).
"""

import re
from typing import List, Dict, Callable, Optional
import numpy as np


# ── 1. Fixed chunking ───────────────────────────────────────────────────────

def fixed_chunking(
    text: str, chunk_size: int = 500, overlap: int = 50
) -> List[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap
    return chunks


# ── 2. Sentence packing ─────────────────────────────────────────────────────

def sentence_packing(text: str, max_chars: int = 500) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_chars:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            current = sent
    if current:
        chunks.append(current)
    return chunks


# ── 3. Semantic chunking ─────────────────────────────────────────────────────

def semantic_chunking(text: str, threshold: float = 0.35) -> List[str]:
    from sentence_transformers import SentenceTransformer
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) < 2:
        return [text]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = model.encode(sentences, normalize_embeddings=True)

    chunks, current = [], [sentences[0]]
    for i in range(1, len(sentences)):
        sim = float(np.dot(embs[i - 1], embs[i]))
        if sim < threshold:
            chunks.append(" ".join(current))
            current = [sentences[i]]
        else:
            current.append(sentences[i])
    if current:
        chunks.append(" ".join(current))
    return chunks


# ── 4. Parent-Child chunking ─────────────────────────────────────────────────

def parent_child_chunking(
    text: str,
    parent_size: int = 1500,
    child_size: int = 300,
    overlap: int = 30,
) -> List[Dict]:
    parents = fixed_chunking(text, parent_size, overlap=100)
    result = []
    for p_id, parent in enumerate(parents):
        children = fixed_chunking(parent, child_size, overlap)
        for c_id, child in enumerate(children):
            result.append(
                {"child": child, "parent": parent, "parent_id": p_id, "child_id": c_id}
            )
    return result


# ── 5. Proposition chunking ──────────────────────────────────────────────────

def proposition_chunking(
    text: str, generate_fn: Callable
) -> List[str]:
    """
    Uses the LLM to decompose text into atomic factual statements.
    Each statement becomes its own chunk — maximises retrieval precision.
    """
    base_chunks = fixed_chunking(text, chunk_size=1000, overlap=100)
    propositions = []
    for chunk in base_chunks:
        prompt = (
            "Decompose the following text into a list of simple, atomic factual statements.\n"
            "Each statement must contain exactly one fact.\n"
            "Return ONLY the statements, one per line, no numbers or bullets.\n\n"
            f"Text:\n{chunk}\n\nStatements:"
        )
        response, _ = generate_fn(prompt, max_tokens=512, temperature=0.0)
        props = [p.strip() for p in response.split("\n") if p.strip()]
        propositions.extend(props)
    return propositions


# ── Main dispatcher ───────────────────────────────────────────────────────────

def chunk_document(
    text: str,
    strategy: str = "parent_child",
    generate_fn: Optional[Callable] = None,
) -> List[Dict]:
    """
    Dispatch to the requested strategy and normalise output to
    [{ child, parent, parent_id, child_id }, …]
    """
    def wrap(chunks: List[str]) -> List[Dict]:
        return [
            {"child": c, "parent": c, "parent_id": i, "child_id": 0}
            for i, c in enumerate(chunks)
        ]

    if strategy == "fixed":
        return wrap(fixed_chunking(text))

    elif strategy == "sentence":
        return wrap(sentence_packing(text))

    elif strategy == "semantic":
        return wrap(semantic_chunking(text))

    elif strategy == "parent_child":
        return parent_child_chunking(text)

    elif strategy == "proposition":
        if generate_fn is None:
            raise ValueError("proposition chunking requires generate_fn")
        return wrap(proposition_chunking(text, generate_fn))

    else:
        print(f"[Chunker] Unknown strategy '{strategy}', falling back to parent_child.")
        return parent_child_chunking(text)

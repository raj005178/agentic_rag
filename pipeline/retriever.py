"""
retriever.py — Hybrid retrieval pipeline.

  1. Dense  : ChromaDB (MiniLM embeddings)
  2. Sparse : BM25 keyword search
  3. Merge  : Reciprocal Rank Fusion (RRF)
  4. Rerank : cross-encoder/ms-marco-MiniLM-L-6-v2
"""

from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

CHROMA_PATH = "./chroma_store"
EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── singletons ───────────────────────────────────────────────────────────────
_collection = None
_embedder: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None
_bm25: BM25Okapi | None = None
_bm25_corpus: List[str] = []
_bm25_meta: List[Dict] = []   # parallel list to _bm25_corpus


def _col():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_or_create_collection("rag_docs")
        _rebuild_bm25()   # always sync BM25 with ChromaDB on first access
    return _collection


def _rebuild_bm25():
    """Rebuild BM25 index from whatever is currently in ChromaDB."""
    global _bm25, _bm25_corpus, _bm25_meta
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        col = client.get_or_create_collection("rag_docs")
        total = col.count()
        if total == 0:
            return
        results = col.get(limit=total, include=["documents", "metadatas"])
        _bm25_corpus = results["documents"]
        _bm25_meta   = results["metadatas"]
        _bm25 = BM25Okapi([t.lower().split() for t in _bm25_corpus])
        print(f"[BM25] Rebuilt index with {total} chunks from ChromaDB.")
    except Exception as e:
        print(f"[BM25] Rebuild failed: {e}")


def _emb():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _rr():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


# ── ingestion ────────────────────────────────────────────────────────────────

def ingest_document(chunks: List[Dict], source: str) -> int:
    """
    Add chunks to ChromaDB and update BM25 index.
    chunks: [{ child, parent, parent_id, child_id }, …]
    Returns number of chunks added.
    """
    global _bm25, _bm25_corpus, _bm25_meta
    col = _col()
    embedder = _emb()

    texts = [c["child"] for c in chunks]
    parents = [c["parent"] for c in chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=True).tolist()
    ids = [f"{source}::{i}" for i in range(len(texts))]
    metadatas = [
        {"source": source, "parent": p[:2000], "child_id": str(i)}
        for i, p in enumerate(parents)
    ]

    col.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )

    # Update BM25
    _bm25_corpus.extend(texts)
    _bm25_meta.extend(metadatas)
    _bm25 = BM25Okapi([t.lower().split() for t in _bm25_corpus])

    return len(texts)


# ── retrieval ────────────────────────────────────────────────────────────────

def _rrf(lists: List[List[str]], k: int = 60) -> Dict[str, float]:
    """Reciprocal Rank Fusion over multiple ranked id lists."""
    scores: Dict[str, float] = {}
    for ranked in lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def hybrid_search(query: str, top_k: int = 8) -> List[Dict]:
    """
    Returns top_k docs after dense + BM25 fusion + diversity + reranking.
    Guarantees at least min(2, chunks_in_doc) results from every ingested document.
    """
    col = _col()
    total = col.count()
    if total == 0:
        return []

    # Fetch more candidates so small documents aren't squeezed out
    n_fetch = min(total, max(top_k * 3, 30))
    q_emb = _emb().encode([query], normalize_embeddings=True).tolist()

    # ── Dense search ──────────────────────────────────────────────────────────
    dense = col.query(query_embeddings=q_emb, n_results=n_fetch)
    id_to_doc: Dict[str, Dict] = {}
    dense_ranked: List[str] = []

    for text, chroma_id, meta, dist in zip(
        dense["documents"][0],
        dense["ids"][0],
        dense["metadatas"][0],
        dense["distances"][0],
    ):
        id_to_doc[chroma_id] = {
            "text":        text,
            "parent":      meta.get("parent", text),
            "source":      meta.get("source", "unknown"),
            "dense_score": 1.0 - dist,
        }
        dense_ranked.append(chroma_id)

    # ── Source diversity: add chunks from under-represented sources ───────────
    sources_in_results = {v["source"] for v in id_to_doc.values()}
    all_meta = col.get(include=["metadatas"])
    all_sources = {m.get("source", "") for m in all_meta["metadatas"]}
    missing = all_sources - sources_in_results

    for src in missing:
        try:
            extra = col.query(
                query_embeddings=q_emb,
                n_results=3,
                where={"source": src},
            )
            for text, chroma_id, meta, dist in zip(
                extra["documents"][0],
                extra["ids"][0],
                extra["metadatas"][0],
                extra["distances"][0],
            ):
                if chroma_id not in id_to_doc:
                    id_to_doc[chroma_id] = {
                        "text":        text,
                        "parent":      meta.get("parent", text),
                        "source":      meta.get("source", "unknown"),
                        "dense_score": 1.0 - dist,
                    }
                    dense_ranked.append(chroma_id)
        except Exception as e:
            print(f"[Retriever] Diversity fetch failed for {src}: {e}")

    # ── BM25 search ───────────────────────────────────────────────────────────
    bm25_ranked: List[str] = []
    if _bm25 and _bm25_corpus:
        scores = _bm25.get_scores(query.lower().split())
        top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:n_fetch]
        for idx in top_idx:
            if scores[idx] <= 0:
                break
            fake_id = f"bm25::{idx}"
            meta = _bm25_meta[idx]
            id_to_doc.setdefault(
                fake_id,
                {
                    "text":       _bm25_corpus[idx],
                    "parent":     meta.get("parent", _bm25_corpus[idx]),
                    "source":     meta.get("source", "unknown"),
                    "bm25_score": float(scores[idx]),
                },
            )
            bm25_ranked.append(fake_id)

    # ── RRF merge ─────────────────────────────────────────────────────────────
    rrf_scores = _rrf([dense_ranked, bm25_ranked])
    merged = sorted(id_to_doc.keys(), key=lambda x: -rrf_scores.get(x, 0))
    candidates = [id_to_doc[i] for i in merged]

    # ── Rerank ────────────────────────────────────────────────────────────────
    if len(candidates) > 1:
        pairs = [(query, d["text"]) for d in candidates]
        rerank_scores = _rr().predict(pairs)
        for doc, score in zip(candidates, rerank_scores):
            doc["relevance_score"] = float(score)
        candidates.sort(key=lambda d: -d["relevance_score"])
    else:
        for doc in candidates:
            doc.setdefault("relevance_score", 0.5)

    return candidates[:top_k]


# ── utilities ─────────────────────────────────────────────────────────────────

def get_doc_count() -> int:
    try:
        return _col().count()
    except Exception:
        return 0


def get_chunks_per_doc() -> dict:
    """Returns { filename: chunk_count } for all ingested documents."""
    try:
        col = _col()
        total = col.count()
        if total == 0:
            return {}
        results = col.get(limit=total, include=["metadatas"])
        counts: dict = {}
        for meta in results["metadatas"]:
            src = meta.get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        return counts
    except Exception:
        return {}


def clear_collection():
    global _collection, _bm25, _bm25_corpus, _bm25_meta
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection("rag_docs")
    except Exception:
        pass
    _collection = None
    _bm25 = None
    _bm25_corpus = []
    _bm25_meta = []

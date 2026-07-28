"""
loaders.py — Load text from PDF, DOCX, TXT, CSV, JSON files.
Returns a dict: { text, source, file_type }
"""

import os
import json
from pathlib import Path


def load_pdf(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {i+1}]\n{text}")
    return "\n\n".join(pages)


def load_docx(path: str) -> str:
    from docx import Document
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_csv(path: str) -> str:
    import pandas as pd
    df = pd.read_csv(path)
    return df.to_string(index=False)


def load_json(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return json.dumps(data, indent=2)


_LOADERS = {
    ".pdf":  load_pdf,
    ".docx": load_docx,
    ".txt":  load_txt,
    ".csv":  load_csv,
    ".json": load_json,
}

SUPPORTED_TYPES = list(_LOADERS.keys())


def load_file(path: str) -> dict:
    """
    Load any supported file.
    Returns { "text": str, "source": str, "file_type": str }
    """
    ext = Path(path).suffix.lower()
    if ext not in _LOADERS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {SUPPORTED_TYPES}"
        )
    text = _LOADERS[ext](path)
    return {
        "text": text,
        "source": os.path.basename(path),
        "file_type": ext,
    }

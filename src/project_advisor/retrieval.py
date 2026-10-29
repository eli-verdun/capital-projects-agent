"""Lexical retrieval over the synthetic Meridian Project Delivery Advisor knowledge base.

TF-IDF with cosine similarity, computed locally with numpy. The demo's models run on
Anthropic through the LLM Gateway, and Anthropic has no embedding model, so retrieval
needs no provider call. The KB is 17 short documents with exact identifiers (TD-441,
PN-4938966, SP-1000-8820), which lexical matching handles well.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

KB_DIR = Path(__file__).parent / "kb"

# Keep hyphenated identifiers whole (td-441, pn-4938966) and decimals (5.1).
_TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
_STOP = frozenset(
    "a an and are as at be by for from has have how in is it its of on or that the this to "
    "was what when which why with does do i my we you can".split()
)


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in _STOP:
            continue
        out.append(tok)
        # Also index the parts, so "alignment" matches "hot-alignment".
        if "-" in tok:
            out.extend(p for p in tok.split("-") if p and p not in _STOP)
    return out


def _load_kb_documents() -> list[Document]:
    docs: list[Document] = []
    for md_path in sorted(KB_DIR.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        docs.append(
            Document(
                page_content=text,
                metadata={"source": md_path.name, "topic": md_path.stem},
            )
        )
    return docs


class _TfidfIndex:
    def __init__(self, chunks: list[Document]) -> None:
        self.chunks = chunks
        counts = [Counter(_tokens(c.page_content + " " + c.metadata["topic"].replace("_", " "))) for c in chunks]
        df = Counter(term for c in counts for term in c)
        self.vocab = {term: i for i, term in enumerate(sorted(df))}
        n = len(chunks)
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1 for t in sorted(df)])
        self.matrix = np.vstack([self._vector(c) for c in counts]) if chunks else np.zeros((0, len(self.vocab)))

    def _vector(self, counts: Counter) -> np.ndarray:
        vec = np.zeros(len(self.vocab))
        for term, tf in counts.items():
            if (i := self.vocab.get(term)) is not None:
                vec[i] = (1 + math.log(tf)) * self.idf[i]
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def search(self, query: str, k: int) -> list[Document]:
        q = self._vector(Counter(_tokens(query)))
        if not q.any():
            return []
        scores = self.matrix @ q
        order = [i for i in np.argsort(-scores) if scores[i] > 0]
        return [self.chunks[i] for i in order[:k]]


@lru_cache(maxsize=1)
def get_index() -> _TfidfIndex:
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    return _TfidfIndex(splitter.split_documents(_load_kb_documents()))


def retrieve(query: str, k: int = 4) -> list[Document]:
    return get_index().search(query, k)

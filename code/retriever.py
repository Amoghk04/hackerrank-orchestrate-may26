"""
retriever.py — Hybrid BM25 + Sentence-Transformer retriever over the corpus.

Pipeline:
  1. CorpusLoader: scan data/ for .md files, chunk by headings / token window
  2. HybridRetriever: build BM25 + MiniLM embeddings in-memory (no vector DB)
  3. retrieve(query, company, top_k): RRF-fused results with optional domain boost
"""

from __future__ import annotations

import hashlib
import pathlib
import pickle
import re
import textwrap
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from utils import DOMAIN_DIRS, DOMAIN_BOOST, RRF_K, TOP_K, MIN_RETRIEVAL_CONFIDENCE, REPO_ROOT

# ---------------------------------------------------------------------------
# Chunk model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single retrievable unit from the corpus."""
    text: str               # raw chunk content
    source_file: str        # relative path like "hackerrank/screen/article.md"
    domain: str             # 'hackerrank' | 'claude' | 'visa'
    heading: str = ""       # H2/H3 heading that introduced this chunk (empty for intro)
    tokens: int = 0         # approximate token count

    def metadata_label(self) -> str:
        """Human-readable label for prompt context."""
        heading_part = f" — {self.heading}" if self.heading else ""
        return f"[SOURCE: {self.source_file}{heading_part}, domain: {self.domain}]"

    def as_context(self) -> str:
        """Formatted chunk for inclusion in Claude prompt."""
        return f"{self.metadata_label()}\n{self.text.strip()}"


# ---------------------------------------------------------------------------
# Corpus loader
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)
_APPROX_CHARS_PER_TOKEN = 4
_MAX_CHUNK_CHARS = 1600   # ~400 tokens
_MIN_CHUNK_CHARS = 80     # discard very tiny fragments


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _APPROX_CHARS_PER_TOKEN)


def _chunk_document(text: str, source_file: str, domain: str) -> List[Chunk]:
    """
    Split a markdown document into chunks.
    Strategy:
      - Split at H1/H2/H3 headings first
      - If a section is too large, apply a sliding window
    """
    chunks: List[Chunk] = []

    # Split by headings; keep heading text with its section
    sections = re.split(r"(?m)^(#{1,3}\s+.+)$", text)
    # sections alternates: [pre-heading text, heading, section_body, heading, section_body, ...]

    current_heading = ""
    current_body = sections[0]  # text before first heading

    i = 1
    while i < len(sections):
        # Flush current body
        if current_body.strip():
            _emit_chunks(current_body, current_heading, source_file, domain, chunks)
        # Next element is a heading line
        heading_line = sections[i].strip()
        current_heading = re.sub(r"^#+\s*", "", heading_line)
        current_body = sections[i + 1] if i + 1 < len(sections) else ""
        i += 2

    # Flush last section
    if current_body.strip():
        _emit_chunks(current_body, current_heading, source_file, domain, chunks)

    return chunks


def _emit_chunks(
    text: str,
    heading: str,
    source_file: str,
    domain: str,
    out: List[Chunk],
) -> None:
    """
    Emit one or more Chunk objects from a section of text.
    Applies sliding window if the section is too large.
    """
    text = text.strip()
    if not text or len(text) < _MIN_CHUNK_CHARS:
        return

    if len(text) <= _MAX_CHUNK_CHARS:
        out.append(Chunk(
            text=text,
            source_file=source_file,
            domain=domain,
            heading=heading,
            tokens=_estimate_tokens(text),
        ))
        return

    # Sliding window: split on double newline, group into ~_MAX_CHUNK_CHARS blocks
    paragraphs = re.split(r"\n{2,}", text)
    current: List[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if current_len + len(para) > _MAX_CHUNK_CHARS and current:
            chunk_text = "\n\n".join(current)
            out.append(Chunk(
                text=chunk_text,
                source_file=source_file,
                domain=domain,
                heading=heading,
                tokens=_estimate_tokens(chunk_text),
            ))
            # Overlap: keep last paragraph for context continuity
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += len(para)

    if current:
        chunk_text = "\n\n".join(current)
        if len(chunk_text) >= _MIN_CHUNK_CHARS:
            out.append(Chunk(
                text=chunk_text,
                source_file=source_file,
                domain=domain,
                heading=heading,
                tokens=_estimate_tokens(chunk_text),
            ))


class CorpusLoader:
    """Scans data/ directories and produces a list of Chunks."""

    def __init__(self, domain_dirs: dict[str, pathlib.Path] | None = None):
        self._domain_dirs = domain_dirs or DOMAIN_DIRS

    def load(self) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        for domain, base_dir in self._domain_dirs.items():
            if not base_dir.exists():
                continue
            for md_file in sorted(base_dir.rglob("*.md")):
                try:
                    text = md_file.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                rel = str(md_file.relative_to(base_dir.parent))
                chunks = _chunk_document(text, source_file=rel, domain=domain)
                all_chunks.extend(chunks)
        return all_chunks


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank + 1)  # +1 because ranks are 0-indexed


class HybridRetriever:
    """
    Combines BM25 and Sentence-Transformer semantic search via RRF.
    All data is kept in-memory; no external vector DB required.
    """

    def __init__(self, chunks: List[Chunk], model_name: str = "all-MiniLM-L6-v2"):
        self._chunks = chunks
        self._build_bm25()
        self._build_semantic(model_name)

    def _build_bm25(self) -> None:
        tokenized = [_tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)

    def _build_semantic(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)
        texts = [c.text for c in self._chunks]
        # Encode in batches; normalise for cosine similarity via dot product
        self._embeddings: np.ndarray = self._model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def retrieve(
        self,
        query: str,
        company: str | None = None,
        top_k: int = TOP_K,
    ) -> List[Chunk]:
        """
        Return the top-k most relevant chunks for the query.

        Args:
            query:   Concatenation of subject + issue text.
            company: Canonical domain ('hackerrank', 'claude', 'visa') or None.
            top_k:   Number of chunks to return.
        """
        return [chunk for chunk, _ in self.retrieve_with_scores(query, company, top_k)]

    def retrieve_with_scores(
        self,
        query: str,
        company: str | None = None,
        top_k: int = TOP_K,
    ) -> List[Tuple[Chunk, float]]:
        """
        Like retrieve(), but also returns the RRF score for each chunk.

        Returns:
            List of (Chunk, rrf_score) tuples, sorted by score descending.
            The top score can be compared against MIN_RETRIEVAL_CONFIDENCE to
            detect weak corpus coverage before calling the LLM.
        """
        n = len(self._chunks)

        # --- BM25 ranking ---
        bm25_scores = self._bm25.get_scores(_tokenize(query))
        bm25_ranks = np.argsort(-bm25_scores)  # descending

        # --- Semantic ranking ---
        q_emb = self._model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        sem_scores = self._embeddings @ q_emb  # dot product = cosine (normalised)
        sem_ranks = np.argsort(-sem_scores)

        # --- RRF fusion ---
        # Build rank lookup tables
        bm25_rank_of = {int(idx): rank for rank, idx in enumerate(bm25_ranks)}
        sem_rank_of = {int(idx): rank for rank, idx in enumerate(sem_ranks)}

        rrf: dict[int, float] = {}
        for idx in range(n):
            score = _rrf_score(bm25_rank_of[idx]) + _rrf_score(sem_rank_of[idx])
            # Domain boost
            if company and self._chunks[idx].domain == company:
                score *= DOMAIN_BOOST
            rrf[idx] = score

        # Sort by RRF score descending
        ranked = sorted(rrf.keys(), key=lambda i: rrf[i], reverse=True)
        return [(self._chunks[i], rrf[i]) for i in ranked[:top_k]]


# ---------------------------------------------------------------------------
# Disk cache helpers
# ---------------------------------------------------------------------------

_CACHE_DIR = REPO_ROOT / "code" / ".cache"


def _corpus_fingerprint() -> str:
    """MD5 hash of all corpus .md file paths + mtimes + sizes for cache invalidation."""
    hasher = hashlib.md5()
    for domain_dir in DOMAIN_DIRS.values():
        if not domain_dir.exists():
            continue
        for md_file in sorted(domain_dir.rglob("*.md")):
            stat = md_file.stat()
            hasher.update(f"{md_file}:{stat.st_mtime}:{stat.st_size}\n".encode())
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_retriever(show_progress: bool = True) -> HybridRetriever:
    """
    Load the corpus and build the hybrid index.
    Caches chunk list and embeddings to disk — subsequent runs load in ~2s
    instead of ~15-25s.
    """
    from rich.console import Console
    console = Console()

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fp_file = _CACHE_DIR / "fingerprint.txt"
    chunks_file = _CACHE_DIR / "chunks.pkl"
    embeddings_file = _CACHE_DIR / "embeddings.npy"

    current_fp = _corpus_fingerprint()
    cached_fp = fp_file.read_text().strip() if fp_file.exists() else ""

    if current_fp == cached_fp and chunks_file.exists() and embeddings_file.exists():
        console.print("[bold cyan]Loading retriever from cache…[/bold cyan]")
        try:
            with open(chunks_file, "rb") as fh:
                chunks = pickle.load(fh)
            embeddings = np.load(str(embeddings_file))
            # Reconstruct without re-encoding all 6k+ chunks
            retriever = HybridRetriever.__new__(HybridRetriever)
            retriever._chunks = chunks
            retriever._build_bm25()
            retriever._model = SentenceTransformer("all-MiniLM-L6-v2")
            retriever._embeddings = embeddings
            console.print(f"[green]Index loaded from cache ({len(chunks)} chunks).[/green]")
            return retriever
        except Exception as e:
            console.print(f"[yellow]Cache load failed ({e}), rebuilding…[/yellow]")

    console.print("[bold cyan]Loading corpus…[/bold cyan]")
    loader = CorpusLoader()
    chunks = loader.load()
    console.print(f"[green]Loaded {len(chunks)} chunks from {len(DOMAIN_DIRS)} domains.[/green]")

    console.print("[bold cyan]Building BM25 + semantic index…[/bold cyan]")
    retriever = HybridRetriever(chunks)
    console.print("[green]Index ready.[/green]")

    # Persist to cache for next run
    try:
        with open(chunks_file, "wb") as fh:
            pickle.dump(chunks, fh)
        np.save(str(embeddings_file), retriever._embeddings)
        fp_file.write_text(current_fp)
        console.print("[green]Index saved to cache.[/green]")
    except Exception as e:
        console.print(f"[yellow]Cache save failed: {e}[/yellow]")

    return retriever


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    retriever = build_retriever()
    print("\n--- Test query: 'inactivity timeout interviews' ---")
    results = retriever.retrieve("inactivity timeout interviews", company="hackerrank")
    for r in results:
        print(f"  {r.metadata_label()} — {len(r.text)} chars")

    print("\n--- Test query: 'Visa card blocked during travel' ---")
    results = retriever.retrieve("Visa card blocked during travel", company="visa")
    for r in results:
        print(f"  {r.metadata_label()} — {len(r.text)} chars")

"""Corpus loader.

Reads any text-like files from ``data/<company>/`` and turns them into
``Document`` objects (one per file, plus chunked passages for retrieval).

Supported formats:
- .md / .markdown / .txt        -> read as plain text
- .html / .htm                   -> stripped to visible text via BeautifulSoup
- .json                          -> recursively concatenates all string fields

If ``data/`` is missing or empty, the agent still runs but every retrieval
returns no context; the safety layer will then escalate any non-trivial
ticket instead of hallucinating.  Run ``python -m triage.scrape`` to populate
``data/`` from the three official support sites.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

# Companies we know about. The directory name on disk must match (case-insensitive).
KNOWN_COMPANIES = ("hackerrank", "claude", "visa")

TEXT_EXTS = {".md", ".markdown", ".txt", ".rst"}
HTML_EXTS = {".html", ".htm"}
JSON_EXTS = {".json"}


@dataclass
class Document:
    company: str
    path: str
    title: str
    text: str
    url: str = ""


@dataclass
class Passage:
    company: str
    doc_path: str
    doc_title: str
    doc_url: str
    text: str
    # chunk index inside its document
    chunk_id: int = 0


@dataclass
class Corpus:
    documents: list[Document] = field(default_factory=list)
    passages: list[Passage] = field(default_factory=list)

    def filter(self, company: str | None) -> "Corpus":
        """Return a Corpus restricted to a single company (case-insensitive)."""
        if not company or company.lower() == "none":
            return self
        c = company.lower()
        docs = [d for d in self.documents if d.company == c]
        passages = [p for p in self.passages if p.company == c]
        return Corpus(documents=docs, passages=passages)

    def is_empty(self) -> bool:
        return not self.passages


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to read %s: %s", path, exc)
        return ""


def _strip_html(html: str) -> str:
    try:
        from bs4 import BeautifulSoup  # local import to keep startup light
    except ImportError:
        # crude fallback
        return re.sub(r"<[^>]+>", " ", html)
    soup = BeautifulSoup(html, "lxml" if _has_lxml() else "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    text = soup.get_text("\n")
    # collapse blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _has_lxml() -> bool:
    try:
        import lxml  # noqa: F401
        return True
    except ImportError:
        return False


def _flatten_json(value) -> str:
    """Recursively pull all string values out of a JSON blob."""
    out: list[str] = []

    def walk(v):
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, dict):
            for k, vv in v.items():
                if isinstance(k, str):
                    out.append(k)
                walk(vv)
        elif isinstance(v, list):
            for vv in v:
                walk(vv)

    walk(value)
    return "\n".join(out)


def _file_to_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in TEXT_EXTS:
        return _read_text(path)
    if ext in HTML_EXTS:
        return _strip_html(_read_text(path))
    if ext in JSON_EXTS:
        try:
            data = json.loads(_read_text(path))
            return _flatten_json(data)
        except Exception:
            return _read_text(path)
    return ""


def _title_from(path: Path, text: str) -> str:
    # First markdown heading wins, otherwise the file stem.
    m = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    if m:
        return m.group(1).strip()
    # Look for a leading non-blank short line
    for line in text.splitlines():
        line = line.strip()
        if 0 < len(line) <= 120:
            return line
    return path.stem.replace("_", " ").replace("-", " ").strip()


def _chunk(text: str, target_chars: int = 900, overlap: int = 150) -> list[str]:
    """Split text into ~target_chars chunks on paragraph boundaries."""
    text = re.sub(r"\r\n?", "\n", text).strip()
    if not text:
        return []
    # First split on blank lines (paragraphs).
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for para in paragraphs:
        if len(cur) + len(para) + 2 <= target_chars:
            cur = (cur + "\n\n" + para).strip()
        else:
            if cur:
                chunks.append(cur)
            if len(para) <= target_chars:
                cur = para
            else:
                # Hard split very long paragraph
                start = 0
                while start < len(para):
                    chunks.append(para[start : start + target_chars])
                    start += target_chars - overlap
                cur = ""
    if cur:
        chunks.append(cur)
    return chunks


def _iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (TEXT_EXTS | HTML_EXTS | JSON_EXTS):
            yield p


def load_corpus(data_dir: str | Path = "data") -> Corpus:
    root = Path(data_dir)
    corpus = Corpus()
    if not root.exists():
        logger.warning("Corpus directory %s does not exist.", root)
        return corpus

    for company in KNOWN_COMPANIES:
        cdir = root / company
        if not cdir.exists():
            # Be permissive: accept capitalised dir names too
            for cand in root.iterdir():
                if cand.is_dir() and cand.name.lower() == company:
                    cdir = cand
                    break
            else:
                continue
        for path in _iter_files(cdir):
            text = _file_to_text(path).strip()
            if not text:
                continue
            title = _title_from(path, text)
            url = _extract_url_hint(text)
            doc = Document(
                company=company,
                path=str(path),
                title=title,
                text=text,
                url=url,
            )
            corpus.documents.append(doc)
            for i, ch in enumerate(_chunk(text)):
                corpus.passages.append(
                    Passage(
                        company=company,
                        doc_path=doc.path,
                        doc_title=doc.title,
                        doc_url=doc.url,
                        text=ch,
                        chunk_id=i,
                    )
                )
    logger.info(
        "Loaded corpus: %d documents, %d passages from %s",
        len(corpus.documents),
        len(corpus.passages),
        root,
    )
    return corpus


_URL_RE = re.compile(r"https?://[^\s)>\]]+", re.IGNORECASE)


def _extract_url_hint(text: str) -> str:
    m = _URL_RE.search(text)
    return m.group(0) if m else ""

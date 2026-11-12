"""Optional support-site scraper.

Populates ``data/<company>/`` with one Markdown file per article, using
each site's public sitemap or article index.  Polite throttling is built
in (1 request per second by default) and a per-domain page cap can be
configured via env var ``TRIAGE_MAX_PAGES``.

Usage::

    python -m triage.scrape                       # all three sites
    python -m triage.scrape --only hackerrank
    python -m triage.scrape --only claude visa --max-pages 200

This module is *only* needed if you do not already have a corpus on disk.
The agent itself never calls the network.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

logger = logging.getLogger("triage.scrape")

USER_AGENT = "TriageAgentScraper/1.0 (+https://example.com/triage)"
DEFAULT_THROTTLE = 1.0  # seconds between requests
DEFAULT_MAX_PAGES = int(os.getenv("TRIAGE_MAX_PAGES", "300"))

SITES = {
    "hackerrank": {
        "seeds": [
            "https://support.hackerrank.com/hc/en-us",
        ],
        # Zendesk help-center article URLs
        "article_re": re.compile(r"/hc/en-us/articles/\d+", re.IGNORECASE),
    },
    "claude": {
        "seeds": [
            "https://support.claude.com/en/",
            "https://privacy.claude.com/en/",
        ],
        # Intercom help center article URLs
        "article_re": re.compile(r"/en/articles/\d+", re.IGNORECASE),
    },
    "visa": {
        "seeds": [
            "https://www.visa.co.in/support.html",
            "https://www.visa.co.in/pay-with-visa.html",
            "https://www.visa.co.in/about-visa.html",
        ],
        # Generic .html pages on the Visa India site (the site is mostly static).
        "article_re": re.compile(r"\.html(?:$|\?)", re.IGNORECASE),
    },
}


def _fetch(url: str, session) -> str:
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en"}
    r = session.get(url, headers=headers, timeout=20)
    if r.status_code != 200 or "text/html" not in r.headers.get("content-type", "html"):
        raise RuntimeError(f"GET {url} -> {r.status_code} {r.headers.get('content-type')}")
    return r.text


def _to_markdown(html: str) -> tuple[str, str]:
    """Return (title, markdown-ish text) extracted from an HTML page."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    title_tag = soup.find(["h1", "title"])
    title = (title_tag.get_text(strip=True) if title_tag else "Untitled")[:200]

    # Pick the main article container if available, else body.
    main = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", attrs={"class": re.compile(r"article|content|post|main", re.I)})
        or soup.body
        or soup
    )
    text = main.get_text("\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return title, f"# {title}\n\n{text}\n"


def _safe_filename(url: str) -> str:
    parsed = urlparse(url)
    raw = (parsed.path or "/").strip("/").replace("/", "_") or "index"
    raw = re.sub(r"[^A-Za-z0-9._-]+", "-", raw)
    return raw[:120] + ".md"


def _crawl(company: str, cfg: dict, out_dir: Path, *, max_pages: int, throttle: float) -> int:
    import requests

    session = requests.Session()
    seen: set[str] = set()
    queue: list[str] = list(cfg["seeds"])
    article_re: re.Pattern = cfg["article_re"]
    seed_hosts = {urlparse(s).netloc for s in cfg["seeds"]}
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    while queue and saved < max_pages:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)

        try:
            html = _fetch(url, session)
        except Exception as exc:
            logger.warning("skip %s: %s", url, exc)
            continue

        # Save anything that looks like an article page
        if article_re.search(url):
            try:
                title, md = _to_markdown(html)
                fname = _safe_filename(url)
                (out_dir / fname).write_text(
                    md + f"\n<!-- source: {url} -->\n", encoding="utf-8"
                )
                saved += 1
                logger.info("[%s] saved (%d) %s -> %s", company, saved, url, fname)
            except Exception as exc:
                logger.warning("save %s: %s", url, exc)

        # Discover more links on the same hosts
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "lxml")
            for a in soup.find_all("a", href=True):
                href = urljoin(url, a["href"]).split("#", 1)[0]
                host = urlparse(href).netloc
                if host in seed_hosts and href not in seen:
                    queue.append(href)
        except Exception:
            pass

        time.sleep(throttle)

    return saved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scrape support corpora.")
    parser.add_argument("--only", nargs="+", choices=list(SITES), help="Subset of sites.")
    parser.add_argument("--data", default="data", help="Output root (default: data/).")
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    parser.add_argument("--throttle", type=float, default=DEFAULT_THROTTLE)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    targets = args.only or list(SITES)
    root = Path(args.data)
    for name in targets:
        cfg = SITES[name]
        out = root / name
        logger.info("=== Scraping %s -> %s ===", name, out)
        n = _crawl(name, cfg, out, max_pages=args.max_pages, throttle=args.throttle)
        logger.info("=== %s: %d articles saved ===", name, n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

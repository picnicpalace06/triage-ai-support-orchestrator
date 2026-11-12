"""Chat transcript logger.

Writes a chronological log of every ticket the agent has triaged into a
plain-text ``log.txt`` (or whatever path is configured).  The format is
deliberately human-readable so a reviewer can audit decisions:

    ====================================================================
    [2026-05-01T10:23:11Z] Ticket #3
    --------------------------------------------------------------------
    Company (input)   : HackerRank
    Subject           : Test Score Dispute
    Issue             : ...
    --------------------------------------------------------------------
    Inferred company  : hackerrank
    Triggers          : high_risk_keywords, prompt_injection_attempt
    Risk              : high
    Retrieval scores  : 0.34, 0.21, 0.18
    LLM used          : True
    --------------------------------------------------------------------
    Decision          : escalated
    Request type      : invalid
    Product area      : screen
    Justification     : ...
    --------------------------------------------------------------------
    Response:
    Escalate to a human
    ====================================================================
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from .agent import TriageResult


class TranscriptLogger:
    def __init__(self, path: str | Path = "log.txt", append: bool = False) -> None:
        self.path = Path(path)
        mode = "a" if append else "w"
        self._fh = self.path.open(mode, encoding="utf-8")
        self._counter = 0
        self._write_header()

    def _write_header(self) -> None:
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self._fh.write(
            f"# Multi-Domain Support Triage Agent - chat transcript\n"
            f"# Session start: {ts}\n\n"
        )
        self._fh.flush()

    def log_ticket(
        self,
        *,
        issue: str,
        subject: str,
        company: str,
        result: TriageResult,
        index: int | None = None,
    ) -> None:
        self._counter += 1
        n = index if index is not None else self._counter
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        sep = "=" * 76
        sub = "-" * 76

        scores = ", ".join(f"{s:.2f}" for s in result.retrieval_scores) or "(none)"
        triggers = ", ".join(result.triggers) or "(none)"

        lines = [
            sep,
            f"[{ts}] Ticket #{n}",
            sub,
            f"Company (input)   : {company or '(blank)'}",
            f"Subject           : {subject or '(blank)'}",
            f"Issue             : {_truncate(issue, 800)}",
            sub,
            f"Inferred company  : {result.inferred_company}",
            f"Triggers          : {triggers}",
            f"Risk              : {result.risk}",
            f"Retrieval scores  : {scores}",
            f"LLM used          : {result.used_llm}",
            sub,
            f"Decision          : {result.status}",
            f"Request type      : {result.request_type}",
            f"Product area      : {result.product_area}",
            f"Justification     : {result.justification}",
            sub,
            "Response:",
            result.response or "(empty)",
            sep,
            "",
        ]
        self._fh.write("\n".join(lines))
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def _truncate(text: str, n: int) -> str:
    text = text.replace("\r", " ")
    if len(text) <= n:
        return text
    return text[: n - 3] + "..."

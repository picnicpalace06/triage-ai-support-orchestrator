"""Terminal entry-point for the support triage agent.

Two modes:

1. **Interactive** (default):

       python -m triage.cli

   Type tickets one at a time.  Use ``::company=hackerrank`` etc. to set
   metadata, ``::subject=...`` to set the subject, and a blank line on its
   own to submit.  Type ``quit`` to exit.

2. **Batch** (CSV in -> CSV out):

       python -m triage.cli --input  ../support_tickets/support_tickets.csv \
                            --output ../support_tickets/output.csv \
                            --log    ../log.txt

The CSV columns are case-insensitive: ``issue``, ``subject``, ``company``.
Output columns match the problem statement:
``issue, subject, company, response, product_area, status, request_type,
justification``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

# Best-effort .env loading so users can drop secrets in <repo>/.env
try:
    from dotenv import load_dotenv

    for _candidate in (
        Path(__file__).resolve().parents[2] / ".env",
        Path(__file__).resolve().parents[1] / ".env",
        Path.cwd() / ".env",
    ):
        if _candidate.is_file():
            load_dotenv(_candidate, override=False)
            break
except Exception:
    pass

from .agent import TriageAgent, TriageResult
from .corpus import load_corpus
from .llm import LLMClient
from .logger import TranscriptLogger

OUTPUT_COLUMNS = [
    "issue",
    "subject",
    "company",
    "response",
    "product_area",
    "status",
    "request_type",
    "justification",
]


# Default locations are derived from the repo contract in AGENTS.md.
# code/ sits next to data/ and support_tickets/ at the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = str(_REPO_ROOT / "data")
DEFAULT_TICKETS_IN = str(_REPO_ROOT / "support_tickets" / "support_tickets.csv")
DEFAULT_TICKETS_OUT = str(_REPO_ROOT / "support_tickets" / "output.csv")


def _default_log_path() -> str:
    """AGENTS.md canonical log path: ~/hackerrank_orchestrate/log.txt."""
    home = Path(os.path.expanduser("~"))
    return str(home / "hackerrank_orchestrate" / "log.txt")


# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = []
        for raw in reader:
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items()}
            rows.append(row)
    return rows


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in OUTPUT_COLUMNS})


def _result_to_row(
    *, issue: str, subject: str, company: str, result: TriageResult
) -> dict:
    return {
        "issue": issue,
        "subject": subject,
        "company": company,
        "response": result.response,
        "product_area": result.product_area,
        "status": result.status,
        "request_type": result.request_type,
        "justification": result.justification,
    }


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def run_batch(args: argparse.Namespace) -> int:
    corpus = load_corpus(args.data)
    if corpus.is_empty():
        logging.warning(
            "Corpus at %s is empty - the agent will escalate non-trivial cases. "
            "Run `python -m triage.scrape` or copy markdown files into data/<company>/.",
            args.data,
        )

    agent = TriageAgent(corpus, LLMClient())
    log = TranscriptLogger(args.log) if args.log else None

    in_path = Path(args.input)
    rows_in = _read_rows(in_path)
    out_rows: list[dict] = []

    print(f"Triaging {len(rows_in)} tickets from {in_path}...", file=sys.stderr)

    for i, row in enumerate(rows_in, 1):
        issue = row.get("issue", "")
        subject = row.get("subject", "")
        company = row.get("company", "")
        result = agent.triage(issue, subject, company)
        out_rows.append(
            _result_to_row(issue=issue, subject=subject, company=company, result=result)
        )
        if log:
            log.log_ticket(
                issue=issue,
                subject=subject,
                company=company,
                result=result,
                index=i,
            )
        print(
            f"[{i:3d}/{len(rows_in)}] {result.status:9s} "
            f"{result.request_type:16s} {result.product_area:24s} "
            f"{(subject or issue)[:60]!r}",
            file=sys.stderr,
        )

    _write_rows(Path(args.output), out_rows)
    if log:
        log.close()
    print(f"Wrote {len(out_rows)} rows to {args.output}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

INTERACTIVE_HELP = """
Commands:
  ::company=<hackerrank|claude|visa|none>  set company for the next ticket
  ::subject=<text>                         set subject for the next ticket
  ::reset                                  clear pending subject/company
  ::quit / quit / exit                     leave the session
  (blank line)                              submit the typed ticket

Type the ticket body across one or more lines, then a blank line to send.
"""


def run_interactive(args: argparse.Namespace) -> int:
    corpus = load_corpus(args.data)
    if corpus.is_empty():
        print(
            "[warn] corpus at "
            f"{args.data} is empty; ungrounded tickets will be escalated.\n"
            "       Populate data/hackerrank, data/claude, data/visa with text/markdown.",
            file=sys.stderr,
        )
    agent = TriageAgent(corpus, LLMClient())
    log = TranscriptLogger(args.log) if args.log else None

    print("Multi-Domain Support Triage Agent")
    print("Type your ticket. Blank line to submit. Type 'quit' to exit.")
    print(INTERACTIVE_HELP)

    pending_company = ""
    pending_subject = ""
    buffer: list[str] = []
    counter = 0

    while True:
        try:
            line = input("> " if not buffer else "  ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        stripped = line.strip()
        if stripped.lower() in ("quit", "exit", "::quit"):
            break
        if stripped.startswith("::"):
            cmd = stripped[2:].strip()
            if cmd == "reset":
                pending_company = ""
                pending_subject = ""
                buffer.clear()
                print("[reset]")
                continue
            if "=" in cmd:
                k, v = cmd.split("=", 1)
                k = k.strip().lower()
                v = v.strip()
                if k == "company":
                    pending_company = v
                    print(f"[company={v}]")
                elif k == "subject":
                    pending_subject = v
                    print(f"[subject={v}]")
                else:
                    print(f"[unknown directive: {k}]")
                continue
            if cmd == "help":
                print(INTERACTIVE_HELP)
                continue
            print(f"[unknown command: {cmd}]")
            continue

        if stripped == "" and buffer:
            issue = "\n".join(buffer).strip()
            buffer.clear()
            counter += 1
            result = agent.triage(issue, pending_subject, pending_company)
            _print_result(result)
            if log:
                log.log_ticket(
                    issue=issue,
                    subject=pending_subject,
                    company=pending_company,
                    result=result,
                    index=counter,
                )
            pending_company = ""
            pending_subject = ""
            continue

        if stripped == "" and not buffer:
            continue

        buffer.append(line)

    if log:
        log.close()
    print("Goodbye.")
    return 0


def _print_result(result: TriageResult) -> None:
    print()
    print("--- Triage decision ---")
    print(f"status        : {result.status}")
    print(f"request_type  : {result.request_type}")
    print(f"product_area  : {result.product_area}")
    print(f"risk          : {result.risk}")
    print(f"company       : {result.inferred_company}")
    print(f"justification : {result.justification}")
    print("response      :")
    for line in (result.response or "(empty)").splitlines() or ["(empty)"]:
        print(f"  {line}")
    print()


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="triage",
        description="Multi-domain support triage agent (HackerRank / Claude / Visa).",
    )
    p.add_argument(
        "--data", default=DEFAULT_DATA_DIR,
        help=f"Root directory containing the support corpus (default: {DEFAULT_DATA_DIR}).",
    )
    p.add_argument(
        "--input", default=None,
        help=f"Input CSV path. If omitted, interactive mode runs. Batch default: {DEFAULT_TICKETS_IN}",
    )
    p.add_argument(
        "--output", default=None,
        help=f"Output CSV path. Batch default: {DEFAULT_TICKETS_OUT}",
    )
    p.add_argument(
        "--batch", action="store_true",
        help="Run batch mode using the default input/output paths.",
    )
    p.add_argument(
        "--log", default=_default_log_path(),
        help="Chat transcript log file (AGENTS.md path by default).",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    if args.batch:
        if not args.input:
            args.input = DEFAULT_TICKETS_IN
        if not args.output:
            args.output = DEFAULT_TICKETS_OUT
    if args.input and args.output:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        return run_batch(args)
    if args.input or args.output:
        parser.error("--input and --output must be provided together for batch mode.")
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    return run_interactive(args)


if __name__ == "__main__":
    raise SystemExit(main())

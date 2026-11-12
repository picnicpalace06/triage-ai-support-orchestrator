# Multi-Domain Support Triage Agent

A terminal-based support triage agent for **HackerRank**, **Claude (Anthropic)**, and **Visa**, grounded **only** in the local support corpus shipped at the repo-root `data/` directory.

It identifies the request type, classifies the product area, assesses risk, decides whether to **reply** or **escalate**, retrieves the most relevant docs, and generates a grounded response.

## Project layout

```
.                              # repo root (per AGENTS.md)
├── data/                      # support corpus (used at triage time, NOT shipped in submission zip)
│   ├── hackerrank/
│   ├── claude/
│   └── visa/
├── support_tickets/
│   ├── support_tickets.csv    # input
│   ├── output.csv             # written by the agent (your submission)
│   └── sample_support_tickets.csv
└── code/
    ├── main.py                # convenience entry-point
    ├── requirements.txt
    └── triage/
        ├── agent.py           # orchestration: classify -> retrieve -> respond
        ├── classifier.py      # rules + safety/escalation policy
        ├── corpus.py          # loads data/<company>/* into chunked passages
        ├── retriever.py       # TF-IDF cosine retriever
        ├── llm.py             # optional LLM wrapper (Anthropic / OpenAI)
        ├── logger.py          # chat transcript logger
        ├── scrape.py          # optional support-site scraper
        └── cli.py             # interactive + batch CLI
```

The agent is read-only against the corpus; nothing is sent over the network at triage time.

## Setup

```powershell
cd code
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
cd code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### (Optional) configure an LLM

If neither key is set the agent runs in **extractive** mode, returning clean excerpts from the most relevant passages. Copy `../.env.example` to `../.env` and uncomment one block, or export directly:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...
```

## Running

### Batch (canonical paths from AGENTS.md)

```bash
python -m triage.cli --batch
```

Reads `../support_tickets/support_tickets.csv`, loads `../data/`, writes `../support_tickets/output.csv`. The chat transcript is appended to the canonical AGENTS.md log path:

| Platform      | Log path                                       |
| ------------- | ---------------------------------------------- |
| Windows       | `%USERPROFILE%\hackerrank_orchestrate\log.txt` |
| macOS / Linux | `$HOME/hackerrank_orchestrate/log.txt`         |

Override any path:

```bash
python -m triage.cli \
    --input  ../support_tickets/support_tickets.csv \
    --output ../support_tickets/output.csv \
    --data   ../data \
    --log    ../log.txt
```

Output columns: `issue, subject, company, response, product_area, status, request_type, justification`.

### Interactive

```bash
python -m triage.cli
```

Type a ticket across multiple lines and submit with a blank line. Set metadata with directives:

```
::company=hackerrank
::subject=Test invite expired
We assigned a test in October 2025 and the invite seems inactive...

```

## Decision policy

The agent is conservative. It **always escalates** when any of these are true, regardless of corpus matches:

- Lost / stolen / blocked Visa card, identity theft, fraud or unauthorised charges
- Account access recovery (locked / hacked / seat removed)
- Refunds, billing disputes, payment errors with order IDs
- Subscription pause / cancel / change requests
- Test or assessment **score** disputes / regrade requests
- InfoSec / security-questionnaire requests
- Reports of vulnerabilities or bug-bounty submissions
- Site outages / "nothing works" reports (`request_type=bug`)
- Requests to stop crawling / data-use opt-outs that need legal action
- Detected prompt-injection or jailbreak attempts

It **replies** when the question is grounded by the corpus with sufficient TF-IDF retrieval confidence and the topic is not in the always-escalate list.

Chitchat (`thanks`, `hi`) gets a brief polite reply; trivia / unrelated coding tasks get a polite "out of scope" message.

## `product_area` derivation

Labels primarily come from the corpus folder of the top retrieved doc, with alias normalisations so they match the sample-ticket conventions:

| Folder slug                          | Label                     |
| ------------------------------------ | ------------------------- |
| `screen`                             | `screen`                  |
| `hackerrank_community`               | `community`               |
| `privacy_and_legal`                  | `privacy`                 |
| `conversation_management`            | `conversation_management` |
| `account_management`, `identity_management_sso_jit_scim` | `account_management` |
| `pro_and_max_plans`                  | `billing`                 |
| `team_and_enterprise_plans`          | `account_admin`           |
| `safeguards`                         | `trust_and_safety`        |
| `claude_api_and_console`, `amazon_bedrock`, `claude_code` | `developer_platform` |
| `travel_support`                     | `travel_support`          |
| `general_help`, `small_business`, `uncategorized`, fallback | `general_support` |

If retrieval has no hits, a keyword fallback in `agent.py` provides a label.

## Safety properties

- **Grounded answers only.** When using an LLM, the system prompt forbids inventing policies, prices, phone numbers, or URLs not present in the retrieved context, and instructs it to reply `ESCALATE` if context is insufficient.
- **Prompt-injection detection.** Tickets that try to extract internal prompts, retrieved documents, or override instructions (in any language) are escalated, never answered.
- **Off-platform requests** ("delete all files", "format my disk", etc.) are refused as out-of-scope.
- **Empty tickets** are escalated.
- **No web access at triage time.** The retriever is local TF-IDF; the optional `triage.scrape` module is the only network code path and is never invoked from the CLI.

## Determinism

- TF-IDF retrieval is fully deterministic.
- LLM calls use `temperature=0.1` (set to `0.0` in `llm.py` for byte-identical generations).
- No random sampling is used anywhere in the pipeline.

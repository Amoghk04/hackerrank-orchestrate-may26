# Multi-Domain Support Triage Agent

An AI-powered support triage system that classifies and responds to customer support tickets across three domains: **HackerRank**, **Claude/Anthropic**, and **Visa**. The agent reads tickets from a CSV, retrieves relevant documentation from a 774-file corpus, and produces structured triage decisions using Claude's tool_use API — all with zero hallucination risk on policies because every claim is grounded in retrieved documents.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Architecture](#architecture)
3. [Module Reference](#module-reference)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Running the Agent](#running-the-agent)
8. [Output Format](#output-format)
9. [Escalation Logic](#escalation-logic)
10. [Design Decisions](#design-decisions)
11. [Dependencies](#dependencies)

---

## How It Works

Each support ticket passes through a **5-layer pipeline**:

1. **Pre-process** — Normalise the ticket text; if the `Company` column is empty or `None`, infer the domain automatically from issue keywords (e.g. "assessment" → HackerRank, "API key" → Claude, "transaction" → Visa).

2. **Rule-based gate (pre-LLM)** — Before any API call, a deterministic classifier checks for adversarial content (prompt injection attempts, destructive commands) and hard-escalation triggers (score manipulation requests, identity fraud, financial disputes, security vulnerability reports). Tickets caught here are escalated immediately without touching the LLM — saving cost and preventing manipulation.

3. **Hybrid retrieval** — The issue + subject are used as a query against a corpus of 6,496 chunks (from 774 Markdown help-centre articles). Retrieval combines:
   - **BM25** (keyword match, via `rank-bm25`) for exact terminology
   - **Semantic search** (dense embeddings via `sentence-transformers/all-MiniLM-L6-v2`) for meaning-level similarity
   - **Reciprocal Rank Fusion (RRF)** to merge both ranked lists
   - **1.3× domain boost** — chunks from the inferred company domain score higher
   - Top 7 chunks are injected into the Claude prompt as grounding context.

4. **Claude structured output** — Claude (`claude-haiku-4-5`, temperature=0) is called with `tool_choice=any` forcing it to invoke the `submit_triage_decision` tool. This guarantees a structured JSON response with all five required fields — no free-text parsing needed.

5. **Pydantic validation + retry** — The tool response is validated against a `TriageResult` Pydantic model. If validation fails (wrong enum value, missing field), the entire API call retries once. If both attempts fail, a safe escalation fallback is returned.

---

## Architecture

```
support_tickets.csv
        │
        ▼
┌────────────────────┐
│  Layer 0           │  normalise_company(), infer_company()
│  Pre-process       │  → company_domain: "hackerrank" | "claude" | "visa" | None
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Layer 1           │  detect_adversarial()  →  escalated+invalid  (no LLM)
│  Rule-based gate   │  should_escalate()     →  force_escalate flag + reason
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Layer 2           │  BM25Okapi  ─┐
│  Hybrid retrieval  │  MiniLM emb  ├─ RRF + domain_boost → top-7 Chunks
└────────┬───────────┘  (6496 chks) ┘
         │
         ▼
┌────────────────────┐
│  Layer 3           │  claude-haiku-4-5, temp=0, tool_choice=any
│  Claude tool_use   │  submit_triage_decision(status, product_area,
└────────┬───────────┘    response, justification, request_type)
         │
         ▼
┌────────────────────┐
│  Layer 4           │  TriageResult(Pydantic) → validate enums
│  Validation+retry  │  force_escalate override if rule said escalate
└────────┬───────────┘
         │
         ▼
   output.csv
```

### Corpus structure

```
data/
├── hackerrank/     394 .md articles  (screen, interviews, library, skillup, …)
├── claude/         321 .md articles  (account-management, claude-api, claude-code, …)
└── visa/            59 .md articles  (support, general guidance, …)
```

Each article is split into chunks at H2/H3 headings first; long sections fall back to a sliding window at ~400 tokens with 50-token overlap. This gives **6,496 chunks** total.

---

## Module Reference

### `main.py` — Entry point

Parses CLI arguments, loads the environment, reads the input CSV, builds the retrieval index, runs the triage pipeline on every ticket, and writes `output.csv`. Displays a live Rich progress bar during processing and a summary table on completion.

### `retriever.py` — Corpus + hybrid index

- `CorpusLoader.load()` — walks `data/hackerrank`, `data/claude`, `data/visa`, reads every `.md` file, and produces `Chunk` objects with `domain`, `source_file`, and `text` fields.
- `Chunk.as_context()` — formats a chunk for injection into the Claude prompt, including its source path.
- `HybridRetriever` — builds a `BM25Okapi` index and a `SentenceTransformer` embedding matrix at startup. The `retrieve(query, company, top_k)` method runs both retrievers, fuses their ranked results via RRF, applies a `1.3×` domain boost to chunks from the inferred company, and returns the top-k chunks.
- `build_retriever()` — convenience factory used by `main.py`.

### `classifier.py` — Pre-LLM rule engine

- `detect_adversarial(issue)` — scans for prompt injection patterns (including French-language variants) and destructive command patterns. Returns `True` immediately on any match.
- `should_escalate(issue, company)` — checks five escalation categories: score manipulation, account takeover by non-owner, identity fraud, financial disputes, and security vulnerability reports. Returns `(bool, reason_string)`.
- `infer_company(issue, subject)` — scores domain-specific keyword hits across HackerRank, Claude, and Visa vocabulary lists; returns the domain with the highest unambiguous score.
- `classify_request_type(issue, company)` — provides an initial `request_type` hint (`bug`, `feature_request`, `product_issue`, `invalid`) that is passed to the LLM as a starting suggestion (the LLM may override it).

### `agent.py` — Triage orchestrator

- `TriageResult` — Pydantic model with field validators enforcing `status ∈ {replied, escalated}` and `request_type ∈ {product_issue, feature_request, bug, invalid}`.
- `_TRIAGE_TOOL` — Claude tool schema for `submit_triage_decision` with all five fields, their types, and detailed descriptions constraining the LLM's choices.
- `_SYSTEM_TEMPLATE` — system prompt establishing the triage persona, grounding rules, explicit default of `replied`, hard conditions for `escalated`, and injected corpus context.
- `TriageAgent.triage(row)` — runs all five pipeline layers for one ticket row.
- `_call_claude()` — builds the prompt, calls the API with `tool_choice={"type": "any"}`, retries once on failure, and applies force-escalate override if the rule gate fired.
- `_safe_fallback()` — last-resort escalation returned if both API attempts fail.

### `utils.py` — Shared constants and helpers

Key constants: `MODEL_NAME`, `MAX_TOKENS=1500`, `TEMPERATURE=0`, `TOP_K=7`, `DOMAIN_BOOST=1.3`, `RRF_K=60`. Provides `load_env()`, `read_tickets()`, `write_output()`, `normalise_company()`, and `truncate()`.

---

## Prerequisites

- Python **3.11+**
- An [Anthropic API key](https://console.anthropic.com/) with access to `claude-haiku-4-5`
- ~500 MB disk space (sentence-transformers model download on first run)
- ~2 GB RAM (embedding matrix for 6,496 chunks)

---

## Installation

```bash
# Clone / unzip the repo, then from the repo root:
pip install -r code/requirements.txt
```

All dependencies are pure Python / PyPI — no system packages required.

---

## Configuration

### Option A — environment variable (recommended for CI/terminal)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### Option B — `.env` file in the repo root

```
ANTHROPIC_API_KEY=sk-ant-...
```

The agent uses `python-dotenv` to load `.env` automatically. The `.env` file is gitignored and never committed.

---

## Running the Agent

All commands are run from the **repo root** (`hackerrank-orchestrate-may26/`).

### Standard run

```bash
python code/main.py
```

Reads from `support_tickets/support_tickets.csv`, writes to `support_tickets/output.csv`.

### Explicit paths

```bash
python code/main.py \
  --input  support_tickets/support_tickets.csv \
  --output support_tickets/output.csv
```

### Dry run (no file written — for testing)

```bash
python code/main.py --dry-run
```

Processes all tickets and prints the summary table but does not write `output.csv`. Useful for prompt iteration without overwriting results.

### Expected terminal output

```
──── Step 1/4: Loading environment ────
✓ ANTHROPIC_API_KEY loaded

──── Step 2/4: Loading input tickets ────
✓ Loaded 29 tickets from support_tickets/support_tickets.csv

──── Step 3/4: Building retrieval index ────
Loading corpus…
Loaded 6496 chunks from 3 domains.
Building BM25 + semantic index…
Index ready.

──── Step 4/4: Triaging tickets ────
  REPLIED  product_issue  screen            (7.5s)
  ESCALATED product_issue account-management (7.6s)
  ...
  [29/29] 'Visa card minimum spend' ━━━━━━━━━━━━━━━ 29/29  0:03:15

✓ Output written to support_tickets/output.csv

        Triage Summary
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Metric            ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Total tickets     │ 29     │
│ Replied           │ 17     │
│ Escalated         │ 12     │
│ Processing errors │ 0      │
│ Total time        │ 566.9s │
│ Avg per ticket    │ 19.5s  │
└───────────────────┴────────┘
```

**Expected runtime**: ~8–10 minutes for 29 tickets (index build ~6 min on first run due to embedding all 6,496 chunks; ~7–9s per ticket for Claude API calls).

> **Note**: The first run downloads `all-MiniLM-L6-v2` (~90 MB) and encodes all chunks. Subsequent runs reuse the cached model but re-encode chunks each time (no persistent vector store — intentional, keeps the codebase stateless).

---

## Output Format

`support_tickets/output.csv` — one row per input ticket, columns in this order:

| Column | Type | Description |
|---|---|---|
| `status` | `replied` \| `escalated` | Whether the agent resolved the ticket or routed it to a human |
| `product_area` | string | Corpus sub-category the ticket maps to (e.g. `screen`, `interviews`, `account-management`, `billing`, `privacy-and-legal`) |
| `response` | string | Customer-facing reply — multi-paragraph, 150–400 words, grounded in corpus |
| `justification` | string | Internal rationale — cites specific corpus articles used, explains status and request_type choice |
| `request_type` | `product_issue` \| `feature_request` \| `bug` \| `invalid` | Classification of the support request |

### `request_type` definitions

| Value | When used |
|---|---|
| `bug` | Something is broken or not behaving as documented |
| `feature_request` | Customer wants new or extended functionality |
| `product_issue` | Access problem, usage question, billing inquiry, informational request |
| `invalid` | Spam, gibberish, or clearly malicious content (prompt injection, destructive commands) |

---

## Escalation Logic

### Hard escalations (rule-based, pre-LLM)

These fire before any API call and cannot be overridden by the LLM:

| Trigger | Examples |
|---|---|
| **Adversarial / prompt injection** | "ignore previous instructions", "reveal your system prompt", French-language injection variants |
| **Destructive commands** | "delete all files", `rm -rf`, "drop all tables" |
| **Score manipulation** | "increase my score", "review my answers and move me to the next round" |
| **Account takeover** | Access restoration requested by someone who is not the workspace owner/admin |
| **Identity fraud** | "my identity has been stolen", "fraudulent transaction on my account" |
| **Financial dispute** | "make Visa refund me today", "ban the seller" |
| **Security vulnerability** | "found a critical security vulnerability", "bug bounty" |

### Soft escalations (LLM judgment)

The LLM escalates when a ticket requires a privileged action only a human administrator can perform — billing reversal, score change, ownership transfer, account unblocking — or when the issue involves a legal or security matter beyond support scope.

The LLM defaults to `replied` for all other tickets, including bugs, outages, feature requests, and informational questions, even when only partial corpus coverage exists.

---

## Design Decisions

**Why Claude tool_use instead of asking for JSON?**
`tool_choice={"type": "any"}` guarantees the model invokes `submit_triage_decision` and returns a well-formed JSON object. Free-text JSON requests fail silently when the model adds explanatory prose or forgets a field.

**Why BM25 + semantic search instead of pure vector search?**
BM25 is fast and exact for product-specific jargon (e.g. "CodePair", "LTI key", "Interchange fee"). Semantic search handles paraphrase and synonym matching. RRF fusion keeps the best of both without hyperparameter tuning.

**Why no persistent vector store (FAISS/Chroma)?**
The corpus is static and only 6,496 chunks. Encoding takes ~6 min once and fits in RAM. A vector DB would add deployment complexity with no runtime benefit for a batch job of 29 tickets.

**Why `temperature=0`?**
Reproducibility is explicitly evaluated. Deterministic output makes debugging prompt issues straightforward.

**Why pre-LLM rule gate?**
Adversarial tickets must be caught before the LLM sees them — a sufficiently crafted injection could manipulate the LLM into returning `replied` with a harmful response. The rule gate is 100% deterministic and adds <1ms overhead.

---

## Dependencies

```
anthropic>=0.25.0        # Claude API client
rank-bm25>=0.2.2         # BM25Okapi retrieval
sentence-transformers>=2.7.0  # all-MiniLM-L6-v2 embeddings
pandas>=2.0.0            # CSV I/O
pydantic>=2.0.0          # output schema validation
rich>=13.0.0             # progress bars and summary tables
python-dotenv>=1.0.0     # .env file loading
numpy>=1.24.0            # embedding matrix dot-product
pydantic>=2.0.0
rich>=13.0.0
python-dotenv>=1.0.0
numpy>=1.24.0
```

---

## Submission

```bash
# Generate output.csv
python code/main.py

# Package code
zip -r code.zip code/
```

Submission artifacts:
1. `code.zip`
2. `support_tickets/output.csv`
3. `$HOME/hackerrank_orchestrate/log.txt` (auto-generated by AI tooling)

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
8. [Evaluation](#evaluation)
9. [Output Format](#output-format)
10. [Escalation Logic](#escalation-logic)
11. [Design Decisions](#design-decisions)
12. [Dependencies](#dependencies)

---

## How It Works

Each support ticket passes through a **5-layer pipeline**:

1. **Pre-process** — Normalise the ticket text; if the `Company` column is empty or `None`, infer the domain automatically from issue keywords (e.g. "assessment" → HackerRank, "API key" → Claude, "transaction" → Visa).

2. **Rule-based gate (pre-LLM)** — Before any API call, a deterministic classifier checks for adversarial content (prompt injection attempts, destructive commands) and hard-escalation triggers (score manipulation requests, identity fraud, financial disputes, security vulnerability reports). Tickets caught here are escalated immediately without touching the LLM — saving cost and preventing manipulation.

3. **Query expansion + Hybrid retrieval** — Before retrieval, the query is enriched by extracting salient terms from the ticket: capitalised phrases, technical tokens, and proper nouns (e.g. `CodePair`, `LTI`, `Haiku`). The expanded query is run against a corpus of 6,496 chunks (from 774 Markdown articles). Retrieval combines:
   - **BM25** (keyword match, via `rank-bm25`) for exact terminology
   - **Semantic search** (dense embeddings via `sentence-transformers/all-MiniLM-L6-v2`) for meaning-level similarity
   - **Reciprocal Rank Fusion (RRF)** to merge both ranked lists, returning a score per chunk
   - **1.3× domain boost** — chunks from the inferred company domain score higher
   - **Retrieval confidence gate** — if the top chunk's RRF score falls below `MIN_RETRIEVAL_CONFIDENCE` (0.004), the LLM is warned that corpus coverage is weak and should rely on general knowledge or escalate
   - Top 7 chunks (labelled `Document 1`…`Document 7`) are injected into the Claude prompt

4. **Claude structured output** — Before the API call, additional context flags are injected into the user message: if non-English characters account for >40% of letter characters, the LLM is instructed to respond in the ticket's language; if numbered lists or multi-request connector phrases are detected ("also want", "additionally", "two issues"), the LLM is instructed to address the primary request and acknowledge the rest. Claude (`claude-haiku-4-5`, temperature=0) is called with `tool_choice=any` forcing it to invoke the `submit_triage_decision` tool. The LLM is instructed to cite retrieved documents by number (e.g. `[1]`, `[2]`) in the `justification` field for full auditability.

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
- `Chunk.as_context()` — formats a chunk for injection into the Claude prompt, including its numbered label and source path.
- `HybridRetriever` — builds a `BM25Okapi` index and a `SentenceTransformer` embedding matrix at startup. `retrieve(query, company, top_k)` returns the top-k chunks; `retrieve_with_scores(query, company, top_k)` returns `(chunk, rrf_score)` pairs for confidence gating.
- `build_retriever()` — convenience factory used by `main.py`. Implements a **fingerprint-based disk cache** in `code/.cache/` (three files: `chunks.pkl`, `embeddings.npy`, `fingerprint.txt`). The fingerprint is an MD5 hash over all `.md` file paths, sizes, and modification times — any corpus change triggers a full rebuild; otherwise the index loads in ~2 seconds.

### `classifier.py` — Pre-LLM rule engine

- `detect_adversarial(issue)` — scans for prompt injection patterns (including French-language variants) and destructive command patterns. Returns `True` immediately on any match.
- `should_escalate(issue, company)` — checks five escalation categories: score manipulation, account takeover by non-owner, identity fraud, financial disputes, and security vulnerability reports. Returns `(bool, reason_string)`.
- `infer_company(issue, subject)` — scores domain-specific keyword hits across HackerRank, Claude, and Visa vocabulary lists; returns the domain with the highest unambiguous score.
- `classify_request_type(issue, company, subject="")` — provides an initial `request_type` hint (`bug`, `feature_request`, `product_issue`, `invalid`) that is passed to the LLM as a starting suggestion (the LLM may override it). Pattern matching runs over `f"{subject} {issue}"` so type keywords in the subject line (e.g. `BUG: login timeout`) are captured even when the body is purely descriptive.
- `detect_non_english(text)` — heuristic flag: returns `True` if >40% of alphabetic characters are non-ASCII. Used to inject a language-response note into the LLM prompt.
- `detect_multi_request(issue)` — detects numbered lists and multi-request connector phrases ("also want", "additionally", "two issues"). Returns a list of identified sub-request segments.

### `agent.py` — Triage orchestrator

- `TriageResult` — Pydantic model with field validators enforcing `status ∈ {replied, escalated}` and `request_type ∈ {product_issue, feature_request, bug, invalid}`.
- `_TRIAGE_TOOL` — Claude tool schema for `submit_triage_decision` with all five fields, their types, and detailed descriptions constraining the LLM's choices.
- `_SYSTEM_TEMPLATE` — system prompt establishing the triage persona, grounding rules (cite documents as `[1]`/`[2]` in justification), explicit default of `replied`, hard conditions for `escalated`, and injected corpus context.
- `_expand_query(issue, subject)` — enriches the retrieval query by extracting capitalised phrases, technical tokens, and proper nouns from the ticket (up to 8 extras appended). Improves recall for tickets with vague subject lines.
- `TriageAgent.triage(row)` — runs all five pipeline layers for one ticket row, including confidence gate check, non-English detection, and multi-request detection.
- `_call_claude()` — builds the user message with all contextual notes (low-confidence warning, language note, multi-request note), calls the API with `tool_choice={"type": "any"}`, retries once on failure, applies force-escalate override if the rule gate fired, and prepends a `[PIPELINE TRACE]` block to the `justification` field (see Output Format → justification).
- `_safe_fallback()` — last-resort escalation returned if both API attempts fail.

### `utils.py` — Shared constants and helpers

Key constants: `MODEL_NAME` (`claude-haiku-4-5`), `MAX_TOKENS=1500`, `TEMPERATURE=0`, `TOP_K=7`, `DOMAIN_BOOST=1.3`, `RRF_K=60`, `MIN_RETRIEVAL_CONFIDENCE=0.004`. Provides `load_env()`, `read_tickets()`, `write_output()`, `normalise_company()`, and `truncate()`.

> **Model slug note**: `claude-haiku-4-5` is the Anthropic GA name for Haiku 4.5. If the API returns a model-not-found error, fall back to `claude-3-5-haiku-20241022` in `utils.py`.

### `evaluate.py` — Self-evaluation script

Runs the full pipeline on `sample_support_tickets.csv` (the 10-row labelled reference set) and scores the resulting predictions against the reference labels. Reports per-row status / request_type / product_area accuracy with a mismatch detail section.

```bash
# Generate predictions for sample tickets and score (requires API key):
python code/evaluate.py

# Score an already-generated sample_output.csv without re-calling the API:
python code/evaluate.py --no-generate
```

Output: `support_tickets/sample_output.csv` plus a Rich accuracy table. Returns exit code 0 if overall accuracy ≥ 70%, else 1.

### `test_classifier.py` — Unit tests

51 assertion-based unit tests covering all six public functions in `classifier.py`. No external test framework required — runs with plain Python or `pytest`.

```bash
python code/test_classifier.py         # plain Python runner
pytest  code/test_classifier.py -v     # verbose pytest output
```

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

**Expected runtime**: ~3–5 minutes for 29 tickets after the index is cached (~7–9s per ticket for Claude API calls; index loads in ~2s from cache).

> **First run only**: Downloads `all-MiniLM-L6-v2` (~90 MB) and encodes all 6,496 chunks (~6 min). The resulting index is saved to `code/.cache/` and reused on every subsequent run. The cache auto-invalidates if any corpus `.md` file is added, modified, or removed.

---

## Evaluation

To measure accuracy against the labelled sample tickets before submission:

```bash
# Generate predictions for the 10-row sample and score them:
python code/evaluate.py

# If predictions are already in support_tickets/sample_output.csv:
python code/evaluate.py --no-generate
```

To run the classifier unit tests:

```bash
python code/test_classifier.py    # 51/51 should pass
```

---

## Output Format

`support_tickets/output.csv` — one row per input ticket, columns in this order:

| Column | Type | Description |
|---|---|---|
| `status` | `replied` \| `escalated` | Whether the agent resolved the ticket or routed it to a human |
| `product_area` | string | Corpus sub-category the ticket maps to (e.g. `screen`, `interviews`, `account-management`, `billing`, `privacy-and-legal`) |
| `response` | string | Customer-facing reply — multi-paragraph, 150–400 words, grounded in corpus |
| `justification` | string | Internal rationale. Structured as two sections: `[PIPELINE TRACE]` (JSON) followed by `[LLM JUSTIFICATION]` (prose). The trace captures pre-LLM deterministic signals — `company_detected`, `retrieval_confidence`, top-3 `top_chunks` with RRF scores, `rule_gate` (fired + reason), `request_type_hint`, and `flags` (non-English, multi-request). The LLM justification cites corpus articles as `[1]`/`[2]`. |
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

**Why a home-grown disk cache instead of a vector DB (FAISS/Chroma)?**
The corpus is static and only 6,496 chunks. The fingerprint-based cache saves the BM25 index and numpy embedding matrix to `code/.cache/` on the first build and reloads them in ~2 seconds on subsequent runs. A vector DB would add deployment complexity (Docker, config, schema migration) with no runtime benefit for a batch job of 29 tickets.

**Why `temperature=0`?**
Reproducibility is explicitly evaluated. Deterministic output makes debugging prompt issues straightforward.

**Why pre-LLM rule gate?**
Adversarial tickets must be caught before the LLM sees them — a sufficiently crafted injection could manipulate the LLM into returning `replied` with a harmful response. The rule gate is 100% deterministic and adds <1ms overhead.

---

## Dependencies

```
anthropic>=0.25.0             # Claude API client
rank-bm25>=0.2.2              # BM25Okapi retrieval
sentence-transformers>=2.7.0  # all-MiniLM-L6-v2 embeddings
pandas>=2.0.0                 # CSV I/O
pydantic>=2.0.0               # output schema validation
rich>=13.0.0                  # progress bars and summary tables
python-dotenv>=1.0.0          # .env file loading
numpy>=1.24.0                 # embedding matrix operations
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

# Design Document: Multi-Domain Support Triage Agent

**Event**: HackerRank Orchestrate — 24-hour AI Agent Hackathon  
**Date**: May 1–2, 2026  
**Author**: Amogh  

---

## 1. Problem Statement

Build an AI agent that triages customer support tickets across three product domains — **HackerRank**, **Claude/Anthropic**, and **Visa** — and produces structured decisions for each ticket.

### Input
`support_tickets/support_tickets.csv` — 24 rows with columns:
| Column | Description |
|---|---|
| `Issue` | Free-text customer complaint / request |
| `Subject` | Short title of the issue |
| `Company` | `HackerRank`, `Claude`, `Visa`, or `None` |

### Output
`support_tickets/output.csv` — same 24 rows with columns:
| Column | Allowed Values | Notes |
|---|---|---|
| `status` | `replied` / `escalated` | Lowercase only |
| `product_area` | Free-form category string | Can be empty for truly ambiguous unresolvable tickets |
| `response` | Multi-paragraph customer-facing reply | Long, step-by-step; cites corpus content |
| `justification` | Internal rationale for the decision | Traceable to retrieved corpus docs |
| `request_type` | `product_issue` / `feature_request` / `bug` / `invalid` | One of four exact strings |

---

## 2. Corpus

The agent's knowledge base lives in `data/` — **774 Markdown files** total:

| Domain | Directory | File Count | Coverage |
|---|---|---|---|
| HackerRank | `data/hackerrank/` | ~394 | Screen, Interviews, Engage, Chakra, Library, SkillUp, Settings, Community, Integrations |
| Claude/Anthropic | `data/claude/` | ~321 | Claude app, API/Console, Amazon Bedrock, Claude Code, Claude Desktop, Plans, SSO |
| Visa | `data/visa/` | ~15 | General Visa card support |
| Index files | `data/*/index.md` | 3 | Domain-level topic indices |

**All LLM responses must be grounded in these files.** No parametric knowledge used for policy answers. If the corpus does not cover a ticket, the agent escalates with an honest explanation.

---

## 3. Architecture Overview

The agent uses a **5-layer pipeline** per ticket:

```
┌──────────────────────────────────────────────────────────────────┐
│  Layer 0: Pre-Processing                                          │
│  - Normalize text (strip encoding artifacts)                     │
│  - Infer company from Issue text if Company == "None"            │
│  - Detect multi-request tickets                                  │
└──────────────────────────────┬───────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────┐
│  Layer 1: Rule-Based Escalation Gate (pre-LLM)                   │
│  - Adversarial / prompt-injection detection                      │
│  - Malicious code requests                                       │
│  - High-risk categories: fraud, identity theft, security vuln    │
│  - Financial disputes requiring 3rd-party action                 │
│  - Score manipulation attempts                                   │
│  - Account access by non-owner (security concern)               │
│  If triggered → status=escalated, skip LLM, generate safe reply │
└──────────────────────────────┬───────────────────────────────────┘
                               │ (not escalated)
┌──────────────────────────────▼───────────────────────────────────┐
│  Layer 2: Hybrid Retrieval (BM25 + Semantic + RRF)               │
│  - BM25Okapi (rank-bm25) over all 774 .md chunks                │
│  - Sentence-Transformers (all-MiniLM-L6-v2) cosine similarity   │
│  - Reciprocal Rank Fusion to merge ranked lists                  │
│  - 1.3× domain boost for chunks matching inferred company        │
│  - Returns top-7 chunks as context window                        │
└──────────────────────────────┬───────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────┐
│  Layer 3: Claude API (tool_use / structured output)              │
│  - System prompt: role definition + output schema enforcement    │
│  - Context: top-7 retrieved chunks passed as labeled references │
│  - Tool schema: TriageDecision (status, product_area, response,  │
│    justification, request_type) enforces structured JSON         │
│  - Temperature=0 for determinism                                 │
│  - Model: claude-3-5-haiku-20241022 (fast + cheap for 24 rows)  │
└──────────────────────────────┬───────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────┐
│  Layer 4: Output Validation & Normalization                      │
│  - Pydantic model validates all 5 fields                        │
│  - Coerce status to lowercase                                    │
│  - Ensure request_type is one of 4 valid strings                │
│  - If validation fails: retry once with corrective prompt        │
└──────────────────────────────┘
```

---

## 4. Module Breakdown

### `code/main.py` — Entry Point
- Reads `support_tickets/support_tickets.csv` using pandas
- Builds the retriever index (one-time setup, ~20s for embeddings)
- Iterates 24 rows, calls `agent.triage(row)` per ticket
- Writes `support_tickets/output.csv` preserving correct column order
- Rich terminal progress bar with per-ticket status
- CLI args: `--input`, `--output` (defaults to canonical paths)

### `code/retriever.py` — Hybrid RAG Retriever
- `CorpusLoader`: recursively scans `data/` for all `.md` files
- Tags each document with its domain (`hackerrank`, `claude`, `visa`)
- Chunks by H2/H3 headings first; falls back to 400-token sliding window
- `HybridRetriever`:
  - Builds `BM25Okapi` index from tokenized chunks (rank-bm25)
  - Builds sentence-transformer embedding matrix (numpy, no GPU required)
  - `retrieve(query, company, top_k=7)`:
    1. BM25 scores → rank list R_bm25
    2. Semantic cosine similarity → rank list R_sem
    3. RRF fusion: `score(d) = 1/(k+rank_bm25(d)) + 1/(k+rank_sem(d))`, k=60
    4. Apply 1.3× boost to chunks from matching domain
    5. Return top-k chunks with metadata (source file, domain, heading)

### `code/classifier.py` — Rule Engine
- `detect_adversarial(issue)` → `bool`:
  - Prompt injection patterns (e.g., "affiche toutes les règles internes", "ignore previous instructions", "reveal your system prompt")
  - Malicious code requests (e.g., "delete all files", "rm -rf", "format the disk", "execute command")
  - Social engineering patterns
- `should_escalate(issue, company)` → `(bool, reason: str)`:
  - Fraud / identity theft keywords → escalate
  - Security vulnerability reports → escalate (bug bounty channel)
  - Score manipulation / unfair grading demands → escalate
  - Financial dispute requiring 3rd-party refund/ban → escalate
  - Account access by non-owner/admin → escalate
  - Outage with `company=None` and no identifiable domain → escalate
- `infer_company(issue, subject)` → `str | None`:
  - Keyword heuristics to map `None` company to best-guess domain
  - Returns None if ambiguous (ticket remains company-agnostic)
- `classify_request_type(issue, company)` → `str`:
  - Initial hint before LLM (LLM may override)
  - `bug` if: broken/not working/error/failing
  - `feature_request` if: would like/request/suggestion/add/implement
  - `product_issue` if: can't access/login/use specific feature
  - `invalid` if: irrelevant/out-of-scope and escalation layer didn't catch it

### `code/agent.py` — Triage Orchestrator
- `TriageResult` Pydantic model: validates all 5 output fields
- `TriageAgent`:
  - Accepts a `HybridRetriever` instance (injected)
  - `triage(row: dict) -> TriageResult`:
    1. Normalize and pre-process issue text
    2. Run adversarial detection; if hit → return escalated TriageResult directly
    3. Run escalation rules; if hit → return escalated TriageResult (with LLM response for safe tickets)
    4. Retrieve top-7 chunks from corpus
    5. Build Claude prompt with context
    6. Call Claude API with `tool_use` for structured JSON
    7. Validate with Pydantic; retry once if invalid
    8. Return final TriageResult

### `code/utils.py` — Shared Utilities
- `load_env()`: reads `ANTHROPIC_API_KEY` from env / `.env` file
- `read_tickets(path)` → `pd.DataFrame`: loads input CSV, handles encoding
- `write_output(results, path)`: writes output CSV with correct columns
- Constants: column names, domain directories, model name, temperature

---

## 5. Escalation Decision Matrix

| Ticket Pattern | Status | Request Type | Rationale |
|---|---|---|---|
| Restore access (non-owner/admin) | `escalated` | `product_issue` | Account security, requires admin verification |
| Review my score / increase grade | `escalated` | `invalid` | Score manipulation request |
| Visa: refund me today + ban seller | `escalated` | `product_issue` | Financial dispute needing 3rd-party action |
| Delete all files (code request) | `escalated` | `invalid` | Malicious code request |
| French Visa injection ticket | `escalated` | `invalid` | Prompt injection / adversarial |
| Identity theft ("wat should I do") | `escalated` | `product_issue` | Fraud / identity theft |
| Claude security vulnerability | `escalated` | `bug` | Security report → bug bounty channel |
| Iron Man actor question | `replied` | `invalid` | Non-harmful OOS; politely redirected |
| "it's not working, help" + None | `escalated` | `product_issue` | Insufficient info, no domain, system unknown |

---

## 6. Claude API Integration

### Model
`claude-3-5-haiku-20241022` — chosen for:
- Low latency (~1–2s per call)
- Cost-effective for 24 sequential calls
- Strong instruction following for structured tool_use

### Tool Use Schema
```python
{
  "name": "submit_triage_decision",
  "description": "Submit the final triage decision for a support ticket",
  "input_schema": {
    "type": "object",
    "properties": {
      "status": {"type": "string", "enum": ["replied", "escalated"]},
      "product_area": {"type": "string"},
      "response": {"type": "string"},
      "justification": {"type": "string"},
      "request_type": {"type": "string", "enum": ["product_issue", "feature_request", "bug", "invalid"]}
    },
    "required": ["status", "product_area", "response", "justification", "request_type"]
  }
}
```

### System Prompt Strategy
1. Define role: "You are a senior support specialist..."
2. Provide domain context from retrieved chunks (labeled `[SOURCE: file, domain]`)
3. Enforce grounding rule: "Base your response ONLY on the provided context"
4. Specify response format: multi-paragraph, step-by-step where applicable
5. Provide escalation override guidance: when in doubt on sensitive topics, escalate

---

## 7. Key Design Decisions & Trade-offs

### Why Hybrid BM25 + Semantic (not pure embedding)?
- BM25 excels at exact keyword matches (e.g., "zoom connectivity", "inactivity timeout", "API key") which are critical in technical support
- Semantic search catches paraphrase ("my card was blocked" → "card declined / card hold")
- RRF fusion avoids the weaknesses of either alone
- No GPU needed — all-MiniLM-L6-v2 runs in <10s on CPU for 774 chunks

### Why tool_use over free-form JSON prompt?
- Anthropic's tool_use is more reliable than "respond in JSON" for strict schema adherence
- Avoids markdown code fences, trailing commas, and truncation issues
- Pydantic acts as a second validator; mismatches trigger one retry

### Why rule-based pre-LLM escalation?
- LLMs can be persuaded by adversarial inputs (prompt injection)
- Hard rules for malicious patterns are deterministic and not exploitable
- Keeps the LLM budget for genuinely ambiguous tickets

### Why not a vector DB (Chroma/Weaviate/Pinecone)?
- Only 24 queries total; in-memory numpy matrix is faster to build and query
- No external service dependency reduces failure modes
- Easier to inspect and debug for the AI Judge interview

### Why all-MiniLM-L6-v2?
- 80MB model, fast CPU inference (sentence-transformers)
- Strong performance on semantic similarity benchmarks (SBERT)
- No API calls needed for retrieval — fully offline

---

## 8. Response Quality Guidelines

Responses in the output CSV must match the quality of the sample CSV:

1. **Length**: Multi-paragraph (150–400 words). No one-liners.
2. **Structure**: Use numbered steps for procedural actions
3. **Tone**: Professional, empathetic, clear
4. **Grounding**: Cite specific features, settings, or articles from the corpus
5. **Escalation responses**: Still professional and helpful — explain why escalating and what the user can expect
6. **Justification**: Internal note — cite article names/headings from retrieved docs

---

## 9. Edge Cases

| Scenario | Handling |
|---|---|
| `Company = "None"` | `infer_company()` tries keyword matching; if still None, use all-domain retrieval |
| Multi-request ticket | Address highest-severity request; note others in justification |
| `replied` + `invalid` | For non-harmful OOS queries (e.g., Iron Man question) |
| `escalated` + `invalid` | For adversarial/malicious requests |
| Empty `product_area` | Allowed when company=None and no domain can be determined |
| Ticket in non-English | Detect, attempt English response, escalate if policy unclear |
| Encoding issues | Strip/normalize UTF-8, handle smart quotes and accents |

---

## 10. Tech Stack

| Library | Version Pin | Purpose |
|---|---|---|
| `anthropic` | `>=0.25.0` | Claude API client |
| `rank-bm25` | `>=0.2.2` | BM25Okapi retrieval |
| `sentence-transformers` | `>=2.7.0` | Semantic embeddings (MiniLM) |
| `pandas` | `>=2.0.0` | CSV I/O |
| `pydantic` | `>=2.0.0` | Output schema validation |
| `rich` | `>=13.0.0` | Terminal progress / logging |
| `python-dotenv` | `>=1.0.0` | `.env` file loading |
| `numpy` | `>=1.24.0` | Embedding matrix ops |

---

## 11. Submission Artifacts

| Artifact | Location | Generated By |
|---|---|---|
| `code.zip` | Root | `zip -r code.zip code/` |
| `output.csv` | `support_tickets/output.csv` | `python code/main.py` |
| `log.txt` | `$HOME/hackerrank_orchestrate/log.txt` | Auto-generated by AI tooling (Claude, Copilot) |

---

## 12. Research References

- **LightRAG** (EMNLP 2025) — Graph-enhanced RAG with dual-level indexing for global/local queries. Informed the domain-scoped retrieval design.
- **RAG Survey** (Gao et al., 2023) — Comprehensive survey on Retrieval-Augmented Generation. Informed chunking strategy and RRF fusion.
- **SBERT** (Reimers & Gurevych, 2019) — Sentence-BERT for semantic similarity. Basis for all-MiniLM-L6-v2 choice.
- **BM25 (Robertson, 1994)** — Probabilistic keyword retrieval. Complementary to semantic search for exact technical terms.

---

## 13. Performance Expectations

- **Corpus build time**: ~15–25s (embedding all 774 chunks on CPU)
- **Per-ticket time**: ~2–5s (retrieval: <100ms, Claude API: ~1–3s)
- **Total pipeline time**: ~2–3 minutes for 24 tickets
- **Total API cost**: ~$0.05–0.10 (Haiku pricing at ~1K tokens/call × 24 calls)

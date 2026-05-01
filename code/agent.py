"""
agent.py — Triage orchestrator.

Coordinates all pipeline layers:
  Layer 0: Pre-process and normalise ticket
  Layer 1: Rule-based escalation gate (classifier.py)
  Layer 2: Hybrid retrieval (retriever.py)
  Layer 3: Claude API with tool_use structured output
  Layer 4: Pydantic validation + single retry on failure
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import anthropic
from pydantic import BaseModel, field_validator, model_validator

from classifier import detect_adversarial, should_escalate, infer_company, classify_request_type, detect_non_english, detect_multi_request
from retriever import HybridRetriever, Chunk
from utils import (
    MODEL_NAME, MAX_TOKENS, TEMPERATURE, TOP_K, MIN_RETRIEVAL_CONFIDENCE,
    VALID_STATUSES, VALID_REQUEST_TYPES, truncate, normalise_company,
)

# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

class TriageResult(BaseModel):
    status: str
    product_area: str
    response: str
    justification: str
    request_type: str

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in VALID_STATUSES:
            raise ValueError(f"status must be one of {VALID_STATUSES}, got {v!r}")
        return v

    @field_validator("request_type")
    @classmethod
    def validate_request_type(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in VALID_REQUEST_TYPES:
            raise ValueError(f"request_type must be one of {VALID_REQUEST_TYPES}, got {v!r}")
        return v

    def to_dict(self) -> Dict[str, str]:
        return {
            "status": self.status,
            "product_area": self.product_area,
            "response": self.response,
            "justification": self.justification,
            "request_type": self.request_type,
        }


# ---------------------------------------------------------------------------
# Claude tool schema
# ---------------------------------------------------------------------------

_TRIAGE_TOOL: Dict[str, Any] = {
    "name": "submit_triage_decision",
    "description": (
        "Submit the final structured triage decision for a support ticket. "
        "All fields are required."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["replied", "escalated"],
                "description": "Whether the ticket can be resolved ('replied') or must be sent to a human ('escalated').",
            },
            "product_area": {
                "type": "string",
                "enum": [
                    "screen", "community", "privacy", "conversation_management",
                    "travel_support", "general_support", ""
                ],
                "description": (
                    "The product sub-area. Choose ONLY from the enum values. "
                    "HackerRank tests/assessments \u2192 'screen'; HackerRank Community platform \u2192 'community'. "
                    "Claude data privacy, personal data deletion, conversation deletion, GDPR, or any ticket asking to "
                    "remove/delete/access personal data or conversations for privacy reasons \u2192 'privacy'. "
                    "Claude chat sessions, conversation history, or managing ongoing chats (NOT involving privacy/deletion) \u2192 'conversation_management'. "
                    "When in doubt between 'privacy' and 'conversation_management', prefer 'privacy' if the ticket mentions data, deletion, or private information. "
                    "Visa travel-related issues (including traveller's cheques, cards used abroad) \u2192 'travel_support'; "
                    "Visa general card questions or reporting (lost/stolen card, where-to-report) \u2192 'general_support'. "
                    "Use empty string ONLY when Company is Unknown."
                ),
            },
            "response": {
                "type": "string",
                "description": (
                    "Customer-facing reply. Must be multi-paragraph (150-400 words). "
                    "For procedural issues: numbered step-by-step instructions. "
                    "For escalations: professional explanation of next steps. "
                    "Ground every claim in the retrieved corpus context."
                ),
            },
            "justification": {
                "type": "string",
                "description": (
                    "Internal decision rationale (not shown to customer). "
                    "Cite the specific corpus articles/headings used to produce the response. "
                    "Explain why the chosen status and request_type were selected."
                ),
            },
            "request_type": {
                "type": "string",
                "enum": ["product_issue", "feature_request", "bug", "invalid"],
                "description": (
                    "Classification: 'bug' if something is broken, not working, down, inaccessible, or not behaving as expected; "
                    "'feature_request' if they want new or extended functionality; "
                    "'product_issue' if an access, usage, billing, or informational problem; "
                    "'invalid' ONLY for spam, gibberish, clearly out-of-scope questions (e.g. trivia, unrelated topics), "
                    "or clearly malicious content. A 'thank you' or empty-meaning message is also 'invalid'. "
                    "Never use 'invalid' for a legitimate business question, even if complex."
                ),
            },
        },
        "required": ["status", "product_area", "response", "justification", "request_type"],
    },
}

# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """You are a senior customer support specialist for a multi-domain support platform. \
You handle tickets for HackerRank, Claude/Anthropic, and Visa.

Your task is to triage each support ticket and produce a structured decision using the \
`submit_triage_decision` tool. You must fill in all five fields.

**DEFAULT: Set status="replied".** The vast majority of tickets can and should be answered \
directly. Only escalate when a hard condition below is met — do NOT escalate just because \
the context is incomplete or you are slightly uncertain.

**Grounding rule**: Base your response primarily on the provided reference documents. \
If the documents partially cover the topic, use what is available to give a genuinely \
helpful, actionable response. You may use reasonable general product knowledge to fill \
minor gaps — but do not invent specific prices, SLA timelines, or proprietary internal \
procedures that are absent from the documents.

**Citation rule**: Each reference document below is labelled "Document 1", "Document 2", \
etc. When you write the `justification` field, cite the documents you used by number, \
e.g. "[1]", "[2]". This makes the decision auditable.

**When to set status="replied" (the default for most tickets)**:
- How-to questions, feature explanations, UI navigation, troubleshooting steps
- Informational questions about product capabilities, settings, or availability
- Bug reports and outage reports — always reply with troubleshooting steps or a workaround, even if partial
- Feature requests — always reply acknowledging the request and explaining current behavior
- Sales/onboarding inquiries — reply with available information and direct to appropriate next step
- Billing/account questions where general guidance exists in the corpus
- Any request where you can provide substantively useful, actionable information
- **Out-of-scope, nonsensical, or trivial questions** (request_type='invalid') — reply with a polite "this is outside our scope" message. Do NOT escalate these; a human does not need to see them.
- Thank-you messages, greetings, or empty-meaning messages — reply briefly.

**When to set status="escalated" (hard conditions only)**:
- The ticket has been explicitly flagged with an IMPORTANT escalation note in this message
- The request requires a privileged human action that ONLY a human admin can perform: billing reversal, score/grade change, ownership transfer, account unblocking by request of a non-owner
- Security incident, fraud, identity theft, or legal matter requiring a specialist team
- You genuinely cannot provide ANY useful information even after consulting the documents (extremely rare)

**NEVER escalate** just because the issue is complex, the customer sounds frustrated, the tool is down/broken, you can only partially answer, or the request is out of scope — in those cases, still reply with whatever help or "out of scope" message is appropriate.

**Response quality**:
- Responses must be multi-paragraph (at least 150 words), never a single sentence.
- For procedural issues, use numbered steps with specific UI/setting names from the corpus.
- Tone: professional, empathetic, clear. Avoid jargon the customer would not know.
- For escalations, still write a full, helpful response explaining what will happen next.

**Reference documents**:
{context}
"""

# ---------------------------------------------------------------------------
# Escalation response templates (pre-LLM, for adversarial cases)
# ---------------------------------------------------------------------------

_ADVERSARIAL_RESPONSE = (
    "Thank you for reaching out. Unfortunately, your request cannot be processed through "
    "this support channel as it falls outside the scope of permitted support interactions.\n\n"
    "If you believe this is a mistake and you have a legitimate support need, please submit "
    "a new request with a clear description of your issue, and our team will be happy to assist you."
)

_ADVERSARIAL_JUSTIFICATION = (
    "Ticket flagged by pre-LLM adversarial detection (prompt injection or malicious code request). "
    "No LLM call made. Ticket is not actionable through support."
)


# ---------------------------------------------------------------------------
# Query expansion helper
# ---------------------------------------------------------------------------

def _expand_query(issue: str, subject: str) -> str:
    """
    Build an enriched retrieval query by appending salient terms extracted from
    the ticket text.  This improves BM25 keyword recall for tickets whose
    subject line is vague or noisy.

    Strategy:
      1. Start with "<subject>: <first 300 chars of issue>"
      2. Extract capitalised multi-word phrases (product names, proper nouns)
      3. Extract numeric codes and version strings (e.g., "API key", "LTI 1.3")
      4. De-duplicate and append up to 8 extra tokens as a boosted tail
    """
    base = f"{subject}: {issue[:300]}"

    # Capitalised phrases (2+ consecutive title-case words)
    cap_phrases = re.findall(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)\b", issue)
    # Single technical tokens: camelCase, acronyms, numbers attached to letters
    tech_tokens = re.findall(r"\b(?:[A-Z]{2,}|[a-z]+[A-Z][a-z]+|[A-Za-z]+\d[\w.-]*)\b", issue)
    # Domain-specific short proper nouns (e.g., "CodePair", "Haiku", "Sonnet")
    proper_nouns = re.findall(r"\b[A-Z][a-z]{3,}\b", issue)

    extras: list[str] = []
    seen: set[str] = set()
    for term in cap_phrases + tech_tokens + proper_nouns:
        lower = term.lower()
        if lower not in seen and lower not in base.lower() and len(term) > 3:
            extras.append(term)
            seen.add(lower)
        if len(extras) >= 8:
            break

    if extras:
        return f"{base} | {' '.join(extras)}"
    return base


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TriageAgent:
    """
    Orchestrates the full 5-layer triage pipeline for a single ticket.
    Inject a HybridRetriever and an Anthropic client at construction time.
    """

    def __init__(self, retriever: HybridRetriever, client: anthropic.Anthropic):
        self._retriever = retriever
        self._client = client

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def triage(self, row: Dict[str, str]) -> TriageResult:
        """
        Process one support ticket row and return a validated TriageResult.

        The row dict must contain 'issue', 'subject', 'company' keys
        (lowercase, as produced by utils.read_tickets).
        """
        issue = str(row.get("issue", "")).strip()
        subject = str(row.get("subject", "")).strip()
        raw_company = str(row.get("company", "None")).strip()

        # --- Layer 0: Pre-process ---
        company_domain = normalise_company(raw_company)
        if company_domain is None:
            company_domain = infer_company(issue, subject)

        # --- Layer 1a: Adversarial detection ---
        if detect_adversarial(issue):
            request_type = classify_request_type(issue, company_domain)
            return TriageResult(
                status="escalated",
                product_area=self._infer_product_area(company_domain, []),
                response=_ADVERSARIAL_RESPONSE,
                justification=_ADVERSARIAL_JUSTIFICATION,
                request_type=request_type,
            )

        # --- Layer 1b: Rule-based escalation ---
        escalate, reason = should_escalate(issue, company_domain)
        request_type_hint = classify_request_type(issue, company_domain)

        # If company is unknown and the ticket is not simply invalid/out-of-scope,
        # we cannot route or action it — escalate for human triage.
        if not escalate and company_domain is None and request_type_hint != "invalid":
            escalate = True
            reason = "Company unknown and cannot be inferred — unable to route or provide actionable support."

        # --- Non-English and multi-request flags (used in Layer 3 prompt) ---
        is_non_english = detect_non_english(issue)
        multi_requests = detect_multi_request(issue)

        # --- Layer 2: Hybrid retrieval with confidence gate ---
        query = _expand_query(issue, subject)
        scored_chunks = self._retriever.retrieve_with_scores(query, company=company_domain, top_k=TOP_K)
        chunks = [c for c, _ in scored_chunks]
        top_score = scored_chunks[0][1] if scored_chunks else 0.0
        low_confidence = top_score < MIN_RETRIEVAL_CONFIDENCE

        # --- Layer 3 + 4: LLM triage ---
        result = self._call_claude(
            issue=issue,
            subject=subject,
            company_domain=company_domain,
            chunks=chunks,
            force_escalate=escalate,
            escalation_reason=reason,
            request_type_hint=request_type_hint,
            low_confidence=low_confidence,
            is_non_english=is_non_english,
            multi_requests=multi_requests,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_context(self, chunks: List[Chunk]) -> str:
        """Format retrieved chunks for inclusion in the system prompt."""
        if not chunks:
            return "(No relevant documentation found in the corpus.)"
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(f"--- Document {i} ---\n{chunk.as_context()}")
        return "\n\n".join(parts)

    def _infer_product_area(self, company_domain: str | None, chunks: List[Chunk]) -> str:
        """
        Infer a reasonable product_area from retrieved chunk paths
        when a rule-based escalation fires before the LLM call.
        Normalizes hyphens to underscores to match canonical taxonomy.
        """
        if chunks:
            # Use the top chunk's sub-directory as product_area hint
            top_chunk = chunks[0]
            parts = top_chunk.source_file.replace("\\", "/").split("/")
            # parts[0]=domain, parts[1]=sub-area (if present)
            if len(parts) >= 3:
                return parts[1].replace("-", "_")  # e.g. "screen", "account_management"
            if len(parts) == 2:
                return parts[1].replace(".md", "").replace("-", "_")
        # Fallback: no chunks — use a generic area for the domain
        if company_domain == "visa":
            return "general_support"
        if company_domain in ("hackerrank", "claude"):
            return "general_help"
        return ""

    def _call_claude(
        self,
        issue: str,
        subject: str,
        company_domain: str | None,
        chunks: List[Chunk],
        force_escalate: bool,
        escalation_reason: str,
        request_type_hint: str,
        low_confidence: bool = False,
        is_non_english: bool = False,
        multi_requests: list[str] | None = None,
    ) -> TriageResult:
        """
        Build the prompt, call Claude with tool_use, validate the result.
        Retries once if the first response fails validation.
        """
        context = self._build_context(chunks)
        system = _SYSTEM_TEMPLATE.format(context=context)

        # User message
        escalation_note = ""
        if force_escalate:
            escalation_note = (
                f"\n\n**IMPORTANT**: This ticket MUST be escalated. Reason: {escalation_reason} "
                f"Set status='escalated' in your decision."
            )

        confidence_note = ""
        if low_confidence:
            confidence_note = (
                "\n\n**⚠ Retrieval confidence is low**: The corpus documents above may not "
                "directly cover this ticket's topic. Use them as background context only and "
                "rely on general product knowledge where documents are insufficient. If no "
                "useful guidance can be given, escalate."
            )

        lang_note = ""
        if is_non_english:
            lang_note = (
                "\n\n**Note**: This ticket appears to be written in a non-English language. "
                "Please respond in the same language as the ticket if possible, or in English "
                "if you cannot reliably detect the language."
            )

        multi_note = ""
        if multi_requests:
            multi_note = (
                f"\n\n**Note**: This ticket contains {len(multi_requests)} distinct requests. "
                "Address the highest-priority request in the `response` field and briefly "
                "acknowledge the others. Note all requests in the `justification` field."
            )

        user_message = (
            f"Please triage the following support ticket.\n\n"
            f"**Company**: {company_domain or 'Unknown'}\n"
            f"**Subject**: {subject}\n"
            f"**Issue**:\n{issue}"
            f"{escalation_note}"
            f"{confidence_note}"
            f"{lang_note}"
            f"{multi_note}\n\n"
            f"Initial classification hint (you may override): request_type={request_type_hint!r}\n\n"
            "Call the `submit_triage_decision` tool with your decision."
        )

        for attempt in range(2):
            try:
                response = self._client.messages.create(
                    model=MODEL_NAME,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    system=system,
                    tools=[_TRIAGE_TOOL],
                    tool_choice={"type": "any"},
                    messages=[{"role": "user", "content": user_message}],
                )
                result = self._parse_response(response)
                if result:
                    # If company is unknown, product_area cannot be determined
                    if company_domain is None:
                        if result.product_area != "":
                            result = TriageResult(
                                status=result.status,
                                product_area="",
                                response=result.response,
                                justification=result.justification,
                                request_type=result.request_type,
                            )
                    # Fill blank product_area from chunks only when company is known
                    elif not result.product_area.strip():
                        inferred = self._infer_product_area(company_domain, chunks)
                        if inferred:
                            result = TriageResult(
                                status=result.status,
                                product_area=inferred,
                                response=result.response,
                                justification=result.justification,
                                request_type=result.request_type,
                            )
                    # Post-process: if force_escalate but LLM returned 'replied', override
                    if force_escalate and result.status == "replied":
                        result = TriageResult(
                            status="escalated",
                            product_area=result.product_area,
                            response=result.response,
                            justification=f"{result.justification} [Overridden to escalated: {escalation_reason}]",
                            request_type=result.request_type,
                        )
                    return result
            except anthropic.APIError as e:
                if attempt == 0:
                    # Retry once on API error
                    continue
                # Second failure: return safe escalation
                return self._safe_fallback(company_domain, chunks, str(e))

        return self._safe_fallback(company_domain, chunks, "LLM response parsing failed after 2 attempts")

    def _parse_response(self, response: anthropic.types.Message) -> TriageResult | None:
        """Extract and validate the tool_use block from Claude's response."""
        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_triage_decision":
                raw: Dict[str, Any] = block.input
                try:
                    return TriageResult(**raw)
                except Exception:
                    return None
        return None

    def _safe_fallback(
        self,
        company_domain: str | None,
        chunks: List[Chunk],
        reason: str,
    ) -> TriageResult:
        """Last-resort escalation when LLM calls fail completely."""
        return TriageResult(
            status="escalated",
            product_area=self._infer_product_area(company_domain, chunks),
            response=(
                "We apologise for any inconvenience. Your ticket has been escalated to our "
                "support team who will review it and get back to you as soon as possible. "
                "Please expect a response within 1–2 business days."
            ),
            justification=f"Safe fallback triggered: {reason}",
            request_type="product_issue",
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from utils import load_env
    from retriever import build_retriever

    api_key = load_env()
    client = anthropic.Anthropic(api_key=api_key)
    retriever = build_retriever()

    agent = TriageAgent(retriever=retriever, client=client)

    test_ticket = {
        "issue": "My mock interviews stopped in between, please give me the refund asap",
        "subject": "Why are my mock interviews not working",
        "company": "HackerRank",
    }
    result = agent.triage(test_ticket)
    import json as _json
    print(_json.dumps(result.to_dict(), indent=2))

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
from typing import Any, Dict

import anthropic
from pydantic import BaseModel, field_validator, model_validator

from classifier import detect_adversarial, should_escalate, infer_company, classify_request_type
from retriever import HybridRetriever, Chunk
from utils import (
    MODEL_NAME, MAX_TOKENS, TEMPERATURE, TOP_K,
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
                "description": (
                    "The specific product sub-area the ticket relates to "
                    "(e.g., 'screen', 'interviews', 'billing', 'account-management', 'privacy-and-legal'). "
                    "Use the corpus source paths as a guide. May be empty string only if the domain is truly unknown."
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
                    "Classification: 'bug' if something is broken or not working as expected; "
                    "'feature_request' if they want new or extended functionality; "
                    "'product_issue' if an access, usage, billing, or informational problem; "
                    "'invalid' ONLY for spam, gibberish, or clearly malicious content "
                    "(asking to run harmful code, reveal system internals, etc.). "
                    "Never use 'invalid' for a legitimate business question, even if complex or outside normal scope."
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

**When to set status="replied" (the default for most tickets)**:
- How-to questions, feature explanations, UI navigation, troubleshooting steps
- Informational questions about product capabilities, settings, or availability
- Bug reports and outage reports — always reply with troubleshooting steps or a workaround, even if partial
- Feature requests — always reply acknowledging the request and explaining current behavior
- Sales/onboarding inquiries — reply with available information and direct to appropriate next step
- Billing/account questions where general guidance exists in the corpus
- Any request where you can provide substantively useful, actionable information

**When to set status="escalated" (hard conditions only)**:
- The ticket has been explicitly flagged with an IMPORTANT escalation note in this message
- The request requires a privileged human action that ONLY a human admin can perform: billing reversal, score/grade change, ownership transfer, account unblocking by request of a non-owner
- Security incident, fraud, identity theft, or legal matter requiring a specialist team
- You genuinely cannot provide ANY useful information even after consulting the documents (extremely rare)

**NEVER escalate** just because the issue is complex, the customer sounds frustrated, the tool is down/broken, or you can only partially answer — in those cases, still reply with whatever help you can provide.

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

        # --- Layer 2: Hybrid retrieval ---
        query = f"{subject}: {issue[:300]}"
        chunks = self._retriever.retrieve(query, company=company_domain, top_k=TOP_K)

        # --- Layer 3 + 4: LLM triage ---
        result = self._call_claude(
            issue=issue,
            subject=subject,
            company_domain=company_domain,
            chunks=chunks,
            force_escalate=escalate,
            escalation_reason=reason,
            request_type_hint=request_type_hint,
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
        """
        if not chunks:
            return ""
        # Use the top chunk's sub-directory as product_area hint
        top_chunk = chunks[0]
        parts = top_chunk.source_file.replace("\\", "/").split("/")
        # parts[0]=domain, parts[1]=sub-area (if present)
        if len(parts) >= 3:
            return parts[1]  # e.g. "screen", "interviews", "amazon-bedrock"
        if len(parts) == 2:
            return parts[1].replace(".md", "")
        return company_domain or ""

    def _call_claude(
        self,
        issue: str,
        subject: str,
        company_domain: str | None,
        chunks: List[Chunk],
        force_escalate: bool,
        escalation_reason: str,
        request_type_hint: str,
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

        user_message = (
            f"Please triage the following support ticket.\n\n"
            f"**Company**: {company_domain or 'Unknown'}\n"
            f"**Subject**: {subject}\n"
            f"**Issue**:\n{issue}"
            f"{escalation_note}\n\n"
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
                    # Post-process: if force_escalate but LLM returned 'replied', override
                    if force_escalate and result.status == "replied":
                        result = TriageResult(
                            status="escalated",
                            product_area=result.product_area or self._infer_product_area(company_domain, chunks),
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

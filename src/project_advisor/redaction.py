"""Sensitive-data trace redaction for the Meridian Project Delivery Advisor capital-project assistant.

Rule-based masking (docs: langsmith/mask-inputs-outputs) applied at the storage
boundary so no raw customer or partner PII is persisted to LangSmith. Sensitive
identifiers (contact SSN, business tax id/EIN, phone, email, client project id) are
masked; catalog and record ids (CP-/SP-/PP-/EL-/HX-/TD-/PN-/SC-/SR-/TQ-/VND-) are deliberately
left intact - they are non-PII and needed for the demo.

`make_redacting_tracer()` returns a LangChainTracer backed by a Client with the
anonymizer attached. The graph attaches it as a callback (graph.py); the load
generator uses it as the *sole* tracer so the bulk demo traces are guaranteed
single and masked regardless of the dev server's own tracer.
"""

from __future__ import annotations

from langchain_core.tracers.langchain import LangChainTracer
from langsmith import AgentAddress, Client
from langsmith.anonymizer import create_anonymizer

REDACTION_RULES = [
    {"pattern": r"\b\d{3}-\d{2}-\d{4}\b", "replace": "<ssn>"},                        # contact SSN
    {"pattern": r"\b\d{2}-\d{7}\b", "replace": "<tin>"},                              # business tax id / EIN
    {"pattern": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "replace": "<email>"},
    {"pattern": r"(?:\(\d{3}\)|\b\d{3})[\s.\-]?\d{3}[\s.\-]?\d{4}\b", "replace": "<phone>"},
    {"pattern": r"\bPRJ-\d{6,}\b", "replace": "<project-id>"},                    # synthetic client project id
]


def make_redacting_tracer(address: AgentAddress | None = None) -> LangChainTracer | None:
    """Return a LangChainTracer whose Client masks sensitive data before storage.

    With no ``address`` the tracer follows LANGSMITH_AGENT_ID and
    LANGSMITH_AGENT_ENVIRONMENT (agent addressing needs langchain-core>=1.6.11).
    Pass an address to send one process to another environment, for example
    ``local`` for load tests.

    Returns None if a client can't be built (e.g. no API key) so callers can
    fail open.
    """
    try:
        client = Client(anonymizer=create_anonymizer(REDACTION_RULES))
        return LangChainTracer(client=client, address=address)
    except Exception:
        return None

"""Context Hub guardrail middleware for the synthetic capital-project delivery demo.
The seeded classifier blocks injection only; approval, sign-off and equipment
requests slip through intentionally. Strengthen guardrail.md and promote production to fix it.
The tools remain read-only. Trace redaction is an independent storage control.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Literal

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from project_advisor.llm import guardrail_model, openai_gateway_kwargs

# The guardrail classifier prompt lives in the Context Hub agent repo as this file,
# alongside AGENTS.md and the skills - so all three fixable surfaces are Context Hub
# files (edit + re-promote :production to ship).
GUARDRAIL_FILE = "guardrail.md"
# Guardrail classifier model, routed through the gateway. Default
# anthropic/claude-haiku-5-5; compare models via the GUARDRAIL_MODEL env var.
_GUARDRAIL_MODEL = guardrail_model()

# The guardrail prompt text is NOT in this code - it lives in Context Hub as
# guardrail.md (seeded from context_hub_seed/guardrail.md). The seed is
# intentionally WEAK (injection-only, silent on approvals, sign-offs and equipment actions);
# strengthening it is the demo fix - edit guardrail.md and re-promote :production.

_REFUSAL = (
    "I can't take that action. I can review package evidence, explain test deviations and "
    "draft a simulated technical query. I cannot accept tests, waive hold points, approve "
    "change orders, sign certificates or permits, or operate equipment."
)


class GuardrailDecision(BaseModel):
    decision: Literal["ALLOWED", "BLOCKED"]
    reason: str = Field(description="One sentence explaining the decision.")


# Strict json_schema response format - the model is constrained to emit exactly this
# shape, so parsing can't drift. Hand-written (not model_json_schema()) to guarantee
# additionalProperties:false + all-required, which OpenAI strict mode demands.
_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "guardrail_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": ["ALLOWED", "BLOCKED"]},
                "reason": {"type": "string"},
            },
            "required": ["decision", "reason"],
            "additionalProperties": False,
        },
    },
}


def get_guardrail_prompt() -> str:
    """Pull the guardrail classifier prompt from LangSmith Context Hub (guardrail.md).

    Read from the same agent repo as AGENTS.md/skills, at the CONTEXT_HUB_VERSION
    commit (``production`` by default). There is **no local fallback**: the hub is
    the runtime source of truth, so this raises if the file is missing/unreachable.
    Seed it first with ``python -m scripts.setup_context_hub`` (the seed text lives
    in ``context_hub_seed/guardrail.md``, never in code, and is never read at runtime).

    SECURITY: ``pull_agent`` reads markdown files only - it does not deserialize a
    model/manifest, so the prompt text is untrusted-but-inert.
    """
    from langsmith import Client

    from project_advisor.context import CONTEXT_HUB_REPO, CONTEXT_HUB_VERSION

    version = CONTEXT_HUB_VERSION or None
    try:
        agent = Client().pull_agent(CONTEXT_HUB_REPO, version=version)
        entry = agent.files.get(GUARDRAIL_FILE) if hasattr(agent.files, "get") else agent.files[GUARDRAIL_FILE]
        content = getattr(entry, "content", "") or ""
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not read '{GUARDRAIL_FILE}' from Context Hub repo "
            f"'{CONTEXT_HUB_REPO}' (version='{version or 'latest'}'): {exc}. "
            f"Seed it with 'python -m scripts.setup_context_hub'."
        ) from exc
    if not content:
        raise RuntimeError(
            f"Context Hub repo '{CONTEXT_HUB_REPO}' has no '{GUARDRAIL_FILE}' content "
            f"at version='{version or 'latest'}'. Seed it with "
            f"'python -m scripts.setup_context_hub'."
        )
    return content


# Short TTL cache (not a permanent cache) so a Context Hub edit + re-promote shows
# up on a RUNNING server within GUARDRAIL_PROMPT_TTL seconds - no restart. Pulled
# lazily on first use (not at import), so the module imports before the hub is seeded.
_GUARDRAIL_PROMPT_TTL = float(os.getenv("GUARDRAIL_PROMPT_TTL", "15"))
_prompt_cache: dict[str, Any] = {"text": None, "at": 0.0}


def _guardrail_prompt() -> str:
    now = time.monotonic()
    if _prompt_cache["text"] is None or (now - _prompt_cache["at"]) > _GUARDRAIL_PROMPT_TTL:
        _prompt_cache["text"] = get_guardrail_prompt()
        _prompt_cache["at"] = now
    return _prompt_cache["text"]


def _make_classifier() -> ChatOpenAI:
    # Routed through the LangSmith LLM Gateway (openai_gateway_kwargs). Strict
    # json_schema response format on a single NON-streaming call. Non-streaming is
    # deliberate: under langgraph's messages stream a streaming classifier would emit
    # its raw {"decision",...} JSON to the chat UI as a message. One clean ChatOpenAI
    # run; the decision routes internally.
    return ChatOpenAI(
        model=_GUARDRAIL_MODEL,
        model_kwargs={"response_format": _RESPONSE_FORMAT},
        **openai_gateway_kwargs(),
    )


def _messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _guardrail_prompt()},
        {"role": "user", "content": ""},
    ]


def _parse(resp: Any) -> GuardrailDecision:
    content = resp.content if isinstance(resp.content, str) else " ".join(
        b.get("text", "") for b in resp.content if isinstance(b, dict) and b.get("type") == "text"
    )
    return GuardrailDecision(**json.loads(content))


# Tag the classifier run "nostream" so langgraph's messages stream EXCLUDES it
# (otherwise its raw {"decision",...} JSON would render as a chat bubble). The run
# is still traced - it stays a visible single nano LLM run in LangSmith. ("guardrails"
# is a human-readable label, mirroring chat-langchain's guardrails middleware.)
_CLASSIFIER_CONFIG = {"tags": ["nostream", "guardrails"]}


def _classify(text: str) -> GuardrailDecision:
    """Run the strict-JSON classifier and validate it into a GuardrailDecision (sync)."""
    msgs = _messages()
    msgs[1]["content"] = text
    return _parse(_make_classifier().invoke(msgs, config=_CLASSIFIER_CONFIG))


async def _aclassify(text: str) -> GuardrailDecision:
    """Async classify - used by the served/async path so it never blocks the event loop."""
    # Context Hub SDK uses synchronous HTTP; keep its cached prompt read off the
    # ASGI event loop. Model inference itself uses the async gateway client.
    import asyncio
    msgs = await asyncio.to_thread(_messages)
    msgs[1]["content"] = text
    return _parse(await _make_classifier().ainvoke(msgs, config=_CLASSIFIER_CONFIG))


def _last_user_text(state: Any) -> str:
    messages = (state or {}).get("messages") if isinstance(state, dict) else getattr(state, "messages", None)
    for msg in reversed(messages or []):
        if isinstance(msg, HumanMessage):
            content = msg.content
        elif isinstance(msg, dict) and msg.get("role") in ("user", "human"):
            content = msg.get("content")
        else:
            continue
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _record(decision: GuardrailDecision) -> None:
    try:
        from langsmith import get_current_run_tree

        run = get_current_run_tree()
        if run is not None:
            run.metadata["guardrail_decision"] = decision.decision
            run.metadata["guardrail_reason"] = decision.reason
    except Exception:
        pass


class GuardrailsMiddleware(AgentMiddleware):
    """Classify each request and block approval/sign-off actions and injection (weak by seed)."""

    def _result(self, decision: GuardrailDecision) -> dict[str, Any] | None:
        _record(decision)
        if decision.decision == "BLOCKED":
            return {"jump_to": "end", "messages": [AIMessage(content=_REFUSAL)]}
        return None

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state, runtime) -> dict[str, Any] | None:
        text = _last_user_text(state)
        if not text.strip():
            return None
        try:
            decision = _classify(text)
        except Exception:
            return None  # fail open
        return self._result(decision)

    @hook_config(can_jump_to=["end"])
    async def abefore_agent(self, state, runtime) -> dict[str, Any] | None:
        # Async path (langgraph server / aevaluate): await the classifier so the
        # ASGI event loop is never blocked by a sync OpenAI call.
        text = _last_user_text(state)
        if not text.strip():
            return None
        try:
            decision = await _aclassify(text)
        except Exception:
            return None  # fail open
        return self._result(decision)

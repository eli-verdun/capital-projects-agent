"""Shared LLM wiring: route every model call through the LangSmith LLM Gateway.

Every `ChatOpenAI` in this project spreads `openai_gateway_kwargs()`, so requests go
to the gateway's provider-routed Chat Completions endpoint (`<gateway>/v1`). The model
name carries the provider, for example `anthropic/claude-sonnet-5-5`, or names a saved
model configuration, for example `custom/claude-haiku-5-5`. The gateway translates the request, resolves the provider key from
the workspace Provider Secrets (for example `ANTHROPIC_API_KEY`), enforces policies
and traces the call. No local provider key is needed.

Credentials:
  - `LANGSMITH_GATEWAY_API_KEY` authenticates gateway requests. Any workspace-scoped
    LangSmith key from the same region works.
  - `LANGSMITH_API_KEY` is used for LangSmith tracing (read by the SDK).

The gateway host follows the LANGSMITH_ENDPOINT region (see endpoints.py).
See https://docs.langchain.com/langsmith/llm-gateway-api-formats
"""

from __future__ import annotations

import os

from project_advisor.endpoints import gateway_url as _gateway_base_url

# Defaults for the three model roles. Override with AGENT_MODEL, GUARDRAIL_MODEL and
# EVAL_JUDGE_MODEL to compare models across experiments.
# ``custom/<name>`` is a saved model configuration (Settings > Model configurations).
DEFAULT_AGENT_MODEL = "custom/claude-haiku-5-5"
DEFAULT_GUARDRAIL_MODEL = "custom/claude-haiku-5-5"
DEFAULT_JUDGE_MODEL = "custom/claude-haiku-5-5"


def _gateway_model(value: str) -> str:
    """The gateway's /v1 endpoint needs <provider>/<model>; a bare name means OpenAI."""
    return value if "/" in value else f"openai/{value}"


def agent_model() -> str:
    return _gateway_model(os.getenv("AGENT_MODEL", "").strip() or DEFAULT_AGENT_MODEL)


def guardrail_model() -> str:
    return _gateway_model(os.getenv("GUARDRAIL_MODEL", "").strip() or DEFAULT_GUARDRAIL_MODEL)


def judge_model() -> str:
    return _gateway_model(os.getenv("EVAL_JUDGE_MODEL", "").strip() or DEFAULT_JUDGE_MODEL)


# Output budget sent on every gateway request. A saved model configuration can carry
# its own max_tokens (often 1024), which truncates the synthesis mid-answer; the request
# value overrides it. Override with LLM_MAX_TOKENS.
DEFAULT_MAX_TOKENS = 4096


def max_tokens() -> int:
    return int(os.getenv("LLM_MAX_TOKENS", "").strip() or DEFAULT_MAX_TOKENS)


def split_provider(model: str) -> tuple[str, str]:
    """Split a gateway model id into (provider, model). A bare name means OpenAI."""
    provider, sep, name = model.partition("/")
    return (provider, name) if sep else ("openai", model)


def openai_gateway_kwargs() -> dict[str, str | int]:
    """Constructor kwargs that point langchain-openai at the LLM Gateway.

    Returns ``{"base_url": "<gateway>/v1", "api_key": <LANGSMITH_GATEWAY_API_KEY>,
    "max_tokens": <LLM_MAX_TOKENS>}``.
    Falls back to ``{}`` (direct OpenAI via `OPENAI_API_KEY`) when
    `LANGSMITH_GATEWAY_API_KEY` is unset.
    """
    api_key = os.getenv("LANGSMITH_GATEWAY_API_KEY")
    if not api_key:
        return {}
    return {"base_url": f"{_gateway_base_url()}/v1", "api_key": api_key, "max_tokens": max_tokens()}

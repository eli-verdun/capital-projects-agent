"""Shared LLM wiring: route OpenAI SDK calls through the LangSmith LLM Gateway.

Instead of calling OpenAI directly with an `OPENAI_API_KEY`, every `ChatOpenAI`
and `OpenAIEmbeddings` constructor in this project spreads `openai_gateway_kwargs()`
so requests are proxied through the LangSmith LLM Gateway. The gateway authenticates
with `LANGSMITH_GATEWAY_API_KEY`, resolves the real provider key from workspace
Provider Secrets, enforces spend/PII/secrets policies, and traces the call - so we
never need a local OpenAI key.

Two distinct credentials are used:
  - `LANGSMITH_GATEWAY_API_KEY` authenticates gateway (proxy) requests.
  - `LANGSMITH_API_KEY` is used only for LangSmith tracing (read by the SDK).

See https://docs.langchain.com/langsmith/llm-gateway-quickstart
"""

from __future__ import annotations

import os

# The gateway host follows the LANGSMITH_ENDPOINT region (apac. / eu. / US); see
# endpoints.py. LLM_GATEWAY_BASE_URL overrides it for a self-hosted gateway.
from project_advisor.endpoints import gateway_url as _gateway_base_url


def openai_gateway_kwargs() -> dict[str, str]:
    """Constructor kwargs that point langchain-openai at the LLM Gateway.

    Returns ``{"base_url": ".../openai/v1", "api_key": <LANGSMITH_GATEWAY_API_KEY>}``
    so `ChatOpenAI` / `OpenAIEmbeddings` proxy through the gateway. Falls back to ``{}``
    (direct OpenAI via `OPENAI_API_KEY`) when `LANGSMITH_GATEWAY_API_KEY` is unset, so
    local experiments without gateway access still work.

    This is separate from `LANGSMITH_API_KEY`, which is used only for tracing.
    """
    api_key = os.getenv("LANGSMITH_GATEWAY_API_KEY")
    if not api_key:
        return {}
    return {"base_url": f"{_gateway_base_url()}/openai/v1", "api_key": api_key}

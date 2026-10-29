"""LangSmith region endpoints, derived from LANGSMITH_ENDPOINT.

The SDK reads LANGSMITH_ENDPOINT itself. The raw REST calls in the setup scripts,
the experiment UI links and the LLM Gateway host must follow the same region, or a
non-US workspace answers 403. Set one variable for the region:

    LANGSMITH_ENDPOINT=https://apac.api.smith.langchain.com   # or eu., or unset for US

Region hosts (https://docs.langchain.com/langsmith/cloud):

    API      https://<region>.api.smith.langchain.com
    UI       https://<region>.smith.langchain.com
    Gateway  https://<region>.gateway.smith.langchain.com

Override the derived hosts with LANGSMITH_UI_URL or LLM_GATEWAY_BASE_URL only for a
self-hosted or custom installation. These are trusted, developer-controlled values
from .env, not request input.
"""

from __future__ import annotations

import os
import re

_US_API = "https://api.smith.langchain.com"
# https://<region>.api.smith.langchain.com, where <region> is empty for US.
_CLOUD_API = re.compile(r"^https://(?:(?P<region>[a-z0-9-]+)\.)?api\.smith\.langchain\.com$")


def api_url() -> str:
    return (os.getenv("LANGSMITH_ENDPOINT", "").strip() or _US_API).rstrip("/")


def rest_api() -> str:
    """Base URL for raw REST calls, e.g. ``f"{rest_api()}/runs/rules"``."""
    return f"{api_url()}/api/v1"


def _region() -> str | None:
    """Cloud region prefix ('' for US), or None for a self-hosted endpoint."""
    match = _CLOUD_API.match(api_url())
    return (match.group("region") or "") if match else None


def ui_url() -> str:
    if override := os.getenv("LANGSMITH_UI_URL", "").strip():
        return override.rstrip("/")
    region = _region()
    if region is None:
        # Self-hosted: the UI and API usually share a host, with the API under /api.
        return re.sub(r"/api/?$", "", api_url())
    return f"https://{region + '.' if region else ''}smith.langchain.com"


def gateway_url() -> str:
    if override := os.getenv("LLM_GATEWAY_BASE_URL", "").strip():
        return override.rstrip("/")
    region = _region() or ""
    return f"https://{region + '.' if region else ''}gateway.smith.langchain.com"

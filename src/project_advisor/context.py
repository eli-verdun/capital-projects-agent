"""Agent context - pulled from LangSmith Context Hub at runtime.

The agent's system prompt (``AGENTS.md``) lives in the LangSmith Context
Hub, NOT in this repo. By default the runtime pulls the commit promoted to the
``production`` environment tag (promote in the Context Hub UI, or via
``scripts/setup_context_hub.py``), so promoting a fix to Production is what
ships it - no code redeploy. Override the tag/commit with the
``CONTEXT_HUB_VERSION`` env var (e.g. ``staging``, a commit hash, or an empty
string to pull the latest commit).

Look here to find where the agent's context comes from - but the prompt itself
is edited in the Context Hub UI, not in this repo. The seed text lives as
markdown under ``context_hub_seed/`` (data, not code) and is only pushed to the
hub by ``setup_context_hub``; it is never read at runtime.
"""

from __future__ import annotations

import os

from langsmith import Client

CONTEXT_HUB_REPO = "capital-projects-agent"

# Environment tag (or commit hash) the runtime resolves. Defaults to the
# promoted ``production`` commit; set CONTEXT_HUB_VERSION="" to pull the latest.
CONTEXT_HUB_VERSION = os.getenv("CONTEXT_HUB_VERSION", "production")


def get_prompt() -> str:
    """Return the agent's system prompt, pulled from LangSmith Context Hub.

    Resolves the commit tagged ``CONTEXT_HUB_VERSION`` (``production`` by
    default). There is **no local fallback**: the runtime source of truth is the
    Context Hub, so if the hub is unreachable or ``AGENTS.md`` is missing this
    raises. Seed/promote the hub first with ``python -m scripts.setup_context_hub``
    (the seed text lives in ``context_hub_seed/AGENTS.md``, never in code).
    """
    version = CONTEXT_HUB_VERSION or None
    try:
        agent = Client().pull_agent(CONTEXT_HUB_REPO, version=version)
        content = getattr(agent.files["AGENTS.md"], "content", "")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not pull AGENTS.md from Context Hub repo '{CONTEXT_HUB_REPO}' "
            f"(version='{version or 'latest'}'): {exc}. Seed it with "
            f"'python -m scripts.setup_context_hub'."
        ) from exc
    if not content:
        raise RuntimeError(
            f"Context Hub repo '{CONTEXT_HUB_REPO}' has no AGENTS.md content at "
            f"version='{version or 'latest'}'. Seed it with "
            f"'python -m scripts.setup_context_hub'."
        )
    return content

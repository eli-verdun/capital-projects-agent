"""LangSmith Context Hub setup-time helper for the Meridian Project Delivery Advisor demo.

`seed_agent_repo()` pushes ALL three fixable surfaces into ONE Context Hub agent
repo and promotes it to `production`:
  - `AGENTS.md`              - the system prompt
  - `guardrail.md`           - the intentionally-WEAK guardrail classifier prompt
  - `skills/<name>/SKILL.md`  - the skills (one intentionally WRONG)

The prompt content is NOT embedded in this code - it lives as markdown files under
``context_hub_seed/`` (data, not code). This module just reads those files and
pushes them. At runtime the agent reads AGENTS.md + skills via its Context-Hub-backed
filesystem and the guardrail reads guardrail.md from the same repo - so every fix is
"edit a file in Context Hub, re-promote :production". No Prompt Hub, no local prompts
in code.

The raw-REST calls set a workspace handle + `source=internal` so the repo shows
in the Context Hub UI listing (the SDK doesn't expose those yet).
"""

from __future__ import annotations

import os
from pathlib import Path

import requests
from langsmith import Client
from langsmith.schemas import FileEntry

from project_advisor.context import CONTEXT_HUB_REPO

from project_advisor.endpoints import rest_api

_API = rest_api()

# Seed prompts live here as markdown (bootstrap data, never read at runtime).
_SEED_DIR = Path(__file__).resolve().parent.parent.parent / "context_hub_seed"


def _rest_headers() -> dict:
    headers = {"x-api-key": os.getenv("LANGSMITH_API_KEY"), "Content-Type": "application/json"}
    if ws := os.getenv("LANGSMITH_WORKSPACE_ID", "").strip():
        headers["X-Tenant-Id"] = ws
    return headers


def _agent_repo_files() -> dict:
    """Every *.md under context_hub_seed/, keyed by its repo-relative path."""
    if not _SEED_DIR.is_dir():
        raise SystemExit(f"Seed dir not found: {_SEED_DIR}")
    files = {}
    for p in sorted(_SEED_DIR.rglob("*.md")):
        rel = p.relative_to(_SEED_DIR).as_posix()  # AGENTS.md, guardrail.md, skills/<n>/SKILL.md
        files[rel] = FileEntry(content=p.read_text(encoding="utf-8"))
    return files


def _promote_to_production(client: Client) -> None:
    """Point the ``production`` tag at the latest commit, MOVING it if it already exists."""
    try:
        agent = client.pull_agent(CONTEXT_HUB_REPO)
        try:
            client._create_commit_tags(f"-/{CONTEXT_HUB_REPO}", str(agent.commit_id), "production")
        except Exception:
            # Tag exists on an older commit; delete it then re-create on this commit.
            requests.delete(f"{_API}/repos/-/{CONTEXT_HUB_REPO}/tags/production", headers=_rest_headers())
            client._create_commit_tags(f"-/{CONTEXT_HUB_REPO}", str(agent.commit_id), "production")
        print(f"  Promoted commit {agent.commit_hash[:8]} to 'production'.")
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: could not promote to 'production' ({exc}).")


def seed_agent_repo() -> None:
    """Seed the Context Hub agent repo from context_hub_seed/, then promote to production."""
    files = _agent_repo_files()
    print(f"\n[*] Seeding Context Hub repo '{CONTEXT_HUB_REPO}' with {len(files)} files...")
    headers = _rest_headers()

    settings = requests.get(f"{_API}/settings", headers=headers).json()
    if not settings.get("tenant_handle"):
        requests.post(f"{_API}/settings/handle", headers=headers, json={"tenant_handle": "capital-projects-agent"})
        print("  Set workspace tenant_handle to 'capital-projects-agent'.")

    requests.post(
        f"{_API}/repos/",
        headers=headers,
        json={
            "repo_handle": CONTEXT_HUB_REPO,
            "repo_type": "agent",
            "source": "internal",
            "is_public": False,
            "description": (
                "Meridian Project Delivery Advisor capital-project assistant (AGENTS.md + guardrail.md + skills; "
                "buggy, for the Insights demo)"
            ),
        },
    )

    client = Client()
    client.push_agent(CONTEXT_HUB_REPO, files=files)
    print(f"  Pushed: {', '.join(sorted(files))}.")
    _promote_to_production(client)

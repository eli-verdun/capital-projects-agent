"""Associate LangSmith resources with the registered agent.

An agent-based workspace has no per-resource "attach to agent" endpoint. Registering
the agent creates the system tag key ``Agent`` with a value equal to the agent id, and
tagging a resource with that value associates it with the agent. This is what the UI
does when you attach a dataset. See
https://docs.langchain.com/langsmith/administration-overview#agents-and-applications
"""

from __future__ import annotations

import os

import requests

from project_advisor.agent_address import agent_id
from project_advisor.endpoints import rest_api
from project_advisor.governance import AGENT_TAG_KEY


def headers() -> dict:
    h = {"x-api-key": os.environ["LANGSMITH_API_KEY"], "Content-Type": "application/json"}
    if ws := os.getenv("LANGSMITH_WORKSPACE_ID", "").strip():
        h["X-Tenant-Id"] = ws
    return h


def tag_keys(h: dict) -> list[dict]:
    resp = requests.get(f"{rest_api()}/workspaces/current/tags", headers=h)
    resp.raise_for_status()
    body = resp.json()
    return body.get("items", body) if isinstance(body, dict) else body


def agent_tag_value_id(h: dict) -> str | None:
    """The id of the ``Agent`` tag value for this agent, or None outside an agent-based workspace.

    Registering the agent creates the value; creating it here is only a fallback.
    """
    key = next((k for k in tag_keys(h) if k.get("key") == AGENT_TAG_KEY), None)
    if key is None:
        return None
    for value in key.get("values") or []:
        if value.get("value") == agent_id():
            return value["id"]
    resp = requests.post(
        f"{rest_api()}/workspaces/current/tag-keys/{key['id']}/tag-values",
        headers=h,
        json={"value": agent_id()},
    )
    return resp.json()["id"] if resp.status_code in (200, 201) else None


def associate(h: dict, tag_value_id: str, resource_type: str, resource_id: str) -> str:
    """Tag one resource with the agent value. Returns 'associated', 'already' or an error."""
    resp = requests.post(
        f"{rest_api()}/workspaces/current/taggings",
        headers=h,
        json={"tag_value_id": tag_value_id, "resource_type": resource_type, "resource_id": resource_id},
    )
    if resp.status_code in (200, 201):
        return "associated"
    if resp.status_code == 409 or "exist" in resp.text.lower():
        return "already"
    return f"error {resp.status_code}: {resp.text[:150]}"


def associate_with_agent(resource_type: str, resource_id: str) -> str:
    """One-call helper for scripts: associate a resource, or explain why not."""
    h = headers()
    tag_value_id = agent_tag_value_id(h)
    if tag_value_id is None:
        return f"skipped (no '{AGENT_TAG_KEY}' tag for '{agent_id()}'; run scripts.setup_agent)"
    return associate(h, tag_value_id, resource_type, resource_id)

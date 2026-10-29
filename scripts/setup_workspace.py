"""Provision the workspace grouping + feedback-review queue for the demo.

Run after the agent is registered (``python -m scripts.setup_agent``) and after the
datasets are restored. It:

  1. Ensures the "Meridian Project Delivery Advisor: Negative Feedback Review" annotation queue exists.
  2. Tags the agent environment's backing project, the four datasets, the queue, and
     the Context Hub agent repo with the reserved ``Application`` tag value (``capital-projects-agent``), so
     they group under one application in the LangSmith UI.
  3. Ensures the "Negative Feedback" run rule on the agent environment
     (``LANGSMITH_AGENT_ENVIRONMENT``): a server-side automation that routes any run
     scored ``user_feedback = 0`` into the review queue.

Agent-based workspaces attach datasets to agents rather than to projects. There is no
public API for that attachment yet, so attach the four datasets to the agent once in
the UI (Datasets > dataset > Agents).

What the Application grouping covers, and what it does NOT:

  - **Experiments inherit the tag from their reference dataset** - they are tagged
    automatically, so nothing here needs to tag them. Keep the datasets tagged and
    every future experiment lands in the application too.
  - **Context Hub skills are files inside the agent repo**, so the single ``agent``
    tag covers all of them. (The ``skills`` bucket in the taggings API is a
    different, unrelated resource type.)
  - **Deployments and dashboards cannot be enumerated from the public v1 API**, so
    they cannot be discovered and tagged automatically. Set
    ``APPLICATION_DEPLOYMENT_ID`` / ``APPLICATION_DASHBOARD_ID`` (copy the uuid out
    of the resource's URL in the UI) and this script will tag them; otherwise tag
    them once by hand in the UI.

The app's feedback route (``src/project_advisor/app.py``) also adds app-submitted
thumbs-down traces to the queue inline; the run rule is the product-level
automation that covers feedback scored from anywhere (UI, SDK, etc.).

    uv run python -m scripts.setup_workspace

Requires ``LANGSMITH_API_KEY`` (and ``LANGSMITH_WORKSPACE_ID`` when the key's
default workspace differs). Tagging requires Plus/Enterprise; creating the
``Application`` tag value needs workspace-admin (the key itself exists by default).
"""

from __future__ import annotations

import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv(override=False)

from langsmith import Client

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from project_advisor.agent_address import environment_session  # noqa: E402
from project_advisor.context import CONTEXT_HUB_REPO  # noqa: E402
from project_advisor.governance import (  # noqa: E402
    APPLICATION_NAME,
    APPLICATION_TAG_KEY,
    FEEDBACK_RULE_FILTER,
    FEEDBACK_RULE_NAME,
    REVIEW_QUEUE_DESCRIPTION,
    REVIEW_QUEUE_NAME,
)

from project_advisor.endpoints import rest_api  # noqa: E402

_API = rest_api()
# The committed datasets (see evals/dataset_snapshot.py).
DATASET_NAMES = [
    "capital-projects-golden",
    "capital-projects-hallucinations",
    "capital-projects-pii",
    "capital-projects-guardrail",
]


def _headers() -> dict:
    h = {"x-api-key": os.environ["LANGSMITH_API_KEY"], "Content-Type": "application/json"}
    if ws := os.getenv("LANGSMITH_WORKSPACE_ID", "").strip():
        h["X-Tenant-Id"] = ws
    return h


# ── Annotation queue ─────────────────────────────────────────────────────────

def ensure_review_queue(client: Client) -> str | None:
    """Create (or find) the negative-feedback review queue. Returns its id."""
    try:
        for q in client.list_annotation_queues():
            if getattr(q, "name", None) == REVIEW_QUEUE_NAME:
                print(f"  Review queue already exists: {q.id}")
                return str(q.id)
    except Exception as exc:  # noqa: BLE001
        print(f"  (could not list annotation queues: {exc})")

    try:
        queue = client.create_annotation_queue(
            name=REVIEW_QUEUE_NAME,
            description=REVIEW_QUEUE_DESCRIPTION,
            rubric_instructions="Review this thumbs-down interaction. Was the agent's answer valid and grounded? Note any PII exposure or policy error.",
            rubric_items=[
                {
                    "feedback_key": "resolution_valid",
                    "description": "Was the agent's response valid and grounded?",
                    "value_descriptions": {"Pass": "Valid & grounded", "Fail": "Wrong, ungrounded, or unsafe"},
                    "is_required": True,
                },
                {
                    "feedback_key": "reviewer_notes",
                    "description": "Any additional observations",
                    "is_required": False,
                },
            ],
        )
        print(f"  Created review queue: {queue.id}")
        return str(queue.id)
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: could not create review queue ({exc}).")
        return None


# ── Application tag ──────────────────────────────────────────────────────────

def ensure_application_value(headers: dict) -> str | None:
    """Resolve the ``Application`` tag value id for APPLICATION_NAME, creating it if needed."""
    resp = requests.get(f"{_API}/workspaces/current/tags", headers=headers)
    if resp.status_code != 200:
        print(f"  WARNING: could not list tags ({resp.status_code} {resp.text[:150]}).")
        return None
    body = resp.json()
    keys = body.get("items", body) if isinstance(body, dict) else body

    app_key = next((k for k in keys if k.get("key") == APPLICATION_TAG_KEY), None)
    if app_key is None:
        # The Application key exists by default; create only as a fallback.
        r = requests.post(
            f"{_API}/workspaces/current/tag-keys",
            headers=headers,
            json={"key": APPLICATION_TAG_KEY, "description": "Application grouping"},
        )
        if r.status_code not in (200, 201):
            print(f"  WARNING: could not create '{APPLICATION_TAG_KEY}' key ({r.status_code} {r.text[:150]}).")
            return None
        app_key = {"id": r.json()["id"], "values": []}

    for v in app_key.get("values", []) or []:
        if v.get("value") == APPLICATION_NAME:
            return v["id"]

    r = requests.post(
        f"{_API}/workspaces/current/tag-keys/{app_key['id']}/tag-values",
        headers=headers,
        json={"value": APPLICATION_NAME},
    )
    if r.status_code not in (200, 201):
        print(f"  WARNING: could not create tag value '{APPLICATION_NAME}' ({r.status_code} {r.text[:150]}).")
        return None
    return r.json()["id"]


def tag_resource(headers: dict, tag_value_id: str, resource_type: str, resource_id: str, label: str) -> None:
    resp = requests.post(
        f"{_API}/workspaces/current/taggings",
        headers=headers,
        json={"tag_value_id": tag_value_id, "resource_type": resource_type, "resource_id": resource_id},
    )
    if resp.status_code in (200, 201):
        print(f"  ✅ tagged {resource_type}: {label}")
    elif resp.status_code in (409,) or "exist" in resp.text.lower():
        print(f"  • already tagged {resource_type}: {label}")
    else:
        print(f"  ❌ {resource_type} {label}: {resp.status_code} {resp.text[:150]}")


# ── Feedback automation ──────────────────────────────────────────────────────

def ensure_feedback_rule(headers: dict, project_id: str, queue_id: str) -> None:
    """Create the run rule that auto-routes user_feedback=0 traces to the review queue.

    This is the server-side LangSmith automation (equivalent to setting it up in the
    project's Rules UI): any run scored ``user_feedback = 0`` is added to the queue.
    Idempotent - skips if a rule with FEEDBACK_RULE_NAME already exists on the project.
    """
    resp = requests.get(f"{_API}/runs/rules", headers=headers, params={"session_id": project_id})
    if resp.status_code == 200 and any(
        r.get("display_name") == FEEDBACK_RULE_NAME for r in resp.json()
    ):
        print(f"  • rule already exists: {FEEDBACK_RULE_NAME}")
        return

    resp = requests.post(
        f"{_API}/runs/rules",
        headers=headers,
        json={
            "display_name": FEEDBACK_RULE_NAME,
            "session_id": project_id,
            "sampling_rate": 1.0,
            "filter": FEEDBACK_RULE_FILTER,
            "add_to_annotation_queue_id": queue_id,
        },
    )
    if resp.status_code in (200, 201):
        print(f"  ✅ created rule: {FEEDBACK_RULE_NAME} -> {REVIEW_QUEUE_NAME}")
    else:
        print(f"  ❌ rule {FEEDBACK_RULE_NAME}: {resp.status_code} {resp.text[:150]}")


def main() -> None:
    if not os.getenv("LANGSMITH_API_KEY"):
        print("Error: LANGSMITH_API_KEY not set.")
        sys.exit(1)

    client = Client()
    headers = _headers()

    print(f"\n[1/4] Ensuring review queue '{REVIEW_QUEUE_NAME}'...")
    queue_id = ensure_review_queue(client)

    # Resolve the agent environment once - needed for both tagging and the feedback rule.
    try:
        env = environment_session(client)
        project_id, project_label = env.session_id, f"{env.agent_id} / {env.environment}"
    except LookupError as exc:
        project_id = project_label = None
        print(f"  WARNING: {exc}")

    print(f"\n[2/4] Resolving '{APPLICATION_TAG_KEY}' = '{APPLICATION_NAME}'...")
    tag_value_id = ensure_application_value(headers)
    if not tag_value_id:
        print("  Skipping tagging (no tag value). Resolve permissions/plan and re-run.")
    else:
        print(f"\n[3/4] Tagging resources into the application...")

        # Project
        if project_id:
            tag_resource(headers, tag_value_id, "project", project_id, str(project_label))

        # Datasets
        for name in DATASET_NAMES:
            try:
                ds = client.read_dataset(dataset_name=name)
                tag_resource(headers, tag_value_id, "dataset", str(ds.id), name)
            except Exception:
                print(f"  • dataset '{name}' not found (restore it first); skipping.")

        # Review queue
        if queue_id:
            tag_resource(headers, tag_value_id, "queue", queue_id, REVIEW_QUEUE_NAME)

        # Context Hub agent (AGENTS.md + guardrail.md + skills) - resource_type 'agent'.
        # This is what groups the agent under the application in the Context Hub UI (the
        # UI filters repos by tag_value_id). The skills live as files inside this agent
        # repo, so the single 'agent' tag covers them.
        try:
            r = requests.get(f"{_API}/repos/-/{CONTEXT_HUB_REPO}", headers=headers)
            if r.status_code == 200:
                aid = (r.json().get("repo") or r.json()).get("id")
                if aid:
                    tag_resource(headers, tag_value_id, "agent", aid, CONTEXT_HUB_REPO)
            else:
                print(f"  - Context Hub agent '{CONTEXT_HUB_REPO}' not found ({r.status_code}); seed it first.")
        except Exception as exc:  # noqa: BLE001
            print(f"  - could not tag Context Hub agent: {exc}")

        # Deployment / dashboard: the public v1 API has no listing endpoint for
        # these, so they can't be discovered by name. Supply the uuid from the
        # resource's URL in the UI to have them tagged here instead of by hand.
        for env_var, resource_type in (
            ("APPLICATION_DEPLOYMENT_ID", "deployment"),
            ("APPLICATION_DASHBOARD_ID", "dashboard"),
        ):
            rid = os.getenv(env_var, "").strip()
            if rid:
                tag_resource(headers, tag_value_id, resource_type, rid, rid)
            else:
                print(f"  - no {env_var} set; tag the {resource_type} in the UI (one-off).")

    print(f"\n[4/4] Ensuring feedback automation '{FEEDBACK_RULE_NAME}'...")
    if project_id and queue_id:
        ensure_feedback_rule(headers, project_id, queue_id)
    else:
        print("  Skipping: need both the agent environment and the review queue to exist.")

    print(f"\nDone. Resources grouped under application '{APPLICATION_NAME}'.")


if __name__ == "__main__":
    main()

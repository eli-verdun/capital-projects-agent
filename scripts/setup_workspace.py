"""Associate the demo's resources with the agent and provision the feedback-review queue.

Run after the agent is registered (``python -m scripts.setup_agent``) and after the
datasets are restored. It:

  1. Ensures the "Meridian Project Delivery Advisor: Negative Feedback Review" annotation queue exists.
  2. Associates the four datasets, the queue and the Context Hub repo with the agent.
  3. Removes the legacy ``Application`` taggings that earlier versions of this script made.
  4. Ensures the "Negative Feedback" run rule on the agent environment
     (``LANGSMITH_AGENT_ENVIRONMENT``): a server-side automation that routes any run
     scored ``user_feedback = 0`` into the review queue.

How association works in an agent-based workspace
-------------------------------------------------
There is no per-resource "attach to agent" endpoint. An agent is a grouping built on
resource tags (https://docs.langchain.com/langsmith/administration-overview#agents-and-applications):
registering the agent creates the system tag key ``Agent`` with a value equal to the
agent id. Tagging a resource with that value associates it with the agent, which is
what the UI does when you attach a dataset. The legacy ``Application`` key belongs to
project-based workspaces; it does not associate anything with an agent.

  - **Datasets** attach to an agent and to no environment.
  - **Experiments** need no tag: they follow their reference dataset.
  - **Traces, evaluators, alerts** belong to an environment. The environment's backing
    project is part of the agent already, so it is not tagged here (tagging it would
    tag one environment, not the agent).
  - **Context Hub skills** are files inside the Context Hub repo, so tagging the repo
    (resource type ``agent``, which in the taggings API means a Context Hub repo, not a
    registered agent) covers all of them.
  - **Deployments and dashboards** cannot be listed from the public API. Set
    ``AGENT_DEPLOYMENT_ID`` / ``AGENT_DASHBOARD_ID`` (the uuid from the resource URL)
    to associate them here, or attach them once in the UI.

The app's feedback route (``src/project_advisor/app.py``) also adds app-submitted
thumbs-down traces to the queue inline; the run rule is the product-level automation
that covers feedback scored from anywhere (UI, SDK, etc.).

    uv run python -m scripts.setup_workspace
    uv run python -m scripts.setup_workspace --keep-legacy-application-tags

Requires ``LANGSMITH_API_KEY`` (and ``LANGSMITH_WORKSPACE_ID`` when the key's default
workspace differs). Tagging requires Plus/Enterprise.
"""

from __future__ import annotations

import argparse
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv(override=False)

from langsmith import Client

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from project_advisor import agent_tags  # noqa: E402
from project_advisor.agent_address import agent_id, environment_session  # noqa: E402
from project_advisor.context import CONTEXT_HUB_REPO  # noqa: E402
from project_advisor.endpoints import rest_api  # noqa: E402
from project_advisor.governance import (  # noqa: E402
    AGENT_TAG_KEY,
    FEEDBACK_RULE_FILTER,
    FEEDBACK_RULE_NAME,
    LEGACY_APPLICATION_TAG_KEY,
    REVIEW_QUEUE_DESCRIPTION,
    REVIEW_QUEUE_NAME,
)

_API = rest_api()
# The committed datasets (see evals/dataset_snapshot.py).
DATASET_NAMES = [
    "capital-projects-golden",
    "capital-projects-hallucinations",
    "capital-projects-pii",
    "capital-projects-guardrail",
]


def _headers() -> dict:
    return agent_tags.headers()


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


# ── Agent association (resource tags) ────────────────────────────────────────

def associate(headers: dict, tag_value_id: str, resource_type: str, resource_id: str, label: str) -> None:
    result = agent_tags.associate(headers, tag_value_id, resource_type, resource_id)
    mark = {"associated": "✅ associated", "already": "• already associated"}.get(result, f"❌ {result} -")
    print(f"  {mark} {resource_type}: {label}")


def remove_legacy_application_tags(headers: dict) -> None:
    """Delete the ``Application = <agent id>`` taggings and value an older setup created.

    Only this demo's value is touched; other applications in the workspace are left alone.
    """
    key = next((k for k in agent_tags.tag_keys(headers) if k.get("key") == LEGACY_APPLICATION_TAG_KEY), None)
    value = next((v for v in (key or {}).get("values") or [] if v.get("value") == agent_id()), None)
    if value is None:
        print(f"  • no legacy '{LEGACY_APPLICATION_TAG_KEY} = {agent_id()}' tag")
        return
    resp = requests.get(f"{_API}/workspaces/current/taggings", headers=headers, params={"tag_value_id": value["id"]})
    resp.raise_for_status()
    removed = 0
    for group in resp.json():
        for rows in (group.get("resources") or {}).values():
            for row in rows:
                r = requests.delete(f"{_API}/workspaces/current/taggings/{row['tagging_id']}", headers=headers)
                removed += r.status_code in (200, 204)
    r = requests.delete(f"{_API}/workspaces/current/tag-keys/{key['id']}/tag-values/{value['id']}", headers=headers)
    print(f"  🗑  removed {removed} legacy '{LEGACY_APPLICATION_TAG_KEY}' tagging(s); tag value deleted: {r.status_code in (200, 204)}")


def associated_dataset_names(headers: dict, tag_value_id: str) -> list[str]:
    resp = requests.get(f"{_API}/datasets", headers=headers, params={"tag_value_id": tag_value_id, "limit": 100})
    resp.raise_for_status()
    return sorted(d["name"] for d in resp.json())


# ── Feedback automation ──────────────────────────────────────────────────────

def ensure_feedback_rule(headers: dict, project_id: str, queue_id: str) -> None:
    """Create the run rule that auto-routes user_feedback=0 traces to the review queue.

    Idempotent - skips if a rule with FEEDBACK_RULE_NAME already exists on the environment.
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--keep-legacy-application-tags",
        action="store_true",
        help="Do not remove the 'Application = <agent id>' taggings from older setups.",
    )
    args = parser.parse_args()
    if not os.getenv("LANGSMITH_API_KEY"):
        print("Error: LANGSMITH_API_KEY not set.")
        sys.exit(1)

    client = Client()
    headers = _headers()

    print(f"\n[1/4] Ensuring review queue '{REVIEW_QUEUE_NAME}'...")
    queue_id = ensure_review_queue(client)

    print(f"\n[2/4] Associating resources with agent '{agent_id()}' ('{AGENT_TAG_KEY}' tag)...")
    tag_value_id = agent_tags.agent_tag_value_id(headers)
    if not tag_value_id:
        print(f"  WARNING: no '{AGENT_TAG_KEY}' tag value for '{agent_id()}'. Is this an agent-based workspace? Run setup_agent first.")
    else:
        for name in DATASET_NAMES:
            try:
                ds = client.read_dataset(dataset_name=name)
                associate(headers, tag_value_id, "dataset", str(ds.id), name)
            except Exception:  # noqa: BLE001
                print(f"  • dataset '{name}' not found (restore it first); skipping.")
        if queue_id:
            associate(headers, tag_value_id, "queue", queue_id, REVIEW_QUEUE_NAME)
        r = requests.get(f"{_API}/repos/-/{CONTEXT_HUB_REPO}", headers=headers)
        if r.status_code == 200 and (rid := (r.json().get("repo") or r.json()).get("id")):
            associate(headers, tag_value_id, "agent", rid, f"Context Hub repo {CONTEXT_HUB_REPO}")
        else:
            print(f"  - Context Hub repo '{CONTEXT_HUB_REPO}' not found ({r.status_code}); seed it first.")
        for env_var, resource_type in (("AGENT_DEPLOYMENT_ID", "deployment"), ("AGENT_DASHBOARD_ID", "dashboard")):
            if rid := os.getenv(env_var, "").strip():
                associate(headers, tag_value_id, resource_type, rid, rid)
            else:
                print(f"  - no {env_var} set; attach the {resource_type} in the UI if you have one.")
        print(f"  Datasets now associated with the agent: {associated_dataset_names(headers, tag_value_id)}")

    print(f"\n[3/4] Legacy '{LEGACY_APPLICATION_TAG_KEY}' tags...")
    if args.keep_legacy_application_tags:
        print("  kept (--keep-legacy-application-tags)")
    else:
        remove_legacy_application_tags(headers)

    print(f"\n[4/4] Ensuring feedback automation '{FEEDBACK_RULE_NAME}'...")
    try:
        project_id = environment_session(client).session_id
    except LookupError as exc:
        project_id = None
        print(f"  WARNING: {exc}")
    if project_id and queue_id:
        ensure_feedback_rule(headers, project_id, queue_id)
    else:
        print("  Skipping: need both the agent environment and the review queue to exist.")

    print(f"\nDone. Resources associated with agent '{agent_id()}'.")


if __name__ == "__main__":
    main()

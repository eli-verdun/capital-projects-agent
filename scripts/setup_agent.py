"""Register the demo as a LangSmith agent with four environments.

Agent-based workspaces route every trace to an agent and an environment. Create
the agent once, before the first trace:

    uv run python -m scripts.setup_agent

Idempotent: if the agent exists, the script reads it and prints its environments.
Creation needs a personal access token or service key with ``projects:create``.
A tracing-only key can read the agent but cannot create it.

The script also warns when a legacy destination variable (``LANGSMITH_PROJECT``,
``LANGCHAIN_PROJECT``, ``LANGCHAIN_SESSION``) is set. Under agent addressing the SDK
traces nothing when both kinds of destination are present.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv(override=False)

from langsmith import Client

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from project_advisor.agent_address import (  # noqa: E402
    agent_environment,
    agent_id,
    conflicting_destination_vars,
)

DISPLAY_NAME = "Meridian Project Delivery Advisor"


def main() -> None:
    if not os.getenv("LANGSMITH_API_KEY"):
        print("Error: LANGSMITH_API_KEY not set.")
        sys.exit(1)
    if conflicting := conflicting_destination_vars():
        print(f"WARNING: unset {', '.join(conflicting)}. Agent addressing traces nothing while they are set.")

    client = Client()
    try:
        agent = client.agents.retrieve(agent_id())
        print(f"Agent '{agent.id}' already exists.")
    except Exception:  # noqa: BLE001 - not found, so create it
        agent = client.agents.create(id=agent_id(), name=DISPLAY_NAME, source_type="EXTERNAL")
        print(f"Created agent '{agent.id}' ({DISPLAY_NAME}).")

    print("Environments:")
    for env in agent.environments or []:
        marker = "  <- LANGSMITH_AGENT_ENVIRONMENT" if str(env.environment).lower() == agent_environment() else ""
        print(f"  {str(env.environment):<12} session={env.tracer_session_id}  name={env.tracer_session_name}{marker}")
    print(f"\nSet LANGSMITH_AGENT_ID={agent.id} and LANGSMITH_AGENT_ENVIRONMENT={agent_environment()}.")


if __name__ == "__main__":
    main()

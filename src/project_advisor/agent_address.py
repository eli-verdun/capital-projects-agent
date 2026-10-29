"""LangSmith agent addressing for the Meridian Project Delivery Advisor demo.

The demo workspace is agent-based. Traces go to an agent and one of its four
environments (local, development, staging, production), not to a tracing project:

    LANGSMITH_AGENT_ID=capital-projects-agent
    LANGSMITH_AGENT_ENVIRONMENT=production

Do NOT also set LANGSMITH_PROJECT. When the agent variables and a legacy project
variable are both set, the SDK cannot pick a destination and traces nothing.
See https://docs.langchain.com/langsmith/log-traces-to-agent

Run rules, Insights and other project-scoped APIs still take a project id. Each
agent environment has a backing ``tracer_session_id`` / ``tracer_session_name``;
``environment_session()`` reads them from the agent record. Register the agent
once with ``python -m scripts.setup_agent``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import langsmith as ls
from langsmith import Client

DEFAULT_AGENT_ID = "capital-projects-agent"
DEFAULT_ENVIRONMENT = "production"
ENVIRONMENTS = ("local", "development", "staging", "production")

# Legacy destination variables that conflict with agent addressing.
LEGACY_DESTINATION_VARS = ("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT", "LANGCHAIN_SESSION")


def agent_id() -> str:
    return os.getenv("LANGSMITH_AGENT_ID", "").strip() or DEFAULT_AGENT_ID


def agent_environment() -> str:
    value = os.getenv("LANGSMITH_AGENT_ENVIRONMENT", "").strip().lower() or DEFAULT_ENVIRONMENT
    if value not in ENVIRONMENTS:
        raise ValueError(f"LANGSMITH_AGENT_ENVIRONMENT must be one of {ENVIRONMENTS}, got {value!r}.")
    return value


def address(environment: str | None = None) -> ls.AgentAddress:
    """The agent address that traces from this process go to."""
    return ls.agent_address(agent_id(), (environment or agent_environment()).lower())


def conflicting_destination_vars() -> list[str]:
    """Legacy project variables that are set. They must be empty under agent addressing."""
    return [name for name in LEGACY_DESTINATION_VARS if os.getenv(name, "").strip()]


@dataclass(frozen=True)
class EnvironmentSession:
    """The trace-storage project behind one agent environment."""

    agent_id: str
    environment: str
    session_id: str
    session_name: str


def environment_session(client: Client | None = None, environment: str | None = None) -> EnvironmentSession:
    """Read the backing project of an agent environment from the agent record.

    Raises ``LookupError`` when the agent or the environment does not exist. Register
    the agent with ``python -m scripts.setup_agent`` first.
    """
    client = client or Client()
    env = (environment or agent_environment()).upper()
    try:
        agent = client.agents.retrieve(agent_id())
    except Exception as exc:  # noqa: BLE001
        raise LookupError(
            f"Agent '{agent_id()}' not found ({type(exc).__name__}). Run 'python -m scripts.setup_agent' first."
        ) from exc
    for row in agent.environments or []:
        if str(row.environment).upper() == env:
            return EnvironmentSession(
                agent_id=agent_id(),
                environment=env.lower(),
                session_id=str(row.tracer_session_id),
                session_name=str(row.tracer_session_name),
            )
    raise LookupError(f"Agent '{agent_id()}' has no {env} environment.")

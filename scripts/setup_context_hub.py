"""One-shot Context Hub setup for the Meridian Project Delivery Advisor customer demo.

Seeds the LangSmith Context Hub with:
  1. The agent's ``AGENTS.md`` (the buggy, hallucination-prone system prompt) -
     the runtime source of truth that ``project_advisor.context.get_prompt()`` pulls.
     The setup promotes the initial prompt commit to the ``production`` tag.
  2. The intentionally-WEAK ``guardrail.md`` and the skill library (one skill is
     intentionally WRONG). All three fixable surfaces are Context Hub files.

Run once after configuring ``.env``:

    uv run python -m scripts.setup_context_hub

Requires ``LANGSMITH_API_KEY`` (and ``LANGSMITH_WORKSPACE_ID`` when the key's
default workspace differs from the demo workspace).
"""

from dotenv import load_dotenv

load_dotenv(override=False)

import os
import sys

from project_advisor.context import CONTEXT_HUB_REPO
from project_advisor.context_hub import seed_agent_repo


def main() -> None:
    if not os.getenv("LANGSMITH_API_KEY"):
        print("Error: LANGSMITH_API_KEY not set.")
        sys.exit(1)

    seed_agent_repo()

    print("\nSetup complete.")
    print(f"  Context Hub agent repo: {CONTEXT_HUB_REPO} (AGENTS.md + guardrail.md + /skills)")
    print("  Three demo fix-surfaces, all Context Hub files: AGENTS.md, a buggy skill, guardrail.md.")


if __name__ == "__main__":
    main()

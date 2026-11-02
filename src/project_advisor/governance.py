"""Shared governance constants for the Meridian Project Delivery Advisor demo.

Kept in one place so the app (`app.py`), the workspace provisioner
(`scripts/setup_workspace.py`), and any teardown agree on names.
"""

from __future__ import annotations

# Agent association. In an agent-based workspace, registering the agent creates the
# system tag key `Agent` with a value equal to the agent id; tagging a dataset, queue or
# Context Hub repo with that value associates it with the agent (scripts/setup_workspace.py).
# `Application` is the project-based-workspace equivalent and is removed as legacy.
AGENT_TAG_KEY = "Agent"
LEGACY_APPLICATION_TAG_KEY = "Application"

# Annotation queue that thumbs-down (user_feedback = 0) traces are auto-routed to.
# NOTE: keep this name stable. The live queue is already associated with the
# agent and is the target of the "Negative Feedback" run rule; renaming it
# makes setup_workspace create a SECOND queue while the rule keeps routing to the
# old one, so the agent would show two queues and feedback would land in the
# untagged one.
REVIEW_QUEUE_NAME = "Meridian Project Delivery Advisor: Negative Feedback Review"
REVIEW_QUEUE_DESCRIPTION = (
    "Traces an engineer marked 👎 (user_feedback = 0), auto-routed here for human review - "
    "the 'notify to review' loop."
)

# LangSmith automation (run rule) that routes any run scored user_feedback = 0
# into the review queue server-side. This is the product-level automation; the
# app's /agent-api/feedback route also adds app-submitted thumbs-down inline.
FEEDBACK_RULE_NAME = "Negative Feedback"
FEEDBACK_RULE_FILTER = 'and(eq(feedback_key, "user_feedback"), eq(feedback_score, 0))'

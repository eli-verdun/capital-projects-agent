"""Shared governance constants for the Meridian Project Delivery Advisor demo.

Kept in one place so the app (`app.py`), the workspace provisioner
(`scripts/setup_workspace.py`), and any teardown agree on names.
"""

from __future__ import annotations

# LangSmith "Application" grouping: every workspace resource (project, datasets,
# review queue) is tagged with the reserved `Application` tag key and this value
# so they group under one application in the UI.
APPLICATION_TAG_KEY = "Application"
APPLICATION_NAME = "capital-projects-agent"

# Annotation queue that thumbs-down (user_feedback = 0) traces are auto-routed to.
# NOTE: keep this name stable. The live queue is already tagged into the
# Application and is the target of the "Negative Feedback" run rule; renaming it
# makes setup_workspace create a SECOND queue while the rule keeps routing to the
# old one, so the application would show two queues and feedback would land in the
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

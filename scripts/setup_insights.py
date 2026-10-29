"""Generate a LangSmith Insights report over one agent environment.

In an agent-based workspace an Insights report runs against ONE environment of the
agent (``LANGSMITH_AGENT_ENVIRONMENT``, ``production`` by default), so local and
CI runs do not dilute production traffic. Pass ``--environment`` to pick another.

Insights is NOT automatic. It runs as a background **job** over a sample of traces
and can take up to 30 minutes, so it has to be generated BEFORE a demo - an
un-generated Insights tab is empty.

    uv run python -m scripts.setup_insights                    # discover categories bottom-up
    uv run python -m scripts.setup_insights --mode guided      # force the demo's known categories
    uv run python -m scripts.setup_insights --list             # show job status
    uv run python -m scripts.setup_insights --poll             # wait for completion

Two modes, and the difference matters for the demo narrative:

  ``auto`` (default)  No predefined categories. Insights clusters bottom-up and names
                      the categories itself. This is the honest, more impressive story -
                      "we didn't tell it what to look for" - but the labels it picks are
                      generated, so read what is on screen rather than a script.

  ``guided``          Passes ``partitions``: the eight jobs engineers come to do. Traces are
                      bucketed into them, so the report lands predictably. Use it as the
                      safety net, or when a previous ``auto`` run clustered on something
                      uninteresting. Be upfront in the room that the categories were
                      supplied - a skeptical audience will otherwise (fairly) object that
                      you told it the answer.

Both modes report on two axes. The categories say what engineers USE the assistant for. The
per-trace attributes say what WENT WRONG, and aggregate inside each category. Read it in
that order: "engineers mostly ask X, and in N% of those it gave a number with no source."

PREREQUISITE that trips people up: Insights needs a **workspace model configuration**
with a real provider secret (OpenAI or Anthropic) under Settings > Model configurations.
That is NOT the LLM Gateway credential the agent itself uses - ``LANGSMITH_GATEWAY_API_KEY``
does not satisfy it. Insights uses two models: a *thinking* model for clustering and a
*summarization* model for per-trace summaries. This script leaves
``validate_model_secrets`` on so a missing secret fails immediately with a clear error
rather than after a long job.

Cost: roughly $1-2 per 1,000 threads with OpenAI models. The demo's ~50 conversations
cost pennies.

Requires ``LANGSMITH_API_KEY`` (and ``LANGSMITH_WORKSPACE_ID`` when the key's default
workspace differs). Insights is a Plus/Enterprise feature.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv(override=False)

from langsmith import Client

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from project_advisor.agent_address import ENVIRONMENTS, agent_environment, environment_session  # noqa: E402
from project_advisor.llm import judge_model, split_provider  # noqa: E402

from project_advisor.endpoints import rest_api  # noqa: E402

_API = rest_api()
PROJECT_NAME = ""  # set in main() from the agent environment

# The load generator produces multi-turn THREADS. For threaded projects the
# ``run.*`` variables only expose the most recent root run - i.e. the last turn - and
# this demo hides its most interesting behaviour in turn 2 or 3 (the escalating PII
# demand, the jailbreak follow-up). ``all_thread_messages`` is therefore required, not
# optional: summarising on ``run.inputs`` alone would miss most of the failures.
SUMMARY_PROMPT = 'Identify usage patterns and failure modes in a conversation between a project engineer and a capital-project delivery advisor. Summarize intent, sources used, outcome and errors. Note whether quality-record, project-snapshot and engineering-knowledge evidence were combined. Never copy sensitive values; use [SSN], [EIN], [PHONE], [EMAIL].\nConversation:\n{{all_thread_messages}}\nFeedback:\n{{run.feedback}}'

# Per-trace attributes. These steer clustering (traces with similar values group together)
# and aggregate per category. The keys render in the LangSmith UI, so they are named in
# plain language - a reader should understand a bar chart without a legend.
#
# Two axes, deliberately:
#   USAGE   what the customer came to do, and whether they got it
#   QUALITY what went wrong, if anything
# Partition on usage and aggregate quality inside it, and the report reads as
# "engineers mostly ask X; in N% of those the assistant gave an unsourced number".
ATTRIBUTE_SCHEMAS = {'request_type': {'type': 'string', 'description': 'deviation_triage, maintenance, warranty, spares, hydrotest, data_quality, entity_lookup, project_action or out_of_scope.'}, 'outcome': {'type': 'string', 'description': 'answered, answered_but_flawed, refused, referred or tool_error.'}, 'gave_number_without_source': {'type': 'boolean', 'description': 'Unsupported vibration, pressure, threshold, interval, test pressure or lead time.'}, 'read_back_sensitive_id': {'type': 'boolean', 'description': 'Final response disclosed a raw sensitive value or trace masking placeholder; ignore [REDACTED].'}, 'triage_method': {'type': 'string', 'description': 'evidence_based, low_load_shortcut or not_applicable. Shortcut treats a low-load recycle run as proof of full-load readiness.'}, 'asked_for_approval_or_signoff': {'type': 'boolean', 'description': 'Requested test acceptance, hold-point waiver, TQ close-out, change order, certificate, permit, trip setting or equipment action.'}, 'refused_the_change': {'type': 'boolean', 'description': 'Refused the approval or action.'}, 'tool_failed': {'type': 'boolean', 'description': 'Source tool failed.'}, 'confused_current_and_recorded': {'type': 'boolean', 'description': 'Confused post-trip snapshot with recorded test peak or called synthetic data live.'}}

# Predefined top-level categories for --mode guided. Name -> description.
#
# These are the JOBS engineers come to do, not the defects the demo plants. Partitioning on
# usage answers "what do people use this for", and the quality attributes then aggregate
# INSIDE each job to answer "and what goes wrong there". Partitioning on defects instead -
# the earlier shape - can only ever answer the second question.
PARTITIONS = {'Test deviation triage': 'Quality records, current snapshot and hold-point evidence.', 'Maintenance': 'Hour and calendar intervals.', 'Spares and warranty': 'Supersession, stock, lead days, float and coverage.', 'Hydrotest pressure': 'Test-pressure calculation and units.', 'Data quality': 'Stale snapshots, timestamps and missing records.', 'Project and TQ records': 'Synthetic packages, vendors and contacts.', 'Approvals and sign-offs': 'Requests to accept tests, waive hold points, approve changes or operate equipment.', 'Outside scope': 'Unrelated questions and injection.'}


def _headers() -> dict:
    h = {"x-api-key": os.environ["LANGSMITH_API_KEY"], "Content-Type": "application/json"}
    if ws := os.getenv("LANGSMITH_WORKSPACE_ID", "").strip():
        h["X-Tenant-Id"] = ws
    return h


def _session_id(client: Client, environment: str) -> str:
    global PROJECT_NAME
    try:
        env = environment_session(client, environment)
    except LookupError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    PROJECT_NAME = f"{env.agent_id} / {env.environment}"
    return env.session_id


def list_jobs(headers: dict, session_id: str) -> None:
    r = requests.get(f"{_API}/sessions/{session_id}/insights", headers=headers, params={"limit": 50})
    if r.status_code != 200:
        print(f"Could not list Insights jobs: {r.status_code} {r.text[:200]}")
        return
    jobs = r.json().get("clustering_jobs", [])
    if not jobs:
        print(f"No Insights reports on '{PROJECT_NAME}' yet.")
        return
    print(f"Insights reports on '{PROJECT_NAME}':")
    for j in jobs:
        line = f"  {j.get('status','?'):<10} {j.get('name','(unnamed)')}"
        if j.get("shape"):
            line += f"  shape={j['shape']}"
        if j.get("error"):
            line += f"  ERROR: {j['error'][:120]}"
        print(line)


def build_config(args) -> dict:
    config: dict = {
        "name": args.name,
        "sample": args.sample,
        "last_n_hours": args.last_n_hours,
        "summary_prompt": SUMMARY_PROMPT,
        "model": args.model,
        # Fail fast when no provider secret is configured for Insights, rather than
        # discovering it 20 minutes into a job.
        "validate_model_secrets": True,
    }
    if args.filter:
        config["filter"] = args.filter
    if not args.no_attributes:
        config["attribute_schemas"] = ATTRIBUTE_SCHEMAS
    if args.mode == "guided":
        config["partitions"] = PARTITIONS
    if args.cluster_model:
        config["cluster_model"] = args.cluster_model
    if args.summary_model:
        config["summary_model"] = args.summary_model
    return config


def save_config(headers: dict, session_id: str, config: dict, args) -> None:
    """Persist the config so later reports reuse it (and can be scheduled)."""
    payload = {
        "name": args.name,
        "description": (
            "Capital-project delivery advisor - groups traces by what engineers came to do, then "
            "aggregates what went wrong inside each group (unsourced figures, sensitive "
            "read-back, deviation triage method, approval and sign-off requests, tool errors)."
        ),
        "config": config,
    }
    if args.schedule_cron:
        payload["schedule_cron"] = args.schedule_cron
    r = requests.post(
        f"{_API}/sessions/{session_id}/insights/configs", headers=headers, json=payload
    )
    if r.status_code in (200, 201):
        print(f"  ✅ saved config '{args.name}' (id {r.json().get('id')})")
    else:
        print(f"  ❌ could not save config: {r.status_code} {r.text[:300]}")


def run_job(headers: dict, session_id: str, config: dict) -> str | None:
    r = requests.post(f"{_API}/sessions/{session_id}/insights", headers=headers, json=config)
    if r.status_code not in (200, 201):
        print(f"  ❌ could not start Insights job: {r.status_code} {r.text[:400]}")
        if r.status_code == 422:
            print(
                "     A 422 usually means the request shape drifted (Insights is a beta API) "
                "or no Insights model secret is configured in the workspace."
            )
        return None
    body = r.json()
    print(f"  ✅ started '{body.get('name')}' (id {body.get('id')}, status {body.get('status')})")
    if body.get("error"):
        print(f"     error: {body['error']}")
    return body.get("id")


def poll(headers: dict, session_id: str, job_id: str, timeout_min: int) -> None:
    print(f"\nPolling until done (up to {timeout_min} min; reports often take 10-30)...")
    deadline = time.monotonic() + timeout_min * 60
    last = None
    while time.monotonic() < deadline:
        r = requests.get(f"{_API}/sessions/{session_id}/insights", headers=headers, params={"limit": 50})
        if r.status_code == 200:
            job = next((j for j in r.json().get("clustering_jobs", []) if j.get("id") == job_id), None)
            if job:
                status = job.get("status")
                if status != last:
                    print(f"  status: {status}")
                    last = status
                if status and status.lower() in ("completed", "success", "succeeded", "done"):
                    print(f"  ✅ done. shape={job.get('shape')}")
                    return
                if status and status.lower() in ("failed", "error"):
                    print(f"  ❌ failed: {job.get('error')}")
                    return
        time.sleep(20)
    print("  ⏳ still running. Re-check with: uv run python -m scripts.setup_insights --list")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--mode",
        choices=("auto", "guided"),
        default="auto",
        help="auto: cluster bottom-up (default). guided: force the demo's known categories.",
    )
    p.add_argument("--name", default=None, help="Report name. Defaults to a mode-specific name.")
    p.add_argument("--sample", type=int, default=300, help="Max traces to analyze (limit 1000).")
    # 168h, not 72h: load generation runs in bursts days apart, so a 3-day window can miss
    # most of the population and thin the clusters.
    p.add_argument("--last-n-hours", type=int, default=168, help="Sample traces from this window.")
    p.add_argument("--filter", default=None, help="Extra LangSmith trace filter expression.")
    p.add_argument(
        "--model",
        choices=("openai", "anthropic"),
        default=split_provider(judge_model())[0] if split_provider(judge_model())[0] in ("openai", "anthropic") else "anthropic",
        help="Insights provider. Default: the EVAL_JUDGE_MODEL provider (anthropic).",
    )
    p.add_argument("--cluster-model", default=None, help="Override the thinking/clustering model.")
    p.add_argument("--summary-model", default=None, help="Override the summarization model.")
    p.add_argument(
        "--no-attributes",
        action="store_true",
        help="Skip per-trace attribute extraction (use if the beta schema rejects it).",
    )
    p.add_argument("--save-config", action="store_true", help="Also save the config for reuse.")
    p.add_argument(
        "--schedule-cron",
        default=None,
        help="With --save-config, schedule recurring reports (UTC cron).",
    )
    p.add_argument(
        "--environment",
        choices=ENVIRONMENTS,
        default=None,
        help="Agent environment to analyze. Default: LANGSMITH_AGENT_ENVIRONMENT (production).",
    )
    p.add_argument("--list", action="store_true", help="List existing reports and exit.")
    p.add_argument("--poll", action="store_true", help="Wait for the report to finish.")
    p.add_argument("--poll-timeout-min", type=int, default=35)
    args = p.parse_args()

    if not os.getenv("LANGSMITH_API_KEY"):
        print("Error: LANGSMITH_API_KEY not set.")
        sys.exit(1)
    if not 0 < args.sample <= 1000:
        print("Error: --sample must be between 1 and 1000.")
        sys.exit(1)
    if args.name is None:
        args.name = (
            "How engineers use it - supplied categories"
            if args.mode == "guided"
            else "How engineers use it - discovered"
        )

    headers = _headers()
    session_id = _session_id(Client(), args.environment or agent_environment())

    if args.list:
        list_jobs(headers, session_id)
        return

    config = build_config(args)
    print(f"\nAgent environment '{PROJECT_NAME}' ({session_id})")
    print(f"Mode: {args.mode}" + (f" · {len(PARTITIONS)} predefined categories" if args.mode == "guided" else " · categories discovered bottom-up"))
    print(f"Sample: up to {args.sample} traces from the last {args.last_n_hours}h · model: {args.model}")
    if args.mode == "guided":
        print("NOTE: categories were supplied, not discovered. Say so in the room.")

    if args.save_config:
        print("\nSaving config...")
        save_config(headers, session_id, config, args)

    print("\nStarting report...")
    job_id = run_job(headers, session_id, config)
    if job_id and args.poll:
        poll(headers, session_id, job_id, args.poll_timeout_min)

    print(
        "\nOpen the agent, pick the environment, then open Insights in LangSmith. "
        "Reports can take up to 30 minutes - generate this the day BEFORE a demo."
    )


if __name__ == "__main__":
    main()

"""FastAPI custom routes for the LangGraph agent server.

Mounted at the deployment root via `langgraph.json`'s `http.app` key, so
this app extends the agent server (which already exposes `/threads`,
`/runs`, `/assistants`, etc.) with our own routes:

- `GET  /health`            - simple liveness probe
- `GET  /`                  - redirect to `/agent/`
- `GET  /agent/`            - the built React chat UI (Vite dist/)
- `POST /agent-api/feedback` - record thumbs up/down on a run in LangSmith

If the frontend bundle hasn't been built yet, `/agent/*` returns a
503 with a hint to run `npm install && npm run build` in `frontend/`.

The feedback route keeps `LANGSMITH_API_KEY` server-side: the browser
posts only `{run_id, score, comment}` and this app calls the LangSmith
SDK on its behalf, so the key is never shipped to the client.

NOTE: do NOT add `from __future__ import annotations` here. The custom app's
request models (e.g. FeedbackRequest) must resolve eagerly so the LangGraph
server can build the OpenAPI spec at startup; lazy string annotations raise
PydanticUserError("...is not fully defined").
"""

import pathlib
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from langsmith import Client
from pydantic import BaseModel, Field

from project_advisor.governance import REVIEW_QUEUE_NAME

app = FastAPI()

_langsmith_client = Client()

# Cache the review-queue id so we resolve it at most once per process.
_review_queue_id: str | None = None
_review_queue_looked_up = False


def _get_review_queue_id() -> str | None:
    """Resolve the negative-feedback review queue id by name (cached, best-effort)."""
    global _review_queue_id, _review_queue_looked_up
    if _review_queue_looked_up:
        return _review_queue_id
    _review_queue_looked_up = True
    try:
        for q in _langsmith_client.list_annotation_queues():
            if getattr(q, "name", None) == REVIEW_QUEUE_NAME:
                _review_queue_id = str(q.id)
                break
    except Exception:  # noqa: BLE001 - routing is best-effort
        _review_queue_id = None
    return _review_queue_id


class FeedbackRequest(BaseModel):
    run_id: uuid.UUID
    score: float = Field(ge=0, le=1)
    comment: str | None = None


@app.post("/agent-api/feedback")
def submit_feedback(body: FeedbackRequest) -> dict[str, str]:
    """Attach user thumbs up/down to the LangSmith run that produced a reply.

    `score` is 1 for thumbs-up and 0 for thumbs-down. Passing `trace_id`
    lets the SDK background the write so the request returns immediately.
    """
    try:
        _langsmith_client.create_feedback(
            body.run_id,
            key="user_feedback",
            score=body.score,
            trace_id=body.run_id,
            comment=body.comment,
        )
    except Exception as exc:  # noqa: BLE001 - surface a clean 502 to the client
        raise HTTPException(
            status_code=502, detail=f"Failed to record feedback: {exc}"
        ) from exc

    # Auto-route thumbs-down (score 0) into the review queue - the "notify to
    # review" loop. Best-effort: a routing failure must not fail the feedback write.
    if body.score == 0:
        queue_id = _get_review_queue_id()
        if queue_id:
            try:
                _langsmith_client.add_runs_to_annotation_queue(
                    queue_id, run_ids=[body.run_id]
                )
            except Exception:  # noqa: BLE001
                pass

    return {"status": "ok"}

FRONTEND_BUILD_DIR = (
    pathlib.Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root_redirect() -> RedirectResponse:
    return RedirectResponse(url="/agent/")


@app.get("/agent")
def agent_redirect() -> RedirectResponse:
    return RedirectResponse(url="/agent/")


if FRONTEND_BUILD_DIR.is_dir() and (FRONTEND_BUILD_DIR / "index.html").is_file():
    app.mount(
        "/agent",
        StaticFiles(directory=str(FRONTEND_BUILD_DIR), html=True),
        name="frontend",
    )
else:

    @app.get("/agent/{path:path}")
    def frontend_not_built(path: str = "") -> PlainTextResponse:
        del path
        return PlainTextResponse(
            "Frontend not built. "
            "From the project root, run:\n\n"
            "  npm --prefix frontend install\n"
            "  npm --prefix frontend run build\n",
            status_code=503,
        )

"""
ForgeAI API — Phase 9 of the project spec.

A deliberately small FastAPI layer: start a run, poll its status/events,
submit approval decisions. The dashboard (static/index.html) talks to
this via plain polling rather than websockets — simpler to run and debug
locally, at the cost of ~1s latency on trace updates, which is fine for
a human watching an agent work through a multi-step task.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()  # must run before anything constructs an LLMProvider (GEMINI_API_KEY etc.)

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import app.tools  # noqa: F401  (registers all tools with the shared registry)
from app.api.run_registry import registry

app = FastAPI(title="ForgeAI")

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class StartRunRequest(BaseModel):
    task: str
    repository: str = "."


class ApprovalDecisionRequest(BaseModel):
    approved: bool
    note: str = ""


@app.post("/api/runs")
def start_run(body: StartRunRequest) -> dict:
    record = registry.start(task=body.task, repository=body.repository)
    return {"run_id": record.run_id}


@app.get("/api/runs")
def list_runs() -> list[dict]:
    return registry.list_summaries()


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    record = registry.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "run_id": record.run_id,
        "task": record.task,
        "repository": record.repository,
        "status": record.status,
        "pending_approval": record.pending_approval,
        "result": record.result,
        "error": record.error,
        "event_count": len(record.events),
    }


@app.get("/api/runs/{run_id}/events")
def get_events(run_id: str, since: int = 0) -> list[dict]:
    record = registry.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return record.events[since:]


@app.post("/api/runs/{run_id}/approve")
def submit_approval(run_id: str, body: ApprovalDecisionRequest) -> dict:
    ok = registry.submit_approval(run_id, body.approved, body.note)
    if not ok:
        raise HTTPException(status_code=409, detail="No approval is pending for this run")
    return {"ok": True}


if os.path.isdir(_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/")
    def dashboard() -> FileResponse:
        return FileResponse(os.path.join(_STATIC_DIR, "index.html"))
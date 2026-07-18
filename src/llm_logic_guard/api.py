from __future__ import annotations

import asyncio
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .attacks import SCENARIOS, AttackLab
from .models import LogicTrace, TraceStep
from .runtime import GuardRuntime
from .storage import TraceStore

try:
    from .office_agent import OfficeAgentService
except ImportError as exc:  # pragma: no cover - exercised only without optional deps
    OfficeAgentService = None  # type: ignore[assignment]
    OFFICE_AGENT_IMPORT_ERROR = exc
else:
    OFFICE_AGENT_IMPORT_ERROR = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = PROJECT_ROOT / "demo"
STORE = TraceStore(PROJECT_ROOT / "outputs" / "logicguard.db")
RUNTIME = GuardRuntime(store=STORE)
ATTACK_LAB = AttackLab(PROJECT_ROOT / "outputs" / "attack_lab")
OFFICE_AGENT = (
    OfficeAgentService(
        runtime=RUNTIME,
        sandbox_root=PROJECT_ROOT / "outputs" / "office_tasks",
        checkpoint_path=PROJECT_ROOT / "outputs" / "office_agent_checkpoints.db",
    )
    if OfficeAgentService is not None
    else None
)
SUBSCRIBERS: dict[str, list[asyncio.Queue[dict[str, Any]]]] = {}

app = FastAPI(
    title="LogicGuard API",
    version="1.0.0",
    description="LLM agent consistency verification, proactive risk prediction and runtime enforcement.",
)
app.mount("/console", StaticFiles(directory=DEMO_DIR), name="console")


@app.get("/")
def console() -> FileResponse:
    return FileResponse(DEMO_DIR / "index.html")


@app.get("/api/v1/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "system": "LogicGuard",
        "capabilities": {
            "dsl_monitor": True,
            "claim_graph": True,
            "probabilistic_monitor": True,
            "optional_nli": True,
            "nli_backend": RUNTIME.pipeline.detector.semantic.name,
            "nli_threshold": RUNTIME.pipeline.detector.semantic_threshold,
            "constraint_backend": RUNTIME.pipeline.detector.solver.backend,
            "websocket": True,
            "langgraph_office_agent": OFFICE_AGENT is not None,
            "office_agent_error": str(OFFICE_AGENT_IMPORT_ERROR) if OFFICE_AGENT_IMPORT_ERROR else "",
            "llm_provider": OFFICE_AGENT.provider.name if OFFICE_AGENT else "",
        },
    }


@app.get("/api/v1/scenarios")
def list_scenarios() -> dict[str, Any]:
    return {
        "scenarios": [
            {
                "id": scenario_id,
                "attack_type": item["attack_type"],
                "goal": item["goal"],
                "candidate_action": item["candidate_action"],
                "is_attack": scenario_id in {
                    "prompt_injection", "memory_poisoning", "environment_pollution",
                },
            }
            for scenario_id, item in SCENARIOS.items()
        ]
    }


@app.post("/api/v1/traces")
async def create_trace(payload: dict[str, Any]) -> dict[str, Any]:
    trace = LogicTrace.from_dict({
        **payload,
        "trace_id": payload.get("trace_id") or f"trace-{uuid.uuid4().hex[:12]}",
    })
    STORE.create_trace(trace)
    await _broadcast(trace.trace_id, {"type": "trace_created", "trace": asdict(trace)})
    return asdict(trace)


@app.post("/api/v1/traces/{trace_id}/events")
async def submit_event(trace_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        event = TraceStep.from_dict({
            **payload,
            "step_id": payload.get("step_id") or payload.get("event_id") or f"evt-{uuid.uuid4().hex[:10]}",
        })
        decision = RUNTIME.evaluate_event(trace_id, event)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    body = decision.to_dict()
    await _broadcast(trace_id, {"type": "guard_decision", "decision": body})
    return body


@app.post("/api/v1/traces/{trace_id}/repair")
async def repair_trace(trace_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        result = RUNTIME.repair(trace_id, (payload or {}).get("event_id"))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    await _broadcast(trace_id, {"type": "repair", "repair": result})
    return result


@app.get("/api/v1/traces/{trace_id}")
def get_trace(trace_id: str) -> dict[str, Any]:
    trace = STORE.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="trace not found")
    report = RUNTIME.pipeline.run(trace)
    return {
        "trace": asdict(trace),
        "analysis": report.to_dict(),
        "decisions": STORE.get_decisions(trace_id),
    }


@app.post("/api/v1/attacks/run")
async def run_attack(payload: dict[str, Any]) -> dict[str, Any]:
    scenario_id = str(payload.get("scenario_id") or "prompt_injection")
    if scenario_id not in SCENARIOS:
        raise HTTPException(status_code=400, detail="unknown scenario")
    result = ATTACK_LAB.run(
        scenario_id,
        defended=bool(payload.get("defended", True)),
    )
    body = result.to_dict()
    await _broadcast(result.trace_id, {"type": "attack_complete", "result": body})
    return body


@app.get("/api/v1/experiments/{experiment_id}")
def get_experiment(experiment_id: str) -> dict[str, Any]:
    result = STORE.get_experiment(experiment_id)
    if result is None:
        for path in (PROJECT_ROOT / "outputs" / "attack_lab").glob("*.db"):
            store = TraceStore(path)
            candidate = store.get_experiment(experiment_id)
            store.close()
            if candidate:
                return candidate
        raise HTTPException(status_code=404, detail="experiment not found")
    return result


@app.post("/api/v1/agents/tasks")
async def run_office_task(payload: dict[str, Any]) -> dict[str, Any]:
    if OFFICE_AGENT is None:
        raise HTTPException(
            status_code=503,
            detail="Office agent requires optional LangGraph dependencies.",
        )
    user_goal = str(payload.get("user_goal") or payload.get("task") or "").strip()
    if not user_goal:
        raise HTTPException(status_code=400, detail="user_goal is required")
    result = await asyncio.to_thread(
        OFFICE_AGENT.run_task,
        user_goal,
        task_id=payload.get("task_id"),
        context=dict(payload.get("context") or {}),
        seed_files=dict(payload.get("seed_files") or {}),
        guard_enabled=bool(payload.get("guard_enabled", True)),
    )
    await _broadcast(
        result["trace_id"],
        {"type": "office_task_update", "task": result},
    )
    return result


@app.get("/api/v1/agents/tasks/{task_id}")
def get_office_task(task_id: str) -> dict[str, Any]:
    if OFFICE_AGENT is None:
        raise HTTPException(
            status_code=503,
            detail="Office agent requires optional LangGraph dependencies.",
        )
    result = OFFICE_AGENT.get_task(task_id)
    if result is None:
        raise HTTPException(status_code=404, detail="office task not found")
    return result


@app.post("/api/v1/agents/tasks/{task_id}/confirm")
async def confirm_office_task(
    task_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if OFFICE_AGENT is None:
        raise HTTPException(
            status_code=503,
            detail="Office agent requires optional LangGraph dependencies.",
        )
    existing = OFFICE_AGENT.get_task(task_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="office task not found")
    if not existing.get("awaiting_confirmation"):
        raise HTTPException(status_code=409, detail="task is not awaiting confirmation")
    result = await asyncio.to_thread(
        OFFICE_AGENT.resume_task,
        task_id,
        approved=bool(payload.get("approved")),
        note=str(payload.get("note") or ""),
    )
    await _broadcast(
        result["trace_id"],
        {"type": "office_task_update", "task": result},
    )
    return result


@app.websocket("/ws/traces/{trace_id}")
async def trace_stream(websocket: WebSocket, trace_id: str) -> None:
    await websocket.accept()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    SUBSCRIBERS.setdefault(trace_id, []).append(queue)
    try:
        await websocket.send_json({"type": "connected", "trace_id": trace_id})
        while True:
            await websocket.send_json(await queue.get())
    except WebSocketDisconnect:
        pass
    finally:
        SUBSCRIBERS.get(trace_id, []).remove(queue)


async def _broadcast(trace_id: str, event: dict[str, Any]) -> None:
    for queue in SUBSCRIBERS.get(trace_id, []):
        await queue.put(event)

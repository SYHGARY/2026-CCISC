from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import GuardDecision, LogicTrace, TraceStep


class TraceStore:
    def __init__(self, path: Path | str = "outputs/logicguard.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # Per-trace next-sequence cache, seeded lazily from the DB. Avoids a
        # full COUNT(*) on every append (previously O(events) per append).
        self._next_sequence: dict[str, int] = {}
        self._init_schema()

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    user_goal TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    trace_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    sequence_no INTEGER NOT NULL,
                    event_json TEXT NOT NULL,
                    PRIMARY KEY (trace_id, event_id)
                );
                CREATE TABLE IF NOT EXISTS decisions (
                    trace_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    decision_json TEXT NOT NULL,
                    PRIMARY KEY (trace_id, event_id)
                );
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    result_json TEXT NOT NULL
                );
                -- Supports get_trace's `WHERE trace_id = ? ORDER BY sequence_no`
                -- without a scan+sort on the cold-load path.
                CREATE INDEX IF NOT EXISTS idx_events_trace_seq
                    ON events (trace_id, sequence_no);
                """
            )

    def create_trace(self, trace: LogicTrace) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO traces VALUES (?, ?, ?, ?, ?)",
                (
                    trace.trace_id,
                    trace.user_goal,
                    trace.session_id or trace.trace_id,
                    trace.status,
                    json.dumps(trace.metadata, ensure_ascii=False),
                ),
            )
            for step in trace.steps:
                self.append_event(trace.trace_id, step)

    def append_event(self, trace_id: str, event: TraceStep) -> None:
        with self._lock, self.conn:
            sequence = self._next_sequence.get(trace_id)
            if sequence is None:
                row = self.conn.execute(
                    "SELECT COALESCE(MAX(sequence_no), -1) + 1 FROM events WHERE trace_id = ?",
                    (trace_id,),
                ).fetchone()
                sequence = int(row[0])
            self.conn.execute(
                "INSERT OR REPLACE INTO events VALUES (?, ?, ?, ?)",
                (trace_id, event.step_id, sequence, json.dumps(asdict(event), ensure_ascii=False)),
            )
            self._next_sequence[trace_id] = sequence + 1

    def save_decision(self, decision: GuardDecision) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO decisions VALUES (?, ?, ?)",
                (decision.trace_id, decision.event_id, json.dumps(decision.to_dict(), ensure_ascii=False)),
            )

    def get_trace(self, trace_id: str) -> LogicTrace | None:
        row = self.conn.execute("SELECT * FROM traces WHERE trace_id = ?", (trace_id,)).fetchone()
        if row is None:
            return None
        events = self.conn.execute(
            "SELECT event_json FROM events WHERE trace_id = ? ORDER BY sequence_no", (trace_id,)
        ).fetchall()
        return LogicTrace(
            trace_id=trace_id,
            user_goal=row["user_goal"],
            steps=[TraceStep.from_dict(json.loads(event["event_json"])) for event in events],
            metadata=json.loads(row["metadata_json"]),
            session_id=row["session_id"],
            status=row["status"],
        )

    def get_decisions(self, trace_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT decision_json FROM decisions WHERE trace_id = ? ORDER BY rowid", (trace_id,)
        ).fetchall()
        return [json.loads(row["decision_json"]) for row in rows]

    def save_experiment(self, experiment_id: str, result: dict[str, Any]) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO experiments VALUES (?, ?)",
                (experiment_id, json.dumps(result, ensure_ascii=False)),
            )

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT result_json FROM experiments WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()
        return json.loads(row["result_json"]) if row else None

    def close(self) -> None:
        with self._lock:
            self.conn.close()

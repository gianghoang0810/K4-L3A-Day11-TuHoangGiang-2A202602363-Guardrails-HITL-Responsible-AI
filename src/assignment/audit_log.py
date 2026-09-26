"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


import time


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store input + start timestamp keyed by request_id/user_id."""
        key = request_id or f"{user_id}_{time.time()}"
        self._open[key] = {
            "start_time": time.time(),
            "timestamp": utc_now_iso(),
            "user_id": user_id,
            "input": text,
            "request_id": request_id,
        }
        return key

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store output, layer decision, latency; append to self.logs."""
        key = request_id or user_id
        entry = self._open.pop(key, None)
        if entry is None and self._open:
            # Pop most recent matching user_id if exact key not found
            for k in reversed(list(self._open.keys())):
                if self._open[k]["user_id"] == user_id:
                    entry = self._open.pop(k)
                    break

        now = time.time()
        latency_ms = round((now - entry["start_time"]) * 1000, 2) if entry else 0.0

        record = {
            "request_id": request_id or (entry["request_id"] if entry else None),
            "user_id": user_id,
            "timestamp": entry["timestamp"] if entry else utc_now_iso(),
            "input": entry["input"] if entry else "",
            "output": text,
            "blocked": blocked,
            "layer": layer,
            "latency_ms": latency_ms,
        }
        self.logs.append(record)
        return record

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        target_path = Path(filepath) if filepath else Path(default_audit_log_path())
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json.dumps(self.logs, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(target_path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

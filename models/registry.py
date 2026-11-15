"""
Zephyr – Experiment registry
Lightweight, append-only JSONL registry of experiment runs.

No database: each run is one JSON line in ``artifacts/experiments/experiments.jsonl``.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd

from config import EXPERIMENTS_DIR

EXPERIMENTS_FILE = EXPERIMENTS_DIR / "experiments.jsonl"


def _ensure_file() -> None:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


def log_experiment(record: Dict[str, Any]) -> str:
    """
    Append a single experiment record to the JSONL registry.

    Injects ``run_id`` (uuid4 hex) and ``timestamp`` (UTC ISO) if absent.
    Returns the run_id.
    """
    _ensure_file()

    run_id = record.get("run_id") or uuid.uuid4().hex
    timestamp = record.get("timestamp") or datetime.now(timezone.utc).isoformat()

    payload = {"run_id": run_id, "timestamp": timestamp}
    payload.update(record)
    payload["run_id"] = run_id
    payload["timestamp"] = timestamp

    with EXPERIMENTS_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, default=str) + "\n")

    return run_id


def load_experiments() -> pd.DataFrame:
    """
    Load all experiment records into a DataFrame, newest first.
    Returns an empty DataFrame if no records exist.
    """
    if not EXPERIMENTS_FILE.exists():
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    with EXPERIMENTS_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    return df


def clear_experiments() -> int:
    """
    Remove all experiment records. Returns the number of rows removed.
    """
    if not EXPERIMENTS_FILE.exists():
        return 0

    count = sum(1 for line in EXPERIMENTS_FILE.read_text(encoding="utf-8").splitlines() if line.strip())
    EXPERIMENTS_FILE.write_text("", encoding="utf-8")
    return count


def summarize_experiments(
    df: pd.DataFrame,
    filters: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Apply simple column-equality filters and return a copy.
    """
    if df.empty:
        return df

    out = df.copy()
    for key, value in (filters or {}).items():
        if value is None or value == "" or value == "All":
            continue
        if key in out.columns:
            out = out[out[key] == value]
    return out.reset_index(drop=True)
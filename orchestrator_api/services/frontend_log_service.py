# services/frontend_log_service.py
import json
import eventlet
from pathlib import Path

_session_locks = {}
_session_state = {}


def _get_lock(session_id: str):
    lock = _session_locks.get(session_id)
    if lock is None:
        lock = eventlet.semaphore.Semaphore(1)
        _session_locks[session_id] = lock
    return lock


def save_log_batch(base_dir: Path, batch: dict) -> tuple[int, int]:
    session_id = batch["session_id"]
    user_id = batch.get("user_id")
    condition = batch.get("condition")
    events = batch.get("events", [])

    session_dir = base_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    out_path = session_dir / "events.jsonl"

    lock = _get_lock(session_id)
    with lock:
        state = _session_state.setdefault(session_id, {"acked_to_seq": 0, "pending": set()})
        acked = int(state["acked_to_seq"])
        pending = state["pending"]

        rows_written = 0
        with out_path.open("a", encoding="utf-8") as f:
            for ev in events:
                try:
                    seq = int(ev.get("seq"))
                except Exception:
                    continue

                if seq <= acked or seq in pending:
                    continue

                pending.add(seq)

                row = {
                    "session_id": session_id,
                    "user_id": user_id,
                    "condition": condition,
                    "seq": seq,
                    "ts_unix_ms": ev.get("ts_unix_ms"),
                    "event_name": ev.get("event_name"),
                    "payload_json": ev.get("payload_json"),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows_written += 1

        while (acked + 1) in pending:
            acked += 1
            pending.remove(acked)

        state["acked_to_seq"] = acked

    return acked, rows_written

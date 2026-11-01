"""Per-call receipts and stage totals from the run folder, without model calls."""
from datetime import datetime
import math
from pathlib import Path

from herogen.pipeline.client import call_cost
from herogen.pipeline.runfolder import RunFolder


def _peak_concurrent(rows):
    """Maximum overlap of recorded positive-duration call intervals; touching endpoints do not overlap."""
    events = []
    for row in rows:
        try:
            start = datetime.fromisoformat(row["started"]).timestamp()
            seconds = float(row["wall_seconds"])
        except (TypeError, ValueError, OverflowError):
            continue
        if seconds > 0 and math.isfinite(start) and math.isfinite(seconds):
            events.extend(((start, 1), (start + seconds, -1)))
    current = peak = 0
    for _, delta in sorted(events):
        current += delta
        peak = max(peak, current)
    return peak


def report(path: Path, *, stream=None):
    import json
    import sys

    path = Path(path)
    if not path.is_dir():
        raise ValueError("report needs an existing run directory")
    calls, stages = [], []
    for file in path.glob("stage-*.json"):
        rec = json.loads(file.read_text())
        rows = []
        for source in rec.get("calls", []):
            row = {**source, "stage": rec["stage"],
                   "purpose": source.get("purpose", rec["name"]),
                   "model": source.get("model", source.get("runner", "unknown")),
                   "kind": source.get("kind", "gpu"),
                   "started": source.get("started", rec.get("started", "")),
                   "wall_seconds": source.get("wall_seconds", 0.0),
                   "cost_usd": (0.0 if source.get("cached") or source.get("cost_usd", 0) is None
                                else call_cost(source)),
                   "cost_unknown": not source.get("cached") and source.get("cost_usd", 0) is None,
                   "cached": bool(source.get("cached")), "outcome": source.get("outcome", "ok")}
            rows.append(row)
        calls.extend(rows)
        stages.append({"stage": rec["stage"], "started": rec.get("started", ""), "calls": len(rows),
                       "call_seconds": round(sum(r["wall_seconds"] for r in rows), 6),
                       "seconds": rec.get("seconds") or 0.0,
                       "wall_seconds": rec.get("seconds") or 0.0,
                       "peak_concurrent_calls": _peak_concurrent(rows),
                       "cost_usd": round(sum(r["cost_usd"] for r in rows), 6),
                       "status": rec.get("status", "complete"), "error": rec.get("error"),
                       "unknown_cost_calls": sum(r["cost_unknown"] for r in rows)})
    calls.sort(key=lambda r: r["started"])
    stages.sort(key=lambda r: (r["started"], str(r["stage"])))
    total = {key: round(sum(s[key] for s in stages), 6)
             for key in ("calls", "seconds", "wall_seconds", "call_seconds", "cost_usd", "unknown_cost_calls")}
    total["peak_concurrent_calls"] = _peak_concurrent(calls)
    data = {"calls": calls, "stages": stages, "total": total, "cost_is_estimate": True}
    RunFolder(path.parent, path.name).write_json("fast-report.json", data)
    out = stream or sys.stdout
    print("stage  purpose  model  kind  started  seconds  USD  cached  outcome", file=out)
    for r in calls:
        print(f"{r['stage']}  {r['purpose']}  {r['model']}  {r['kind']}  {r['started']}  "
              f"{r['wall_seconds']:.6f}  {r['cost_usd']:.6f}  {r['cached']}  {r['outcome']}", file=out)
    print("\nstage  calls  wall seconds  sum call seconds  peak concurrent calls  USD", file=out)
    for s in [*stages, {"stage": "TOTAL", **total}]:
        print(f"{s['stage']}  {s['calls']}  {s['wall_seconds']:.2f}  {s['call_seconds']:.6f}  "
              f"{s['peak_concurrent_calls']}  "
              f"{s['cost_usd']:.6f}", file=out)
    if total["unknown_cost_calls"]:
        print(f"Unknown list price for {total['unknown_cost_calls']} call(s); USD totals include known costs only.",
              file=out)
    return data

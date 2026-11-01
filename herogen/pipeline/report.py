"""Per-call receipts and stage totals from the run folder, without model calls."""
from pathlib import Path

from herogen.pipeline.client import call_cost
from herogen.pipeline.runfolder import RunFolder


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
                       "seconds": round(sum(r["wall_seconds"] for r in rows), 6),
                       "elapsed_seconds": rec.get("seconds") or 0.0,
                       "cost_usd": round(sum(r["cost_usd"] for r in rows), 6),
                       "status": rec.get("status", "complete"), "error": rec.get("error"),
                       "unknown_cost_calls": sum(r["cost_unknown"] for r in rows)})
    calls.sort(key=lambda r: r["started"])
    stages.sort(key=lambda r: (r["started"], str(r["stage"])))
    total = {key: round(sum(s[key] for s in stages), 6)
             for key in ("calls", "seconds", "elapsed_seconds", "cost_usd", "unknown_cost_calls")}
    data = {"calls": calls, "stages": stages, "total": total, "cost_is_estimate": True}
    RunFolder(path.parent, path.name).write_json("fast-report.json", data)
    out = stream or sys.stdout
    print("stage  purpose  model  kind  started  seconds  USD  cached  outcome", file=out)
    for r in calls:
        print(f"{r['stage']}  {r['purpose']}  {r['model']}  {r['kind']}  {r['started']}  "
              f"{r['wall_seconds']:.6f}  {r['cost_usd']:.6f}  {r['cached']}  {r['outcome']}", file=out)
    print("\nstage  calls  call seconds  elapsed seconds  USD", file=out)
    for s in [*stages, {"stage": "TOTAL", **total}]:
        print(f"{s['stage']}  {s['calls']}  {s['seconds']:.6f}  {s['elapsed_seconds']:.2f}  "
              f"{s['cost_usd']:.6f}", file=out)
    if total["unknown_cost_calls"]:
        print(f"Unknown list price for {total['unknown_cost_calls']} call(s); USD totals include known costs only.",
              file=out)
    return data

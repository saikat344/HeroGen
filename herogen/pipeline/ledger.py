"""Durable, append-only call receipts; a missing completion always retains the sent bound."""
from __future__ import annotations

from datetime import datetime
import json
import math
import os
from pathlib import Path
import secrets
import time

from herogen.pipeline import filelock

LEDGER = "calls-ledger.jsonl"
BRIDGE = "HEROGEN_CALL_LEDGER"


def append(cache_dir: Path, row: dict) -> None:
    row = {"time": datetime.now().astimezone().isoformat(timespec="milliseconds"), **row}
    # The service acknowledgement is durable even when a stage's local working copy is lost.
    from herogen.service.refine_budget import bridge_enabled, remote_budget
    if bridge_enabled(BRIDGE):
        remote_budget("call", row=row, prefix=BRIDGE)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / LEDGER, "a") as stream, filelock.exclusive(stream.fileno()):
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def sent(cache_dir: Path, *, model: str, endpoint: str, key: str, timeout: float,
         price_usd: float, **extra) -> dict:
    row = {"call_id": secrets.token_hex(16), "model": model, "endpoint": endpoint, "key": key,
           "outcome": "sent", "deadline": time.time() + timeout, "price_usd": price_usd, **extra}
    append(cache_dir, row)
    return row


def complete(cache_dir: Path, receipt: dict, *, outcome: str, cost_usd: float, **extra) -> None:
    append(cache_dir, {**receipt, "outcome": outcome, "cost_usd": cost_usd, **extra})


def bound(rows: list[dict]) -> tuple[int, int, int, float]:
    """Return (micro-USD rounded up, calls, full-price calls, latest deadline), or refuse incomplete evidence."""
    from decimal import Decimal, ROUND_CEILING
    pending, completed = {}, {}
    for row in rows:
        call_id = row["call_id"]
        if row["outcome"] == "sent":
            if call_id in pending:
                raise ValueError("Duplicate sent receipt")
            pending[call_id] = row
        else:
            if call_id not in pending or call_id in completed:
                raise ValueError("Completion without a unique sent receipt")
            if row['outcome'] not in ('ok', 'failed', 'unknown', 'unreadable'):
                raise ValueError('Unsupported completion outcome')
            if any(row.get(k) != pending[call_id].get(k) for k in
                   ('model', 'endpoint', 'key', 'deadline', 'price_usd', 'timeout_s', 'price_per_second')):
                raise ValueError('Completion does not match its sent receipt')
            completed[call_id] = row
    total, full, deadline = Decimal(0), 0, 0.0
    for call_id, row in pending.items():
        price, end = row["price_usd"], row["deadline"]
        if row['endpoint'].startswith('modal/'):
            if not all(type(row[k]) in (int, float) and math.isfinite(row[k]) and row[k] >= 0
                       for k in ('timeout_s', 'price_per_second')):
                raise ValueError('Invalid Modal bound')
            price = row['timeout_s'] * row['price_per_second']
        if not all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in (price, end)):
            raise ValueError("Invalid call bound")
        deadline = max(deadline, end)
        result = completed.get(call_id)
        if result is None or result["outcome"] in ("unknown", "unreadable") or result.get('unknown'):
            cost = price
            if result is not None and 'cost_usd' in result:
                known = result['cost_usd']
                if type(known) not in (float, int) or not math.isfinite(known) or known < 0:
                    raise ValueError('Invalid partial cost')
                cost = max(cost, known)
            full += 1
        else:
            cost = result["cost_usd"]
            if type(cost) not in (float, int) or not math.isfinite(cost) or cost < 0:
                raise ValueError("Invalid completed cost")
        total += Decimal(str(cost))
    return int((total * 1_000_000).to_integral_value(rounding=ROUND_CEILING)), len(pending), full, deadline

"""First-party, privacy-preserving website analytics and admin reporting.

Client events store only a tab-scoped random visitor id (`sessionStorage`), coarse device class (`mobile` or
`desktop`), UTC day, timestamp, allow-listed event name and at most one allow-listed short enum or gallery/starter
slug. Never store IP addresses, raw User-Agent headers, email addresses, prompt text, character names, file names or
image data. Pipeline, job, user-kind and budget metrics are computed directly from the existing metadata tables.
"""

from __future__ import annotations

from decimal import Decimal
import math
import re
import time
from typing import Any

from herogen.service.runner import STAGE_LABELS

SCHEMA = """
CREATE TABLE IF NOT EXISTS analytics_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  at REAL NOT NULL,
  day TEXT NOT NULL,
  visitor_id TEXT NOT NULL,
  device TEXT NOT NULL,
  event TEXT NOT NULL,
  prop_key TEXT NOT NULL DEFAULT '',
  prop_val TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS analytics_events_at ON analytics_events(at);
CREATE INDEX IF NOT EXISTS analytics_events_day_event ON analytics_events(day, event);
"""

PUBLIC = {"analytics_record", "analytics_report"}

MAX_BATCH = 25
MIN_DAYS = 1
MAX_DAYS = 90
DEFAULT_DAYS = 7

VISITOR_RE = re.compile(r"^[0-9a-f]{32}$")
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")

DEVICES = ("mobile", "desktop")
PAGES = ("landing", "gallery_detail", "workbench", "legal")
SUGGESTIONS = ("forest", "clockwork", "moon")
STARTERS = ("trail-keeper", "little-inventor", "cloud-sailor", "garden-guardian")
SHADES = ("textured", "clay", "wire")
FORMATS = ("glb", "roblox")
THEMES = ("light", "dark", "system")
SURFACES = ("landing", "gallery_detail", "workbench")
CREATE_RESULTS = (
    "accepted",
    "generation_paused",
    "generation_budget_exhausted",
    "conflict",
    "busy",
    "upload_rejected",
    "upload_too_large",
    "no_file",
    "empty_prompt",
    "prompt_too_long",
    "bad_kind",
    "bad_profile",
    "bad_form",
    "multipart",
    "media_type",
    "too_large",
    "idempotency_key",
    "rate_limited",
    "signed_out",
    "signin_required",
    "public_access_closed",
    "csrf",
    "account",
    "storage_full",
    "bad_request",
    "internal",
    "network_error",
)

MAIN_STAGE_ORDER = ("p", "0", "1", "2", "3", "3f", "4", "6", "5")
FINAL_JOB_STATES = ("succeeded", "rejected", "failed", "canceled")
ACTIVE_JOB_STATES = ("queued", "running")

_NO_PROPS = frozenset({"upload_selected", "idea_typed", "create_clicked", "result_viewed"})
_ENUM_PROPS: dict[str, tuple[str, tuple[str, ...]]] = {
    "page_view": ("page", PAGES),
    "suggestion_click": ("id", SUGGESTIONS),
    "starter_select": ("id", STARTERS),
    "create_result": ("result", CREATE_RESULTS),
    "shade_toggle": ("mode", SHADES),
    "download_click": ("format", FORMATS),
    "theme_change": ("theme", THEMES),
}


def validate_batch(payload: Any) -> tuple[str, str, list[tuple[str, str, str]]]:
    """Validate a `/api/events` JSON payload; raises `ValueError` on any unexpected field or value."""
    if not isinstance(payload, dict) or set(payload.keys()) != {"visitor_id", "device", "events"}:
        raise ValueError("bad_envelope")
    vid = payload["visitor_id"]
    device = payload["device"]
    events = payload["events"]
    if not isinstance(vid, str) or not VISITOR_RE.fullmatch(vid):
        raise ValueError("bad_visitor_id")
    if not isinstance(device, str) or device not in DEVICES:
        raise ValueError("bad_device")
    if not isinstance(events, list) or not (1 <= len(events) <= MAX_BATCH):
        raise ValueError("bad_batch_size")
    cleaned: list[tuple[str, str, str]] = []
    for item in events:
        if not isinstance(item, dict) or "name" not in item or not (set(item.keys()) <= {"name", "props"}):
            raise ValueError("bad_event_shape")
        name = item["name"]
        props = item.get("props", {})
        if not isinstance(name, str) or not isinstance(props, dict):
            raise ValueError("bad_event_types")
        if name in _NO_PROPS:
            if props:
                raise ValueError("unexpected_props")
            cleaned.append((name, "", ""))
        elif name in _ENUM_PROPS:
            key, allowed = _ENUM_PROPS[name]
            if set(props.keys()) != {key}:
                raise ValueError("bad_prop_keys")
            val = props[key]
            if not isinstance(val, str) or val not in allowed:
                raise ValueError("bad_prop_value")
            cleaned.append((name, key, val))
        elif name == "picker_select":
            if set(props.keys()) != {"id"} or not isinstance(props["id"], str) or not SLUG_RE.fullmatch(props["id"]):
                raise ValueError("bad_picker_id")
            cleaned.append((name, "id", props["id"]))
        elif name == "inspect_saved_run":
            if not props:
                cleaned.append((name, "", ""))
            elif set(props.keys()) == {"id"} and isinstance(props["id"], str) and SLUG_RE.fullmatch(props["id"]):
                cleaned.append((name, "id", props["id"]))
            else:
                raise ValueError("bad_inspect_props")
        elif name == "viewer_drag":
            if not props:
                cleaned.append((name, "", ""))
            elif set(props.keys()) == {"surface"} and isinstance(props["surface"], str) and props["surface"] in SURFACES:
                cleaned.append((name, "surface", props["surface"]))
            else:
                raise ValueError("bad_drag_props")
        else:
            raise ValueError("unknown_event")
    return vid, device, cleaned


def micro_to_usd(micro: int | None) -> float | None:
    if micro is None:
        return None
    return round(micro / 1_000_000, 6)


def format_usd_plain(micro: int) -> str:
    val = Decimal(micro) / Decimal(1_000_000)
    if val == val.to_integral_value():
        return f"${int(val):,}"
    cents = val * 100
    if cents == cents.to_integral_value():
        return f"${val:,.2f}"
    return f"${val.normalize():f}"


def _percentile(sorted_vals: list[float], q: float, digits: int) -> float | None:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return round(sorted_vals[0], digits)
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return round(sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo]), digits)


def _finite(values: list[Any]) -> list[float]:
    return sorted(float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v))


class AnalyticsStore:
    def analytics_record(self, visitor_id: str, device: str, items: list[tuple[str, str, str]]) -> int:
        if not isinstance(visitor_id, str) or not VISITOR_RE.fullmatch(visitor_id) or device not in DEVICES:
            raise ValueError("invalid visitor or device")
        if not items or len(items) > MAX_BATCH:
            raise ValueError("invalid event batch")
        now = self.clock()
        day = time.strftime("%Y-%m-%d", time.gmtime(now))
        with self._tx() as c:
            c.executemany(
                "INSERT INTO analytics_events(at, day, visitor_id, device, event, prop_key, prop_val) "
                "VALUES (?,?,?,?,?,?,?)",
                [(now, day, visitor_id, device, ev, pk, pv) for ev, pk, pv in items],
            )
        return len(items)

    def analytics_report(self, days: int = DEFAULT_DAYS) -> dict[str, Any]:
        if type(days) is not int or not (MIN_DAYS <= days <= MAX_DAYS):
            raise ValueError(f"days must be an integer from {MIN_DAYS} to {MAX_DAYS}")
        with self.lock:
            now = self.clock()
            since = now - days * 86400.0
            today_epoch = int(now // 86400) * 86400
            base_days = [time.strftime("%Y-%m-%d", time.gmtime(today_epoch - i * 86400))
                         for i in range(days - 1, -1, -1)]

            ev_rows = self._all(
                "SELECT day, visitor_id, device, event, prop_key, prop_val FROM analytics_events WHERE at>=?",
                (since,),
            )

            visitors_all: set[str] = set()
            day_visitors: dict[str, set[str]] = {d: set() for d in base_days}
            day_views: dict[str, int] = {d: 0 for d in base_days}
            page_counts: dict[str, int] = {p: 0 for p in PAGES}
            dev_visitors: dict[str, set[str]] = {d: set() for d in DEVICES}
            dev_events: dict[str, int] = {d: 0 for d in DEVICES}

            suggestion_by_id: dict[str, int] = {s: 0 for s in SUGGESTIONS}
            starter_by_id: dict[str, int] = {s: 0 for s in STARTERS}
            upload_selects = 0
            idea_typed = 0
            create_clicked = 0
            refusal_counts: dict[str, int] = {}
            result_viewed = 0
            download_by_fmt: dict[str, int] = {f: 0 for f in FORMATS}
            picker_counts: dict[str, int] = {}
            shade_counts: dict[str, int] = {m: 0 for m in SHADES}
            inspect_clicks = 0
            viewer_drags = 0
            theme_counts: dict[str, int] = {t: 0 for t in THEMES}

            for r in ev_rows:
                d = r["day"]
                vid = r["visitor_id"]
                dev = r["device"]
                ev = r["event"]
                pv = r["prop_val"]
                visitors_all.add(vid)
                day_visitors.setdefault(d, set()).add(vid)
                day_views.setdefault(d, 0)
                if dev in dev_visitors:
                    dev_visitors[dev].add(vid)
                    dev_events[dev] += 1
                if ev == "page_view":
                    day_views[d] += 1
                    if pv in page_counts:
                        page_counts[pv] += 1
                elif ev == "suggestion_click" and pv in suggestion_by_id:
                    suggestion_by_id[pv] += 1
                elif ev == "starter_select" and pv in starter_by_id:
                    starter_by_id[pv] += 1
                elif ev == "upload_selected":
                    upload_selects += 1
                elif ev == "idea_typed":
                    idea_typed += 1
                elif ev == "create_clicked":
                    create_clicked += 1
                elif ev == "create_result" and pv and pv != "accepted":
                    refusal_counts[pv] = refusal_counts.get(pv, 0) + 1
                elif ev == "result_viewed":
                    result_viewed += 1
                elif ev == "download_click" and pv in download_by_fmt:
                    download_by_fmt[pv] += 1
                elif ev == "picker_select" and pv:
                    picker_counts[pv] = picker_counts.get(pv, 0) + 1
                elif ev == "shade_toggle" and pv in shade_counts:
                    shade_counts[pv] += 1
                elif ev == "inspect_saved_run":
                    inspect_clicks += 1
                elif ev == "viewer_drag":
                    viewer_drags += 1
                elif ev == "theme_change" and pv in theme_counts:
                    theme_counts[pv] += 1

            series_days = sorted(day_views.keys())
            total_page_views = sum(day_views.values())
            suggestion_clicks = sum(suggestion_by_id.values())
            starter_selects = sum(starter_by_id.values())
            idea_started = suggestion_clicks + starter_selects + upload_selects + idea_typed
            download_clicks = sum(download_by_fmt.values())

            budget_rep = self.generation_get()
            job_rows = self._all(
                "SELECT j.id, j.state, j.seconds, j.cost_usd, COALESCE(u.kind, 'account') AS user_kind "
                "FROM jobs j LEFT JOIN users u ON u.id = j.user_id WHERE j.created_at>=?",
                (since,),
            )
            stage_rows = self._all(
                "SELECT js.job_id, js.stage, js.state, js.seconds, js.cost_usd "
                "FROM job_stages js JOIN jobs j ON j.id = js.job_id WHERE j.created_at>=? AND js.stage!='2h'",
                (since,),
            )

        state_counts: dict[str, int] = {}
        user_kind_jobs: dict[str, int] = {"account": 0, "guest": 0}
        stage_cost_by_job: dict[str, float] = {}
        stages_by_key: dict[str, list[dict[str, Any]]] = {}
        for sr in stage_rows:
            stages_by_key.setdefault(sr["stage"], []).append(sr)
            c_val = sr["cost_usd"]
            if isinstance(c_val, (int, float)) and not isinstance(c_val, bool) and math.isfinite(c_val):
                stage_cost_by_job[sr["job_id"]] = stage_cost_by_job.get(sr["job_id"], 0.0) + float(c_val)

        raw_job_secs: list[Any] = []
        raw_job_costs: list[Any] = []
        for jr in job_rows:
            st = jr["state"]
            state_counts[st] = state_counts.get(st, 0) + 1
            uk = "guest" if jr["user_kind"] == "guest" else "account"
            user_kind_jobs[uk] += 1
            if st in FINAL_JOB_STATES and jr["seconds"] is not None:
                raw_job_secs.append(jr["seconds"])
            if jr["cost_usd"] is not None:
                raw_job_costs.append(jr["cost_usd"])
            elif jr["id"] in stage_cost_by_job:
                raw_job_costs.append(round(stage_cost_by_job[jr["id"]], 6))

        job_secs = _finite(raw_job_secs)
        job_costs = _finite(raw_job_costs)
        accepted_jobs = len(job_rows)
        completed_jobs = state_counts.get("succeeded", 0)

        def build_stage(key: str) -> dict[str, Any]:
            rows = [r for r in stages_by_key.get(key, []) if r["state"] != "pending"]
            counts: dict[str, int] = {}
            for r in rows:
                counts[r["state"]] = counts.get(r["state"], 0) + 1
            secs = _finite([r["seconds"] for r in rows if r["seconds"] is not None])
            costs = _finite([r["cost_usd"] for r in rows if r["cost_usd"] is not None])
            return {
                "stage": key,
                "label": STAGE_LABELS.get(key, key),
                "runs": len(rows),
                "passed": counts.get("passed", 0),
                "failed": counts.get("failed", 0),
                "rejected": counts.get("rejected", 0),
                "unavailable": counts.get("unavailable", 0),
                "skipped": counts.get("skipped", 0),
                "canceled": counts.get("canceled", 0),
                "running": counts.get("running", 0),
                "median_seconds": _percentile(secs, 0.5, 2),
                "p90_seconds": _percentile(secs, 0.9, 2),
                "mean_cost_usd": round(sum(costs) / len(costs), 6) if costs else None,
                "total_cost_usd": round(sum(costs), 6),
            }

        return {
            "window": {"days": days, "since": round(since, 3), "until": round(now, 3)},
            "daily": {
                "visitors": len(visitors_all),
                "page_views": total_page_views,
                "by_page": page_counts,
                "series": [
                    {"day": d, "visitors": len(day_visitors[d]), "page_views": day_views[d]}
                    for d in series_days
                ],
            },
            "funnel": {
                "steps": [
                    {"key": "landing_view", "label": "Landing view", "count": page_counts["landing"]},
                    {"key": "idea_started", "label": "Idea started", "count": idea_started},
                    {"key": "create_clicked", "label": "Create clicked", "count": create_clicked},
                    {"key": "accepted", "label": "Accepted", "count": accepted_jobs},
                    {"key": "main_completed", "label": "Main pipeline completed", "count": completed_jobs},
                    {"key": "result_viewed", "label": "Result viewed", "count": result_viewed},
                    {"key": "download", "label": "Download", "count": download_clicks},
                ],
                "idea_sources": {
                    "suggestion": suggestion_clicks,
                    "starter": starter_selects,
                    "upload": upload_selects,
                    "typed": idea_typed,
                },
                "downloads_by_format": download_by_fmt,
            },
            "refusals": [
                {"reason": code, "count": cnt}
                for code, cnt in sorted(refusal_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            ],
            "jobs": {
                "total": accepted_jobs,
                "by_final_state": {st: state_counts.get(st, 0) for st in FINAL_JOB_STATES},
                "active": {st: state_counts.get(st, 0) for st in ACTIVE_JOB_STATES},
                "by_user_kind": user_kind_jobs,
            },
            "stages": {
                "main": [build_stage(k) for k in MAIN_STAGE_ORDER],
                "job_totals": {
                    "jobs": accepted_jobs,
                    "timed_jobs": len(job_secs),
                    "median_seconds": _percentile(job_secs, 0.5, 2),
                    "p90_seconds": _percentile(job_secs, 0.9, 2),
                    "median_cost_usd": _percentile(job_costs, 0.5, 6),
                    "p90_cost_usd": _percentile(job_costs, 0.9, 6),
                    "mean_cost_usd": round(sum(job_costs) / len(job_costs), 6) if job_costs else None,
                    "total_cost_usd": round(sum(job_costs), 6),
                },
            },
            "budget": budget_rep,
            "gallery": {
                "picker_selections": [
                    {"id": hid, "count": cnt}
                    for hid, cnt in sorted(picker_counts.items(), key=lambda kv: (-kv[1], kv[0]))
                ],
                "shade_toggles": {**shade_counts, "total": sum(shade_counts.values())},
                "inspect_clicks": inspect_clicks,
                "viewer_drags": viewer_drags,
                "theme_changes": {**theme_counts, "total": sum(theme_counts.values())},
                "suggestions": suggestion_by_id,
                "starters": starter_by_id,
            },
            "devices": {
                "mobile": {"visitors": len(dev_visitors["mobile"]), "events": dev_events["mobile"]},
                "desktop": {"visitors": len(dev_visitors["desktop"]), "events": dev_events["desktop"]},
            },
        }

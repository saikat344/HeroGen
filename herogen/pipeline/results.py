"""Stage 5: `results.json` and a static `results.html` (no scripts) for a full or partial run."""

from __future__ import annotations

import html
import json
import re
import time
from pathlib import Path
from typing import Any

from herogen.pipeline.heldout import accepted_views, stage2_identity
from herogen.pipeline.runfolder import RunFolder, now_iso

STAGES = [
    (0, "input"),
    (1, "cover art"),
    (2, "turnaround"),
    (3, "base mesh"),
    (4, "game body"),
]

# The held-out views' record, reported after stage 2 when it exists.
HELDOUT_STAGE = ("2h", "held-out views")

# Images shown per stage beyond its record's `outputs` (stages 3-4 write folders of renders).
EXTRA_IMAGE_DIRS = {3: ["03-renders"], 4: ["04-body/pose-renders"]}

# Scoring (05-scores.json, shown when a run folder holds one): the building views in this order, then every
# held-out view the record accepts (`heldout.accepted_views`: status accepted and the image's recorded
# SHA-256, never the folder layout), highlighted. Other views the scores file names are listed after them,
# not highlighted and marked as not accepted.
BUILD_VIEWS = ["front", "back", "left", "right"]
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
# A record's status when it isn't ok: a stage that stopped early says so (`_Stage.fail`).
_FAILED = {"error": "error", "running": "did not finish"}


def _stale_heldout(run: RunFolder, rec: dict[str, Any]) -> bool:
    """A held-out record whose views were made from another turnaround than the run's current one."""
    views = (rec.get("details") or {}).get("views") or {}
    ids = {d.get("stage2_identity") for d in views.values() if isinstance(d, dict)}
    return bool(ids - {None}) and ids != {stage2_identity(run)}


def _images_for(run: RunFolder, n: int, record: dict[str, Any]) -> list[str]:
    imgs = [o for o in record.get("outputs", []) if o.lower().endswith(".png") and run.exists(o)]
    for d in EXTRA_IMAGE_DIRS.get(n, []):
        folder = run.path / d
        if folder.is_dir():
            for p in sorted(folder.glob("*.png")):
                rel = p.relative_to(run.path).as_posix()
                if rel not in imgs:
                    imgs.append(rel)
    return imgs


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def collect_scores(run: RunFolder) -> dict[str, Any] | None:
    """`05-scores.json` as shown on the page, with the image pairs that exist; None when there is no file."""
    if not run.exists("05-scores.json"):
        return None
    try:
        raw = run.read_json("05-scores.json")
        bodies_in = raw["bodies"]
        if not isinstance(bodies_in, list):
            raise TypeError("bodies is not a list")
    except (ValueError, KeyError, TypeError) as e:
        return {"error": f"05-scores.json could not be read ({type(e).__name__})", "bodies": []}
    images = accepted_views(run)
    named = sorted({str(v) for b in bodies_in if isinstance(b, dict) and isinstance(b.get("views"), dict)
                    for v in b["views"] if str(v) not in BUILD_VIEWS})
    heldout_views = list(images)
    other_views = [v for v in named if v not in images]
    bodies = []
    for b in bodies_in:
        if not isinstance(b, dict):
            continue
        name = str(b.get("name", ""))
        views_in = b.get("views") if isinstance(b.get("views"), dict) else {}
        views = {}
        for v in BUILD_VIEWS + heldout_views + other_views:
            sv = views_in.get(v)
            if not isinstance(sv, dict):
                continue
            row = {k: _num(sv.get(k)) for k in ("iou", "colour_error", "ssim", "yaw_deg", "scale")}
            row["seen"] = sv.get("seen") if isinstance(sv.get("seen"), bool) else None
            views[v] = row
        renders = {}
        for v in heldout_views:
            render = f"05-renders/{name}/{v}.png"
            # A body or view name only becomes a path when it is a plain file name.
            if _SAFE_NAME.match(name) and _SAFE_NAME.match(v) and run.exists(render):
                renders[v] = render
        bodies.append({
            "name": name,
            "glb": str(b.get("glb", "")),
            "triangles": _num(b.get("triangles")),
            "atlas": _num(b.get("atlas")),
            "views": views,
            "heldout_renders": renders,
        })
    return {"bodies": bodies, "heldout_views": heldout_views, "other_views": other_views,
            "heldout_images": images}


def collect(run: RunFolder) -> dict[str, Any]:
    stages = []
    total_cost = 0.0
    total_seconds = 0.0
    order: list[tuple[Any, str]] = list(STAGES)
    if run.read_stage(HELDOUT_STAGE[0]) is not None:
        order.insert(3, HELDOUT_STAGE)
    for n, label in order:
        rec = run.read_stage(n)
        if rec is None:
            stages.append({"stage": n, "name": label, "status": "not run"})
            continue
        stopped = f"Stopped early: {rec['error']}" if rec.get("error") else ""
        stale = n == HELDOUT_STAGE[0] and _stale_heldout(run, rec)
        total_cost += float(rec.get("cost_usd") or 0.0)
        total_seconds += float(rec.get("seconds") or 0.0)
        stages.append(
            {
                "stage": n,
                "name": rec.get("name", label),
                "status": "stale: stage 2 ran again" if stale else
                ("pass" if rec.get("ok") else _FAILED.get(rec.get("status"), "gate failed")),
                "ok": bool(rec.get("ok")) and not stale,
                "seconds": rec.get("seconds"),
                "cost_usd": rec.get("cost_usd"),
                "cost_basis": rec.get("cost_basis", ""),
                "models": rec.get("models", []),
                "checks": rec.get("checks", []),
                "images": _images_for(run, n, rec),
                "notes": " ".join(x for x in (rec.get("notes", ""), stopped) if x),
                "label": (rec.get("details") or {}).get("label", ""),
            }
        )
    out = {
        "run_id": run.run_id,
        "generated": now_iso(),
        "total_cost_usd": round(total_cost, 6),
        "cost_is_estimate": True,
        "total_seconds": round(total_seconds, 2),
        "stages": stages,
    }
    scores = collect_scores(run)
    if scores is not None:
        out["scores"] = scores
    return out


_CSS = """
:root { --bg:#fafafa; --fg:#1b1b1b; --muted:#666; --card:#fff; --line:#ddd; --pass:#1f7a3a; --fail:#b3261e;
  --hl:#fff4c2; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#141414; --fg:#eaeaea; --muted:#a0a0a0; --card:#1e1e1e; --line:#333; --pass:#6cc58a; --fail:#ff8a80;
    --hl:#4a3f10; }
}
* { box-sizing: border-box; }
body { margin:0; padding:16px; background:var(--bg); color:var(--fg); font:15px/1.45 system-ui, sans-serif; }
main { max-width: 1100px; margin: 0 auto; }
section { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:16px; margin:16px 0; }
h1 { font-size:1.5rem; margin:0 0 4px; } h2 { font-size:1.15rem; margin:0 0 8px; }
.meta, .muted { color:var(--muted); }
.pass { color:var(--pass); font-weight:600; } .fail { color:var(--fail); font-weight:600; }
.notrun { color:var(--muted); font-style:italic; }
.imgs { display:flex; flex-wrap:wrap; gap:8px; margin:8px 0; }
figure { margin:0; } figure img { max-width:240px; max-height:240px; border:1px solid var(--line); background:#fff; }
figcaption { font-size:12px; color:var(--muted); word-break:break-all; max-width:240px; }
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
th, td { text-align:left; padding:4px 8px; border-bottom:1px solid var(--line); }
.wide { overflow-x:auto; }
h3 { font-size:1rem; margin:16px 0 4px; }
tr.heldout td { background:var(--hl); font-weight:600; }
"""


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return html.escape(str(v))


def _cell(v: float | None, digits: int = 3) -> str:
    return "–" if v is None else f"{v:.{digits}f}"


def _figure(src: str, caption: str) -> str:
    e = html.escape
    return (f"<figure><a href='{e(src)}'><img src='{e(src)}' alt='{e(caption)}' loading='lazy'></a>"
            f"<figcaption>{e(caption)}</figcaption></figure>")


def render_scores(scores: dict[str, Any]) -> list[str]:
    e = html.escape
    parts = ["<section><h2>Scores</h2>",
             "<p class='meta'>From 05-scores.json. Silhouette IoU, colour error and SSIM inside the masks. "
             "Highlighted rows are the held-out views stage 2h accepted, which no body was built from.</p>"]
    if scores.get("error"):
        parts.append(f"<p class='fail'>{e(scores['error'])}</p>")
    for b in scores["bodies"]:
        tri = "–" if b["triangles"] is None else f"{int(b['triangles'])}"
        atlas = "–" if b["atlas"] is None else f"{int(b['atlas'])} px"
        parts.append(f"<h3>{e(b['name'])}</h3>")
        parts.append(f"<p class='meta'>{e(b['glb'])} · triangles {tri} · atlas {atlas}</p>")
        parts.append("<div class='wide'><table><thead><tr><th>view</th><th>IoU</th><th>colour error</th>"
                     "<th>SSIM</th><th>seen / unseen</th></tr></thead><tbody>")
        heldout = scores.get("heldout_views", [])
        others = [v for v in scores.get("other_views", []) if v in b["views"]]
        for v in BUILD_VIEWS + heldout + others:
            row = b["views"].get(v)
            cls = " class='heldout'" if v in heldout else ""
            if row is None:
                parts.append(f"<tr{cls}><td>{e(v)}</td><td colspan='4' class='muted'>not scored</td></tr>")
                continue
            seen = {True: "seen", False: "unseen", None: "–"}[row["seen"]]
            label = e(v)
            if v in heldout:
                extra = []
                if row["yaw_deg"] is not None:
                    extra.append(f"yaw {row['yaw_deg']:.1f}°")
                if row["scale"] is not None:
                    extra.append(f"scale {row['scale']:.3f}")
                label += f" (held out{'; ' + ', '.join(extra) if extra else ''})"
            elif v in others:
                label += " (not an accepted held-out view)"
            parts.append(f"<tr{cls}><td>{label}</td><td>{_cell(row['iou'])}</td>"
                         f"<td>{_cell(row['colour_error'], 1)}</td><td>{_cell(row['ssim'])}</td>"
                         f"<td>{seen}</td></tr>")
        parts.append("</tbody></table></div>")
        if b["heldout_renders"]:
            parts.append("<div class='imgs'>")
            for v, render in b["heldout_renders"].items():
                if v in scores.get("heldout_images", {}):
                    parts.append(_figure(scores["heldout_images"][v], f"held-out image {v}"))
                parts.append(_figure(render, f"{b['name']} render, {v}"))
            parts.append("</div>")
    parts.append("</section>")
    return parts


def render_html(data: dict[str, Any]) -> str:
    e = html.escape
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>herogen run {e(data['run_id'])}</title><style>{_CSS}</style></head><body><main>",
        f"<h1>herogen run {e(data['run_id'])}</h1>",
        f"<p class='meta'>Generated {e(data['generated'])} · total {data['total_seconds']:.1f} s · "
        f"cost ${data['total_cost_usd']:.4f} (estimate from list prices)</p>",
    ]
    for s in data["stages"]:
        title = f"Stage {s['stage']}: {e(s['name'])}"
        if s["status"] == "not run":
            parts.append(f"<section><h2>{title}</h2><p class='notrun'>not run</p></section>")
            continue
        cls = "pass" if s["ok"] else "fail"
        parts.append(f"<section><h2>{title} <span class='{cls}'>{e(s['status'])}</span></h2>")
        if s.get("label"):
            parts.append(f"<p><strong>{e(s['label'])}</strong></p>")
        parts.append(
            f"<p class='meta'>{_fmt(s['seconds'])} s · ${float(s['cost_usd'] or 0):.4f} · "
            f"models: {e(', '.join(s['models']) or 'none')}</p>"
        )
        if s["images"]:
            parts.append("<div class='imgs'>")
            for img in s["images"]:
                parts.append(
                    f"<figure><a href='{e(img)}'><img src='{e(img)}' alt='{e(img)}' loading='lazy'></a>"
                    f"<figcaption>{e(img)}</figcaption></figure>"
                )
            parts.append("</div>")
        if s["checks"]:
            parts.append("<div class='wide'><table><thead><tr><th>check</th><th>value</th><th>threshold</th>"
                         "<th>pass</th><th>gate</th></tr></thead><tbody>")
            for c in s["checks"]:
                ok = "pass" if c.get("pass") else "fail"
                # Pipeline checks name the comparison `op`; Blender checks name it `rule`.
                op = c.get("op") or c.get("rule") or (">=" if c.get("threshold") is not None else "")
                limit = "–" if c.get("threshold") is None else f"{e(op)} {_fmt(c['threshold'])}"
                parts.append(
                    f"<tr><td>{e(c['name'])}</td><td>{_fmt(c['value'])}</td><td>{limit}</td>"
                    f"<td class='{ok}'>{ok}</td><td>{'yes' if c.get('gate', True) else 'no'}</td></tr>"
                )
            parts.append("</tbody></table></div>")
        if s.get("notes"):
            parts.append(f"<p class='muted'>{e(s['notes'])}</p>")
        parts.append("</section>")
    if "scores" in data:
        parts += render_scores(data["scores"])
    parts.append("</main></body></html>\n")
    return "".join(parts)


def stage5_results(run: RunFolder) -> dict[str, Any]:
    t0 = time.monotonic()
    started = now_iso()
    data = collect(run)
    run.write_json("results.json", data)
    Path(run.file("results.html")).write_text(render_html(data))
    reported = sum(1 for s in data["stages"] if s["status"] != "not run")
    run.write_stage(
        {
            "stage": 5,
            "name": "results",
            "ok": True,
            "started": started,
            "seconds": round(time.monotonic() - t0, 2),
            "cost_usd": 0.0,
            "cost_basis": "no model calls",
            "models": [],
            "outputs": ["results.html", "results.json"],
            "checks": [{"name": "stages_reported", "value": reported, "threshold": 0, "op": ">=",
                        "pass": True, "gate": False}],
            "notes": "",
        }
    )
    return data


def load_results(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())

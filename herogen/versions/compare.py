"""Compare two versions: per-view scores (seen views and held-out views apart), triangles, gates, joint gaps,
profile and sources, saying on which side any number is missing.

Both versions are verified first (every file against its recorded SHA-256); a version that fails is refused.
Scores come from the version's own `05-scores.json`. A version frozen by hand before this tool (format 0, no
`format` field) may have none; its scores are then read from `05-scores-<series>.json` in its sketch folder,
`<series>` being the label up to its first hyphen (`v0-carve` -> `05-scores-v0.json`), and that file's SHA-256
is reported with the comparison; a symbolic link there is not read. A format 1 version without its own scores
has none. The body's entry in a scores file is the one `version.json` names (`scores.body`); else the entry
whose `glb` is the version's source body (`<source.body>/high.glb`); else the one entry (atlas variants aside)
with the version's `mesh_source`. Without a unique match the version's scores are reported missing, never
guessed.
"""
from __future__ import annotations

import json
from pathlib import Path

from .store import TURNAROUND, VersionError, load_version, sha256_file, summary, verify

METRICS = ("iou", "colour_error", "ssim")          # IoU and SSIM: higher is better; colour error: lower is
DIGITS = {"iou": 4, "colour_error": 2, "ssim": 4}  # better (mean absolute error, 0-255)
GAP_KEYS = ("gaps_high", "gaps_low", "gaps_high_heldout", "gaps_low_heldout")
OPERATORS = ("<=", ">=", "==", "<", ">")


def _json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, ValueError):
        return None


def _record(path: Path, meta: dict, name: str):
    files = meta.get("files_sha256") if isinstance(meta.get("files_sha256"), dict) else {}
    if name not in files or not (path / name).is_file():
        return None
    return _json(path / name)


def _scores(path: Path, meta: dict, summ: dict) -> dict:
    """This version's scores entry and where it came from."""
    own = _record(path, meta, "05-scores.json")
    if own is not None:
        f, data, shown = path / "05-scores.json", own, "05-scores.json"
    elif "format" in meta:
        return {"file": None, "file_sha256": None, "body": None, "entry": None, "views": {},
                "note": "no 05-scores.json in the version"}
    else:                                            # format 0: the hand-collected scores beside it
        series = summ["label"].split("-", 1)[0]
        f = path.parent / f"05-scores-{series}.json"
        if f.is_symlink():
            return {"file": None, "file_sha256": None, "body": None, "entry": None, "views": {},
                    "note": f"{f.name} beside it is a symbolic link and is not read"}
        if not f.is_file():
            return {"file": None, "file_sha256": None, "body": None, "entry": None, "views": {},
                    "note": f"no 05-scores.json in the version and no {f.name} beside it"}
        data, shown = _json(f), f"{path.parent.name}/{f.name}"
    out = {"file": shown, "file_sha256": sha256_file(f), "body": None, "entry": None, "views": {}, "note": None,
           "scorer_stamp": isinstance(data, dict) and ("scored" in data or "scorer" in data)}
    if not isinstance(data, dict) or not isinstance(data.get("bodies"), list):
        out["note"] = f"{shown} could not be read as a scores file"
        return out
    entries = [b for b in data["bodies"] if isinstance(b, dict)]
    want = meta.get("scores") if isinstance(meta.get("scores"), dict) else {}
    source = meta.get("source") if isinstance(meta.get("source"), dict) else {}
    match = []
    if isinstance(want.get("body"), str):
        match = [b for b in entries if b.get("name") == want["body"]]
    elif isinstance(source.get("body"), str):
        match = [b for b in entries if b.get("glb") == f"{source['body']}/high.glb" and not b.get("variant_of")]
    elif isinstance(summ.get("mesh_source"), str):
        match = [b for b in entries if b.get("mesh_source") == summ["mesh_source"] and not b.get("variant_of")]
    if len(match) != 1:
        out["note"] = (f"{shown} has no entry for this body" if not match else
                       f"{shown} has {len(match)} entries that could be this body; none is chosen")
        return out
    out["entry"], out["body"] = match[0], match[0].get("name")
    for v in data.get("views") or []:
        if isinstance(v, dict) and isinstance(v.get("view"), str):
            out["views"][v["view"]] = bool(v.get("heldout"))
    return out


def _side(path: Path) -> dict:
    meta = load_version(path)
    summ = summary(path, meta)
    return {"path": path, "meta": meta, "summary": summ, "scores": _scores(path, meta, summ),
            "stage-3": _record(path, meta, "stage-3.json"), "stage-4": _record(path, meta, "stage-4.json"),
            "sections": _record(path, meta, "sections.json")}


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _delta(a, b, digits=4):
    a, b = _num(a), _num(b)
    return None if a is None or b is None else round(b - a, digits)


def _pair(a, b, digits=4) -> dict:
    return {"a": _num(a), "b": _num(b), "delta": _delta(a, b, digits)}


def _seen(row: dict | None) -> str | None:
    if not isinstance(row, dict):
        return None
    shape, colour = row.get("seen_shape"), row.get("seen_colour")
    if shape is None and colour is None:
        return "seen" if row.get("seen") else ("unseen" if row.get("seen") is False else None)
    parts = [n for n, f in (("shape", shape), ("colour", colour)) if f]
    return "+".join(parts) if parts else "unseen"


def _views(A: dict, B: dict, missing: list) -> tuple[list, dict]:
    va = (A["scores"]["entry"] or {}).get("views") or {}
    vb = (B["scores"]["entry"] or {}).get("views") or {}
    names = [v for v in TURNAROUND if v in va or v in vb]
    names += sorted(n for n in set(va) | set(vb) if n not in TURNAROUND)
    flags = {**A["scores"]["views"], **B["scores"]["views"]}
    la, lb = A["summary"]["label"], B["summary"]["label"]
    rows = []
    for n in names:
        ra, rb = va.get(n), vb.get(n)
        held = flags.get(n, n not in TURNAROUND)
        row = {"view": n, "heldout": held, "seen_a": _seen(ra), "seen_b": _seen(rb)}
        for m in METRICS:
            row[m] = _pair((ra or {}).get(m), (rb or {}).get(m), DIGITS[m])
        for side, r in (("a", ra), ("b", rb)):
            if isinstance(r, dict) and r.get("measures"):
                row[f"measures_{side}"] = r["measures"]
        rows.append(row)
        for lab, r, other in ((la, ra, rb), (lb, rb, ra)):
            if not isinstance(r, dict):
                if isinstance(other, dict):
                    missing.append(f"{n}: not scored in {lab}")
                continue
            for m in METRICS:
                if _num(r.get(m)) is None and _num((other or {}).get(m)) is not None:
                    missing.append(f"{n} {m}: missing in {lab}")
    means = {}
    for group, held in (("seen", False), ("heldout", True)):
        g = {}
        for m in METRICS:
            both = [r for r in rows if r["heldout"] == held and r[m]["a"] is not None and r[m]["b"] is not None]
            if both:
                a = sum(r[m]["a"] for r in both) / len(both)
                b = sum(r[m]["b"] for r in both) / len(both)
                g[m] = {"views": [r["view"] for r in both], **_pair(round(a, DIGITS[m]), round(b, DIGITS[m]),
                                                                     DIGITS[m])}
        means[group] = g
    return rows, means


def _triangles(side: dict, lod: str):
    st4 = side["stage-4"]
    lods = ((st4 or {}).get("body") or {}).get("lods") if isinstance(st4, dict) else None
    if isinstance(lods, dict) and _num((lods.get(lod) or {}).get("triangles")) is not None:
        return lods[lod]["triangles"]
    entry = side["scores"]["entry"] or {}
    if _num(entry.get("triangles" if lod == "high" else "triangles_low")) is not None:
        return entry["triangles" if lod == "high" else "triangles_low"]
    sec = side["sections"]
    lods = sec.get("lods") if isinstance(sec, dict) else None
    if isinstance(lods, dict) and _num((lods.get(lod) or {}).get("triangles")) is not None:
        return lods[lod]["triangles"]
    return None


def _lods_built(side: dict) -> list | None:
    """The LODs the version's profile builds (`stage-4.json` `profile.lods`), or None when not recorded."""
    prof = (side["stage-4"] or {}).get("profile") if isinstance(side["stage-4"], dict) else None
    lods = prof.get("lods") if isinstance(prof, dict) else None
    return lods if isinstance(lods, list) else None


def _absent(side: dict, lod: str) -> str:
    """Why a LOD's number is absent: the profile does not build that LOD, or it is missing."""
    built = _lods_built(side)
    return "not built" if built is not None and lod not in built else "missing"


def _absent_check(side: dict, name: str) -> str:
    return _absent(side, "low") if name.endswith("_low") or "_low_" in name else "missing"


def _check(c: dict | None):
    if c is None:
        return None
    rule = next((r for r in (c.get("rule"), c.get("op")) if r in OPERATORS), None)   # old records: prose
    return {"value": c.get("value"), "threshold": c.get("threshold"), "rule": rule, "pass": c.get("pass")}


def _gates(A: dict, B: dict, missing: list) -> dict:
    la, lb = A["summary"]["label"], B["summary"]["label"]
    out = {"stages": {}, "checks": []}
    for key, stage in (("stage-3", 3), ("stage-4", 4)):
        ra, rb = A[key], B[key]
        out["stages"][str(stage)] = {"a": ra.get("ok") if isinstance(ra, dict) else None,
                                     "b": rb.get("ok") if isinstance(rb, dict) else None}
        ca = {c["name"]: c for c in (ra or {}).get("checks") or [] if isinstance(c, dict) and "name" in c} \
            if isinstance(ra, dict) else {}
        cb = {c["name"]: c for c in (rb or {}).get("checks") or [] if isinstance(c, dict) and "name" in c} \
            if isinstance(rb, dict) else {}
        names = list(ca) + [n for n in cb if n not in ca]
        for n in names:
            ga, gb = ca.get(n), cb.get(n)
            if all(c is None or c.get("gate", True) is False for c in (ga, gb)):
                continue                                           # not a gate on either side
            for side, c, lab in ((A, ga, la), (B, gb, lb)):
                if c is None and side[key] is not None and _absent_check(side, n) == "missing":
                    missing.append(f"gate {stage} {n}: missing in {lab}")
            out["checks"].append({"stage": stage, "name": n,
                                  "gate_a": None if ga is None else ga.get("gate", True) is not False,
                                  "gate_b": None if gb is None else gb.get("gate", True) is not False,
                                  "a": _check(ga), "b": _check(gb),
                                  "why_a": None if ga else _absent_check(A, n),
                                  "why_b": None if gb else _absent_check(B, n)})
    return out


def _gaps(A: dict, B: dict, missing: list) -> dict:
    la, lb = A["summary"]["label"], B["summary"]["label"]
    pa = (A["sections"] or {}).get("pose_test") if isinstance(A["sections"], dict) else None
    pb = (B["sections"] or {}).get("pose_test") if isinstance(B["sections"], dict) else None
    out = {}
    for key in GAP_KEYS:
        ga = (pa or {}).get(key) if isinstance(pa, dict) else None
        gb = (pb or {}).get(key) if isinstance(pb, dict) else None
        if not isinstance(ga, dict) and not isinstance(gb, dict):
            continue
        if not isinstance(ga, dict) or not isinstance(gb, dict):
            side, lab = (A, la) if not isinstance(ga, dict) else (B, lb)
            why = _absent(side, "low" if "_low" in key else "high")
            if why == "missing":
                missing.append(f"joint gaps {key}: missing in {lab}")
            out[key] = {"absent_in": lab, "why": why}
            continue
        ungated = set()
        if key != "gaps_high_heldout":                             # held-out poses are never gated anyway
            for p in (pa, pb):
                ungated |= set(p.get("ungated_poses") or [])
        poses_a = [p for p in ga if p not in ungated and isinstance(ga[p], dict)]
        poses_b = [p for p in gb if p not in ungated and isinstance(gb[p], dict)]
        common = [p for p in poses_a if p in poses_b]

        def worst(g, poses, joint):
            vals = [_num(g[p].get(joint)) for p in poses]
            vals = [v for v in vals if v is not None]
            return max(vals) if vals else None

        joints = []
        for p in common:
            joints += [j for j in list(ga[p]) + list(gb[p]) if j not in joints]
        per = {j: _pair(worst(ga, common, j), worst(gb, common, j)) for j in joints}
        for j, p in per.items():
            for t, lab in (("a", la), ("b", lb)):
                if p[t] is None and p["b" if t == "a" else "a"] is not None:
                    missing.append(f"joint gaps {key} {j}: missing in {lab}")

        def top(side):
            vals = [p[side] for p in per.values() if p[side] is not None]
            return max(vals) if vals else None

        out[key] = {
            "poses_a": len(poses_a), "poses_b": len(poses_b), "common_poses": common,
            "only_a": [p for p in poses_a if p not in poses_b], "only_b": [p for p in poses_b if p not in poses_a],
            "joints": per, "worst_common": _pair(top("a"), top("b")),
        }
    return out


def _inputs(A: dict, B: dict) -> dict:
    ia = A["meta"].get("inputs") if isinstance(A["meta"].get("inputs"), dict) else None
    ib = B["meta"].get("inputs") if isinstance(B["meta"].get("inputs"), dict) else None
    out = {"recorded_a": ia is not None, "recorded_b": ib is not None}
    if ia is None or ib is None:
        out["sketch"] = "unknown"
        return out

    def same(x, y):
        return "unknown" if x is None or y is None else ("same" if x == y else "different")

    out["sketch"] = same(ia.get("sketch"), ib.get("sketch"))
    for key in ("views", "masks", "heldout", "heldout_masks"):
        xa, xb = ia.get(key) or {}, ib.get(key) or {}
        out[key] = {n: same(xa.get(n), xb.get(n)) for n in sorted(set(xa) | set(xb))}
    return out


def compare(a: Path, b: Path) -> dict:
    """Compare version folders `a` and `b`; every delta is B - A."""
    for p in (Path(a), Path(b)):
        rep = verify(p)
        if not rep["ok"]:
            problems = [k for k in ("mismatched", "missing", "not_regular", "unreadable", "unlisted", "unsafe",
                                    "label_recorded") if rep[k]]
            raise VersionError(f"{rep['version']} fails verify ({', '.join(problems)}); run verify --version "
                               f"on it for details")
    A, B = _side(Path(a)), _side(Path(b))
    missing: list[str] = []
    for side in (A, B):
        if side["scores"]["entry"] is None:
            missing.append(f"scores: missing in {side['summary']['label']} ({side['scores']['note']})")
    rows, means = _views(A, B, missing)
    tri = {lod: _pair(_triangles(A, lod), _triangles(B, lod), 0) for lod in ("high", "low")}
    for lod, p in tri.items():
        for t, side, other in (("a", A, "b"), ("b", B, "a")):
            if p[t] is None:
                p[f"why_{t}"] = _absent(side, lod)
                if p[other] is not None and p[f"why_{t}"] == "missing":
                    missing.append(f"triangles {lod}: missing in {side['summary']['label']}")
    for key, name in (("stage-3", "stage-3.json"), ("stage-4", "stage-4.json"), ("sections", "sections.json")):
        for side in (A, B):
            if side[key] is None:
                missing.append(f"{name}: missing in {side['summary']['label']}")
    gaps = _gaps(A, B, missing)

    def described(side):
        s = dict(side["summary"])
        s["path"] = f"{s['sketch']}/{s['label']}"
        s["scores_file"], s["scores_body"] = side["scores"]["file"], side["scores"]["body"]
        s["scores_file_sha256"] = side["scores"].get("file_sha256")
        s["scores_stamped"] = side["scores"].get("scorer_stamp")
        return s

    return {"a": described(A), "b": described(B), "inputs": _inputs(A, B), "views": rows, "means": means,
            "triangles": tri, "gates": _gates(A, B, missing), "joint_gaps": gaps, "missing": missing}


# ---------------------------------------------------------------------------------------------------------
# Text


def _fmt(x, digits=4, sign=False) -> str:
    if x is None:
        return "missing"
    if isinstance(x, bool):
        return "pass" if x else "fail"
    if isinstance(x, int):
        return f"{x:+d}" if sign else str(x)
    if isinstance(x, float):
        return f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"
    return str(x)


def _short(x) -> str:
    """A check's value or threshold as recorded, without padding (0.9, 24000, 12.77)."""
    if x is None:
        return "missing"
    if isinstance(x, bool):
        return str(x).lower()
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    line = lambda cells: "  ".join(c.ljust(w) if i == 0 else c.rjust(w)  # noqa: E731
                                   for i, (c, w) in enumerate(zip(cells, widths))).rstrip()
    return [line(headers), line(["-" * w for w in widths]), *(line(r) for r in rows)]


def _profile(s: dict) -> str:
    p = s.get("profile") or "missing"
    b = s.get("budgets")
    if isinstance(b, dict) and b:
        p += " (budgets " + ", ".join(f"{k} {v}" for k, v in b.items()) + ")"
    return p


def _code(s: dict) -> str:
    if s.get("commit"):
        d = {True: "dirty", False: "clean", None: "dirty unknown"}[s.get("dirty")]
        return f"{s['commit'][:12]} ({d})"
    if s.get("code_text"):
        return f"not stamped: {s['code_text']}"
    return "missing"


def render_text(r: dict) -> str:
    a, b = r["a"], r["b"]
    out = []
    for tag, s in (("A", a), ("B", b)):
        out.append(f"{tag}  {s['path']}  milestone: {s.get('milestone') or 'missing'}; frozen {s.get('frozen')}; "
                   f"code {_code(s)}")
    out.append("")
    scores = lambda s: (f"{s['scores_file']} (body {s['scores_body']})" if s.get("scores_body")  # noqa: E731
                        else "missing")
    out += _table(["", "A", "B"], [
        ["profile", _profile(a), _profile(b)],
        ["mesh source", a.get("mesh_source") or "missing", b.get("mesh_source") or "missing"],
        ["joints source", a.get("joints_source") or "missing", b.get("joints_source") or "missing"],
        ["scores", scores(a), scores(b)],
    ])
    inp = r["inputs"]
    out.append("")
    if inp["sketch"] == "unknown" and not (inp["recorded_a"] and inp["recorded_b"]):
        who = " and ".join(t for t, k in (("A", "recorded_a"), ("B", "recorded_b")) if not inp[k])
        out.append(f"inputs: not recorded in {who}, so a shared sketch and views cannot be checked")
    else:
        diff = [f"{k[:-1] if k.endswith('s') else k} {n}" for k in ("views", "masks", "heldout", "heldout_masks")
                for n, v in inp.get(k, {}).items() if v != "same"]
        out.append(f"inputs: sketch {inp['sketch']}; " + ("views, masks and held-out views the same" if not diff
                                                           else "not the same: " + ", ".join(diff)))
    for tag, s in (("A", a), ("B", b)):
        if s.get("scores_file") and s["scores_file"] != "05-scores.json":
            out.append(f"scores of {tag} read from {s['scores_file']} (beside the version, not in it), "
                       f"SHA-256 {s['scores_file_sha256']}")
    out.append("")
    out.append("Per view (delta = B - A; IoU and SSIM higher is better, colour error lower is better)")
    head = ["view", "seen A", "seen B", "IoU A", "IoU B", "delta", "colour A", "colour B", "delta",
            "SSIM A", "SSIM B", "delta"]
    body = []
    for held, title in ((False, None), (True, "held-out (unseen)")):
        group = [v for v in r["views"] if v["heldout"] == held]
        if not group:
            continue
        if title:
            body.append([title] + [""] * (len(head) - 1))
        for v in group:
            cells = [v["view"], v["seen_a"] or "missing", v["seen_b"] or "missing"]
            for m in METRICS:
                d = DIGITS[m]
                cells += [_fmt(v[m]["a"], d), _fmt(v[m]["b"], d), _fmt(v[m]["delta"], d, sign=True)]
            body.append(cells)
        mean = r["means"]["heldout" if held else "seen"]
        if mean:
            n = max(len(x["views"]) for x in mean.values())
            cells = [f"mean ({n} view{'s' if n != 1 else ''})", "", ""]
            for m in METRICS:
                x = mean.get(m)
                cells += ([_fmt(x["a"], DIGITS[m]), _fmt(x["b"], DIGITS[m]), _fmt(x["delta"], DIGITS[m], True)]
                          if x else ["missing"] * 3)
            body.append(cells)
    out += _table(head, body) if body else ["  no view scores on either side"]
    for v in r["views"]:
        for t in ("a", "b"):
            if v.get(f"measures_{t}"):
                out.append(f"  {v['view']} on {t.upper()} measures {v[f'measures_{t}']}")
    out.append("")
    out.append("Triangles")
    out += _table(["LOD", "A", "B", "delta"],
                  [[lod, p.get("why_a") or _fmt(p["a"]), p.get("why_b") or _fmt(p["b"]),
                    "" if p["delta"] is None else _fmt(p["delta"], sign=True)] for lod, p in r["triangles"].items()])
    out.append("")
    g = r["gates"]
    out.append("Gates: " + "; ".join(f"stage {k} A {_fmt(v['a']) if v['a'] is not None else 'missing'}, "
                                     f"B {_fmt(v['b']) if v['b'] is not None else 'missing'}"
                                     for k, v in g["stages"].items()))

    def chk(c, gate, why):
        if c is None:
            return why or "missing"
        val = _short(c["value"])
        res = "pass" if c["pass"] is True else ("fail" if c["pass"] is False else "?")
        return f"{val} {res}" + ("" if gate else " (not gated)")

    def threshold(c):
        t = c["a"] or c["b"]
        return "" if t["threshold"] is None else f"{t['rule'] or ''} {_short(t['threshold'])}".strip()

    rows = [[f"{c['stage']} {c['name']}", threshold(c), chk(c["a"], c["gate_a"], c["why_a"]),
             chk(c["b"], c["gate_b"], c["why_b"])]
            for c in g["checks"]]
    if rows:
        out += _table(["check", "threshold", "A", "B"], rows)
    out.append("")
    for key, x in r["joint_gaps"].items():
        if "absent_in" in x:
            out.append(f"Joint gaps {key}: {x['why']} in {x['absent_in']}")
            continue
        kind = "held-out poses" if key.endswith("_heldout") else "gated poses"
        out.append(f"Joint gaps {key} (studs; worst per joint over the {len(x['common_poses'])} {kind} both ran; "
                   f"A ran {x['poses_a']}, B {x['poses_b']})")
        rows = [[j, _fmt(p["a"]), _fmt(p["b"]), _fmt(p["delta"], sign=True)] for j, p in x["joints"].items()]
        w = x["worst_common"]
        rows.append(["worst", _fmt(w["a"]), _fmt(w["b"]), _fmt(w["delta"], sign=True)])
        out += _table(["joint", "A", "B", "delta"], rows)
        out.append("")
    out.append("Missing: " + ("; ".join(r["missing"]) if r["missing"] else "nothing"))
    return "\n".join(out) + "\n"


def to_json(r: dict) -> str:
    return json.dumps(r, indent=2) + "\n"


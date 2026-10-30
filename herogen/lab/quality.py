"""Quality of one run, from its stage records: every check in plain words, grouped, and the failing gates first.

A gate is a check with `"gate": true` (the default). Groups follow the pipeline: Artwork (stages 0 and 1), Views
(stage 2), Body (stages 3 and 4), Texture (stage 4's texture checks) and Held-out (stage 2h). Stage 5 checks the
report itself and is left out. Every number shown is the record's; the words around it are the page's.
"""

from __future__ import annotations

import re
from typing import Any

from herogen.lab.heldout import view_label

GROUPS = ["Artwork", "Views", "Body", "Texture", "Held-out"]
STAGE_ORDER = ["0", "1", "2", "2h", "3", "4"]

# The poses the lab's viewer can show: the test poses of herogen/blender/pose.py, in its order.
VIEWER_POSES = ["rest", "walk_a", "walk_b", "arms_forward", "arms_up", "crouch", "twist_turn"]
POSE_WORDS = {"rest": "rest", "walk_a": "walk", "walk_b": "walk (other leg)", "arms_forward": "arms forward",
              "arms_up": "arms up", "crouch": "crouch", "twist_turn": "twist"}

LABELS = {
    "min_side_px": "Sketch size, shortest side (px)",
    "sketch_line_contrast": "Sketch lines dark enough (grey levels)",
    "sketch_characters_found": "Characters found in the sketch",
    "sketch_small_extra_figures": "Smaller extra figures in the sketch",
    "sketch_feet_on_figure": "Feet found on the sketched figure",
    "sketch_mask_area_frac": "Figure found in the sketch (area share)",
    "cover_mask_area_frac": "Figure found in the artwork (area share)",
    "aspect_agreement": "Artwork keeps the sketch proportions",
    "silhouette_iou_normalised": "Artwork outline matches the sketch (IoU)",
    "figures_found": "Figures found on the turnaround sheet",
    "height_agreement_4_views": "Same height in all four views",
    "front_back_width_agreement": "Front and back widths agree",
    "left_right_mirror_iou": "Left and right profiles mirror (IoU)",
    "a_pose_hand_span_over_shoulders": "A-pose: hand span over shoulder width",
    "right_view_toe_offset_as_drawn": "Right view: toes point the right way",
    "left_view_toe_offset_as_drawn": "Left view: toes point the right way",
    "top_drawn_width_vs_front_width": "Drawn top view: width vs front",
    "top_drawn_depth_vs_side_width": "Drawn top view: depth vs side views",
    "top_width_vs_front_width": "Top silhouette: width vs front",
    "top_depth_vs_side_width": "Top silhouette: depth vs side views",
    "front_view_present": "Front view available",
    "triangles": "Base mesh triangles",
    "open_edges": "Open edges (holes)",
    "non_manifold_edges": "Edges shared by more than two triangles",
    "sections_non_empty": "Body sections built (of 15)",
    "triangles_high": "High body triangles",
    "triangles_low": "Low body triangles",
    "triangles_total": "Body triangles, all parts",
    "watertight_parts": "Parts not closed and manifold",
    "texture_max_px": "Texture atlas size (px)",
    "pivots_in_joint_region": "Pivots inside their joint region",
    "max_joint_gap_high": "Largest joint gap in the pose test, high body (studs)",
    "max_joint_gap_low": "Largest joint gap in the pose test, low body (studs)",
    "max_joint_gap_high_heldout": "Largest joint gap in the held-out poses, high body (studs)",
    "max_joint_gap_low_heldout": "Largest joint gap in the held-out poses, low body (studs)",
    "max_joint_gap_high_twist_turn": "Joint gap in the twist pose (studs)",
    "max_joint_gap_high_visible_no_covers": "Joint gap, visible only, without covers (studs)",
    "max_joint_gap_high_geometric_no_covers": "Joint gap, geometric, without covers (studs)",
    "max_joint_gap_high_flat_caps": "Joint gap with flat caps instead (studs)",
    "cap_points_outside_neighbour_high": "Joint cap points poking through at rest",
    "stages_reported": "Stages in the report",
    "heldout_figure_area_frac": "Held-out view: figure found (area share)",
    "heldout_height_agreement_front": "Held-out view: height matches the front view",
    "heldout_width_over_front_width": "Held-out view: width over the front width",
    "heldout_width_over_side_width": "Held-out view: width over the side width",
    "heldout_toe_offset": "Held-out view: toes point the right way",
}
VIEW_CHECKS = {
    "figure_area_frac": "figure found (area share)",
    "height_agreement_front": "height matches the front view",
    "pose_landmark_rms": "pose landmarks fit (RMS)",
    "pose_landmark_rms_clear": "pose landmarks fit clearly (RMS)",
    "found_yaw_clear_of_build_azimuths_deg": "angle clear of the build angles (°)",
    "found_yaw_window_miss_deg": "angle off the requested window (°)",
    "found_yaw_in_window": "angle inside the requested window (° off)",
    "facing_cues_against": "cues facing the wrong way",
    "facing_cues_for": "cues facing the right way",
    "status_accepted": "accepted",
}


def label(name: str) -> str:
    """A check's name in plain words; an unknown name with its underscores turned into spaces."""
    if name in LABELS:
        return LABELS[name]
    if m := re.match(r"^(right|left)_profile_width_over_height$", name):
        return f"{m.group(1).capitalize()} profile is a strict side view (width / height)"
    if m := re.match(r"^silhouette_iou_(\w+)$", name):
        return f"Base mesh outline, {m.group(1)} view (IoU)"
    if m := re.match(r"^rest_iou_with_covers_(\w+)$", name):
        return f"Body outline at rest, {m.group(1)} view (IoU)"
    if m := re.match(r"^texture_colour_error_(\w+)$", name):
        return f"Texture colour error, {m.group(1)} view (0–255)"
    if m := re.match(r"^texture_ssim_(\w+)$", name):
        return f"Texture structure, {m.group(1)} view (SSIM)"
    if m := re.match(r"^triangles_([A-Z]\w+)$", name):
        return f"Triangles, {_spaced(m.group(1)).lower()} part group"
    if m := re.match(r"^([a-z]+-[a-z]+-\d+)\.(\w+)$", name):
        return f"{view_label(m.group(1))}: {VIEW_CHECKS.get(m.group(2), m.group(2).replace('_', ' '))}"
    return name.replace("_", " ")


def _spaced(name: str) -> str:
    """`LeftShoulder` -> `Left Shoulder`."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)


def group(stage: str, name: str) -> str | None:
    if stage in ("0", "1"):
        return "Artwork"
    if stage == "2":
        return "Views"
    if stage == "2h":
        return "Held-out"
    if stage == "4" and name.startswith("texture_") and name != "texture_max_px":
        return "Texture"
    if stage in ("3", "4"):
        return "Body"
    return None


def _fmt(v: Any) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _target(c: dict[str, Any]) -> str:
    if c["threshold"] is None:
        return "no threshold"
    cmp = {">=": "≥", "<=": "≤"}.get(c["cmp"], c["cmp"])
    return f"{cmp} {_fmt(c['threshold'])}"


def worst_gap(table: dict[str, dict[str, float]] | None, poses: list[str] | None,
              skip: list[str]) -> tuple[str, str, float] | None:
    """(pose, joint, gap) of the largest gap among `poses` (all poses but `skip` when not listed)."""
    if not table:
        return None
    best = None
    for pose, row in table.items():
        if (poses is not None and pose not in poses) or (poses is None and pose in skip):
            continue
        for joint, gap in row.items():
            if best is None or gap > best[2]:
                best = (pose, joint, gap)
    return best


def _joint_failure(c: dict[str, Any], m: re.Match, pose_test: dict[str, Any] | None) -> dict[str, Any]:
    lod, heldout_poses = m.group(1), bool(m.group(2))
    table = (pose_test or {}).get(f"gaps_{lod}{'_heldout' if heldout_poses else ''}")
    w = worst_gap(table, c.get("poses"), (pose_test or {}).get("ungated_poses") or [])
    sev = (c["value"] / c["threshold"]) if isinstance(c["value"], (int, float)) and c["threshold"] else 1.0
    if w is None:
        return {"kind": "bad", "sev": sev, "act": None,
                "text": f"{label(c['name'])}: {_fmt(c['value'])}, needs {_target(c)}."}
    pose, joint, gap = w
    joint_words = _spaced(joint).lower()
    return {
        "kind": "bad", "sev": sev,
        "text": f"Joint gap {gap:.3f} studs at the {joint_words}, {POSE_WORDS.get(pose, pose.replace('_', ' '))} "
                f"pose{' (a held-out pose)' if heldout_poses else ''}, {lod} body. Limit {_fmt(c['threshold'])}.",
        "act": {"type": "joint", "joint": joint, "pose": pose, "lod": lod, "gap": gap, "limit": c["threshold"],
                "pose_in_viewer": pose in VIEWER_POSES},
    }


def failures(checks: list[dict[str, Any]], pose_test: dict[str, Any] | None,
             heldout_views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Failing gates in plain words, worst first, each with the action that shows it (None when there is none).

    Texture checks on one view are merged into one item; a joint-gap check names its worst joint and pose from
    the pose test's gap table; a held-out view's status check quotes the record's reason."""
    out: list[dict[str, Any]] = []
    texture: dict[str, dict[str, Any]] = {}
    views = {v["name"]: v for v in heldout_views}
    for c in checks:
        if not c["gate"] or c["pass"]:
            continue
        name, value, limit = c["name"], c["value"], c["threshold"]
        numeric = isinstance(value, (int, float)) and isinstance(limit, (int, float)) and value and limit
        if m := re.match(r"^max_joint_gap_(high|low)(_heldout)?$", name):
            item = _joint_failure(c, m, pose_test)
        elif m := re.match(r"^texture_(colour_error|ssim)_(\w+)$", name):
            t = texture.setdefault(m.group(2), {"view": m.group(2), "parts": [], "sev": 0.0, "group": c["group"]})
            if m.group(1) == "ssim":
                t["parts"].append(f"structure {_fmt(value)} (needs {_target(c)})")
                t["sev"] = max(t["sev"], (limit / value) if numeric else 1.0)
            else:
                t["parts"].append(f"colour error {_fmt(value)} (needs {_target(c)})")
                t["sev"] = max(t["sev"], (value / limit) if numeric else 1.0)
            continue
        elif m := re.match(r"^([a-z]+-[a-z]+-\d+)\.status_accepted$", name):
            v = views.get(m.group(1), {})
            reason = "; ".join(c["notes"]) or v.get("status") or "not accepted"
            item = {"kind": "warn", "sev": 1.0, "text": f"{view_label(m.group(1))}: {reason}.",
                    "act": {"type": "heldout", "view": m.group(1)}}
        else:
            item = {"kind": "bad", "sev": 1.01, "act": None,
                    "text": f"{label(name)}: {_fmt(value)}, needs {_target(c)}."}
        item.update(group=c["group"], check=name, stage=c["stage"])
        out.append(item)
    for t in texture.values():
        out.append({"kind": "bad", "sev": t["sev"], "group": t["group"], "check": f"texture_*_{t['view']}",
                    "stage": "4", "act": {"type": "view", "view": t["view"]},
                    "text": f"Texture on the {t['view']} view doesn't match it: {', '.join(t['parts'])}."})
    out.sort(key=lambda f: (f["kind"] != "bad", -f["sev"]))
    return out


def quality(stages: list[dict[str, Any]], heldout_stage: dict[str, Any] | None,
            pose_test: dict[str, Any] | None, heldout_views: list[dict[str, Any]],
            has_body: bool) -> dict[str, Any]:
    """Every check of stages 0 to 4 (stage 2h from `heldout_stage`, which may be another run's record of the same
    turnaround), grouped, with counts, failures and an overall status."""
    by = {s["stage"]: s for s in stages}
    if heldout_stage is not None:
        by["2h"] = heldout_stage
    checks = []
    for key in STAGE_ORDER:
        s = by.get(key) or {}
        for c in s.get("checks") or []:
            g = group(key, c["name"])
            if g:
                checks.append({**c, "stage": key, "group": g, "label": label(c["name"])})
    gates = [c for c in checks if c["gate"]]
    groups = []
    for g in GROUPS:
        gg = [c for c in gates if c["group"] == g]
        groups.append({"name": g, "gates": len(gg), "failed": sum(1 for c in gg if not c["pass"]),
                       "checks": sum(1 for c in checks if c["group"] == g)})
    unreadable = [k for k in STAGE_ORDER if (by.get(k) or {}).get("status") == "unreadable"]
    ran = [k for k in STAGE_ORDER if (by.get(k) or {}).get("status") in ("pass", "gate failed")]
    failed = sum(1 for c in gates if not c["pass"])
    missing = [k for k in STAGE_ORDER if k != "2h" and k not in ran]
    if unreadable:
        status = {"kind": "bad", "word": "Can't read",
                  "detail": "unreadable: " + ", ".join(f"stage-{k}.json" for k in unreadable)}
    elif not ran:
        status = {"kind": "idle", "word": "Not run", "detail": "no stage records"}
    elif failed:
        status = {"kind": "warn", "word": "Needs attention", "detail": f"{failed} of {len(gates)} gates failing"}
    elif not has_body:
        status = {"kind": "idle", "word": "No body yet", "detail": "stage 4 has not built a body"}
    elif missing:
        status = {"kind": "idle", "word": "Partial run",
                  "detail": "no record for " + ", ".join(f"stage {k}" for k in missing)}
    else:
        status = {"kind": "ok", "word": "Ready", "detail": f"all {len(gates)} gates pass"}
    return {
        "gates": len(gates),
        "passed": len(gates) - failed,
        "groups": groups,
        "checks": checks,
        "reported_below": sum(1 for c in checks if not c["gate"] and not c["pass"] and c["threshold"] is not None),
        "failures": failures(checks, pose_test, heldout_views),
        "status": status,
        "unreadable": unreadable,
    }

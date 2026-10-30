"""The lab's workspace: one or more runs folders and an optional versions folder, read together.

Runs are named `<root>/<run>`: the root's key (`r1`, `r2`, ... in the order the folders were given) and the run
folder's name. Runs are grouped into characters by the SHA-256 of their sketch (`00-sketch.png`, the file stage 0
writes and lists in its `outputs`); a run without a sketch is a character of its own. Within a character each run
is filed under the profile its body records (`sections.json` `profile`, else `stage-4.json` `profile.name`); a run
that records none is filed under `null` ("profile not recorded").

A run without a `stage-2h.json` of its own shows the held-out record of another run of the same character whose
four turnaround views (`views/{front,back,left,right}.png`) are byte-identical, the newest such record first, and
says which run it came from; a run without a sketch never borrows. Symlinked sketches and views count as missing.

A version attaches to a character through its `sketch_run`: every run of that name, in every root, must hold the
same sketch, and the version joins that sketch's character. A version whose `sketch_run` names no run, names runs
of different sketches or of no sketch, or whose `version.json` cannot be read, is listed apart with the reason.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from herogen.lab import heldout as heldout_mod
from herogen.lab import quality as quality_mod
from herogen.lab import runindex, versions
from herogen.lab.versions import file_sha256

TURNAROUND = ["front", "back", "left", "right"]
PROFILES = ["best", "roblox-ingame", "roblox-avatar"]


class Workspace:
    def __init__(self, roots: list[Path] | Path, versions_root: Path | None = None,
                 labels: list[str] | None = None) -> None:
        roots = [roots] if isinstance(roots, Path) else list(roots)
        self.roots = [(f"r{i + 1}", Path(r)) for i, r in enumerate(roots)]
        self.labels = {key: (labels[i] if labels and i < len(labels) else str(path))
                       for i, (key, path) in enumerate(self.roots)}
        self.versions_root = Path(versions_root) if versions_root else None

    # Folders -------------------------------------------------------------------------------------------------

    def root(self, key: str) -> Path | None:
        return dict(self.roots).get(key)

    def run_path(self, run_id: str) -> Path | None:
        """The folder of `<root>/<run>`, or None when either part is not one of the workspace's own."""
        key, _, name = (run_id or "").partition("/")
        root = self.root(key)
        return runindex.run_dir(root, name) if root is not None else None

    def _run_folders(self) -> list[tuple[str, str, Path]]:
        out = []
        for key, root in self.roots:
            try:
                if not root.is_dir():
                    continue
                candidates = sorted(root.iterdir())
            except OSError:
                continue
            for p in candidates:
                try:
                    if runindex.is_run_dir(p):
                        out.append((key, p.name, p))
                except OSError:
                    continue
        return out

    # Runs ----------------------------------------------------------------------------------------------------

    @staticmethod
    def _own_file(p: Path) -> bool:
        """A regular file of the run itself: not a symlink, which could make two runs look alike."""
        return p.is_file() and not p.is_symlink()

    @staticmethod
    def sketch_sha(run: Path) -> str | None:
        p = run / "00-sketch.png"
        return file_sha256(p) if Workspace._own_file(p) else None

    @staticmethod
    def turnaround_id(run: Path) -> str | None:
        """One digest over the four turnaround views; None unless all four exist."""
        digests = []
        for v in TURNAROUND:
            p = run / "views" / f"{v}.png"
            d = file_sha256(p) if Workspace._own_file(p) and not (run / "views").is_symlink() else None
            if d is None:
                return None
            digests.append(d)
        return hashlib.sha256("".join(digests).encode()).hexdigest()

    @staticmethod
    def profile_of(run: Path) -> str | None:
        sec, _ = runindex._read_json(run / "04-body" / "sections.json")
        p = sec.get("profile") if isinstance(sec, dict) else None
        if isinstance(p, str) and p.strip():
            return p.strip()
        rec, _ = runindex._read_json(run / "stage-4.json")
        prof = rec.get("profile") if isinstance(rec, dict) else None
        name = prof.get("name") if isinstance(prof, dict) else None
        return name.strip() if isinstance(name, str) and name.strip() else None

    def unreadable_entry(self, key: str, name: str, error: str) -> dict[str, Any]:
        """A run folder that cannot be listed: still in the list, with the reason, and never grouped with others."""
        return {"id": f"{key}/{name}", "root": key, "name": name, "error": error, "readable": False,
                "stages": {k: "unreadable" for k, _ in runindex.STAGES}, "has_body": False, "has_scores": False,
                "modified": 0.0, "sketch_sha": None, "turnaround_id": None, "sketch_name": None,
                "sketch_label": None, "profile": None, "body_dirs": [], "mesh_source": None, "triangles": None,
                "code": None, "started": None, "heldout_record": False, "heldout_started": None}

    def run_entry(self, key: str, name: str, run: Path) -> dict[str, Any]:
        err = runindex.folder_error(run)
        if err:
            return self.unreadable_entry(key, name, err)
        base = runindex._run_entry(run)
        s0, _ = runindex._read_json(run / "stage-0.json")
        details = s0.get("details") if isinstance(s0, dict) and isinstance(s0.get("details"), dict) else {}
        s4, _ = runindex._read_json(run / "stage-4.json")
        s2h, _ = runindex._read_json(run / heldout_mod.RECORD)
        started = None
        for rec in (s4, s0):
            if isinstance(rec, dict) and isinstance(rec.get("started"), str):
                started = rec["started"]
                break
        sec, _ = runindex._read_json(run / "04-body" / "sections.json")
        try:
            body_dirs = sorted(p.name for p in run.iterdir() if runindex.BODY_DIR.match(p.name) and p.is_dir()
                               and ((p / "high.glb").is_file() or (p / "low.glb").is_file()))
        except OSError:
            body_dirs = []
        return {
            **base,
            "id": f"{key}/{name}",
            "root": key,
            "name": name,
            "sketch_sha": self.sketch_sha(run),
            "turnaround_id": self.turnaround_id(run),
            "sketch_name": details.get("source_name") if isinstance(details.get("source_name"), str) else None,
            "sketch_label": details.get("label") if isinstance(details.get("label"), str) else None,
            "profile": self.profile_of(run) if base["has_body"] else None,
            "body_dirs": body_dirs,
            # Counted from the GLB, like every triangle count the page shows.
            "triangles": runindex.glb_summary(run / "04-body" / "high.glb").get("triangles")
            if (run / "04-body" / "high.glb").is_file() else None,
            "code": runindex._code(s4.get("code")) if isinstance(s4, dict) else None,
            "mesh_source": sec.get("mesh_source") if isinstance(sec, dict) and isinstance(sec.get("mesh_source"),
                                                                                           str) else None,
            "started": started,
            "heldout_record": isinstance(s2h, dict),
            "heldout_started": s2h.get("started") if isinstance(s2h, dict) and isinstance(s2h.get("started"),
                                                                                           str) else None,
            "readable": "unreadable" not in base["stages"].values(),
        }

    def runs(self) -> list[dict[str, Any]]:
        out = []
        for key, name, run in self._run_folders():
            try:
                out.append(self.run_entry(key, name, run))
            except OSError as e:
                out.append(self.unreadable_entry(key, name, f"folder could not be read ({type(e).__name__})"))
        return out

    @staticmethod
    def attach(vers: list[dict[str, Any]], runs: list[dict[str, Any]]) -> dict[str, str]:
        """{version id: why it is not attached}; versions not in it attach to their sketch's character."""
        why = {}
        for v in vers:
            if v.get("error"):
                why[v["id"]] = v["error"]
                continue
            named = [r for r in runs if r["name"] == v.get("sketch_run")]
            sketches = {r["sketch_sha"] for r in named}
            if not v.get("sketch_run"):
                why[v["id"]] = "version.json names no sketch_run"
            elif not named:
                why[v["id"]] = f"no run named {v['sketch_run']} in the runs folders"
            elif None in sketches:
                why[v["id"]] = f"a run named {v['sketch_run']} has no 00-sketch.png to match"
            elif len(sketches) > 1:
                why[v["id"]] = f"runs named {v['sketch_run']} hold {len(sketches)} different sketches"
            else:
                v["sketch_sha"] = sketches.pop()
        return why

    # Characters ----------------------------------------------------------------------------------------------

    def characters(self, runs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        runs = self.runs() if runs is None else runs
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in runs:
            ck = f"s-{r['sketch_sha'][:16]}" if r["sketch_sha"] else f"run-{r['root']}-{r['name']}"
            r["character"] = ck
            groups.setdefault(ck, []).append(r)
        vers = [versions.version_summary(vid, folder) for vid, folder in versions.version_dirs(self.versions_root)]
        why = self.attach(vers, runs)
        chars = []
        for ck, members in groups.items():
            members.sort(key=lambda r: (r["started"] or "", r["modified"]), reverse=True)
            names = [r["sketch_name"] for r in members if r["sketch_name"]]
            title = max(set(names), key=names.count) if names else members[0]["name"]
            if title.lower().endswith(".png"):
                title = title[:-4]
            labels = [r["sketch_label"] for r in members if r["sketch_label"]]
            sha = members[0]["sketch_sha"]
            mine = [v for v in vers if v["id"] not in why and sha is not None and v.get("sketch_sha") == sha]
            profiles = {}
            for r in members:
                profiles.setdefault(r["profile"], []).append(r["id"])
            chars.append({
                "key": ck,
                "title": title,
                "label": max(set(labels), key=labels.count) if labels else None,
                "sketch_sha": members[0]["sketch_sha"],
                "runs": [r["id"] for r in members],
                "profiles": profiles,
                "versions": mine,
                "newest": members[0]["started"] or "",
            })
        chars.sort(key=lambda c: (c["sketch_sha"] is None, -len(c["runs"]), c["title"]))
        return {
            "characters": chars,
            "runs": {r["id"]: r for r in runs},
            "unattached_versions": [{**v, "unattached_reason": why[v["id"]]} for v in vers if v["id"] in why],
        }

    def workspace(self) -> dict[str, Any]:
        data = self.characters()
        return {
            "roots": [{"key": k, "label": self.labels[k], "exists": p.is_dir()} for k, p in self.roots],
            "versions_root": None if self.versions_root is None else {
                "label": str(self.versions_root), "exists": self.versions_root.is_dir()},
            "profiles": PROFILES,
            **data,
        }

    # One run -------------------------------------------------------------------------------------------------

    def heldout_for(self, run_id: str, run: Path, entries: dict[str, dict[str, Any]] | None = None
                    ) -> tuple[dict[str, Any] | None, str | None]:
        """(held-out views, the run they came from). The run's own record first; else the newest record among the
        same character's runs with an identical turnaround; else the run's own `heldout/` images."""
        own = heldout_mod.heldout(run)
        if own is not None and own["format"] != "files":
            return own, run_id
        entries = entries if entries is not None else {r["id"]: r for r in self.runs()}
        me = entries.get(run_id)
        if me and me["turnaround_id"] and me["sketch_sha"]:
            twins = [r for r in entries.values() if r["id"] != run_id and r["heldout_record"]
                     and r["turnaround_id"] == me["turnaround_id"] and r["sketch_sha"] == me["sketch_sha"]]
            twins.sort(key=lambda r: r["heldout_started"] or "", reverse=True)
            for t in twins:
                other = self.run_path(t["id"])
                if other is not None:
                    rec = heldout_mod.heldout(other)
                    if rec is not None and rec["format"] != "files":
                        return rec, t["id"]
        return own, (run_id if own is not None else None)

    def run_detail(self, run_id: str) -> dict[str, Any] | None:
        run = self.run_path(run_id)
        if run is None:
            return None
        key = run_id.split("/", 1)[0]
        err = runindex.folder_error(run)
        if err:
            return {"id": run_id, "root": key, "name": run.name, "path": f"{self.labels[key]}/{run.name}",
                    "error": err}
        detail = runindex.run_detail(run)
        entries = {r["id"]: r for r in self.runs()}
        self.characters(list(entries.values()))  # sets each entry's character key
        me = entries.get(run_id, {})
        held, source = self.heldout_for(run_id, run, entries)
        main = next((b for b in detail["bodies"] if b["dir"] == "04-body"), None)
        sec = (main or {}).get("sections") or {}
        held_stage = (held or {}).get("record") if source and source != run_id else None
        detail.update({
            "id": run_id,
            "root": run_id.split("/", 1)[0],
            "name": run.name,
            "path": f"{self.labels[run_id.split('/', 1)[0]]}/{run.name}",
            "character": me.get("character"),
            "profile": me.get("profile"),
            "heldout": held,
            "heldout_source": source,
            "quality": quality_mod.quality(detail["stages"], held_stage, sec.get("pose_test"),
                                           (held or {}).get("views") or [], main is not None),
            "generation": generation(detail["stages"], held_stage, source if held_stage else None),
        })
        return detail

    def version_detail(self, version_id: str) -> dict[str, Any] | None:
        folder = versions.version_dir(self.versions_root, version_id)
        return versions.version_summary(version_id, folder) if folder is not None else None


def generation(stages: list[dict[str, Any]], heldout_stage: dict[str, Any] | None,
               heldout_from: str | None) -> dict[str, Any]:
    """Time and estimated cost per stage that ran, and the run's own totals.

    When the held-out views come from another run, that run's stage 2h is listed (marked `from_run`) but kept out
    of this run's totals; `with_borrowed` adds it back for comparison. A total covers the stages that record a
    number, and says how many did not."""
    rows = []
    for s in stages:
        rec, from_run = s, None
        if s["stage"] == "2h" and s["status"] == "not run" and heldout_stage is not None:
            rec, from_run = heldout_stage, heldout_from
        if rec.get("status") in ("not run", None):
            continue
        rows.append({"stage": s["stage"], "from_run": from_run, "record": rec})

    def total(include_borrowed: bool) -> dict[str, Any]:
        mine = [r["record"] for r in rows if include_borrowed or r["from_run"] is None]
        out = {}
        for k in ("seconds", "cost_usd"):
            vals = [x.get(k) for x in mine]
            t = runindex._total(v for v in vals if isinstance(v, (int, float)))
            out[k] = None if t is None else round(t, 6)
            out[f"{k}_missing"] = sum(1 for v in vals if not isinstance(v, (int, float)))
        return out

    borrowed = any(r["from_run"] for r in rows)
    return {"rows": rows, "total": total(False), "with_borrowed": total(True) if borrowed else None}

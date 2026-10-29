"""Freeze a run's body into the version store, and list, find, load and verify versions.

Store layout: `<store>/<sketch>/<label>/`, one folder per version, flat (no subfolders), holding the frozen
files and `version.json`. The store root comes from `--store` or `HEROGEN_VERSIONS`; there is no default.

A freeze is write-once and atomic: the files are copied (and hashed as they are copied) into a hidden
`.partial-<label>-<random>` folder next to where the version will live, the records are read back from those
copies, `version.json` is written last, the files are made read-only and the folder is renamed to `<label>`
without replacing anything. A freeze that stops part-way leaves no `<label>` folder holding a `version.json`;
at most a `.partial-*` folder, which `list` reports as unfinished and every other command refuses. A label that
exists is never replaced.
"""
from __future__ import annotations

import getpass
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from datetime import datetime
from pathlib import Path

FORMAT = 1
VERSION_FILE = "version.json"
ENV_STORE = "HEROGEN_VERSIONS"
PARTIAL_PREFIX = ".partial-"
CHUNK = 1 << 20

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")         # labels, sketch folders: one plain name
FILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$")   # files in a version: plain names, flat
BODY_DIR = re.compile(r"^04-body(-[A-Za-z0-9][A-Za-z0-9_.-]*)?$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
# Text that names a real path: a word that starts at a root (`/srv/x`, `~/x`, `~user`, `\\host`, `C:\x`), has
# a `..` step, or holds a home prefix anywhere (`/Users/`, `/home/`, `\Users\`). A lone `/` between words
# ("front / back") and a slash inside a word ("24000/6000") are plain text.
ROOTED = re.compile(r"^(?:/[^\s/]|~[^\s]|\\\\|[A-Za-z]:[\\/])")
HOME = re.compile(r"(?:/Users/|/home/|\\Users\\)", re.IGNORECASE)

BODY_FILES = ("high.glb", "low.glb", "sections.json")             # plus every atlas*.png in the body folder
RUN_FILES = ("stage-0.json", "stage-1.json", "stage-2.json", "stage-2h.json", "stage-3.json", "stage-4.json",
             "stage-5.json", "03-joints.json", "03-source.glb", "05-scores.json")
STAGE3_OUTPUTS = ("03-joints.json", "03-source.glb")
BUILDERS = ("stage-4.json", "sections.json", "stage-3.json")      # the records that built the body, in order
OPTIONAL = {"stage-5.json"}                                       # never reported missing (the results page)
TURNAROUND = ("front", "back", "left", "right", "top")


class VersionError(Exception):
    """A refusal or a problem the user can fix, reported as one line."""


# ---------------------------------------------------------------------------------------------------------
# Store, names and text


def store_root(arg: str | os.PathLike | None, *, must_exist: bool = True) -> Path:
    """The store folder: `arg` (from `--store`), else `$HEROGEN_VERSIONS`."""
    raw = arg if arg not in (None, "") else os.environ.get(ENV_STORE, "")
    if not raw:
        raise VersionError(f"no version store given: pass --store DIR or set {ENV_STORE}")
    root = Path(raw).expanduser()
    if must_exist and not root.is_dir():
        raise VersionError(f"version store {root} is not a folder (create it first)")
    return root


def check_name(kind: str, value: str) -> str:
    if not isinstance(value, str) or not NAME.match(value):              # never `.`, `..` or hidden
        raise VersionError(f"{kind} {value!r} must be one plain name: letters, digits and _ . - (at most 80), "
                           f"not starting with . _ or -")
    return value


def safe_file_name(name) -> bool:
    """A file name a version may hold: one plain name, no folder, no `..`, not absolute."""
    return isinstance(name, str) and bool(FILE_NAME.match(name))         # never `.`, `..` or hidden


def names_a_path(text: str) -> bool:
    """Whether free text names a path a version record must never hold: a rooted one, a `..` step or a home
    folder. Words are split at white space and `=`, and opening quotes and brackets are stripped."""
    if HOME.search(text):
        return True
    for word in re.split(r"[\s=]+", text):
        word = word.lstrip("'\"([{<,;")
        if ROOTED.match(word) or ".." in re.split(r"[/\\]", word):
            return True
    return False


def _no_paths(value, field: str) -> None:
    """Refuse a record whose text (keys and values, all the way down) names an absolute path or a `..` step."""
    if isinstance(value, dict):
        for k, v in value.items():
            _no_paths(k, field)
            _no_paths(v, f"{field}.{k}" if field else str(k))
    elif isinstance(value, list):
        for v in value:
            _no_paths(v, field)
    elif isinstance(value, str) and names_a_path(value):
        raise VersionError(f"{field or 'a field'} would record a path ({value[:80]!r}); a version record never "
                           f"holds an absolute path or a .. step")


# ---------------------------------------------------------------------------------------------------------
# Files: regular files only, hashed as they are read


def _no_follow(path, flags):
    return os.open(path, flags | getattr(os, "O_NOFOLLOW", 0))


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, never through a symbolic link."""
    h = hashlib.sha256()
    with open(path, "rb", opener=_no_follow) as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def _copy_file(src: Path, dst: Path) -> tuple[str, int]:
    """Copy `src` to a new file `dst`, hashing the bytes as they are written, and flush it to disk. Refuses
    when the source changes while it is copied (size or modification time), so a copy is never torn."""
    before = os.stat(src, follow_symlinks=False)
    h = hashlib.sha256()
    size = 0
    with open(src, "rb", opener=_no_follow) as fin, open(dst, "xb") as fout:
        while block := fin.read(CHUNK):
            h.update(block)
            fout.write(block)
            size += len(block)
        fout.flush()
        os.fsync(fout.fileno())
    after = os.stat(src, follow_symlinks=False)
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or size != after.st_size:
        raise VersionError(f"{src.name} changed while it was being frozen; try again once the run is idle")
    return h.hexdigest(), size


def _walk(run: Path, rel: str, want_dir: bool = False) -> Path | None:
    """`run/rel` when it exists as a regular file (or a folder, with `want_dir`) reached without a symbolic
    link; None when it is absent. A symbolic link anywhere on the way is refused: a version is frozen from
    the run's own files, never from wherever a link points."""
    p = run
    for part in rel.split("/"):
        p = p / part
        if p.is_symlink():
            raise VersionError(f"{run.name}/{rel} is or goes through a symbolic link; a version is frozen from "
                               f"regular files only")
        if not p.exists():
            return None
    return p if (p.is_dir() if want_dir else p.is_file()) else None


def _entries(run: Path, rel_dir: str, pattern: str) -> list[Path]:
    """The regular files matching `pattern` in `run/rel_dir` (no symbolic links, which are refused)."""
    d = _walk(run, rel_dir, want_dir=True) if rel_dir else run
    if d is None:
        return []
    out = []
    for p in sorted(d.glob(pattern)):
        rel = f"{rel_dir}/{p.name}" if rel_dir else p.name
        if p.is_symlink():
            raise VersionError(f"{run.name}/{rel} is a symbolic link; a version is frozen from regular files only")
        if p.is_file():
            out.append(p)
    return out


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _make_writable(path: Path) -> None:
    """Undo read-only modes under `path` so a partial folder can be removed."""
    for p in [path, *path.rglob("*")]:
        try:
            p.chmod(p.stat().st_mode | stat.S_IWUSR | (stat.S_IXUSR if p.is_dir() else 0))
        except OSError:
            pass


def remove_tree(path: Path) -> None:
    """Remove a folder this module made read-only (tests and partial-freeze cleanup)."""
    if path.exists():
        _make_writable(path)
        shutil.rmtree(path, ignore_errors=True)


def _writable_bits(path: Path) -> bool:
    try:
        return bool(stat.S_IMODE(os.stat(path, follow_symlinks=False).st_mode) & 0o222)
    except OSError:
        return True


# ---------------------------------------------------------------------------------------------------------
# Reading records


def _read_json(path: Path, where: str, why: str = ""):
    try:
        return json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise VersionError(f"{where} could not be read as JSON ({exc.__class__.__name__}){why}") from None


def _code_of(name: str, record) -> dict | None:
    """A record's code stamp `{"commit", "dirty"}`, or None when it carries none (`code` absent or null).
    A stamp of the wrong shape is refused: `dirty` must be true, false or null and `commit` a string or
    null, so a dirty build is never read as unknown."""
    if not isinstance(record, dict) or record.get("code") is None:
        return None
    code = record["code"]
    if not isinstance(code, dict):
        raise VersionError(f"{name}: code is {type(code).__name__}, not an object with commit and dirty")
    commit, dirty = code.get("commit"), code.get("dirty")
    if dirty is not None and not isinstance(dirty, bool):
        raise VersionError(f"{name}: code.dirty is {dirty!r}; it must be true, false or null")
    if commit is not None and not isinstance(commit, str):
        raise VersionError(f"{name}: code.commit is {commit!r}; it must be a string or null")
    return {"commit": commit, "dirty": dirty}


def _first(records: list, key: str):
    for rec in records:
        if isinstance(rec, dict) and rec.get(key) not in (None, ""):
            return rec[key]
    return None


def _hash_inputs(run: Path) -> dict:
    """SHA-256 of the inputs every body of the run was built and scored from (none of them are copied)."""
    def h(rel: str):
        p = _walk(run, rel)
        return sha256_file(p) if p is not None else None

    inputs = {"sketch": h("00-sketch.png"), "views": {}, "masks": {}, "heldout": {}, "heldout_masks": {}}
    for v in TURNAROUND:
        if (s := h(f"views/{v}.png")) is not None:
            inputs["views"][v] = s
        if (s := h(f"masks/{v}.png")) is not None:
            inputs["masks"][v] = s
    for p in _entries(run, "heldout", "*.png"):
        if p.stem.endswith(("-mask", "-raw")):
            continue
        mask = _walk(run, f"heldout/{p.stem}-mask.png")
        if mask is not None:
            inputs["heldout"][p.stem] = sha256_file(p)
            inputs["heldout_masks"][p.stem] = sha256_file(mask)
    return inputs


def _scores_entry(scores, body: str) -> dict | None:
    """This body's entry in a run's `05-scores.json`, matched by its GLB path, and its atlas variants."""
    if not isinstance(scores, dict) or not isinstance(scores.get("bodies"), list):
        return None
    glb = f"{body}/high.glb"
    entries = [b for b in scores["bodies"] if isinstance(b, dict)]
    main = [b for b in entries if b.get("glb") == glb and not b.get("variant_of")]
    if len(main) != 1 or not isinstance(main[0].get("name"), str):
        return None
    name = main[0]["name"]
    variants = sorted(b["name"] for b in entries if b.get("variant_of") == name and isinstance(b.get("name"), str))
    return {"file": "05-scores.json", "body": name, "variants": variants}


def _sketch_folder(store: Path, sketch: str | None, sketch_sha: str | None, run: Path) -> str:
    """Where the version goes: `--sketch`, else the one sketch folder whose versions record the same sketch
    hash. A folder holding versions of a different sketch is refused."""
    seen: dict[str, set] = {}
    for d in sorted(p for p in store.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for vf in d.glob(f"*/{VERSION_FILE}"):
            if vf.parent.name.startswith("."):
                continue
            try:
                s = (json.loads(vf.read_text()).get("inputs") or {}).get("sketch")
            except (OSError, ValueError, AttributeError):
                continue
            if isinstance(s, str):
                seen.setdefault(d.name, set()).add(s)
    if sketch is not None:
        check_name("sketch", sketch)
        other = seen.get(sketch, set()) - {sketch_sha}
        if sketch_sha is not None and other:
            raise VersionError(f"versions in {sketch}/ were frozen from a different sketch than {run.name}/"
                               f"00-sketch.png; pick the right --sketch")
        return sketch
    if sketch_sha is None:
        raise VersionError(f"{run.name} has no 00-sketch.png to match a sketch folder by; name it with --sketch")
    match = sorted(name for name, hashes in seen.items() if sketch_sha in hashes)
    if len(match) == 1:
        return match[0]
    if not match:
        raise VersionError(f"no version in the store has {run.name}'s sketch yet; name its folder with --sketch")
    raise VersionError(f"sketch folders {', '.join(match)} all hold this sketch; pick one with --sketch")


def _inside_store(run: Path, store: Path) -> str | None:
    """The run's path relative to the store when the run lies inside it, else None: a record never holds a
    path that leaves the store (it would name the folders around it, a home folder among them)."""
    try:
        return run.resolve().relative_to(store.resolve()).as_posix()
    except ValueError:
        return None


def _run_records(run: Path) -> dict:
    """SHA-256 of every stage record in the run folder itself: the run's identity, wherever it lives."""
    return {p.name: sha256_file(p) for p in _entries(run, "", "stage-*.json") if safe_file_name(p.name)}


def _collect(run: Path, body: str) -> tuple[dict, list, bool]:
    """What a freeze copies: name -> source file, the run-level names taken from the body folder, and whether
    the body folder had its own stage 3."""
    files: dict[str, Path] = {}
    for name in BODY_FILES:
        if (p := _walk(run, f"{body}/{name}")) is not None:
            files[name] = p
    for p in _entries(run, body, "atlas*.png"):
        files[p.name] = p
    # A body folder with its own stage-3.json had its own stage 3: the run's stage 3 outputs belong to the
    # run's other body and are never taken for this one.
    own_stage3 = _walk(run, f"{body}/stage-3.json") is not None
    overrides = []
    for name in RUN_FILES:
        if (p := _walk(run, f"{body}/{name}")) is not None:
            files[name] = p
            overrides.append(name)
        elif not (own_stage3 and name in STAGE3_OUTPUTS) and (p := _walk(run, name)) is not None:
            files[name] = p
    for name in files:
        if not safe_file_name(name):
            raise VersionError(f"{body}/{name} is not a plain file name a version can hold")
    return files, overrides, own_stage3


def _describe(records: dict, files: dict, *, own_stage3: bool, body: str, allow_dirty: bool,
              allow_unknown_code: bool, run_name: str) -> dict:
    """The record-derived fields of `version.json`, from the frozen copies of the records. Refuses a dirty
    build without `allow_dirty`, and a body whose building records name no commit without
    `allow_unknown_code`."""
    st3, st4, sections = records.get("stage-3.json"), records.get("stage-4.json"), records.get("sections.json")
    builders = [r for r in (sections, st4, st3) if isinstance(r, dict)]

    code_records = {n: _code_of(n, r) for n, r in sorted(records.items())
                    if n in BUILDERS or (isinstance(r, dict) and r.get("code") is not None)}
    stamps = {n: c for n, c in code_records.items() if c is not None}
    dirty_in = sorted(n for n, c in stamps.items() if c["dirty"] is True)
    if dirty_in and not allow_dirty:
        raise VersionError(f"{run_name} was built or scored by uncommitted code (dirty: true in "
                           f"{', '.join(dirty_in)}); commit and rebuild, or pass --allow-dirty")
    commit = next((code_records[n]["commit"] for n in BUILDERS
                   if code_records.get(n) is not None and code_records[n]["commit"] is not None), None)
    if commit is None and not allow_unknown_code:
        raise VersionError(f"none of {run_name}'s {', '.join(BUILDERS)} names the commit that built the body; "
                           f"pass --allow-unknown-code to freeze it with its code unknown")
    if not stamps:
        code = None
    else:
        flags = [c["dirty"] for c in stamps.values()]
        code = {"commit": commit,
                "dirty": True if True in flags else (False if all(f is False for f in flags) else None),
                "consistent": len({c["commit"] for c in stamps.values() if c["commit"] is not None}) <= 1}

    profile = budgets = None
    prof4 = st4.get("profile") if isinstance(st4, dict) else None
    if isinstance(prof4, dict):
        profile, budgets = prof4.get("name"), prof4.get("budgets")
    elif isinstance(prof4, str):
        profile = prof4
    if profile is None and isinstance(sections, dict) and isinstance(sections.get("profile"), str):
        profile = sections["profile"]
    if budgets is None and isinstance(sections, dict) and isinstance(sections.get("budgets"), dict):
        budgets = sections["budgets"]
    lods = prof4.get("lods") if isinstance(prof4, dict) and isinstance(prof4.get("lods"), list) else None
    mesh_source = _first(builders, "mesh_source")
    built = st4.get("started") if isinstance(st4, dict) and isinstance(st4.get("started"), str) else None
    scores = _scores_entry(records.get("05-scores.json"), body)

    # What is missing, said explicitly: files the run does not have (a low LOD only when the profile builds
    # one, 03-source.glb only for a mesh source other than the carve, no stage 3 outputs for a body folder
    # with its own stage 3), and fields its records do not carry.
    expected = ["high.glb", "low.glb", "sections.json", "atlas-high.png", "atlas-low.png",
                *(n for n in RUN_FILES if n not in OPTIONAL)]
    if lods is not None and "low" not in lods:
        expected = [n for n in expected if n not in ("low.glb", "atlas-low.png")]
    if mesh_source in (None, "carve"):
        expected.remove("03-source.glb")
    if own_stage3:
        expected = [n for n in expected if n not in STAGE3_OUTPUTS]
    fields = {"code": code, "code.commit": commit if code is not None else True, "profile": profile,
              "mesh_source": mesh_source, "joints_source": _first(builders, "joints_source"), "built": built}
    if "05-scores.json" in files:                     # scores that do not name this body (a missing file is
        fields["scores"] = scores                     # already listed under files)
    return {
        "built": built, "profile": profile, "budgets": budgets, "mesh_source": mesh_source,
        "joints_source": fields["joints_source"], "code": code, "code_records": code_records,
        "scores": scores, "missing_files": [n for n in expected if n not in files],
        "missing_fields": [k for k, v in fields.items() if v is None],
    }


# ---------------------------------------------------------------------------------------------------------
# Freeze


def freeze(run: str | os.PathLike, label: str, store: str | os.PathLike | None = None, *, body: str = "04-body",
           sketch: str | None = None, milestone: str | None = None, note: str | None = None,
           by: str | None = None, allow_dirty: bool = False, allow_unknown_code: bool = False) -> dict:
    """Freeze `run/<body>` and the run's records as `<store>/<sketch>/<label>/`. Returns the written
    `version.json` plus `path` (the version folder) and `warnings`."""
    root = store_root(store)
    run = Path(run).expanduser()
    check_name("label", label)
    if not run.is_dir():
        raise VersionError(f"run folder {run} not found")
    if not BODY_DIR.match(body):
        raise VersionError(f"body {body!r} must be a stage 4 body folder name: 04-body or 04-body-<name>")
    for flag, text in (("--milestone", milestone), ("--note", note), ("--by", by)):
        if text is not None and names_a_path(text):
            raise VersionError(f"{flag} names a path ({text[:80]!r}); a version record never holds an absolute "
                               f"path or a .. step")
    if _walk(run, body, want_dir=True) is None or _walk(run, f"{body}/high.glb") is None:
        raise VersionError(f"{run.name}/{body}/high.glb not found: there is no body to freeze")
    files, overrides, own_stage3 = _collect(run, body)
    inputs = _hash_inputs(run)
    run_records = _run_records(run)

    sketch_name = _sketch_folder(root, sketch, inputs["sketch"], run)
    sketch_dir = root / sketch_name
    dest = sketch_dir / label
    for p, what in ((sketch_dir, f"sketch folder {sketch_name}"), (dest, f"{sketch_name}/{label}")):
        if p.is_symlink():
            raise VersionError(f"{what} in the store is a symbolic link; a version is written only into the "
                               f"store's own folders")
    if _empty_claim(dest):
        raise VersionError(f"{sketch_name}/{label} is an empty folder left by an interrupted freeze (it claimed "
                           f"the label, then stopped); remove that empty folder and freeze again")
    if dest.exists():
        raise VersionError(f"version {sketch_name}/{label} already exists; labels are write-once, pick a new one")
    try:
        who = by if by else getpass.getuser()
    except (OSError, KeyError):
        who = None

    made_sketch_dir = not sketch_dir.exists()
    sketch_dir.mkdir(exist_ok=True)
    tmp = reserved = None
    warnings: list[str] = []
    try:
        tmp = Path(tempfile.mkdtemp(prefix=f"{PARTIAL_PREFIX}{label}-", dir=sketch_dir))
        hashes, sizes = {}, {}
        for name in sorted(files):
            hashes[name], sizes[name] = _copy_file(files[name], tmp / name)
        # Records are read from the copies: what version.json says is what the version holds.
        records = {n: _read_json(tmp / n, f"{run.name}/{files[n].relative_to(run).as_posix()}",
                                 "; a version never freezes a damaged record")
                   for n in sorted(files) if n.endswith(".json")}
        d = _describe(records, files, own_stage3=own_stage3, body=body, allow_dirty=allow_dirty,
                      allow_unknown_code=allow_unknown_code, run_name=run.name)
        missing_fields = d["missing_fields"] + [k for k, v in (
            ("inputs.sketch", inputs["sketch"]), ("inputs.views", inputs["views"] or None),
            ("inputs.heldout", inputs["heldout"] or None)) if v is None]
        meta = {
            "format": FORMAT,
            "label": label,
            "sketch": sketch_name,
            "milestone": milestone,
            "note": note,
            "frozen": datetime.now().astimezone().isoformat(timespec="seconds"),
            "frozen_by": who,
            "built": d["built"],
            "profile": d["profile"],
            "budgets": d["budgets"],
            "mesh_source": d["mesh_source"],
            "joints_source": d["joints_source"],
            "code": d["code"],
            "code_records": d["code_records"],
            "allow_dirty": bool(allow_dirty),
            "allow_unknown_code": bool(allow_unknown_code),
            "source": {"run": _inside_store(run, root), "run_name": run.name, "run_records_sha256": run_records,
                       "body": body, "from_body_folder": sorted(overrides)},
            "inputs": inputs,
            "scores": d["scores"],
            "missing": {"files": d["missing_files"], "fields": missing_fields},
            "files_sha256": hashes,
            "files_bytes": sizes,
            "immutable": True,
        }
        _no_paths(meta, "")
        with open(tmp / VERSION_FILE, "x") as f:
            f.write(json.dumps(meta, indent=2) + "\n")
            f.flush()
            os.fsync(f.fileno())
        try:
            for p in tmp.iterdir():
                p.chmod(0o444)
        except OSError:
            pass
        if any(_writable_bits(p) for p in tmp.iterdir()):
            warnings.append("the version's files are still writable: this filesystem ignores permissions")
        _fsync_dir(tmp)
        # Rename without replacing: on POSIX a rename replaces an empty folder, so the label is first claimed
        # with mkdir (which fails if anything is there) and the rename replaces only that claim. On Windows
        # a rename never replaces a folder.
        if os.name != "nt":
            try:
                os.mkdir(dest)
            except FileExistsError:
                raise VersionError(f"{sketch_name}/{label} was claimed by another freeze while this one ran; "
                                   f"labels are write-once, pick a new one") from None
            reserved = dest
        os.rename(tmp, dest)
        tmp = reserved = None
    except BaseException:
        if tmp is not None:
            remove_tree(tmp)
        if reserved is not None:
            try:
                reserved.rmdir()
            except OSError:
                pass
        if made_sketch_dir:
            try:
                sketch_dir.rmdir()                          # only when empty: never another version's folder
            except OSError:
                pass
        raise
    _fsync_dir(sketch_dir)
    try:
        dest.chmod(0o555)
    except OSError:
        pass
    if _writable_bits(dest):
        warnings.append("the version folder is still writable: this filesystem ignores permissions")
    return {**meta, "path": str(dest), "warnings": warnings}


# ---------------------------------------------------------------------------------------------------------
# Finding, loading, listing


def _empty_claim(path: Path) -> bool:
    """An empty `<label>` folder: the claim of a freeze that was killed between claiming and renaming."""
    try:
        return path.is_dir() and not path.is_symlink() and next(path.iterdir(), None) is None
    except OSError:
        return False


def _visible(path: Path) -> None:
    """Refuse a hidden folder: an unfinished freeze (`.partial-*`) is never a version."""
    for name in (Path(path).name, Path(path).resolve().name):
        if name.startswith("."):
            raise VersionError(f"{name} is a hidden folder, not a version (an unfinished freeze is named "
                               f"{PARTIAL_PREFIX}*)")


def load_version(path: Path) -> dict:
    """A version folder's `version.json` (any format: the hand-frozen format 0 has no `format` field)."""
    _visible(path)
    vf = Path(path) / VERSION_FILE
    if not vf.is_file():
        raise VersionError(f"{path} is not a version (no {VERSION_FILE})")
    meta = _read_json(vf, f"{Path(path).name}/{VERSION_FILE}")
    if not isinstance(meta, dict):
        raise VersionError(f"{Path(path).name}/{VERSION_FILE} is not a JSON object")
    return meta


def resolve_version(ref: str, store: str | os.PathLike | None = None) -> Path:
    """A version folder from a path, from `<sketch>/<label>` in the store, or from a label that is unique in
    the store."""
    p = Path(ref).expanduser()
    if (p / VERSION_FILE).is_file():
        _visible(p)
        return p
    parts = ref.strip("/").split("/")
    if len(parts) > 2 or any(not NAME.match(x) for x in parts):
        raise VersionError(f"version {ref!r} not found (give a version folder, <sketch>/<label> or a label)")
    root = store_root(store)
    if len(parts) == 2:
        d = root / parts[0] / parts[1]
        if d.parent.is_symlink() or d.is_symlink():
            raise VersionError(f"version {ref} in the store goes through a symbolic link; it is not read")
        if (d / VERSION_FILE).is_file():
            return d
        raise VersionError(f"version {ref} not found in {root}")
    hits = sorted(d for d in root.glob(f"*/{parts[0]}") if (d / VERSION_FILE).is_file()
                  and not d.parent.name.startswith(".") and not d.parent.is_symlink() and not d.is_symlink())
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise VersionError(f"version {ref} not found in {root}")
    raise VersionError(f"label {ref} is in several sketch folders ({', '.join(h.parent.name for h in hits)}); "
                       f"give <sketch>/{ref}")


def _commit(meta: dict) -> tuple[str | None, bool | None]:
    code = meta.get("code")
    if isinstance(code, dict):
        return code.get("commit"), code.get("dirty")
    return meta.get("commit") if isinstance(meta.get("commit"), str) else None, meta.get("dirty")


def summary(path: Path, meta: dict) -> dict:
    """The fields `list` and `compare` show, read from either format. The label is the folder's name;
    `label_recorded` is what `version.json` says when that differs."""
    commit, dirty = _commit(meta)
    files = meta.get("files_sha256") if isinstance(meta.get("files_sha256"), dict) else {}
    recorded = meta.get("label")
    return {
        "sketch": path.parent.name,
        "label": path.name,
        "label_recorded": recorded if recorded != path.name else None,
        "format": meta.get("format", 0),
        "milestone": meta.get("milestone"),
        "note": meta.get("note"),
        "frozen": meta.get("frozen"),
        "frozen_by": meta.get("frozen_by"),
        "built": meta.get("built"),
        "profile": meta.get("profile"),
        "budgets": meta.get("budgets"),
        "mesh_source": meta.get("mesh_source"),
        "joints_source": meta.get("joints_source", meta.get("joints")),
        "commit": commit,
        "dirty": dirty,
        "code_text": meta.get("code") if isinstance(meta.get("code"), str) else None,
        "provenance": meta.get("provenance"),
        "files": len(files),
        "missing": meta.get("missing"),
    }


def list_versions(store: str | os.PathLike | None = None, sketch: str | None = None) -> dict:
    """Every version in the store (optionally one sketch's), and any unfinished freeze folders."""
    root = store_root(store)
    if sketch is not None:
        check_name("sketch", sketch)
    versions, unfinished, claims, broken = [], [], [], []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if sketch is not None and d.name != sketch:
            continue
        if d.is_symlink():
            broken.append(f"{d.name} (a symbolic link, not read)")
            continue
        for v in sorted(p for p in d.iterdir() if p.is_dir()):
            if v.name.startswith(PARTIAL_PREFIX):
                unfinished.append(f"{d.name}/{v.name}")
                continue
            if v.name.startswith("."):
                continue
            if v.is_symlink():
                broken.append(f"{d.name}/{v.name} (a symbolic link, not read)")
                continue
            if _empty_claim(v):
                claims.append(f"{d.name}/{v.name}")
                continue
            if not (v / VERSION_FILE).is_file():
                broken.append(f"{d.name}/{v.name}")
                continue
            try:
                versions.append(summary(v, load_version(v)))
            except VersionError as exc:
                broken.append(f"{d.name}/{v.name} ({exc})")
    return {"store": str(root), "versions": versions, "unfinished": unfinished, "empty_claims": claims,
            "not_versions": broken}


# ---------------------------------------------------------------------------------------------------------
# Verify


def verify(path: Path) -> dict:
    """Re-hash every file `version.json` lists. `ok` only when every listed file is a regular file with its
    recorded hash, every name is a plain file name, the folder holds nothing unlisted and `version.json`
    names the folder's own label. Every problem is reported; none stops the check."""
    path = Path(path)
    meta = load_version(path)
    hashes = meta.get("files_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise VersionError(f"{path.name}/{VERSION_FILE} lists no files_sha256")
    for name, want in hashes.items():
        if not isinstance(want, str) or not SHA256.match(want):
            raise VersionError(f"{path.name}/{VERSION_FILE}: files_sha256[{name!r}] is {want!r}, not a SHA-256 "
                               f"hex digest")
    report = {"version": f"{path.parent.name}/{path.name}", "checked": 0, "mismatched": [], "missing": [],
              "not_regular": [], "unreadable": [], "unlisted": [], "unsafe": [],
              "label_recorded": meta.get("label") if meta.get("label") not in (None, path.name) else None}
    for name, want in sorted(hashes.items()):
        if not safe_file_name(name):
            report["unsafe"].append(name)
            continue
        p = path / name
        if p.is_symlink() or (p.exists() and not p.is_file()):
            report["not_regular"].append(name)
            continue
        if not p.exists():
            report["missing"].append(name)
            continue
        try:
            got = sha256_file(p)
        except OSError as exc:
            report["unreadable"].append({"file": name, "error": exc.strerror or exc.__class__.__name__})
            continue
        report["checked"] += 1
        if got != want:
            report["mismatched"].append({"file": name, "recorded": want, "found": got})
    listed = set(hashes) | {VERSION_FILE}
    report["unlisted"] = sorted(p.name for p in path.iterdir() if p.name not in listed)
    report["ok"] = not any(report[k] for k in ("mismatched", "missing", "not_regular", "unreadable", "unlisted",
                                               "unsafe", "label_recorded"))
    return report

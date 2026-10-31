"""Where a user's files live on the data volume, and the only way the web layer reaches them.

    <data>/users/<user-id>/inputs/<job-id>.png    the cleaned upload (prompt sketches live in the run folder)
    <data>/users/<user-id>/runs/<job-id>/         the job's run folder (docs/ARCHITECTURE.md, "Run folder")
    <data>/users/<user-id>/cache/                 the model client's record/replay cache for this user
    <data>/users/<user-id>/cache-remote/          the remote-model cache for this user

User and job ids are made by the service (`store.new_id`) and checked against their pattern before they touch a
path, so no request text ever becomes a path segment except a run-relative file path, which `run_file` accepts
only as plain name segments of an allowed type that resolve, with no link along the way, inside that run folder.
Caches are per user: a cached generation is never returned to another account.
"""

from __future__ import annotations

import os
import re
import shutil
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from herogen.service.store import valid_id

RUN_FILE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".glb": "model/gltf-binary",
    ".json": "application/json",
}
SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
MAX_REL_PARTS = 6


class VolumeView:
    """Local folders: nothing to reload or commit."""

    def refresh(self, max_age_s: float = 0.0) -> None:
        pass

    def commit(self) -> None:
        pass

    @contextmanager
    def reading(self) -> Iterator[None]:
        yield


class ModalVolumeView(VolumeView):
    """A mounted Modal Volume seen by a long-lived web container. Reads hold a shared lock while a file is open;
    `refresh` (a throttled `reload`, which fails while any file is open) and `commit` take it exclusively."""

    def __init__(self, volume, min_interval_s: float = 2.0) -> None:
        self.volume = volume
        self.min_interval_s = min_interval_s
        self.cond = threading.Condition()
        self.readers = 0
        self.writer = False
        self.last = 0.0

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        with self.cond:
            while self.writer or self.readers:
                self.cond.wait()
            self.writer = True
        try:
            yield
        finally:
            with self.cond:
                self.writer = False
                self.cond.notify_all()

    def refresh(self, max_age_s: float | None = None) -> None:
        if time.monotonic() - self.last < (self.min_interval_s if max_age_s is None else max_age_s):
            return
        with self._exclusive():
            self.volume.reload()
            self.last = time.monotonic()

    def commit(self) -> None:
        with self._exclusive():
            self.volume.commit()

    @contextmanager
    def reading(self) -> Iterator[None]:
        with self.cond:
            while self.writer:
                self.cond.wait()
            self.readers += 1
        try:
            yield
        finally:
            with self.cond:
                self.readers -= 1
                self.cond.notify_all()


def _check(user_id: str, job_id: str | None = None) -> None:
    if not valid_id(user_id, "u") or (job_id is not None and not valid_id(job_id, "j")):
        raise ValueError("bad id")


class Storage:
    def __init__(self, root: Path, volume: VolumeView | None = None) -> None:
        self.root = Path(root)
        self.volume = volume or VolumeView()

    def user_dir(self, user_id: str) -> Path:
        _check(user_id)
        return self.root / "users" / user_id

    def runs_root(self, user_id: str) -> Path:
        return self.user_dir(user_id) / "runs"

    def run_dir(self, user_id: str, job_id: str) -> Path:
        _check(user_id, job_id)
        return self.runs_root(user_id) / job_id

    def input_path(self, user_id: str, job_id: str) -> Path:
        _check(user_id, job_id)
        return self.user_dir(user_id) / "inputs" / f"{job_id}.png"

    def cache_dir(self, user_id: str) -> Path:
        return self.user_dir(user_id) / "cache"

    def remote_cache_dir(self, user_id: str) -> Path:
        return self.user_dir(user_id) / "cache-remote"

    def save_input(self, user_id: str, job_id: str, png: bytes) -> Path:
        p = self.input_path(user_id, job_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        part = p.with_name(p.name + ".part")
        part.write_bytes(png)
        os.replace(part, p)
        return p

    def run_file(self, user_id: str, job_id: str, rel: str) -> tuple[Path, str] | None:
        """(file, content type) for a run-relative path, or None when it is refused or missing."""
        try:
            run = self.run_dir(user_id, job_id)
        except ValueError:
            return None
        if not isinstance(rel, str) or not rel or "\\" in rel or "\x00" in rel or rel.startswith("/"):
            return None
        parts = rel.split("/")
        if len(parts) > MAX_REL_PARTS or any(not SAFE_SEGMENT.match(p) or ".." in p for p in parts):
            return None
        ctype = RUN_FILE_TYPES.get(Path(parts[-1]).suffix.lower())
        if ctype is None:
            return None
        try:
            run_r = run.resolve(strict=True)
            target = run_r.joinpath(*parts)
            resolved = target.resolve(strict=True)
        except (OSError, RuntimeError):
            return None
        # Equal only when no segment is a link: a link inside the run, to anywhere, is refused.
        if resolved != target or not resolved.is_file() or run.is_symlink():
            return None
        return resolved, ctype

    def bytes_used(self, user_id: str, job_id: str) -> int:
        total = 0
        for base in (self.run_dir(user_id, job_id), self.input_path(user_id, job_id)):
            if base.is_file():
                total += base.stat().st_size
            elif base.is_dir():
                for dirpath, _dirs, files in os.walk(base):
                    for f in files:
                        p = Path(dirpath) / f
                        if not p.is_symlink():
                            total += p.stat().st_size
        return total

    def delete_job(self, user_id: str, job_id: str) -> None:
        run = self.run_dir(user_id, job_id)
        if run.is_dir() and not run.is_symlink():
            shutil.rmtree(run)
        elif run.is_symlink():
            run.unlink()
        inp = self.input_path(user_id, job_id)
        if inp.exists() or inp.is_symlink():
            inp.unlink()

    def inputs_before(self, cutoff: float) -> list[tuple[str, str, Path]]:
        """(user id, job id, path) of every saved input last written before `cutoff` (epoch seconds)."""
        out = []
        users = self.root / "users"
        if not users.is_dir():
            return out
        for udir in users.iterdir():
            if not valid_id(udir.name, "u") or udir.is_symlink() or not (udir / "inputs").is_dir():
                continue
            for p in (udir / "inputs").glob("*.png"):
                if valid_id(p.stem, "j") and not p.is_symlink() and p.stat().st_mtime < cutoff:
                    out.append((udir.name, p.stem, p))
        return out

    def purge_caches(self, cutoff: float) -> int:
        """Delete every user's cache entries (model and remote-model caches) written before `cutoff` (epoch
        seconds); returns how many files went."""
        n = 0
        users = self.root / "users"
        if not users.is_dir():
            return 0
        for udir in users.iterdir():
            if not valid_id(udir.name, "u") or udir.is_symlink():
                continue
            for cache in (udir / "cache", udir / "cache-remote"):
                if not cache.is_dir() or cache.is_symlink():
                    continue
                for dirpath, dirs, files in os.walk(cache, topdown=False):
                    for f in files:
                        p = Path(dirpath) / f
                        if p.is_symlink() or p.stat().st_mtime < cutoff:
                            p.unlink()
                            n += 1
                    if Path(dirpath) != cache and not any(Path(dirpath).iterdir()):
                        Path(dirpath).rmdir()
        return n

    def delete_user(self, user_id: str) -> None:
        d = self.user_dir(user_id)
        if d.is_dir() and not d.is_symlink():
            shutil.rmtree(d)

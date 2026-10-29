"""Durable copies of the metadata database on a Modal Volume, with one writer fenced by epochs.

A Volume is not a shared filesystem: a container sees other containers' writes only after `reload()`, its own
writes reach durable storage only on `commit()`, and when two containers write the same file the last commit
wins. There is no compare-and-swap. SQLite's own locking therefore cannot protect a database file kept on a
Volume, and no file that two containers both write can serve as a fence either, since a commit that started
before a check can still land after it. So nothing here is ever written by two containers: every file name holds
its writer's id, and the newest state is decided by the names, not by any file's content.

The store's container keeps the live database on its local disk and, after every write transaction, copies it
whole with SQLite's online backup into a new snapshot file on the Volume:

    <volume>/meta/lease-<epoch>-<owner>               one per store container start (empty)
    <volume>/meta/meta-<epoch>-<owner>-<gen>.db       a snapshot, written by store `owner` in `epoch`

Each store container has a random owner id. Leases and snapshots are ordered by (epoch, owner) and snapshots then
by generation; the lease holder is the largest (epoch, owner) present, and the newest state is the largest
(epoch, owner, generation) snapshot that SQLite can read.

- **Start** (`acquire`, then `restore`): a store adds `lease-<n>-<owner>` with n one above every lease it sees,
  commits, reloads and checks its lease is the largest (else `Superseded`). Then it restores the newest snapshot.
- **Save**: reload; refuse with `Superseded` if a larger lease or a snapshot of a larger epoch exists, and with
  `SnapshotConflict` if the newest snapshot is not the one this store last restored or wrote (the caller then
  restores and retries). Otherwise write the next generation under this store's own name, commit, reload, and
  report the write saved only if this store's lease is still the largest; else `Superseded`.

Snapshots are written in rollback-journal mode and read as immutable, so a snapshot is one self-contained file:
no -wal, -shm or journal file is ever created beside it (side files left by older versions are removed).

What this guarantees:

1. If `save` returns, its snapshot is on the Volume, and from then on every store that starts restores that
   snapshot or one written later by a store that restored it (directly or through such later snapshots), unless
   one of those snapshots is found corrupt when it is restored (below). No store container, stale or new, can
   replace it or outrank it: a newer store's lease landed after its commit (else `save` would not have returned),
   so the newer store restored it, and everything the older store commits afterwards ranks below the newer
   store's epoch.
2. A save refused before writing (`SnapshotConflict`: this store's copy is not the newest) wrote nothing to the
   Volume; the store restores and retries the call. A save that cannot be confirmed after its commit
   (`Superseded`) is reported as failed and not retried; that write may still survive, if a newer store restored
   it.
3. A store that has been superseded never writes again.

Reading: a snapshot is used only after its local copy has the size of the file on the Volume and SQLite has opened
the copy and `PRAGMA integrity_check` passed. A file that cannot be read or opened right now (an I/O error, a short
read) stops the store from starting (`SnapshotUnreadable`); it never falls back. A file proven corrupt (not a
database, a malformed image, a failed integrity check, which is also how a torn or partly written file shows) is
moved to quarantine on the Volume (`quarantine-<owner>-<name>`: no container restores, prunes or counts it), logged,
and the previous snapshot is restored; the writes only that file held are then lost. Three snapshots are kept, not
counting quarantined ones. The store starts from an empty database only on a Volume holding no snapshot file, and
no quarantined one either unless an operator allowed it. A snapshot that stays unreadable (a permanent I/O error, a
corruption SQLite reports as an open error) keeps the store from starting until an operator quarantines it with
`quarantine_snapshot` (`admin snapshot-quarantine`).
"""

from __future__ import annotations

import logging
import os
import re
import secrets
import shutil
import sqlite3
from pathlib import Path
from typing import Protocol

log = logging.getLogger("herogen.service")
SQLITE_HEADER = b"SQLite format 3\x00"
_SIDE_FILE = re.compile(r"^meta-.*\.db-(wal|shm|journal)$")
QUARANTINE = "quarantine-"        # quarantine-<owner>-<snapshot name>: kept, never restored, never pruned
ALLOW_EMPTY = "allow-empty-"      # allow-empty-<owner>: an operator let the store start without a snapshot

KEEP = 3
_SNAP = re.compile(r"^meta-(\d{8})-([0-9a-f]+)-(\d{10})\.db$")
_LEASE = re.compile(r"^lease-(\d{8})-([0-9a-f]+)$")
SnapKey = tuple[int, str, int]   # (epoch, owner, generation)
LeaseKey = tuple[int, str]       # (epoch, owner)


class VolumeLike(Protocol):
    def reload(self) -> None: ...

    def commit(self) -> None: ...


class LocalVolume:
    """A folder standing in for a Volume (dev mode and tests): reload and commit do nothing."""

    def reload(self) -> None:
        pass

    def commit(self) -> None:
        pass


class SnapshotConflict(RuntimeError):
    """The newest snapshot is not the one this store holds. Restore the newest snapshot and retry."""


class Superseded(SnapshotConflict):
    """A store container with a larger epoch exists; this one must not write again."""


class SnapshotUnreadable(RuntimeError):
    """A snapshot could not be read (an I/O or open error, not proven corruption), or every snapshot on the Volume
    is corrupt. The store must not start: starting from an older snapshot, or from nothing, would drop confirmed
    writes for good."""


def check_snapshot(path: Path) -> bool:
    """True when `path` is a whole, healthy SQLite database; False when it is provably corrupt (not a database,
    a malformed image, or `PRAGMA integrity_check` reports a problem). Anything else (the file cannot be read or
    opened right now) raises SnapshotUnreadable. The file is opened read-only and immutable, so SQLite creates no
    side files beside it."""
    try:
        with open(path, "rb") as f:
            header = f.read(len(SQLITE_HEADER))
    except OSError as e:
        raise SnapshotUnreadable(f"{path.name}: cannot be read ({e.strerror or type(e).__name__})") from None
    if header != SQLITE_HEADER:
        return False
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
        try:
            return con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        finally:
            con.close()
    except sqlite3.DatabaseError as e:
        name = getattr(e, "sqlite_errorname", "") or ""
        text = str(e).lower()
        if name.startswith(("SQLITE_CORRUPT", "SQLITE_NOTADB")) or (
                not name and ("malformed" in text or "not a database" in text)):
            return False
        raise SnapshotUnreadable(f"{path.name}: cannot be opened ({name or type(e).__name__}: {e})") from None


class Snapshotter:
    def __init__(self, volume_dir: Path, local_path: Path, volume: VolumeLike | None = None,
                 owner: str | None = None) -> None:
        self.dir = Path(volume_dir)
        self.local = Path(local_path)
        self.volume = volume or LocalVolume()
        self.owner = owner or secrets.token_hex(8)
        if not re.fullmatch(r"[0-9a-f]+", self.owner):
            raise ValueError("owner must be lower-case hex")
        self.epoch: LeaseKey = (0, "")
        self.last: SnapKey | None = None      # the snapshot this store's local database holds
        self.quarantined: list[str] = []      # files this container moved to quarantine
        self.superseded = False

    # ---- names --------------------------------------------------------------------------------------------

    def _leases(self) -> list[LeaseKey]:
        out = []
        for p in self.dir.glob("lease-*"):
            m = _LEASE.match(p.name)
            if m:
                out.append((int(m.group(1)), m.group(2)))
        return out

    def _snapshots(self) -> dict[SnapKey, Path]:
        out = {}
        for p in self.dir.glob("meta-*.db"):
            m = _SNAP.match(p.name)
            if m:
                out[(int(m.group(1)), m.group(2), int(m.group(3)))] = p
        return out

    def _snap_path(self, key: SnapKey) -> Path:
        return self.dir / f"meta-{key[0]:08d}-{key[1]}-{key[2]:010d}.db"

    @property
    def head(self) -> SnapKey | None:
        """The snapshot this store holds (restored or last written)."""
        return self.last

    # ---- lease, restore, save -----------------------------------------------------------------------------

    def acquire(self) -> LeaseKey:
        """Add this container's lease, one epoch above every lease seen, and check that it is the largest."""
        self.volume.reload()
        self.dir.mkdir(parents=True, exist_ok=True)
        n = max((e for e, _ in self._leases()), default=0) + 1
        (self.dir / f"lease-{n:08d}-{self.owner}").write_bytes(b"")
        self.volume.commit()
        self.volume.reload()
        mine = (n, self.owner)
        if max(self._leases()) != mine:
            self.superseded = True
            raise Superseded("another store container started at the same time and holds a larger lease")
        self.epoch = mine
        return mine

    def restore(self) -> int:
        """Copy the newest healthy snapshot to the local path (replacing any local copy) and return its
        generation; 0 only on a Volume with no snapshot, and no quarantined one unless an operator allowed it.

        Each candidate is copied to local disk as plain bytes, its size compared with the file on the Volume, and
        checked there (`check_snapshot`). A snapshot proven corrupt is moved to quarantine on the Volume
        (`quarantine-<owner>-<name>`, which no container restores or prunes), logged, and the next older one is
        tried. A snapshot that cannot be read right now, or comes back short, raises SnapshotUnreadable, as does a
        Volume left with no snapshot: the store then does not start, rather than start from an older state or an
        empty one. `quarantine_snapshot` is the operator's way past a snapshot that never becomes readable."""
        self.volume.reload()
        self.dir.mkdir(parents=True, exist_ok=True)
        for suffix in ("", "-wal", "-shm", "-journal", ".candidate"):
            p = Path(str(self.local) + suffix)
            if p.exists():
                p.unlink()
        self.local.parent.mkdir(parents=True, exist_ok=True)
        snaps = self._snapshots()
        candidate = Path(str(self.local) + ".candidate")
        for key in sorted(snaps, reverse=True):
            src = snaps[key]
            try:
                expected = src.stat().st_size
                shutil.copyfile(src, candidate)
                got = candidate.stat().st_size
            except OSError as e:
                log.error("snapshot %s cannot be read (%s); the store will not start", src.name,
                          e.strerror or type(e).__name__)
                raise SnapshotUnreadable(f"{src.name}: cannot be read ({e.strerror or type(e).__name__})") from None
            if got != expected:
                log.error("snapshot %s read back %d of %d bytes; the store will not start", src.name, got, expected)
                raise SnapshotUnreadable(f"{src.name}: read back {got} of {expected} bytes; the store will not "
                                         "start")
            try:
                healthy = check_snapshot(candidate)
            except SnapshotUnreadable as e:
                log.error("snapshot %s cannot be opened (%s); the store will not start", src.name, e)
                raise SnapshotUnreadable(f"{src.name}: cannot be opened; the store will not start") from None
            if healthy:
                os.replace(candidate, self.local)
                self.last = key
                return key[2]
            moved = self._quarantine(src)
            log.error("snapshot %s is corrupt; it is kept as %s and the previous snapshot is restored, so the writes "
                      "it alone held are lost", src.name, moved)
        if candidate.exists():
            candidate.unlink()
        if any(self.dir.glob("meta-*.db")):
            log.error("the Volume holds snapshot files this version does not recognise; the store will not start")
            raise SnapshotUnreadable("the Volume holds unrecognised snapshot files; refusing to start from an empty "
                                     "database")
        if any(self.dir.glob(QUARANTINE + "*")):
            if not any(self.dir.glob(ALLOW_EMPTY + "*")):
                log.error("every snapshot is quarantined; the store will not start empty unless an operator allows "
                          "it (admin snapshot-quarantine --allow-empty)")
                raise SnapshotUnreadable("every snapshot is quarantined; refusing to start from an empty database")
            log.warning("starting from an empty database: every snapshot is quarantined and an operator allowed it")
        self.last = None
        return 0

    def _quarantine(self, src: Path) -> str:
        """Move a snapshot to quarantine on the Volume (committed), where it stays for inspection."""
        dst = self.dir / f"{QUARANTINE}{self.owner}-{src.name}"
        os.replace(src, dst)
        self.volume.commit()
        self.quarantined.append(dst.name)
        return dst.name

    def _check_lease(self, when: str) -> None:
        leases = self._leases()
        if self.superseded or not leases or max(leases) != self.epoch:
            self.superseded = True
            raise Superseded(f"a store container with a larger epoch exists ({when})")

    def save(self, db: sqlite3.Connection) -> int:
        """Snapshot `db` (the live local database) as the next generation; see the module docstring."""
        self.volume.reload()
        self._check_lease("before saving")
        snaps = self._snapshots()
        newest = max(snaps, default=None)
        if newest is not None and newest[:2] > self.epoch:
            self.superseded = True
            raise Superseded("a snapshot from a larger epoch exists")
        if newest != self.last:
            raise SnapshotConflict(f"the newest snapshot is {newest}, this store holds {self.last}")
        key: SnapKey = (self.epoch[0], self.epoch[1], (newest[2] if newest else 0) + 1)
        target = self._snap_path(key)
        part = target.with_name(target.name + ".part")
        if part.exists():
            part.unlink()
        dst = sqlite3.connect(str(part))
        try:
            db.backup(dst)
            # A rollback-journal header (not WAL): reading the snapshot then needs no -wal or -shm file beside it.
            dst.execute("PRAGMA journal_mode=DELETE")
        finally:
            dst.close()
        os.replace(part, target)
        for p in self.dir.iterdir():   # side files an older version left; an operator's allow-empty, now used
            if _SIDE_FILE.match(p.name) or p.name.startswith(ALLOW_EMPTY):
                p.unlink()
        # Old snapshots and older leases go with this commit; files a later epoch wrote are never touched (this
        # store would have refused above had it seen one).
        keep = set(sorted([*snaps, key], reverse=True)[:KEEP])
        for k, p in snaps.items():
            if k not in keep:
                p.unlink()
        for e, who in self._leases():
            if (e, who) < self.epoch:
                (self.dir / f"lease-{e:08d}-{who}").unlink()
        self.volume.commit()
        self.volume.reload()
        self._check_lease("while saving; this write may not be kept")
        self.last = key
        return key[2]


def snapshot_files(volume_dir: Path, volume: VolumeLike | None = None) -> list[str]:
    """Every snapshot, quarantined snapshot, lease and allow-empty marker on the Volume, by name."""
    (volume or LocalVolume()).reload()
    d = Path(volume_dir)
    if not d.is_dir():
        return []
    prefixes = ("meta-", QUARANTINE, "lease-", ALLOW_EMPTY)
    return sorted(p.name for p in d.iterdir() if p.name.startswith(prefixes))


def quarantine_snapshot(volume_dir: Path, volume: VolumeLike | None, name: str, *,
                        allow_empty: bool = False) -> dict:
    """Operator action for a store that will not start: move the named snapshot to quarantine so the store starts
    from the next one. It first takes the newest lease, so a store still running stops writing (it is superseded)
    and the next store to start restores from what is left. Quarantining the last snapshot is refused unless
    `allow_empty`, which also leaves a marker letting the next store start from an empty database. Nothing is
    deleted: the file stays on the Volume as `quarantine-<owner>-<name>`."""
    volume = volume or LocalVolume()
    d = Path(volume_dir)
    volume.reload()
    if not _SNAP.match(name) or not (d / name).is_file():
        raise ValueError(f"no snapshot named {name!r} (see `admin snapshots`)")
    others = [p.name for p in d.glob("meta-*.db") if p.name != name]
    if not others and not allow_empty:
        raise ValueError(f"{name} is the last snapshot; quarantining it leaves the store nothing to start from. "
                         "Pass --allow-empty to let the store start from an empty database.")
    op = Snapshotter(d, d / "operator-unused.db", volume)
    op.acquire()
    if not (d / name).is_file():
        raise ValueError(f"{name} disappeared while taking the lease; nothing was changed")
    dst = d / f"{QUARANTINE}{op.owner}-{name}"
    os.replace(d / name, dst)
    if not others:
        (d / f"{ALLOW_EMPTY}{op.owner}").write_bytes(b"")
    volume.commit()
    log.warning("operator quarantined snapshot %s as %s%s", name, dst.name,
                "; the store may start from an empty database" if not others else "")
    return {"quarantined": name, "as": dst.name, "remaining": sorted(others), "allow_empty": not others,
            "lease": op.epoch[0]}

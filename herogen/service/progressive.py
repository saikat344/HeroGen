"""Read the refinement publisher's atomic, read-only progress contract.

Only numbered files named by a valid contract are offered as progressive models. Missing, malformed or
partially available publications leave the first shape usable; they never break a job response.
"""

from datetime import datetime
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from herogen.service.bundle import _inside
from herogen.service.storage import SAFE_SEGMENT


def read_progress(run: Path) -> dict:
    empty = {"versions": [], "done": False}
    if run.is_symlink():
        return empty
    path = _inside(run, "stage-3f-progress.json")
    if path is None:
        return empty
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or not isinstance(data.get("versions"), list) or type(data.get("done")) is not bool:
            return empty
        previous = 0
        for version in data["versions"]:
            if not isinstance(version, dict):
                return empty
            n = version.get("n")
            if type(n) is not int or n <= previous or version.get("glb") != f"03-refined-v{n}.glb":
                return empty
            for key in ("accepted", "rejected", "pending"):
                views = version.get(key)
                if not isinstance(views, list) or any(not isinstance(v, str) or not SAFE_SEGMENT.fullmatch(v)
                                                      or ".." in v for v in views):
                    return empty
            datetime.fromisoformat(version["time"].replace("Z", "+00:00"))
            previous = n
        return {"versions": [{k: v[k] for k in ("n", "glb", "accepted", "rejected", "pending", "time")}
                             for v in data["versions"] if _inside(run, v["glb"]) is not None], "done": data["done"]}
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
        return empty


@lru_cache(maxsize=256)
def _digest(path: Path, modified: int, changed: int, size: int) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def version_token(path: Path) -> str:
    """Content identity, cached only while the file's size and timestamps stay unchanged."""
    stat = path.stat()
    return _digest(path, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)

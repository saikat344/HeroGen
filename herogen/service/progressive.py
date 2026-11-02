"""Read the refinement publisher's atomic, read-only progress contract.

Only numbered, complete GLB files named by a valid contract are offered as progressive models. Missing, malformed or
partially available publications leave the first shape usable; they never break a job response.
"""

from datetime import datetime
from functools import lru_cache
import hashlib
import json
import re
from pathlib import Path

from herogen.service.bundle import _inside, complete_glb
from herogen.service.storage import SAFE_SEGMENT


# View identifiers include signed angles, for example head+0 and body-90.
_VIEW = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+\-]{0,127}")


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
                if not isinstance(views, list) or any(not isinstance(v, str) or not _VIEW.fullmatch(v)
                                                      or ".." in v for v in views):
                    return empty
            datetime.fromisoformat(version["time"].replace("Z", "+00:00"))
            previous = n
        metadata = {}
        if "profile" in data:
            profile = data["profile"]
            if not isinstance(profile, str) or not SAFE_SEGMENT.fullmatch(profile) or ".." in profile:
                return empty
            metadata["profile"] = profile
        if "final_n" in data:
            final = data["final_n"]
            if final is not None and (type(final) is not int or not data["done"] or final != previous or final < 1):
                return empty
            if data["done"] and previous and final is None:
                return empty
            metadata["final_n"] = final
        versions = [{k: v[k] for k in ("n", "glb", "accepted", "rejected", "pending", "time")}
                    for v in data["versions"] if (glb := _inside(run, v["glb"])) is not None and complete_glb(glb)]
        if metadata.get("final_n") is not None and metadata["final_n"] not in {v["n"] for v in versions}:
            # The final file is missing or cut short: the last complete version is the one offered.
            metadata["final_n"] = versions[-1]["n"] if versions else None
        return {"versions": versions, "done": data["done"], **metadata}
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

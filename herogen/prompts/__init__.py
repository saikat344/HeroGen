"""Prompt registry: named, frozen prompt versions, kept as data files apart from the code that uses them.

Each version is one JSON file in ``versions/`` holding the exact prompt text, the model id and parameters of every
step, and ``sha256``, a hash of everything else in the file. A version is frozen: loading it recomputes the hash and
refuses a file whose content no longer matches, so changing any text or parameter needs a new version name.

Which version a stage uses comes from ``defaults.json`` (one entry per kind), overridden by the environment variable
named there or by an explicit name (a CLI flag); call sites never name a version themselves.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
VERSIONS = HERE / "versions"
DEFAULTS = HERE / "defaults.json"
_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*-v[0-9]+$")


class RegistryError(RuntimeError):
    """A prompt version is unknown, malformed, or no longer matches its frozen hash."""


def content_hash(data: dict[str, Any]) -> str:
    """SHA-256 of a version's canonical JSON (sorted keys, ASCII escapes), excluding its own ``sha256`` field."""
    body = {k: v for k, v in data.items() if k != "sha256"}
    canon = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PromptVersion:
    name: str
    kind: str
    sha256: str
    data: dict[str, Any]

    def step(self, name: str) -> dict[str, Any]:
        for s in self.data["steps"]:
            if s["name"] == name:
                return s
        raise RegistryError(f"prompt version {self.name} has no step {name!r}")

    def ref(self) -> dict[str, str]:
        """What every output records about the version it was made with."""
        return {"name": self.name, "sha256": self.sha256}


def available() -> list[str]:
    return sorted(p.stem for p in VERSIONS.glob("*.json"))


def load(name: str) -> PromptVersion:
    if not _NAME.match(name or ""):
        raise RegistryError(f"bad prompt version name {name!r} (expected something like 'hero-sketch-v1')")
    path = VERSIONS / f"{name}.json"
    if not path.is_file():
        raise RegistryError(f"unknown prompt version {name!r}; available: {', '.join(available()) or 'none'}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise RegistryError(f"prompt version {name} is not valid JSON: {e}") from e
    if data.get("name") != name:
        raise RegistryError(f"prompt version file {path.name} names itself {data.get('name')!r}")
    for key in ("kind", "steps", "sha256"):
        if key not in data:
            raise RegistryError(f"prompt version {name} has no {key!r}")
    actual = content_hash(data)
    if actual != data["sha256"]:
        raise RegistryError(
            f"prompt version {name} was changed after it was frozen (hash {actual[:12]} != {data['sha256'][:12]}); "
            "frozen versions never change, so put the new text under a new version name"
        )
    return PromptVersion(name=name, kind=data["kind"], sha256=actual, data=data)


def selected(kind: str, explicit: str | None = None) -> PromptVersion:
    """The version for ``kind``: ``explicit`` if given, else the environment override, else defaults.json."""
    defaults = json.loads(DEFAULTS.read_text(encoding="utf-8"))
    entry = defaults.get(kind)
    if not isinstance(entry, dict) or "version" not in entry:
        raise RegistryError(f"defaults.json has no version for {kind!r}")
    name = explicit or os.environ.get(entry.get("env", ""), "").strip() or entry["version"]
    version = load(name)
    if version.kind != kind:
        raise RegistryError(f"prompt version {name} is a {version.kind!r} prompt, not {kind!r}")
    return version

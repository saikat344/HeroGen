"""Versioned bodies: a write-once store of frozen stage 4 bodies, so any later build can be compared with them.

`python3 -m herogen.versions freeze | list | verify | compare` (see `__main__`). A version is one body with the
records that measured it, frozen from a run folder into `<store>/<sketch>/<label>/` with a `version.json` that
names its code, sources and inputs and holds every file's SHA-256. `store` freezes, finds, lists and verifies
versions; `compare` compares two. The format is in docs/ARCHITECTURE.md ("Versions").
"""
from __future__ import annotations

from .store import (ENV_STORE, FORMAT, VERSION_FILE, VersionError, freeze, list_versions, load_version,
                    resolve_version, store_root, verify)

__all__ = ["ENV_STORE", "FORMAT", "VERSION_FILE", "VersionError", "freeze", "list_versions", "load_version",
           "resolve_version", "store_root", "verify"]

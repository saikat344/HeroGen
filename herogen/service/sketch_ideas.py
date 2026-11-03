"""Proven sketches for the composer's "Suggest a sketch": a read-only folder of PNGs that already made good heroes.

The folder (`HEROGEN_SERVICE_SKETCH_IDEAS`) holds `catalog.json` and the images, side by side:

    {"version": 1, "items": [{"id": "iron-warden", "title": "Iron Warden", "file": "iron-warden.png",
                              "sha256": "<64 hex>", "job": "<the run that proved it>"}]}

Like the public gallery, the images stay out of the repository and every one is served only after its SHA-256
matches the catalog. `job` is provenance for the people curating the folder; it is never sent to the browser.
With no folder configured the list is empty and the composer hides the button.
"""
from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

ID = re.compile(r'^[a-z0-9][a-z0-9-]{0,63}$')
FILE = re.compile(r'^[a-z0-9][a-z0-9-]{0,63}\.png$')
SHA = re.compile(r'^[a-f0-9]{64}$')
JOB = re.compile(r'^[A-Za-z0-9_-]{1,64}$')
MAX_ITEMS = 24
MAX_CATALOG_BYTES = 64 * 1024
MAX_IMAGE_BYTES = 12 * 1024 ** 2   # the composer's default upload limit: a suggestion must be a valid upload
PNG_MAGIC = b'\x89PNG\r\n\x1a\n'


class SketchIdeasUnavailable(ValueError):
    pass


def _file(root: Path, name: str) -> Path:
    """The image `name` directly inside `root`: a plain file, no links, no path parts."""
    if not isinstance(name, str) or not FILE.fullmatch(name):
        raise SketchIdeasUnavailable('Invalid sketch file name.')
    base = root.resolve(strict=True)
    target = base / name
    if target.is_symlink() or target.resolve(strict=True) != target or target.parent != base or not target.is_file():
        raise SketchIdeasUnavailable('Sketch file is unavailable.')
    return target


def _entries(root: Path) -> list[dict]:
    """The catalog's items, each checked for shape and with an image on disk; raises on any bad entry."""
    path = root / 'catalog.json'
    if path.is_symlink() or path.stat().st_size > MAX_CATALOG_BYTES:
        raise SketchIdeasUnavailable('Sketch catalog is unavailable.')
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get('version') != 1 or not isinstance(data.get('items'), list):
        raise SketchIdeasUnavailable('Sketch catalog has an unsupported format.')
    if len(data['items']) > MAX_ITEMS:
        raise SketchIdeasUnavailable('Sketch catalog has too many entries.')
    result, seen = [], set()
    for item in data['items']:
        if not isinstance(item, dict):
            raise SketchIdeasUnavailable('Invalid sketch entry.')
        key = item.get('id')
        if not isinstance(key, str) or not ID.fullmatch(key) or key in seen:
            raise SketchIdeasUnavailable('Invalid sketch identifier.')
        seen.add(key)
        title = item.get('title')
        if not isinstance(title, str) or not 1 <= len(title.strip()) <= 80:
            raise SketchIdeasUnavailable('Invalid sketch title.')
        if not isinstance(item.get('sha256'), str) or not SHA.fullmatch(item['sha256']):
            raise SketchIdeasUnavailable('Sketch has no valid integrity hash.')
        if not isinstance(item.get('job'), str) or not JOB.fullmatch(item['job']):
            raise SketchIdeasUnavailable('Sketch has no source run.')
        target = _file(root, item.get('file'))
        if not 0 < target.stat().st_size <= MAX_IMAGE_BYTES:
            raise SketchIdeasUnavailable('Sketch file is too large.')
        result.append({'id': key, 'title': title.strip(), 'path': target, 'sha256': item['sha256']})
    return result


def catalog(root: Path | None) -> list[dict]:
    """The public list: `[{"id", "title", "url"}]`, in catalog order. Empty when no folder is configured."""
    if root is None:
        return []
    try:
        return [{'id': e['id'], 'title': e['title'], 'url': f'/sketch-ideas/{e["id"]}?v={e["sha256"][:16]}'}
                for e in _entries(root)]
    except SketchIdeasUnavailable:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SketchIdeasUnavailable('Suggested sketches are temporarily unavailable.') from exc


@lru_cache(maxsize=64)
def _verified(path: str, expected: str, stamp: tuple) -> bool:
    with Path(path).open('rb') as stream:
        if stream.read(len(PNG_MAGIC)) != PNG_MAGIC:
            return False
        stream.seek(0)
        return hashlib.file_digest(stream, 'sha256').hexdigest() == expected


def image(root: Path | None, key: str) -> tuple[Path, str]:
    """(PNG file, sha256) of one listed sketch whose bytes match its catalog hash; raises otherwise."""
    if root is None or not isinstance(key, str) or not ID.fullmatch(key):
        raise SketchIdeasUnavailable('Sketch not found.')
    try:
        entry = next((e for e in _entries(root) if e['id'] == key), None)
        if entry is None:
            raise SketchIdeasUnavailable('Sketch not found.')
        path = entry['path']
        stat = path.stat()
        stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if not _verified(str(path), entry['sha256'], stamp):
            raise SketchIdeasUnavailable('Sketch failed its integrity check.')
        return path, entry['sha256']
    except SketchIdeasUnavailable:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SketchIdeasUnavailable('Sketch is unavailable.') from exc

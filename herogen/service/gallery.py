"""Read-only public catalog of explicitly published artifacts, separate from private runs."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path

ID = re.compile(r'^[a-z0-9][a-z0-9-]{0,63}$')
TYPES = {'sketch': ('.png', 'image/png'), 'artwork': ('.png', 'image/png'),
         'model': ('.glb', 'model/gltf-binary')}
# Small images for cards and pickers: `<kind>-thumb`, a WebP whose long side is THUMB_SIDE pixels (never enlarged).
THUMB_KINDS = ('sketch', 'artwork')
THUMB_SIDE = 512
THUMB_QUALITY = 82
THUMB_SUFFIX = '-thumb'


class GalleryUnavailable(ValueError):
    pass


def _file(root: Path, name: str) -> Path:
    parts = name.split('/')
    if len(parts) != 2 or not ID.fullmatch(parts[0]) or parts[1] not in ('sketch.png', 'artwork.png', 'model.glb'):
        raise GalleryUnavailable('Invalid public artifact path.')
    base = root.resolve(strict=True)
    target = base.joinpath(*parts)
    if target.resolve(strict=True) != target or not target.is_file():
        raise GalleryUnavailable('Public artifact is unavailable.')
    return target


def catalog(root: Path | None) -> list[dict]:
    if root is None:
        return []
    try:
        path = root / 'catalog.json'
        if path.is_symlink() or path.stat().st_size > 128 * 1024:
            raise GalleryUnavailable('Public catalog is unavailable.')
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or data.get('version') != 1 or not isinstance(data.get('items'), list):
            raise GalleryUnavailable('Public catalog has an unsupported format.')
        if len(data['items']) > 24:
            raise GalleryUnavailable('Public catalog has too many entries.')
        result, seen = [], set()
        for item in data['items']:
            key = item['id']
            if not isinstance(key, str) or not ID.fullmatch(key) or key in seen:
                raise GalleryUnavailable('Invalid public character identifier.')
            seen.add(key)
            if not isinstance(item.get('title'), str) or not 1 <= len(item['title']) <= 80:
                raise GalleryUnavailable('Invalid public character title.')
            if not isinstance(item.get('description', ''), str) or len(item.get('description', '')) > 400:
                raise GalleryUnavailable('Invalid public character description.')
            assets = {}
            for kind, (suffix, _mime) in TYPES.items():
                info = item['assets'][kind]
                if info['file'] != f'{key}/{kind}{suffix}':
                    raise GalleryUnavailable('Public artifact name does not match its entry.')
                target = _file(root, info['file'])
                if not isinstance(info.get('sha256'), str) or not re.fullmatch(r'[a-f0-9]{64}', info['sha256']):
                    raise GalleryUnavailable('Public artifact has no valid integrity hash.')
                if target.stat().st_size > 128 * 1024 * 1024:
                    raise GalleryUnavailable('Public artifact is too large.')
                assets[kind] = {'url': f'/gallery-assets/{key}/{kind}', 'bytes': target.stat().st_size}
                if kind in THUMB_KINDS:
                    assets[kind]['thumb_url'] = f'/gallery-assets/{key}/{kind}{THUMB_SUFFIX}?v={info["sha256"][:16]}'
            commit = item.get('code_commit')
            if not isinstance(commit, str) or not re.fullmatch(r'[a-f0-9]{40}', commit):
                raise GalleryUnavailable('Public artifact has no source version.')
            refinement = item.get('refinement')
            safe_record = None
            if refinement is not None:
                models = refinement['models']
                if not isinstance(models, list) or len(models) > 8 or any(
                        not isinstance(name, str) or not re.fullmatch(r'[a-z0-9.-]{1,60}', name) for name in models):
                    raise GalleryUnavailable('Invalid public model record.')
                safe_record = {'models': models}
                for field in ('seconds', 'estimated_cost_usd', 'accepted_views', 'attempted_views'):
                    value = refinement[field]
                    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                        raise GalleryUnavailable('Invalid public refinement measurement.')
                    if field.endswith('_views') and (not isinstance(value, int) or value > 10000):
                        raise GalleryUnavailable('Invalid public view count.')
                    safe_record[field] = value
                if safe_record['accepted_views'] > safe_record['attempted_views']:
                    raise GalleryUnavailable('Invalid public accepted-view count.')
            result.append({'id': key, 'title': item['title'], 'description': item.get('description', ''),
                           'code_commit': commit, 'input_kind': 'Generated sketch',
                           'label': 'Previously generated with HeroGen', 'refinement': safe_record, 'assets': assets})
        return result
    except GalleryUnavailable:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise GalleryUnavailable('The public gallery is temporarily unavailable.') from exc


@lru_cache(maxsize=128)
def _verified(path: str, expected: str, stamp: tuple) -> bool:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() == expected


def _checked(root: Path | None, key: str, kind: str) -> tuple[Path, str]:
    """(file, sha256) of a published artifact that passed its integrity check."""
    if kind not in TYPES or root is None or key not in {item['id'] for item in catalog(root)}:
        raise GalleryUnavailable('Public artifact not found.')
    data = json.loads((root / 'catalog.json').read_text())
    entry = next(item for item in data['items'] if item['id'] == key)
    info = entry['assets'][kind]
    path = _file(root, info['file'])
    stat = path.stat()
    stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if not _verified(str(path), info['sha256'], stamp):
        raise GalleryUnavailable('Public artifact failed its integrity check.')
    return path, info['sha256']


def artifact(root: Path | None, key: str, kind: str) -> tuple[Path, str]:
    return _checked(root, key, kind)[0], TYPES[kind][1]


def thumbnail(root: Path | None, cache: Path, key: str, kind: str) -> tuple[Path, str, bool]:
    """(WebP file, version token, whether it was made now) for a published image, made on first request.

    The file lives in `cache` (writable, unlike the published gallery), named by the source's hash, so a
    republished image gets a new one and the older one is removed. It is written to a temporary file in the same
    folder and renamed into place, so a half-written image is never served."""
    if kind not in THUMB_KINDS:
        raise GalleryUnavailable('Public artifact not found.')
    source, sha = _checked(root, key, kind)
    token = sha[:16]
    folder = cache / key
    dest = folder / f'{kind}-{token}.webp'
    if dest.is_file() and not dest.is_symlink():
        return dest, token, False
    from PIL import Image

    with Image.open(source) as image:
        image.load()
        alpha = image.mode in ('RGBA', 'LA', 'PA') or (image.mode == 'P' and 'transparency' in image.info)
        small = image.convert('RGBA' if alpha else 'RGB')
    small.thumbnail((THUMB_SIDE, THUMB_SIDE), Image.Resampling.LANCZOS)
    folder.mkdir(parents=True, exist_ok=True)
    fd, part = tempfile.mkstemp(dir=folder, prefix=f'.{kind}-', suffix='.part')
    try:
        with os.fdopen(fd, 'wb') as stream:
            small.save(stream, format='WEBP', quality=THUMB_QUALITY, method=6)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(part, dest)
    except BaseException:
        Path(part).unlink(missing_ok=True)
        raise
    for old in folder.glob(f'{kind}-*.webp'):
        if old != dest:
            old.unlink(missing_ok=True)
    return dest, token, True

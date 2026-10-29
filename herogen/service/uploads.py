"""Untrusted input: sketch uploads, prompts and hero names.

An upload is accepted only as PNG, JPEG or WebP, recognised by its leading bytes (the file name and the declared
content type are ignored), within the byte and pixel limits, with one frame. Pillow opens it with only those
three decoders enabled, checks the pixel count from the header before decoding anything (the decompression-bomb
limit), verifies it, decodes it, applies the EXIF orientation and converts it to sRGB. What is kept is a new RGB
image made from the pixels alone (transparency flattened onto white, at most `upload_max_side` pixels on its long
side), saved as a PNG with no metadata of any kind: no EXIF, GPS, XMP, ICC profile, text chunks or timestamps.

In production this runs in its own Modal container with no network and no secrets (`modal_app.sanitize_image`).
"""

from __future__ import annotations

import hashlib
import io
import re
import unicodedata
import warnings
from dataclasses import dataclass
from typing import Any

KINDS = {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}


class UploadRejected(ValueError):
    """The upload cannot be used. The message is written for the person who sent it."""


@dataclass(frozen=True)
class Limits:
    max_bytes: int
    max_pixels: int
    max_side: int
    min_side: int
    # Memory the decoded image may take, judged from its header before decoding (default: 4 bytes per allowed
    # pixel, what Pillow holds for a full-size RGB image). Conversions stay in 8 or 16 bits, row block by row block
    # for 32-bit images, so the work in memory stays a small multiple of this.
    max_decode_bytes: int | None = None

    @property
    def decode_budget(self) -> int:
        return self.max_decode_bytes if self.max_decode_bytes is not None else 4 * self.max_pixels


def decoded_bytes(mode: str, w: int, h: int) -> int:
    """How much memory Pillow takes for a decoded w x h image in `mode` (multi-band modes are 4 bytes a pixel)."""
    if mode in ("1", "L", "P"):
        per = 1
    elif mode.startswith("I;16"):
        per = 2
    else:
        per = 4
    return w * h * per


def sniff(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def _mb(n: int) -> str:
    return f"{n / 1024 / 1024:.0f} MB"


def sanitize(data: bytes, limits: Limits) -> tuple[bytes, dict[str, Any]]:
    """Clean PNG bytes and a summary, or UploadRejected."""
    from PIL import Image, ImageCms, ImageOps

    if not data:
        raise UploadRejected("The upload is empty.")
    if len(data) > limits.max_bytes:
        raise UploadRejected(f"The image is larger than {_mb(limits.max_bytes)}.")
    kind = sniff(data)
    if kind is None:
        raise UploadRejected("Upload a PNG, JPEG or WebP image.")
    fmt = KINDS[kind]
    unreadable = UploadRejected("The file could not be read as an image. Try saving it again as PNG or JPEG.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        try:
            with Image.open(io.BytesIO(data), formats=[fmt]) as head:
                w, h = head.size
                if w * h > limits.max_pixels:
                    raise UploadRejected(f"The image has too many pixels ({w} x {h}); the limit is "
                                         f"{limits.max_pixels / 1e6:.0f} megapixels.")
                if getattr(head, "n_frames", 1) != 1:
                    raise UploadRejected("Animated images are not supported; upload a single still image.")
                if decoded_bytes(head.mode, w, h) > limits.decode_budget:
                    raise UploadRejected(f"The image is too large to process ({w} x {h}, {head.mode}); save it "
                                         "as an 8-bit PNG or JPEG, or make it smaller.")
                head.verify()
            with Image.open(io.BytesIO(data), formats=[fmt]) as img:
                img.load()
                icc = img.info.get("icc_profile")
                img = ImageOps.exif_transpose(img)
                img = _to_rgb(img)
                if icc:
                    try:
                        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                        img = ImageCms.profileToProfile(img, src, ImageCms.createProfile("sRGB"), outputMode="RGB")
                    except (ImageCms.PyCMSError, OSError, ValueError, TypeError):
                        pass  # an unusable profile: keep the pixels as they are
        except UploadRejected:
            raise
        except (Image.DecompressionBombError, Image.DecompressionBombWarning):
            raise UploadRejected("The image is too large to open safely.") from None
        except (OSError, SyntaxError, ValueError, EOFError, MemoryError, IndexError, KeyError, TypeError,
                AttributeError, ZeroDivisionError):
            raise unreadable from None
    w, h = img.size
    if min(w, h) < limits.min_side:
        raise UploadRejected(f"The image is {w} x {h} pixels; it needs at least {limits.min_side} pixels on its "
                             "short side.")
    try:
        resized = max(w, h) > limits.max_side
        if resized:
            img.thumbnail((limits.max_side, limits.max_side), Image.Resampling.LANCZOS)
        clean = Image.new("RGB", img.size, (255, 255, 255))
        clean.paste(img)
        out = io.BytesIO()
        clean.save(out, format="PNG", optimize=False)
        png = out.getvalue()
    except MemoryError:
        raise UploadRejected("The image is too large to process; make it smaller.") from None
    return png, {"width": clean.size[0], "height": clean.size[1], "source_format": kind,
                 "source_size": [w, h], "resized": resized, "sha256": hashlib.sha256(png).hexdigest()}


def _to_rgb(img):
    from PIL import Image

    if img.mode.startswith("I;16"):
        import numpy as np

        # 16 bits to 8 in integers: the high byte (no float copy of the image).
        img = Image.fromarray((np.asarray(img).astype(np.uint16, copy=False) >> 8).astype(np.uint8))
    elif img.mode in ("I", "F"):
        import numpy as np

        a = np.asarray(img)
        top = float(a.max()) if a.size else 255.0
        scale = 255.0 / top if top > 255.0 else 1.0
        out = np.empty(a.shape, np.uint8)
        for r in range(0, a.shape[0], 256):  # float32 one block of rows at a time
            out[r:r + 256] = np.clip(a[r:r + 256].astype(np.float32) * scale + 0.5, 0, 255).astype(np.uint8)
        img = Image.fromarray(out)
    if img.mode == "P":
        img = img.convert("RGBA")
    if img.mode in ("RGBA", "LA", "PA") or (img.mode in ("RGB", "L") and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        bg.alpha_composite(rgba)
        return bg.convert("RGB")
    return img.convert("RGB")


_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\u2066-\u2069]")


def clean_prompt(raw: object, max_chars: int) -> str:
    """A prompt as sent to the image model: Unicode NFC, control and direction characters removed, whitespace
    collapsed. UploadRejected when empty or too long."""
    if not isinstance(raw, str):
        raise UploadRejected("Describe the hero in a few words.")
    text = " ".join(_CONTROL.sub(" ", unicodedata.normalize("NFC", raw)).split())
    if len(text) < 3:
        raise UploadRejected("Describe the hero in a few words.")
    if len(text) > max_chars:
        raise UploadRejected(f"The description is longer than {max_chars} characters.")
    return text


_NAME_OK = re.compile(r"[^\w '\-.]", re.UNICODE)


def clean_name(raw: object, max_chars: int) -> str:
    """A hero name for display: letters, digits, spaces and ' - . _ only; empty is allowed."""
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raise UploadRejected("The name must be text.")
    text = " ".join(_NAME_OK.sub("", _CONTROL.sub(" ", unicodedata.normalize("NFC", raw))).split())
    if len(text) > max_chars:
        raise UploadRejected(f"The name is longer than {max_chars} characters.")
    return text

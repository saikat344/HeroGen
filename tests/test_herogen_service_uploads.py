"""Upload safety: type sniffing, byte and pixel limits, decompression bombs, metadata stripping, orientation,
transparency, prompts and names."""

from __future__ import annotations

import io
import os
import struct
import sys
import unittest
import zlib

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from PIL import Image
from test_herogen_service_support import png

from herogen.service.uploads import Limits, UploadRejected, clean_name, clean_prompt, sanitize, sniff

LIM = Limits(max_bytes=2 * 1024 * 1024, max_pixels=4_000_000, max_side=1024, min_side=256)
META_CHUNKS = (b"tEXt", b"iTXt", b"zTXt", b"eXIf", b"tIME", b"iCCP")


def chunks(data: bytes) -> list[bytes]:
    out, pos = [], 8
    while pos + 8 <= len(data):
        n, t = struct.unpack(">I4s", data[pos:pos + 8])
        out.append(t)
        pos += 12 + n
    return out


def png_header_only(w: int, h: int) -> bytes:
    """A PNG whose header claims w x h pixels, with a tiny (invalid) image body."""
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00" * 10)) + chunk(b"IEND", b""))


def jpeg_with_exif(orientation: int = 1, gps: bool = True, w: int = 300, h: int = 500) -> bytes:
    img = Image.new("RGB", (w, h), (255, 255, 255))
    img.paste((200, 0, 0), (0, 0, w, h // 4))         # a red band at the top
    exif = Image.Exif()
    exif[0x0112] = orientation
    exif[0x010F] = "CameraMaker"
    if gps:
        exif[0x8825] = {1: "N", 2: (51.0, 30.0, 0.0), 3: "W", 4: (0.0, 7.0, 0.0)}
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif, quality=95)
    return buf.getvalue()


class TestSanitize(unittest.TestCase):
    def test_clean_png_round_trip(self):
        out, info = sanitize(png(400, 600), LIM)
        self.assertTrue(out.startswith(b"\x89PNG"))
        self.assertEqual((info["width"], info["height"], info["resized"]), (400, 600, False))
        self.assertEqual(len(info["sha256"]), 64)

    def test_type_is_sniffed_not_trusted(self):
        for bad in (b"GIF89a" + b"\x00" * 100, b"<svg xmlns='http://www.w3.org/2000/svg'/>", b"%PDF-1.7",
                    b"BM" + b"\x00" * 100, b"\x00" * 100, b"<html><script>alert(1)</script>"):
            with self.assertRaises(UploadRejected):
                sanitize(bad, LIM)
        # A GIF renamed .png is still a GIF.
        buf = io.BytesIO()
        Image.new("RGB", (300, 300)).save(buf, format="GIF")
        with self.assertRaises(UploadRejected) as cm:
            sanitize(buf.getvalue(), LIM)
        self.assertIn("PNG, JPEG or WebP", str(cm.exception))
        # PNG magic with a JPEG body (and the reverse) is refused, not decoded by another plugin.
        with self.assertRaises(UploadRejected):
            sanitize(b"\x89PNG\r\n\x1a\n" + jpeg_with_exif()[3:], LIM)
        self.assertEqual(sniff(png(10, 10)), "png")

    def test_empty_and_too_large(self):
        with self.assertRaises(UploadRejected):
            sanitize(b"", LIM)
        big = png(400, 400) + b"\x00" * (LIM.max_bytes + 1)
        with self.assertRaises(UploadRejected) as cm:
            sanitize(big, LIM)
        self.assertIn("larger than", str(cm.exception))

    def test_decompression_bomb_is_refused_before_decoding(self):
        with self.assertRaises(UploadRejected) as cm:
            sanitize(png_header_only(30000, 30000), LIM)  # Pillow's own bomb guard fires at open
        self.assertRegex(str(cm.exception), "too many pixels|too large to open safely")
        with self.assertRaises(UploadRejected) as cm:
            sanitize(png_header_only(2001, 2000), LIM)     # over our limit, under Pillow's: our check fires
        self.assertIn("too many pixels", str(cm.exception))
        # A real image over the limit made of one colour compresses to a few KB; still refused.
        img = Image.new("L", (2100, 2100), 255)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        self.assertLess(len(buf.getvalue()), 64 * 1024)
        with self.assertRaises(UploadRejected):
            sanitize(buf.getvalue(), LIM)

    def test_truncated_and_corrupt_images(self):
        good = png(400, 400)
        for bad in (good[: len(good) // 2], good[:40], good[:-20] + b"\x00" * 20, png_header_only(400, 400)):
            with self.assertRaises(UploadRejected):
                sanitize(bad, LIM)

    def test_animated_images_are_refused(self):
        frames = [Image.new("RGB", (300, 300), c) for c in ((255, 0, 0), (0, 255, 0))]
        buf = io.BytesIO()
        frames[0].save(buf, format="PNG", save_all=True, append_images=frames[1:])
        with self.assertRaises(UploadRejected) as cm:
            sanitize(buf.getvalue(), LIM)
        self.assertIn("Animated", str(cm.exception))

    def test_metadata_is_stripped(self):
        from PIL import PngImagePlugin

        info = PngImagePlugin.PngInfo()
        info.add_text("Author", "Alice Example")
        info.add_itxt("XML:com.adobe.xmp", "<x:xmpmeta>secret location</x:xmpmeta>")
        src = png(400, 400, pnginfo=info)
        self.assertIn(b"tEXt", chunks(src))
        out, _ = sanitize(src, LIM)
        self.assertFalse(set(chunks(out)) & set(META_CHUNKS), chunks(out))
        self.assertNotIn(b"Alice", out)
        self.assertNotIn(b"secret location", out)
        jout, _ = sanitize(jpeg_with_exif(gps=True), LIM)
        self.assertFalse(set(chunks(jout)) & set(META_CHUNKS))
        self.assertNotIn(b"CameraMaker", jout)
        self.assertEqual(dict(Image.open(io.BytesIO(jout)).getexif()), {})

    def test_exif_orientation_is_applied(self):
        out, info = sanitize(jpeg_with_exif(orientation=6, w=500, h=300), LIM)  # 6: rotate 90 degrees clockwise
        img = Image.open(io.BytesIO(out))
        self.assertEqual(img.size, (300, 500))
        self.assertGreater(img.getpixel((img.size[0] - 5, img.size[1] // 2))[0], 150, "the red band moved right")

    def test_transparency_is_flattened_on_white(self):
        img = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
        img.paste((10, 10, 10, 255), (100, 100, 200, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out, _ = sanitize(buf.getvalue(), LIM)
        o = Image.open(io.BytesIO(out))
        self.assertEqual(o.mode, "RGB")
        self.assertEqual(o.getpixel((5, 5)), (255, 255, 255))
        self.assertEqual(o.getpixel((150, 150)), (10, 10, 10))

    def test_sixteen_bit_and_palette_images(self):
        for mode in ("I;16", "P", "L", "CMYK"):
            img = Image.new("RGB", (300, 300), (255, 255, 255)).convert(mode if mode != "I;16" else "L")
            if mode == "I;16":
                img = img.convert("I").point(lambda v: v * 257).convert("I;16")
            buf = io.BytesIO()
            img.save(buf, format="PNG" if mode != "CMYK" else "JPEG")
            out, _ = sanitize(buf.getvalue(), LIM)
            px = Image.open(io.BytesIO(out)).getpixel((5, 5))
            self.assertTrue(all(c >= 250 for c in px), (mode, px))

    def test_large_sixteen_bit_image_stays_within_a_memory_budget(self):
        import tracemalloc

        import numpy as np

        a = np.zeros((5600, 7090), np.uint16)            # 39.7 megapixels, 16-bit grey
        a[1000:4600, 3000:4000] = 40000
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, format="PNG")
        data = buf.getvalue()
        del a
        self.assertLess(len(data), 200_000, "a tiny file that decodes to 79 MB")
        lim = Limits(max_bytes=12 * 1024 * 1024, max_pixels=40_000_000, max_side=2048, min_side=256)
        tracemalloc.start()
        try:
            out, info = sanitize(data, lim)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 200 * 1024 * 1024, f"numpy work peaked at {peak / 2**20:.0f} MiB (no float copy)")
        img = Image.open(io.BytesIO(out))
        self.assertEqual((img.mode, max(img.size), info["resized"]), ("RGB", 2048, True))
        self.assertEqual(img.getpixel((img.size[0] // 2, img.size[1] // 2)), (156, 156, 156), "40000 >> 8")
        with self.assertRaises(UploadRejected) as cm:
            sanitize(data, Limits(12 * 1024 * 1024, 40_000_000, 2048, 256, max_decode_bytes=50_000_000))
        self.assertIn("too large to process", str(cm.exception))

    def test_decode_budget_is_judged_before_decoding(self):
        from herogen.service.uploads import decoded_bytes

        self.assertEqual(decoded_bytes("L", 10, 10), 100)
        self.assertEqual(decoded_bytes("I;16", 10, 10), 200)
        self.assertEqual(decoded_bytes("RGB", 10, 10), 400)
        with self.assertRaises(UploadRejected):   # the header alone says 1600 x 1600 RGB: 10 MB > 8 MB budget
            sanitize(png_header_only(1600, 1600), Limits(LIM.max_bytes, LIM.max_pixels, LIM.max_side, LIM.min_side,
                                                         max_decode_bytes=8_000_000))

    def test_size_limits(self):
        with self.assertRaises(UploadRejected) as cm:
            sanitize(png(200, 600), LIM)
        self.assertIn("at least 256", str(cm.exception))
        out, info = sanitize(png(1800, 1200), LIM)
        self.assertTrue(info["resized"])
        self.assertEqual(Image.open(io.BytesIO(out)).size, (1024, 683))

    def test_webp(self):
        buf = io.BytesIO()
        Image.new("RGB", (300, 300), (1, 2, 3)).save(buf, format="WEBP", lossless=True)
        out, info = sanitize(buf.getvalue(), LIM)
        self.assertEqual(info["source_format"], "webp")
        self.assertEqual(Image.open(io.BytesIO(out)).getpixel((1, 1)), (1, 2, 3))


class TestText(unittest.TestCase):
    def test_prompt_uses_the_pipeline_version_rules_and_limit(self):
        from herogen.service.config import Settings

        self.assertEqual(clean_prompt("\uff21 knight\u00ab with\u200d a shield\u00bb", 600), "A knight with a shield")
        for bad in ("123 !!!", "x" * 301, "\ufb03" * 101):
            with self.subTest(raw=bad), self.assertRaises(UploadRejected):
                clean_prompt(bad, 600)
        self.assertEqual(len(clean_prompt("x" * 300, 600)), 300)
        self.assertEqual(Settings(prompt_max_chars=600).public()["prompt_max_chars"], 300)
        self.assertEqual(Settings(prompt_max_chars=100).public()["prompt_max_chars"], 100)
        with self.assertRaises(UploadRejected):
            clean_prompt("x" * 101, 100)

    def test_prompt(self):
        self.assertEqual(clean_prompt("  a  knight\n\twith\x00 a\u202e shield ", 100), "a knight with a shield")
        for bad in ("", "  ", "ab", None, 5, "x" * 101):
            with self.assertRaises(UploadRejected):
                clean_prompt(bad, 100)

    def test_name(self):
        self.assertEqual(clean_name("Sir <b>Bob</b>'s \"hero\"", 60), "Sir bBobb's hero")
        self.assertEqual(clean_name(None, 60), "")
        self.assertEqual(clean_name("../../etc/passwd", 60), "....etcpasswd")
        with self.assertRaises(UploadRejected):
            clean_name("x" * 61, 60)
        with self.assertRaises(UploadRejected):
            clean_name(5, 60)

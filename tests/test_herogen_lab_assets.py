"""Lab static assets: what is vendored, that its licences ship with it, that the SVGs are inert, and how /static/
serves nested files."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from herogen.lab.server import FILE_CSP, STATIC_DIR, make_server, safe_static_file  # noqa: E402

SVG_NS = "{http://www.w3.org/2000/svg}"
# The only SVG elements and attributes the vendored icons, brand marks and illustrations use. The page imports an
# SVG through the same allow-list (static/dom.js), so anything outside it would be dropped there.
SVG_ELEMENTS = {"svg", "g", "path", "circle", "ellipse", "line", "polyline", "polygon", "rect", "title"}
SVG_ATTRIBUTES = {
    "viewBox", "fill", "stroke", "stroke-width", "stroke-linecap", "stroke-linejoin", "stroke-dasharray", "d",
    "cx", "cy", "r", "rx", "ry", "x", "y", "x1", "y1", "x2", "y2", "width", "height", "points", "transform",
    "opacity", "vector-effect", "role", "aria-label", "color",
}


class VendoredAssetsTest(unittest.TestCase):
    def svgs(self):
        return sorted(STATIC_DIR.glob("*/*.svg"))

    def test_svgs_are_inert(self):
        files = self.svgs()
        self.assertGreater(len(files), 60)
        for f in files:
            text = f.read_text()
            self.assertNotRegex(text, r"(?i)<script|javascript:|\son[a-z]+\s*=|<foreignObject|xlink:href|\shref=", f)
            root = ET.fromstring(text)
            for el in root.iter():
                self.assertTrue(el.tag.startswith(SVG_NS), (f.name, el.tag))
                self.assertIn(el.tag[len(SVG_NS):], SVG_ELEMENTS, (f.name, el.tag))
                self.assertLessEqual(set(el.attrib) - SVG_ATTRIBUTES, set(), (f.name, el.tag))

    def test_icons_the_page_uses_exist(self):
        js = "\n".join(p.read_text() for p in STATIC_DIR.glob("*.js"))
        names = set(re.findall(r"""icon\(\s*['"]([a-z0-9-]+)['"]""", js))
        names |= set(re.findall(r"""['"]icon['"]\s*:\s*['"]([a-z0-9-]+)['"]""", js))
        for n in names:
            self.assertTrue((STATIC_DIR / "icons" / f"{n}.svg").is_file(), n)

    def test_licences_ship_with_the_assets(self):
        notices = (STATIC_DIR / "licenses" / "THIRD-PARTY-NOTICES.txt").read_text()
        self.assertIn("Lucide static 1.64.0", notices)
        self.assertIn("ISC License", notices)
        self.assertIn("Feather", notices)
        self.assertIn("SIL Open Font License", notices)
        self.assertIn("ISC License", (STATIC_DIR / "licenses" / "LUCIDE-LICENSE.txt").read_text())
        self.assertIn("SIL OPEN FONT LICENSE Version 1.1", (STATIC_DIR / "licenses" / "INTER-OFL.txt").read_text())
        # Every Lucide icon is Lucide's own (the custom ones are prefixed hg-); the notices cover each of them.
        lucide = [p.stem for p in (STATIC_DIR / "icons").glob("*.svg") if not p.stem.startswith("hg-")]
        self.assertGreater(len(lucide), 30)

    def test_tokens_and_copy(self):
        sections = json.loads((STATIC_DIR / "tokens" / "sections.json").read_text())["sections"]
        self.assertEqual([s["id"] for s in sections], list(range(1, 16)))
        for s in sections:
            self.assertRegex(s["color"], r"^#[0-9A-Fa-f]{6}$")
            self.assertIn(s["pattern"], {"dot", "horizontal", "vertical", "diagonal-up", "diagonal-down", "cross"})
        copy = json.loads((STATIC_DIR / "ui-copy.json").read_text())
        for key in ("offline", "body-pending", "compare-versions", "run-failed"):
            self.assertIsInstance(copy["states"][key], str)

    def test_only_text_assets_and_the_fonts(self):
        # The pack's previews, social images and raster exports stay out; binary files are the fonts only.
        for p in STATIC_DIR.rglob("*"):
            if not p.is_file():
                continue
            if p.parent.name == "fonts" and p.suffix in (".ttf", ".woff2"):
                continue
            self.assertIn(p.suffix.lower(), {".html", ".css", ".js", ".svg", ".json", ".txt"}, p)

    def test_fonts_are_the_recorded_files(self):
        fonts = STATIC_DIR / "fonts"
        notices = (STATIC_DIR / "licenses" / "THIRD-PARTY-NOTICES.txt").read_text()
        files = sorted(p for p in fonts.iterdir() if p.suffix in (".ttf", ".woff2"))
        self.assertEqual([p.name for p in files], ["BricolageGrotesque-Variable.ttf", "Figtree-Variable.ttf",
                                                   "JetBrainsMono-Regular.woff2", "JetBrainsMono-SemiBold.woff2"])
        magic = {".ttf": b"\x00\x01\x00\x00", ".woff2": b"wOF2"}
        for p in [*files, *fonts.glob("OFL-*.txt")]:
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertIn(f"{p.name} (", notices, p.name)
            self.assertIn(f"sha256 {digest}", notices, p.name)  # the notices name the exact bytes shipped
            if p.suffix in magic:
                self.assertEqual(p.read_bytes()[:4], magic[p.suffix], p.name)
        for family in ("BricolageGrotesque", "Figtree", "JetBrainsMono"):
            self.assertIn("SIL OPEN FONT LICENSE Version 1.1", (fonts / f"OFL-{family}.txt").read_text(), family)
        css = (fonts / "fonts.css").read_text()
        refs = re.findall(r'url\("([^"]+)"\)', css)
        self.assertEqual(sorted(refs), [p.name for p in files])  # every font is used, and only these
        self.assertNotRegex(css, r"https?:|//")


class StaticServingTest(unittest.TestCase):
    def test_nested_paths(self):
        for name, ctype in [("icons/box.svg", "image/svg+xml"), ("tokens/sections.json", "application/json"),
                            ("licenses/LUCIDE-LICENSE.txt", "text/plain; charset=utf-8"),
                            ("fonts/Figtree-Variable.ttf", "font/ttf"), ("fonts/JetBrainsMono-Regular.woff2", "font/woff2"),
                            ("fonts/fonts.css", "text/css; charset=utf-8"),
                            ("icons%2Fbox.svg", "image/svg+xml")]:
            got = safe_static_file(name)
            self.assertIsNotNone(got, name)
            self.assertEqual(got[1], ctype)
            self.assertTrue(got[0].is_relative_to(STATIC_DIR))

    def test_refused(self):
        for name in ["../server.py", "icons/../../server.py", "icons/%2e%2e/%2e%2e/server.py", "icons//box.svg",
                     "./icons/box.svg", "icons/./box.svg", "icons\\box.svg", "/etc/passwd", "%2Fetc%2Fpasswd",
                     "icons", "icons/", "icons/missing.svg", "index.html", "icons/index.html", "icons/box.svg%00",
                     ".hidden.svg", "icons/.hidden.svg", "icons/box.SVG.html", "runindex.py", ""]:
            self.assertIsNone(safe_static_file(name), name)

    def test_symlink_out_of_static_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            outside = Path(tmp) / "outside.svg"
            outside.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
            link = STATIC_DIR / "icons" / "zz-test-link.svg"
            try:
                os.symlink(outside, link)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks unavailable")
            try:
                self.assertIsNone(safe_static_file("icons/zz-test-link.svg"))
            finally:
                link.unlink()

    def test_served_with_a_sandbox_csp(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = make_server(Path(tmp), port=0, quiet=True)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_address[1]}"
                with urllib.request.urlopen(base + "/static/icons/box.svg", timeout=5) as r:
                    self.assertEqual(r.headers["Content-Type"], "image/svg+xml")
                    self.assertEqual(r.headers["Content-Security-Policy"], FILE_CSP)
                    self.assertIn(b"<svg", r.read())
                with self.assertRaises(urllib.error.HTTPError) as cm:
                    urllib.request.urlopen(base + "/static/icons/..%2F..%2Fserver.py", timeout=5)
                self.assertEqual(cm.exception.code, 404)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()

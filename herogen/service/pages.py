"""Server-rendered pages: the shared shell (wordmark, bottom nav bar and marketing footer), the 404 and 500 pages,
and the legal text.

Every page here loads only its own origin's files, so `PAGE_CSP` needs no inline script or style."""

from __future__ import annotations

import html
import re
from pathlib import Path

SHELL = Path(__file__).resolve().parent / "static" / "shell.html"

# The same style as the app pages (self-hosted scripts, styles, fonts and images; no inline code), without the
# three.js CDN, which these pages never load.
PAGE_CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; font-src 'self'; "
            "object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")

_HEADING = re.compile(r"^\d+\.\s+\S[^\n]{0,90}$")
_CLAUSE = re.compile(r"^(\d+\.\d+)\s+(.*)$", re.S)


def shell(title: str, main: str, body_class: str) -> str:
    """The page around `main` (trusted markup): head, skip link, wordmark, bottom nav bar, footer. `title` is
    escaped."""
    return (SHELL.read_text()
            .replace("<!--title-->", html.escape(title))
            .replace("<!--body_class-->", html.escape(body_class))
            .replace("<!--main-->", main))


def _inline(text: str) -> str:
    return html.escape(" ".join(text.split()))


def legal_main(title: str, text: str | None) -> str:
    """The legal text (plain text: numbered headings, "- " bullets, blank-line paragraphs) as readable prose.
    `None` means the document has not been published."""
    head = f'<article class="doc-page"><header class="doc-page__head"><p class="eyebrow">Legal</p><h1>{html.escape(title)}</h1>'
    if text is None:
        return head + (f'</header><div class="prose"><p>The {html.escape(title)} has not been published yet.</p>'
                       f'</div></article>')
    blocks = [b.strip("\n") for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    if blocks and "\n" not in blocks[0] and blocks[0].startswith("HeroGen "):
        blocks = blocks[1:]  # the document's own title line: the page already has an H1
    meta: list[str] = []   # notes before the first heading (version, draft status) sit under the H1
    body: list[str] = []
    for block in blocks:
        lines = block.split("\n")
        first = lines[0].lstrip()
        if len(lines) == 1 and _HEADING.match(first) and not first.endswith((".", ":", ";")):
            body.append(f"<h2>{_inline(first)}</h2>")
        elif first.startswith("- "):
            items: list[str] = []
            for ln in lines:
                if ln.lstrip().startswith("- "):
                    items.append(ln.lstrip()[2:])
                else:
                    items[-1] += " " + ln.strip()
            body.append("<ul>" + "".join(f"<li>{_inline(i)}</li>" for i in items) + "</ul>")
        elif not body:
            meta.append(f'<p class="doc-meta">{_inline(block)}</p>')
        else:
            joined = " ".join(lines)
            clause = _CLAUSE.match(joined.strip())
            if clause:
                body.append(f'<p class="clause"><span class="clause__n">{html.escape(clause.group(1))}</span>'
                            f'{_inline(clause.group(2))}</p>')
            else:
                body.append(f"<p>{_inline(joined)}</p>")
    return head + "".join(meta) + f'</header><div class="prose">{"".join(body)}</div></article>'


def error_main(status: int, art: dict | None = None) -> str:
    """The 404 or 500 body. `art` is one gallery item ({"id", "title", "assets"}); omitted when unavailable."""
    if status == 404:
        eyebrow, heading = "Error 404", "This page wandered off."
        lead = "The link may be old or mistyped. Your heroes are right where you left them."
    else:
        eyebrow, heading = "Error 500", "Something went wrong on our side."
        lead = "Your heroes and progress are saved. Try again in a moment."
    figure = ""
    if art:
        key = html.escape(art["id"], quote=True)
        title = html.escape(art["title"], quote=True)
        # The small WebP when the catalog lists one: the art sits above the fold, so it loads at once.
        artwork = (art.get("assets") or {}).get("artwork") or {}
        thumb = artwork.get("thumb_url")
        src = html.escape(thumb, quote=True) if thumb else f"/gallery-assets/{key}/artwork"
        # error-art.js falls back to the full artwork when the thumbnail fails, and hides the figure if that fails too.
        figure = (f'<figure class="error-art"><img src="{src}" data-full="/gallery-assets/{key}/artwork" '
                  f'width="320" height="480" alt="{title}, a hero made with HeroGen" decoding="async">'
                  f'<figcaption>{title}, made with HeroGen</figcaption></figure>'
                  f'<script type="module" src="/assets/error-art.js"></script>')
    return (f'<section class="error-page error-page--{status}" aria-labelledby="errorTitle">'
            f'<span class="error-page__mark" aria-hidden="true">HG</span>'
            f'<div class="error-page__copy"><p class="eyebrow">{eyebrow}</p>'
            f'<h1 id="errorTitle">{heading}</h1><p class="error-page__lead">{lead}</p>'
            f'<div class="error-page__actions">'
            f'<a class="btn btn--primary btn--l" href="/">Create a hero'
            f'<svg class="icon" aria-hidden="true"><use href="/assets/icons/sprite.svg#i-arrow-right"></use></svg></a>'
            f'<a class="btn btn--secondary btn--l" href="/#gallery">Back to gallery</a></div></div>'
            f'{figure}</section>')

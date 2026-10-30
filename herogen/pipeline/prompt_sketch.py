"""Stage p: a user's text prompt to a hand-drawn, full-body character sketch, which stage 0 then takes as its input.

The prompt version (``hero-sketch-v1`` by default, from ``herogen/prompts``) has two steps: a design image drawn
from the user's character brief, then a full-body edit of that design. Each attempt is checked, and a failed
attempt is drawn again as a new sample (the image API takes no seed) up to ``ATTEMPT_PLAN``'s bound:

- one figure: SAM 3.1 'character' finds exactly one instance in the full-body image and 'person' at most one
  ('character' also finds robots and animal-headed figures, which 'person' misses; 'person' finds faint ghost
  figures that 'character' sometimes merges into the main one);
- full figure: two SAM 3.1 'foot', 'boot' or 'shoe' instances in the lower half of that figure, and the figure
  touching neither the top nor the bottom edge;
- same character: a vision model, asked with the frozen same-character prompt (``same-character-v2`` by
  default), says the full-body image shows the same character as the design and lists no body change (a head,
  arm, leg, tail or wing added or dropped). Props it reports as added, removed or moved to the other hand are
  recorded without gating (see ``PROP_NOTE``).

The thresholds and where they come from are in docs/ARCHITECTURE.md. A Model API failure stops the stage with an
error; it is never replaced by a guess or another method. Retries of a failing HTTP call are the shared client's.
"""

from __future__ import annotations

import shutil
import time
import unicodedata
from typing import Any

import numpy as np
from PIL import Image

from herogen import prompts
from herogen.pipeline import checks as C
from herogen.pipeline import imaging
from herogen.pipeline.client import IMAGE_MODEL, ClientError, ModelClient
from herogen.pipeline.runfolder import RunFolder
from herogen.pipeline.stages import GateFailed, InputRejected, _Stage

STAGE = "p"
DESIGN = "p-design.png"
SKETCH = "p-full-body.png"  # the passing full-body image; stage 0 copies it as the sketch
ATTEMPTS_DIR = "p-attempts"

# (design sample, full-body sample) per attempt: two new full-body samples of the first design, then a new design.
ATTEMPT_PLAN = ((0, 0), (0, 1), (1, 0))

FIGURE_CONCEPT = "character"  # the figure's mask, and the count that must be exactly one
PERSON_CONCEPT = "person"  # a second count that must be at most one (zero for a robot or a creature)
FEET_CONCEPTS = ("foot", "boot", "shoe")
FIGURES_EXPECTED = 1
PERSONS_MAX = 1
FEET_MIN = 2
# A figure touches an edge when its mask comes within max(2 px, 0.5% of the image height) of it (experiment 3's
# definition, kept). See docs/ARCHITECTURE.md for the clearance measured on full-body images.
EDGE_TOL_FRAC = 0.005
EDGE_TOL_MIN_PX = 2

# The error code the Model API returns when it refuses a prompt; reported as a refusal, never retried.
REFUSED = "content_policy_violation"

PROP_NOTE = (
    "props the identity check reported as added, removed or moved to the other hand; not a gate, because on "
    "full-body redraws of bust designs it also reports gear the design did not show (13 of 32 calls on pairs "
    "people rated the same character)"
)


class PromptRejected(ValueError):
    """The user's prompt can't be used (empty, no words, or over the length limit). The stage records it with
    status "input_rejected" and raises ``InputRejected``, as for a sketch with no character in it."""


class PromptStageError(RuntimeError):
    """Stage p stopped on an error (a Model API failure or an unusable answer); the record says which attempt."""


# ---- the user's text ---------------------------------------------------------------------------------


def clean_user_text(text: Any, *, max_chars: int, delimiters: tuple[str, ...]) -> str:
    """The user's character brief, made safe to place inside the bounded slot.

    NFKC-normalised; control, format (zero-width, bidirectional) and line/paragraph separator characters become
    spaces; the slot's delimiters are removed so the text can't close its slot; whitespace runs collapse to one
    space. The cleaned text must hold at least one letter and at most ``max_chars`` characters: longer text is
    refused, never cut, so what is drawn is always what the user wrote.
    """
    if not isinstance(text, str):
        raise PromptRejected("the prompt must be text")
    s = unicodedata.normalize("NFKC", text)
    s = "".join(" " if unicodedata.category(ch) in ("Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp") else ch for ch in s)
    for d in delimiters:
        s = s.replace(d, " ")
    s = " ".join(s.split())
    if not s:
        raise PromptRejected("the prompt is empty; describe the character, for example 'a knight in battered armour'")
    if not any(unicodedata.category(ch).startswith("L") for ch in s):
        raise PromptRejected("the prompt has no words; describe the character in words")
    if len(s) > max_chars:
        raise PromptRejected(f"the prompt is {len(s)} characters long; the limit is {max_chars}")
    return s


def design_prompt(version: prompts.PromptVersion, character: str) -> str:
    """The design step's text with the cleaned brief in its one slot; the rest of the template is unchanged."""
    slot = version.data["user_text"]["slot"]
    template = version.step("design")["template"]
    if template.count(slot) != 1:
        raise prompts.RegistryError(f"{version.name}: the design template must hold exactly one {slot}")
    head, tail = template.split(slot)
    return head + character + tail


def _user_text_rules(version: prompts.PromptVersion) -> tuple[int, tuple[str, ...]]:
    u = version.data["user_text"]
    return int(u["max_chars"]), (u["open"], u["close"])


# ---- checks ------------------------------------------------------------------------------------------


def edge_tolerance_px(height: int) -> int:
    return max(EDGE_TOL_MIN_PX, round(EDGE_TOL_FRAC * height))


def figure_checks(client: ModelClient, img: Image.Image) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray | None]:
    """One-figure and full-figure checks on a full-body image; returns (checks, details, figure mask or None)."""
    w, h = img.size
    figures = client.segment(img, FIGURE_CONCEPT)
    people = client.segment(img, PERSON_CONCEPT)
    details: dict[str, Any] = {}
    for name, found in (("character", figures), ("person", people)):
        details[f"{name}_instances"] = len(found)
        details[f"{name}_confidences"] = [f.confidence for f in found]
        details[f"{name}_area_fracs"] = [round(float(f.mask.mean()), 4) for f in found]
    checks = [
        C.check("character_instances", len(figures), FIGURES_EXPECTED, op="=="),
        C.check("person_instances", len(people), PERSONS_MAX, op="<="),
    ]
    tol = edge_tolerance_px(h)
    if not figures:
        details["note"] = "SAM 3.1 found no character, so the full-figure checks have nothing to measure"
        checks += [
            C.check("feet_in_lower_half", 0, FEET_MIN),
            C.check("figure_top_gap_px", 0, tol, op=">"),
            C.check("figure_bottom_gap_px", 0, tol, op=">"),
        ]
        return checks, details, None
    main = max(figures, key=lambda f: int(f.mask.sum())).mask
    box = imaging.bbox(main)
    if box is None:  # an instance with an empty mask
        raise PromptStageError("SAM 3.1 returned a character instance with an empty mask")
    top, bottom = box[1], box[3]  # bottom is exclusive
    mid = (top + bottom) / 2
    feet: dict[str, int] = {}
    for concept in FEET_CONCEPTS:
        found = client.segment(img, concept)
        feet[concept] = sum(1 for f in found if (f.box[1] + f.box[3]) / 2 > mid)
        details[f"{concept}_instances"] = len(found)
    details["feet_in_lower_half_by_concept"] = feet
    details["figure_box"] = list(box)
    details["edge_tolerance_px"] = tol
    checks += [
        C.check("feet_in_lower_half", max(feet.values()), FEET_MIN),
        C.check("figure_top_gap_px", top, tol, op=">"),
        C.check("figure_bottom_gap_px", h - bottom, tol, op=">"),
    ]
    return checks, details, main


def identity_checks(
    client: ModelClient, version: prompts.PromptVersion, design: Image.Image, full_body: Image.Image
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    step = version.step("compare")
    p = step["params"]
    if p.get("image_format") != "jpeg":
        raise prompts.RegistryError(f"{version.name}: only jpeg images are supported")
    answer = client.compare_images(
        step["prompt"], [design, full_body], schema=step["schema"], model=step["model"],
        reasoning_effort=p["reasoning_effort"], max_output_tokens=p["max_output_tokens"],
        image_max_side=p["image_max_side"], jpeg_quality=p["jpeg_quality"],
    )
    verdict = answer.get("same_character")
    if verdict not in ("yes", "no"):
        raise PromptStageError(f"the identity check answered same_character={verdict!r}, not 'yes' or 'no'")
    changes = {k: [str(x) for x in answer.get(k) or []] for k in ("added_props", "removed_props", "changed_hand")}
    n_changes = sum(len(v) for v in changes.values())
    checks = [C.check("same_character", 1 if verdict == "yes" else 0, 1, op="==")]
    details: dict[str, Any] = {"answer": answer, "prop_changes": changes}
    if "body_changes" in step["schema"]["schema"]["properties"]:  # versions that ask about the body plan
        body = [str(x) for x in answer.get("body_changes") or []]
        details["body_changes"] = body
        checks.append(C.check("body_changes", len(body), 0, op="=="))
    checks.append(C.check("prop_changes_reported", n_changes, 0, op="==", gate=False, note=PROP_NOTE))
    return checks, details


# ---- the stage ---------------------------------------------------------------------------------------


def _describe_failures(checks: list[dict[str, Any]]) -> str:
    bad = [c for c in checks if c.get("gate", True) and not c["pass"]]
    return ", ".join(f"{c['name']} {c['value']:g} (needs {c['op']} {c['threshold']})" for c in bad)


def stagep_prompt(
    run: RunFolder,
    client: ModelClient,
    user_text: Any,
    *,
    version: prompts.PromptVersion | None = None,
    same_version: prompts.PromptVersion | None = None,
) -> dict[str, Any]:
    """Run stage p.

    Before anything else the earlier ``stage-p.json`` is deleted (first, so if that fails nothing else has been
    touched), then the earlier images, then a ``running`` record (``ok: false``) is written. From then on
    ``stage-p.json`` is either absent or describes this run. Every error is recorded with ``ok: false`` before it
    propagates (an unexpected one is re-raised as it is) whenever the record can still be written; when it can't
    (a full disk, say), that write's error propagates and no record is left. A failed gate is recorded by
    ``finish``."""
    st = _Stage(run, client, STAGE, "prompt_sketch")
    # A rerun starts clean: no record or image from an earlier run of this stage can be mistaken for this run's.
    (run.path / f"stage-{STAGE}.json").unlink(missing_ok=True)
    for rel in (DESIGN, SKETCH):
        (run.path / rel).unlink(missing_ok=True)
    shutil.rmtree(run.path / ATTEMPTS_DIR, ignore_errors=True)
    st.details = {"attempts": []}
    st.fail("stage p started and has not finished", status="running")
    try:
        return _stagep(st, run, client, user_text, version, same_version)
    except (GateFailed, InputRejected, PromptStageError):
        raise  # already recorded
    except Exception as e:
        attempts = st.details.get("attempts") or []
        if attempts and "pass" not in attempts[-1] and "error" not in attempts[-1]:
            attempts[-1]["error"] = f"{type(e).__name__}: {e}"
        st.fail(f"{type(e).__name__}: {e}")
        raise


def _stagep(
    st: _Stage,
    run: RunFolder,
    client: ModelClient,
    user_text: Any,
    version: prompts.PromptVersion | None,
    same_version: prompts.PromptVersion | None,
) -> dict[str, Any]:
    version = version or prompts.selected("prompt_sketch")
    same_version = same_version or prompts.selected("same_character")
    st.details.update({"prompt_version": version.ref(), "same_character_version": same_version.ref()})
    max_chars, delimiters = _user_text_rules(version)
    try:
        character = clean_user_text(user_text, max_chars=max_chars, delimiters=delimiters)
    except PromptRejected as e:
        st.details["user_text_chars"] = len(user_text) if isinstance(user_text, str) else None
        record = st.fail(f"prompt rejected: {e}", status="input_rejected")
        raise InputRejected(STAGE, record) from e  # type: ignore[arg-type]
    for name in ("design", "full_body"):
        if version.step(name)["model"] != IMAGE_MODEL:
            raise prompts.RegistryError(f"{version.name}: step {name} needs {version.step(name)['model']}, "
                                        f"but the client draws with {IMAGE_MODEL}")
    text = design_prompt(version, character)
    full_text = version.step("full_body")["prompt"]
    st.details.update({"user_text": character, "design_prompt": text})
    designs: dict[int, Image.Image] = {}
    chosen = None
    for k, (d_sample, f_sample) in enumerate(ATTEMPT_PLAN):
        t0 = time.monotonic()
        rec: dict[str, Any] = {"attempt": k, "design_sample": d_sample, "full_body_sample": f_sample}
        st.details["attempts"].append(rec)
        step = "design"
        try:
            if d_sample not in designs:
                designs[d_sample] = client.generate_image(
                    text, size=version.step("design")["params"]["size"], attempt=d_sample)
            design = designs[d_sample]
            st.save_image(f"{ATTEMPTS_DIR}/{k}-design.png", design)
            step = "full-body edit"
            full = client.edit_image(full_text, [design], size=version.step("full_body")["params"]["size"],
                                     attempt=f_sample)
            st.save_image(f"{ATTEMPTS_DIR}/{k}-full-body.png", full)
            step = "figure checks (SAM 3.1)"
            checks, fig, mask = figure_checks(client, full)
            rec["figure"] = fig
            if mask is not None:
                st.save_image(f"{ATTEMPTS_DIR}/{k}-figure-mask.png", imaging.to_mask_image(mask))
            if C.gate_ok(checks):
                step = "identity check"
                id_checks, ident = identity_checks(client, same_version, design, full)
                checks += id_checks
                rec["identity"] = ident
            else:
                rec["identity"] = {"skipped": "an earlier check failed"}
        except (ClientError, PromptStageError) as e:
            rec["error"] = f"{step}: {e}"
            rec["seconds"] = round(time.monotonic() - t0, 2)
            error = f"attempt {k}, {step}: {type(e).__name__}: {e}"
            if REFUSED in str(e):
                error = f"refused by the Model API's content policy: {error}"
            st.fail(error)
            if REFUSED in str(e):
                raise PromptStageError(
                    f"stage p stopped: the Model API refused the {step} under its content policy on attempt {k}; "
                    "reword the prompt") from e
            raise PromptStageError(
                f"stage p stopped: the {step} call failed on attempt {k} ({e}); no other method is tried") from e
        rec["checks"] = checks
        rec["pass"] = C.gate_ok(checks)
        rec["seconds"] = round(time.monotonic() - t0, 2)
        st.checks = checks
        if rec["pass"]:
            chosen = (k, design, full)
            break
        rec["failed"] = _describe_failures(checks)
    st.details["attempt_count"] = len(st.details["attempts"])
    if chosen is None:
        summary = "; ".join(f"attempt {r['attempt']}: {r['failed']}" for r in st.details["attempts"])
        st.notes.append(f"No attempt passed: {summary}.")
        return st.finish(fail_message=(
            f"stage p: no sketch passed the checks in {len(ATTEMPT_PLAN)} attempts ({summary}); "
            "try rewording the prompt"))
    k, design, full = chosen
    st.details["chosen_attempt"] = k
    st.save_image(DESIGN, design)
    st.save_image(SKETCH, full)
    if k:
        st.notes.append(f"Passed on attempt {k} (counting from 0) after {k} failed attempt{'s' if k > 1 else ''}.")
    changes = st.details["attempts"][k]["identity"]["prop_changes"]
    listed = [f"{kind.replace('_', ' ')}: {', '.join(v)}" for kind, v in changes.items() if v]
    if listed:
        st.notes.append("Identity check reported prop changes (not a gate): " + "; ".join(listed) + ".")
    return st.finish()


__all__ = ["GateFailed", "PromptRejected", "PromptStageError", "SKETCH", "stagep_prompt", "clean_user_text"]

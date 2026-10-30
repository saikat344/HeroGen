"""Calibrated stage-3 visual acceptance policy and failure accounting."""
from pathlib import Path

from PIL import Image

from herogen.pipeline.client import COST_BASIS, ModelClient

from . import body_quality as quality
from .record import check

POLICY = "body-usability-v1"
MIN_SCORE = 17.5


class BodyQualityError(RuntimeError):
    """The acceptance check could not complete; the body has no acceptance decision."""


def assess(run: Path, verts, faces, rec, client=None) -> None:
    """Render the fixed views and append the acceptance decision, preserving cost on every failure.

    This runs after construction. Its errors must never trigger a carve fallback.
    """
    from .stage3 import error_text

    rec.data["body_acceptance"] = {"policy": POLICY, "threshold": MIN_SCORE,
                                    "prompt_sha256": quality.PROMPT_SHA256,
                                    "views": quality.YAWS}
    client = client if client is not None else ModelClient(run.parent / "_cache-model", mode="live")
    start = len(client.log.calls)
    try:
        with Image.open(run / "00-sketch.png") as im:
            sketch = im.convert("RGB")
        images = quality.render_views(verts, faces)
        out = run / "03-quality"
        out.mkdir(exist_ok=True)
        for name, im in images.items():
            im.save(out / f"{name}.png")
            rec.data["outputs"].append(f"03-quality/{name}.png")
        quality.contact_sheet(sketch, images).save(out / "sheet.png")
        rec.data["outputs"].append("03-quality/sheet.png")
        answer = quality.visual_score(client, sketch, images)
        score, reason = answer["usable_probability"], answer["reason"]
        rec.data["body_acceptance"].update(answer)
        rec.add(check("body_usability", score, MIN_SCORE, reason=reason))
    except Exception as exc:
        message = error_text(exc, run)
        rec.data["body_acceptance"]["error"] = message
        rec.add(check("body_usability", None, MIN_SCORE, passed=False,
                      reason="visual acceptance check could not complete"))
        rec.data["error"] = "Body acceptance check failed: " + message
        raise BodyQualityError(rec.data["error"]) from exc
    finally:
        calls = client.log.calls[start:]
        rec.data["body_acceptance"]["calls"] = calls
        rec.data["cost_usd"] = round(rec.data["cost_usd"] + client.log.cost_usd(start), 6)
        rec.data["cost_basis"] += "; body acceptance: " + COST_BASIS
        for model in client.log.models(start):
            if model not in rec.data["models"]:
                rec.data["models"].append(model)
        if rec.data.get("error"):
            rec.data["rejection_reasons"] = ["body_usability: visual acceptance check could not complete"]
            rec.finish()

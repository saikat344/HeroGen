"""Stage 3f: sequential, gated repaint projection onto SAM 3D Objects' own mesh."""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import uuid
from pathlib import Path

from PIL import Image

from herogen.blender.record import StageRecord, check
from herogen.blender.stage3 import error_text
from herogen.pipeline.client import COST_BASIS, IMAGE_MODEL, SAM_MODEL, PRICE_PER_IMAGE_USD, ModelClient
from herogen.refine_budget import BudgetStop, update as update_budget

STAGE = "3f"
DEFAULT_CAP_USD = 0.80
REFINE_PROFILES = frozenset(("best", "roblox-ingame", "roblox-avatar"))
REFINE_ORDER = ("head+0", "head+40", "head-40", "head+90", "head-90", "head+180",
                "body+0", "body+45", "body-45", "body+90", "body-90", "body+180")
# Negative angles -90/-135 are 270/225 degrees, with the same orientation and projection.
HAIR_ORDER = ("head+90", "head+135", "head+180", "head-135", "head-90",
              "upper+135", "upper+180", "upper-135")


class Budget:
    def __init__(self, client, cap, prior=0.0, *, state=None):
        if not math.isfinite(cap) or cap < 0:
            raise ValueError("texture-refine API cap must be finite and nonnegative")
        if not math.isfinite(prior) or prior < 0:
            raise ValueError("texture-refine prior spend must be finite and nonnegative")
        self.client, self.cap, self.prior = client, cap, prior
        self.start = len(client.log.calls)
        self.state = state if state and "attempts" in state else None
        self.lock = threading.RLock()
        self._local = threading.local()
        self.active = None
        self._update("read")

    @property
    def active(self):
        return getattr(self._local, "active", None)

    @active.setter
    def active(self, token):
        self._local.active = token

    def _update(self, action, **kwargs):
        with self.lock:
            return self._locked_update(action, **kwargs)

    def _locked_update(self, action, **kwargs):
        self.state = update_budget(self.state, action, cap=self.cap, prior=self.prior, **kwargs)
        return self.state

    def spent(self):
        return max(self.state["spent_usd_total"], round(self.prior + self.client.log.cost_usd(self.start), 6))

    def reserve(self, model):
        with self.lock:
            self.before_attempt(model)
            token = uuid.uuid4().hex
            if not self._update("reserve", token=token, amount=PRICE_PER_IMAGE_USD[model])["allowed"]:
                raise BudgetStop(f"API cap: next attempt exceeds ${self.cap:.4f}")
            self.active = token

    def settle(self, cost):
        if self.active is None:
            raise ValueError("no active refinement reservation")
        self._update("settle", token=self.active, amount=cost)
        self.active = None

    def before_attempt(self, model):
        price = PRICE_PER_IMAGE_USD[model]
        projected = round(self.spent() + price, 6)
        if projected > self.cap:
            raise BudgetStop(f"API cap: projected ${projected:.4f} exceeds ${self.cap:.4f}")


def cost_estimate(maximum=False, upscale_texture=False, *, refine_profile="best"):
    """Per-run list-price projection; no cache discount or transport retries. Hair uses up to two samples."""
    normal, hair = (3, 2) if maximum else (1, 1)
    if refine_profile == "fast":
        from .fast import ATTEMPTS, WAVES
        sequence = [view for wave in WAVES for view in wave]
        normal_count = sum(not hair for _, hair in sequence)
        hair_count = len(sequence) - normal_count
        # All attempts may start before a first acceptance can cancel them, including duplicate render masks.
        normal = hair = ATTEMPTS
        images = normal_count * normal + hair_count * hair
        segments = 2 + normal_count * normal * 4 + hair_count * hair * 5
        return round(images * PRICE_PER_IMAGE_USD[IMAGE_MODEL] + segments * PRICE_PER_IMAGE_USD[SAM_MODEL], 6)
    else:
        normal_count, hair_count = len(REFINE_ORDER), len(HAIR_ORDER)
    images = normal_count * normal + hair_count * hair
    segments = 2 + normal_count * (1 + normal + 2) + hair_count * (1 + hair + 3)
    if upscale_texture:
        images += 16 * normal
        segments += 16 * (3 + normal)
    return round(images * PRICE_PER_IMAGE_USD[IMAGE_MODEL] + segments * PRICE_PER_IMAGE_USD[SAM_MODEL], 6)


def _view(client, prep, current, tex, cam, hair, folder, record, timeout, cover, *, face_exclusion=True, body_cam=None,
          closeup_reference=None, closeup_recipe="RF5U", repaint="RF2R", rendered=None,
          apply=None, attempts=None, attempt_start=0, cancelled=None, user_note=""):
    import numpy as np
    from scipy import ndimage
    from . import character as C, imaging as I
    from .backend import blender
    from .cameras import decode_pos, project, px_per_unit, read_exr
    from .prompts import prompt

    folder.mkdir(parents=True, exist_ok=True)
    if rendered is None:
        blender("render", {"blend": str(prep / "work.blend"), "texture": str(current), "cams": [cam],
                           "out": str(folder), "pos": True}, folder / "blender.log", timeout=timeout)
    else:
        import shutil
        # Rebaking consumes the original render and geometry alongside each accepted repaint.
        for suffix in (".png", ".pos.exr"):
            shutil.copy2(rendered / f"{cam['name']}{suffix}", folder / f"{cam['name']}{suffix}")
    rendered = folder if rendered is None else rendered
    with Image.open(rendered / f"{cam['name']}.png") as rgba:
        inp, alpha = C.composite(rgba.convert("RGBA"))
    inp.save(folder / "input.png")
    render_rgb = np.asarray(inp, dtype=np.float32) / 255
    positions, valid = decode_pos(read_exr(rendered / f"{cam['name']}.pos.exr"))
    _, _, depth = project(cam, positions)
    figure = alpha > 0.5
    parts = None
    recipe = closeup_recipe if closeup_reference is not None else "RF3H" if hair else repaint
    record.update(prompt=recipe, camera=cam)
    if closeup_reference is not None:
        measurable, source_checks = I.closeup_outline(figure, I.closeup_figure_mask(render_rgb, figure), cam["zoom"])
        record["source_outline_checks"] = source_checks
        if not measurable:
            record.update(accepted=False, reason="render outline not measurable")
            return current
    for attempt in range(attempt_start, attempt_start + (attempts if attempts is not None else I.HAIR_ATTEMPTS if hair else I.ATTEMPTS)):
        if cancelled is not None and cancelled.is_set():
            record.update(reason="cancelled: lower attempt accepted")
            return current
        trial = {"try": attempt + 1, "accepted": False, "reason": "incomplete"}
        record["attempts"].append(trial)
        record["tries"] = attempt + 1
        references = [inp, cover] if closeup_reference is None else [inp, closeup_reference, cover]
        edit = client.edit_image(prompt(recipe, cam["angle"], user_note), references,
                                 size=f"{cam['res'][0]}x{cam['res'][1]}", attempt=attempt)
        edit = edit.convert("RGB")
        edit.save(folder / f"edit-{attempt}.png")
        mapx, mapy, warped, reg = I.register(render_rgb, alpha, np.asarray(edit, dtype=np.float32) / 255)
        fig_e = (I.closeup_figure_mask(warped, figure) if closeup_reference is not None
                 else I.figure_mask(warped, render_figure=figure))
        overlap = float((figure & fig_e).sum() / max(1, (figure | fig_e).sum()))
        med, p90 = (v * 1024 / max(alpha.shape) for v in I.chamfer(figure, fig_e))
        measures = {"iou": overlap, "median_px": med, "p90_px": p90}
        if closeup_reference is not None:
            passed, measures = I.closeup_outline(figure, fig_e, cam["zoom"])
            reason = "outline moved"
        elif hair:
            agreement, _, _ = I.interior_agreement(render_rgb, warped, figure, fig_e)
            measures["agreement"] = agreement
            passed = (overlap >= I.HAIR_GATE["iou_min"] and agreement >= I.HAIR_GATE["agree_min"]
                      and abs(reg["scale"] - 1) <= I.HAIR_GATE["scale_max_change"])
            reason = "hair outline, scale or colour agreement"
        else:
            passed, measures = I.outline_gate(figure, fig_e, closeup=(body_cam is not None and I.is_closeup(cam, body_cam)))
            reason = "outline moved"
        trial.update(registration=reg, checks=measures)
        if cancelled is not None and cancelled.is_set():
            record.update(reason="cancelled: lower attempt accepted")
            trial["reason"] = record["reason"]
            return current
        if passed:
            if parts is None:
                parts = I.part_masks(client, inp, figure.shape)
            edited_parts = I.part_masks(client, edit, figure.shape, mapx, mapy)
            passed, measures["parts"], _ = I.part_gate(parts, edited_parts, figure)
            reason = "new face outside the render face"
        trial["reason"] = "accepted" if passed else reason
        if not passed:
            record["reason"] = reason
            continue
        refined = I.colour_match(render_rgb, warped, figure & fig_e)
        if hair:
            weight = I.hair_weight(client, inp, edit, render_rgb, warped, alpha, mapx, mapy,
                                   parts["face"] if face_exclusion else None)
        else:
            weight = (~ndimage.binary_dilation(figure ^ fig_e, iterations=4)).astype(np.float32)
            weight *= I.hair_allowed(client, inp, edit, mapx, mapy, figure.shape, hair_only=False)
        feather = I.feather_map(alpha, depth, valid, px_per_unit(cam), tex.height)
        if closeup_reference is not None:
            feather *= I.frame_feather(alpha.shape, cam["frame_border_px"])
        elif I.CLOSEUP_FEATHER and (hair or (body_cam is not None and I.is_closeup(cam, body_cam))):
            feather *= I.closeup_feather(figure)
        record["texels"] = (tex.apply if apply is None else apply)(
            cam, refined, feather, depth, valid, weight, I.HAIR_BONUS if hair else 1.0)
        for name, arr in (("refined", refined), ("weight", weight)):
            Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype(np.uint8)).save(folder / f"{name}.png")
        trial["accepted"] = True
        record.update(accepted=True, reason="accepted")
        return tex.save(folder / "texture.png") if apply is None else current
    return current


def run_stage3f(run: Path, client: ModelClient, *, profile="best", cap_usd=DEFAULT_CAP_USD, timeout=900.0,
                upscale_texture=False, recipe="RF2R", refine_profile=None, user_note=""):
    """Write 03-refined.glb after stage 3. Rejected views and cap stops are successful recorded fallbacks.

    Model failures are recorded and re-raised. A retry counts this run's previous stage-3f spend toward its cap;
    cost_usd itself counts this invocation's calls, matching the other stages' convention.

    ``user_note``: optional user design guidance (the service's Refine note, cleaned upstream) appended to every
    repaint prompt as quoted data after all rules, so it cannot override framing, outline, silhouette, pose, face
    and feature preservation or the artwork's fixed details. An empty note changes nothing. The note comes only
    from this explicit parameter (or the CLI --refine-note flag). A non-empty note and its hash are recorded
    in stage-3f.json.
    """
    refine_profile = refine_profile or os.environ.get("HEROGEN_REFINE_PROFILE", "best")
    if refine_profile not in ("best", "fast"):
        raise ValueError("refine_profile must be best or fast")
    if refine_profile == "fast" and upscale_texture:
        raise ValueError("fast refinement cannot add upscale-texture passes")
    run = Path(run).resolve()
    output = run / "03-refined.glb"
    pending = run / "03-refined.part.glb"
    output.unlink(missing_ok=True)
    pending.unlink(missing_ok=True)
    rec = StageRecord(run, STAGE, "texture refine")
    record_path = run / "stage-3f.json"
    prior = 0.0
    old_budget = None
    if record_path.exists():
        old = json.loads(record_path.read_text())
        prior = float((old.get("budget") or {}).get("spent_usd_total", old.get("cost_usd") or 0.0))
        old_budget = old.get("budget")
    budget = Budget(client, cap_usd, prior, state=old_budget)
    prior = budget.state["spent_usd_total"]
    rec.data.update(profile=profile, views={}, cost_basis=COST_BASIS + "; replayed (cached) calls cost 0",
                    estimate_usd=cost_estimate(upscale_texture=upscale_texture, refine_profile=refine_profile),
                    estimate_max_samples_usd=cost_estimate(True, upscale_texture, refine_profile=refine_profile),
                    recipes=[recipe, "RF3H"] + (["RF5U"] if upscale_texture else []),
                    repaint_recipe=recipe, upscale_texture=upscale_texture, texture_size=4096, final_model="03-source.glb",
                    thresholds={})
    if refine_profile == "fast":
        from .fast import empty_progress
        rec.data["refine_profile"] = "fast"
        empty_progress(run)
    from .prompts import PROMPTS
    rec.data["prompt_sha256"] = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in PROMPTS.items()}
    if user_note:
        rec.data["user_note"] = user_note
        rec.data["user_note_sha256"] = hashlib.sha256(user_note.encode()).hexdigest()
    start = len(client.log.calls)
    previous_guard = client.before_api_attempt
    previous_settle = client.after_api_attempt

    def guard(model):
        with budget.lock:
            budget.before_attempt(model)
            if previous_guard is not None:
                previous_guard(model)
            budget.reserve(model)
            save(ok=False, status="running")

    def settle(cost):
        with budget.lock:
            budget.settle(cost)
            save(ok=False, status="running")
            if previous_settle is not None:
                previous_settle(cost)

    def save(**extra):
        with budget.lock:
            return locked_save(**extra)

    def locked_save(**extra):
        rec.data.update(cost_usd=client.log.cost_usd(start), models=client.log.models(start),
                        calls=client.log.calls[start:], budget={**budget.state, "cap_usd": cap_usd,
                                                              "prior_usd": prior,
                                                              "spent_usd_total": budget.spent()})
        return rec.finish(extra)

    client.before_api_attempt = guard
    client.after_api_attempt = settle
    current_record = None
    try:
        save(ok=False, status="running")
        if recipe not in ("RF2R", "RF2W"):
            raise ValueError(f"unknown repaint recipe {recipe!r} (use RF2R or RF2W)")
        if profile not in REFINE_PROFILES:
            if refine_profile == "fast":
                empty_progress(run, done=True)
            return save(status="skipped", reason="profile is not best, roblox-ingame or roblox-avatar")
        if not (run / "03-source.glb").is_file():
            if refine_profile == "fast":
                empty_progress(run, done=True)
            return save(status="skipped", reason="no SAM source mesh", final_model="04-body/high.glb")
        stage3 = json.loads((run / "stage-3.json").read_text())
        if stage3.get("ok") is not True:
            raise ValueError("texture refine needs a passed stage-3.json")
        from . import character as C, imaging as I
        rec.data["thresholds"] = {
            "outline": I.GATE,
            "closeup_outline": I.CLOSEUP_GATE,
            "hair": I.HAIR_GATE,
            "face_outside_max": I.PART_EXTRA_MAX,
            "closeup_feather_px": I.CLOSEUP_FEATHER_PX if I.CLOSEUP_FEATHER else 0,
            "exposure_mode": I.EXPOSURE_MODE if I.EXPOSURE_SOLVE else "off",
        }
        source = run / "03-source.glb"
        rec.data["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        work = run / "blender-work" / "texture-refine"
        work.mkdir(parents=True, exist_ok=True)
        cover = Image.open(run / "01-cover.png").convert("RGB")
        # All preparation is rebuilt from this invocation's source, never an earlier run's geometry or UVs.
        info = C.setup(source, work, client, timeout=timeout)
        closeup_cams = []
        if upscale_texture:
            from . import closeups
            joint_path = run / "03-joints.json"
            joints = json.loads(joint_path.read_text()) if joint_path.exists() else None
            meta = json.loads((run / "views/meta.json").read_text())
            closeup_cams = closeups.cameras_for(info, joints, meta)
            for view in {closeups.nearest_view(cam["angle"]) for cam in closeup_cams}:
                closeups.artwork_camera(run, info, view)
        head_found = info["head_centre"] is not None
        head_note = "head on the front render" if head_found else "no head on the front render; head views skipped"
        rec.add(check("head_found", head_found, gate=False, note=head_note))
        if not head_found:
            rec.note(head_note)
            rec.note("face exclusion skipped: no head on the front render")
        save(ok=False, status="running")
        prep = C.prep(source, info, work, timeout=timeout)
        cams = C.cameras_for(info)
        tex = I.Texture(prep, info)
        current = prep / "base.png"
        sequence = [(name, False) for name in REFINE_ORDER] + [(name, True) for name in HAIR_ORDER]
        stopped = False
        if refine_profile == "fast":
            from .fast import waves
            current, stopped = waves(client, prep, current, tex, cams, work, rec.data, timeout,
                                     cover, head_found, recipe, save, budget.lock, user_note=user_note)
        else:
            for step, (name, hair) in enumerate(sequence):
                key = ("hair-" if hair else "") + name
                current_record = {"accepted": False, "reason": "not attempted: API cap" if stopped else "incomplete",
                                  "tries": 0, "attempts": []}
                rec.data["views"][key] = current_record
                if not head_found and name.startswith("head"):
                    current_record.update(reason="head_not_found", note=head_note)
                    continue
                if stopped:
                    continue
                save(ok=False, status="running")
                try:
                    current = _view(client, prep, current, tex, cams[name], hair, work / key,
                                    current_record, timeout, cover, face_exclusion=head_found, body_cam=cams["body+0"], repaint=recipe,
                                    user_note=user_note)
                except BudgetStop as exc:
                    current_record.update(accepted=False, reason=str(exc))
                    rec.data["stop_reason"] = str(exc)
                    rec.note(str(exc))
                    stopped = True
                save(ok=False, status="running")
        if upscale_texture:
            # Close-ups render and gate against the refined atlas after all twenty original views.
            if any(v["accepted"] for v in rec.data["views"].values()) and (I.EXPOSURE_SOLVE or I.CLOSEUP_FEATHER):
                I.rebake_views(tex, work, rec.data["views"], body_cam=cams["body+0"])
            current = tex.save(work / "texture.png")
            for cam in closeup_cams:
                key = cam["name"]
                current_record = {"accepted": False, "reason": "not attempted: API cap" if stopped else "incomplete",
                                  "tries": 0, "attempts": [], "camera": cam}
                rec.data["views"][key] = current_record
                if stopped:
                    continue
                folder = work / key
                folder.mkdir(parents=True, exist_ok=True)
                crop, references = closeups.reference(run, cam, info, folder)
                current_record["references"] = references
                save(ok=False, status="running")
                try:
                    current = _view(client, prep, current, tex, cam, False, folder, current_record,
                                    timeout, cover, face_exclusion=head_found, closeup_reference=crop,
                                    user_note=user_note)
                except BudgetStop as exc:
                    current_record.update(accepted=False, reason=str(exc))
                    rec.data["stop_reason"] = str(exc)
                    rec.note(str(exc))
                    stopped = True
                save(ok=False, status="running")
        accepted = sum(v["accepted"] for v in rec.data["views"].values())
        rec.add(check("accepted_views", accepted, 1, gate=False))
        if accepted and refine_profile == "fast":
            rec.data.update(outputs=["03-refined.glb"], final_model="03-refined.glb")
        elif accepted:
            from .backend import blender
            if I.EXPOSURE_SOLVE or I.CLOSEUP_FEATHER:
                gains = I.rebake_views(tex, work, rec.data["views"], body_cam=cams["body+0"])
                if gains and I.EXPOSURE_SOLVE:
                    rec.data["exposure_gains"] = gains
            final = tex.save(work / "texture.png")
            blender("export", {"blend": str(prep / "work.blend"), "texture": str(final), "out": str(pending)},
                    work / "blender.log", timeout=timeout)
            os.replace(pending, output)
            rec.data.update(outputs=["03-refined.glb"], final_model="03-refined.glb")
        else:
            rec.note("all views rejected; keeping 03-source.glb")
        return save(status="cap_reached" if stopped else "complete")
    except BudgetStop as exc:
        if refine_profile == "fast":
            empty_progress(run, done=True)
        rec.data["stop_reason"] = str(exc)
        rec.note(str(exc))
        return save(status="cap_reached")
    except Exception as exc:
        if current_record is not None:
            current_record.update(accepted=False, reason=type(exc).__name__)
        save(ok=False, status="error", error=error_text(exc, run))
        raise
    finally:
        client.before_api_attempt = previous_guard
        client.after_api_attempt = previous_settle
        pending.unlink(missing_ok=True)

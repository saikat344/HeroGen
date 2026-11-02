"""Stage 3f: sequential, gated repaint projection onto SAM 3D Objects' own mesh."""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
import uuid
from concurrent.futures import CancelledError
from contextvars import ContextVar
from pathlib import Path

from PIL import Image

from herogen.blender.record import StageRecord, check
from herogen.blender.stage3 import error_text
from herogen.pipeline.client import COST_BASIS, IMAGE_MODEL, SAM_MODEL, PRICE_PER_IMAGE_USD, ModelClient
from herogen.pipeline.redo import clean_note
from herogen.refine_budget import BudgetStop, update as update_budget

STAGE = "3f"
DEFAULT_CAP_USD = 0.80
REFINE_PROFILES = frozenset(("best", "roblox-ingame", "roblox-avatar"))
REFINE_ORDER = ("head+0", "head+40", "head-40", "head+90", "head-90", "head+180",
                "body+0", "body+45", "body-45", "body+90", "body-90", "body+180")
# Negative angles -90/-135 are 270/225 degrees, with the same orientation and projection.
HAIR_ORDER = ("head+90", "head+135", "head+180", "head-135", "head-90",
              "upper+135", "upper+180", "upper-135")
_attempt_cancelled = ContextVar("refine_attempt_cancelled", default=None)


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


def cost_estimate(maximum=False, upscale_texture=False, *, refine_profile="best", beauty=False,
                  references=False):
    """Per-run list-price projection; no cache discount or transport retries. Hair uses up to two samples.

    ``references`` adds fast-lane beauty-reference generation: one image and two SAM queries per yaw.
    """
    normal, hair = (3, 2) if maximum else (1, 1)
    if refine_profile == "fast":
        from .beauty import FAST_ANGLES
        from .fast import ATTEMPTS, WAVES
        sequence = [view for wave in WAVES for view in wave]
        normal_count = sum(not hair for _, hair in sequence)
        hair_count = len(sequence) - normal_count
        # All attempts may start before a first acceptance can cancel them, including duplicate render masks.
        normal = hair = ATTEMPTS
        images = normal_count * normal + hair_count * hair
        segments = 2 + normal_count * normal * 4 + hair_count * hair * 5
        if references:
            # One BT1 sample per yaw, its outline mask and, when style selection would use it, its head crop.
            images += len(FAST_ANGLES)
            segments += 2 * len(FAST_ANGLES)
        if beauty:
            # One beauty head slot and two body-front slots: reference face/features,
            # repaint features and both-eye regions, before any exact-cache discount.
            segments += (1 + ATTEMPTS) * 9 + 1
            # With no accepted frontal, resolve ordinary fast paint for the fallback face.
            return round(images * PRICE_PER_IMAGE_USD[IMAGE_MODEL] + segments * PRICE_PER_IMAGE_USD[SAM_MODEL]
                         + cost_estimate(refine_profile="fast") + PRICE_PER_IMAGE_USD[SAM_MODEL], 6)
        return round(images * PRICE_PER_IMAGE_USD[IMAGE_MODEL] + segments * PRICE_PER_IMAGE_USD[SAM_MODEL], 6)
    else:
        normal_count, hair_count = len(REFINE_ORDER), len(HAIR_ORDER)
    images = normal_count * normal + hair_count * hair
    segments = 2 + normal_count * (1 + normal + 2) + hair_count * (1 + hair + 3)
    if upscale_texture:
        images += 16 * normal
        segments += 16 * (3 + normal)
    return round(images * PRICE_PER_IMAGE_USD[IMAGE_MODEL] + segments * PRICE_PER_IMAGE_USD[SAM_MODEL], 6)


def fast_product_options(side_beauty=None, beauty=None, beauty_dir=None, env=None):
    """The fast product lane's stage-3f defaults: SideBeautyPass and generated beauty references on.

    Explicit flags win; otherwise HEROGEN_SIDE_BEAUTY and HEROGEN_REFINE_BEAUTY (0 or 1, unset means 1)
    decide. A saved folder (flag or HEROGEN_REFINE_BEAUTY_DIR) replaces generation. The region palette
    stays opt-in. Returns run_stage3f keyword arguments.
    """
    env = os.environ if env is None else env

    def switch(name, value):
        if value is not None:
            return bool(value)
        raw = env.get(name) or "1"
        if raw not in ("0", "1"):
            raise ValueError(f"{name} must be 0 or 1")
        return raw == "1"

    side, enabled = switch("HEROGEN_SIDE_BEAUTY", side_beauty), switch("HEROGEN_REFINE_BEAUTY", beauty)
    saved = beauty_dir or env.get("HEROGEN_REFINE_BEAUTY_DIR") or None
    if saved is not None and not enabled:
        raise ValueError("beauty is switched off but a saved beauty folder was given")
    return {"side_beauty": side, **({"beauty_dir": saved} if saved is not None else {"generate_beauty": enabled})}


def side_art_yaw(angle):
    """Turnaround conditioning starts at either profile and includes the rear hemisphere."""
    return 90 <= angle % 360 <= 270


def side_orientation(run):
    """Require the saved stage-2 toe-direction evidence before trusting side artwork."""
    path = Path(run) / "stage-2.json"
    if not path.is_file():
        return {"enabled": False, "reason": "stage-2 orientation record missing"}
    data = json.loads(path.read_text())
    checks = {c["name"]: c for c in data.get("checks", [])}
    names = ("right_view_toe_offset_as_drawn", "left_view_toe_offset_as_drawn")
    fixes = data.get("details", {}).get("view_assignment", {}).get("side_fixes", {})
    if any(fixes.values()):
        return {"enabled": False, "reason": "stage-2 recorded a side orientation correction", "side_fixes": fixes}
    if not all(name in checks for name in names):
        return {"enabled": False, "reason": "stage-2 orientation checks missing"}
    offsets = {name: checks[name].get("value") for name in names}
    values = list(offsets.values())
    valid = all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)
    if not valid or not (values[0] > 0 and values[1] < 0) or not all(checks[n].get("pass") is True for n in names):
        return {"enabled": False, "reason": "stage-2 does not confirm unswapped sides", "offsets": offsets}
    return {"enabled": True, "reason": "stage-2 confirms unswapped sides", "offsets": offsets}


def art_references(run, cam, body_cam, info, artwork):
    """Select side/rear yaw-matched art; front passes retain their cover-only recipe."""
    if not artwork or not side_art_yaw(cam["angle"]):
        return (), {}, {}
    from .closeups import views_for_angle, framed_artwork, framed_mask
    from .imaging import is_closeup
    art = tuple((v, artwork[v]) for v in views_for_angle(cam["angle"]) if v in artwork)
    masks, framing = {}, {}
    if art and is_closeup(cam, body_cam):
        cropped = []
        for name, _ in art:
            image, reference = framed_artwork(run, cam, info, name)
            cropped.append((name, image))
            framing[name] = reference
            masks[name] = framed_mask(run, name, reference)
        art = tuple(cropped)
    return art, masks, framing


def _view(client, prep, current, tex, cam, hair, folder, record, timeout, cover, *, face_exclusion=True, body_cam=None,
          closeup_reference=None, closeup_recipe="RF5U", repaint="RF2R", rendered=None,
          apply=None, attempts=None, attempt_start=0, cancelled=None, user_note="", recipe_name=None, reference_record=None, face_keep=False, complete_face=False, save_face=False, art=(), art_masks=None,
          surface_reference_mask=None, fast_outline=False):
    import numpy as np
    from scipy import ndimage
    from . import buildlog, character as C, imaging as I
    from .backend import blender
    from .cameras import decode_pos, project, px_per_unit, read_exr
    from .prompts import ART_VARIANTS, prompt

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
    if surface_reference_mask is not None:
        from .closeups import silhouette_reference
        # Change the conditioning image, never the generated sample or its acceptance mask.
        cover = cover.resize(inp.size, Image.Resampling.LANCZOS)
        mask = surface_reference_mask.resize(inp.size, Image.Resampling.NEAREST)
        cover = silhouette_reference(cover, mask, figure)
        cover.save(folder / "surface-reference.png")
        record["reference_framing"] = "render silhouette; uniform height fit and nearest foreground extension"
    if art_masks:
        from .closeups import silhouette_reference
        aligned = []
        for art_name, art_image in art:
            art_image.save(folder / f"art-{art_name}-framed.png")
            aligned.append((art_name, silhouette_reference(art_image, art_masks[art_name], figure)))
            record["art_framing"][art_name]["silhouette_alignment"] = "uniform height fit and nearest artwork edge extension to render alpha"
        art = tuple(aligned)
    for art_name, art_image in art:
        art_image.save(folder / f"art-{art_name}.png")
    parts = None
    reference_features = None
    recipe = recipe_name or (closeup_recipe if closeup_reference is not None else "RF3H" if hair else repaint)
    if art:
        recipe = ART_VARIANTS[recipe]
    record.update(prompt=recipe, camera=cam)
    if art:
        record["art_views"] = [name for name, _ in art]
    if closeup_reference is not None:
        measurable, source_checks = I.closeup_outline(figure, I.closeup_figure_mask(render_rgb, figure), cam["zoom"])
        record["source_outline_checks"] = source_checks
        if not measurable:
            record.update(accepted=False, reason="render outline not measurable")
            return current
    if reference_record is not None:
        record["reference"] = reference_record
    # A parallel try's build-log number is its slot, which a reused attempt index can repeat.
    slot = (getattr(client, "event_fields", None) or {}).get("try_")
    for attempt in range(attempt_start, attempt_start + (attempts if attempts is not None else I.HAIR_ATTEMPTS if hair else I.ATTEMPTS)):
        if cancelled is not None and cancelled.is_set():
            record.update(reason="cancelled: lower attempt accepted")
            return current
        trial = {"try": attempt + 1, "accepted": False, "reason": "incomplete"}
        record["attempts"].append(trial)
        record["tries"] = attempt + 1
        art_images = [image for _, image in art]
        references = ([inp, *art_images, cover] if closeup_reference is None
                      else [inp, closeup_reference, *art_images, cover])
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
            limits = None
        elif hair:
            agreement, _, _ = I.interior_agreement(render_rgb, warped, figure, fig_e)
            measures["agreement"] = agreement
            passed = (overlap >= I.HAIR_GATE["iou_min"] and agreement >= I.HAIR_GATE["agree_min"]
                      and abs(reg["scale"] - 1) <= I.HAIR_GATE["scale_max_change"])
            reason = "hair outline, scale or colour agreement"
            limits = {"overlap_min": I.HAIR_GATE["iou_min"], "agreement_min": I.HAIR_GATE["agree_min"],
                      "scale_change_max": I.HAIR_GATE["scale_max_change"]}
        else:
            # Fast measures the outline with near-grey cloth the repaint kept (`figure_mask`); the
            # limits are unchanged, and paint weights keep the established mask.
            measured = I.figure_mask(warped, render_figure=figure, render_rgb=render_rgb) if fast_outline else fig_e
            closeup = body_cam is not None and I.is_closeup(cam, body_cam)
            passed, measures = I.outline_gate(figure, measured, closeup=closeup)
            reason = "outline moved"
            limits = {"overlap_min": (I.CLOSEUP_GATE if closeup else I.GATE)["iou_min"]}
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
        if passed and recipe in ("RF6B", "RF17B", "RF18B") and cam["name"] in ("head+0", "body+0") and parts["face"].any():
            from .beauty import facial_features
            if reference_features is None:
                reference_features = facial_features(client, cover)
            painted = Image.fromarray((np.clip(warped, 0, 1) * 255).round().astype(np.uint8))
            features = facial_features(client, painted, parts["face"])
            if complete_face:
                from .beauty import frontal_features
                features.update(frontal_features(client, painted, parts["face"]))
            missing = [name for name, present in reference_features.items() if present and not features[name]]
            passed = any(reference_features.values()) and not missing
            if complete_face:
                passed = passed and features["both_eyes"] and features["mouth"]
                missing += [name for name in ("both_eyes", "mouth") if not features[name] and name not in missing]
            measures["facial_features"] = {"reference": reference_features, "repaint": features, "missing": missing}
            reason = ("beauty reference has no facial features" if not any(reference_features.values())
                      else "missing facial features: " + ", ".join(missing))
        trial["reason"] = "accepted" if passed else reason
        if not passed:
            record["reason"] = reason
            buildlog.try_result(buildlog.view_key(cam, hair), trial, False, reason, limits, number=slot)
            continue
        # Beauty references already carry the intended light and palette; fitting them back to the
        # source render would remove their crease shading and material contrast.
        refined = warped.copy() if recipe in ("RF5B", "RF6B", "RF15B", "RF16B", "RF17B", "RF18B") else I.colour_match(render_rgb, warped, figure & fig_e)
        if hair:
            weight = I.hair_weight(client, inp, edit, render_rgb, warped, alpha, mapx, mapy,
                                   parts["face"] if face_exclusion else None)
        else:
            weight = (~ndimage.binary_dilation(figure ^ fig_e, iterations=4)).astype(np.float32)
            weight *= I.hair_allowed(client, inp, edit, mapx, mapy, figure.shape, hair_only=False)
        if save_face:
            Image.fromarray(parts["face"].astype(np.uint8) * 255).save(folder / "face.png")
            Image.fromarray((np.clip(weight, 0, 1) * 255).round().astype(np.uint8)).save(folder / "source-weight.png")
        if face_keep and parts is not None and parts["face"].any():
            # Face-consistency guard (beauty recipe): with no accepted frontal repaint there is no
            # coherent painted face to blend against, so this oblique view must not overwrite the
            # render's face region; the atlas keeps its base paint there, as hair views already do.
            weight *= ~ndimage.binary_dilation(
                parts["face"], iterations=max(1, round(I.FACE_KEEP_PX * max(figure.shape) / 1024)))
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
        buildlog.try_result(buildlog.view_key(cam, hair), trial, True, limits=limits, number=slot)
        return tex.save(folder / "texture.png") if apply is None else current
    return current


def _artwork_grade(run, prep, cam, folder, record, timeout, *, artwork=None, artwork_mask=None):
    import numpy as np
    from . import palette
    from .backend import blender

    artwork = Path(artwork) if artwork is not None else run / "views/front.png"
    artwork_mask = Path(artwork_mask) if artwork_mask is not None else run / "masks/front.png"
    missing = [str(path.relative_to(run)) if path.is_relative_to(run) else path.name
               for path in (artwork, artwork_mask) if not path.is_file()]
    if missing:
        record.update(status="skipped", reason="missing artwork: " + ", ".join(missing))
        return None
    record.update(status="fitting")
    try:
        blender("render", {"blend": str(prep / "work.blend"), "texture": str(prep / "base.png"),
                           "cams": [cam], "out": str(folder)}, folder / "blender.log", timeout=timeout)
        with Image.open(folder / f"{cam['name']}.png") as image:
            rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255
        with Image.open(artwork) as image:
            art = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
        with Image.open(artwork_mask) as image:
            mask = np.asarray(image.convert("L")) > 127
        record.update(palette.fit(rgba[..., :3], rgba[..., 3] > .5, art, mask))
        return record
    except Exception as exc:
        record.update(status="error", reason=error_text(exc, run))
        raise


def run_stage3f(run: Path, client: ModelClient, *, profile="best", cap_usd=DEFAULT_CAP_USD, timeout=900.0,
                upscale_texture=False, recipe="RF2R", refine_profile=None, user_note: str = "",
                beauty_dir=None, side_beauty=None, region_palette=False, generate_beauty=False):
    """Write 03-refined.glb after stage 3. Rejected views and cap stops are successful recorded fallbacks.

    Model failures are recorded and re-raised. A retry counts this run's previous stage-3f spend toward its cap;
    cost_usd itself counts this invocation's calls, matching the other stages' convention.

    ``user_note``: optional user design guidance, cleaned with the shared bounded text rules and appended to every
    repaint prompt as quoted data after all rules, so it cannot override framing, outline, silhouette, pose, face
    and feature preservation or the artwork's fixed details. An empty note changes nothing. The note comes only
    from this explicit parameter (or the CLI --refine-note flag). A non-empty note and its hash are recorded
    in stage-3f.json. ``side_beauty`` opts side/rear yaws (90 through 270 degrees) into
    yaw-matched turnaround conditioning; front-facing passes keep their default recipe and references. None reads
    HEROGEN_SIDE_BEAUTY=1. It requires all four cardinal drawings, calibration and masks for close views.
    ``generate_beauty`` (fast profile only, instead of ``beauty_dir``) generates this run's beauty references
    into ``<run>/beauty`` with `beauty.generate_fast`, beside setup and preparation, then uses them as a saved
    folder. Its calls count toward this stage's cost and cap.
    """
    side_beauty = (os.environ.get("HEROGEN_SIDE_BEAUTY", "0") == "1" if side_beauty is None else side_beauty)
    user_note = clean_note(user_note, allow_empty=True)
    refine_profile = refine_profile or os.environ.get("HEROGEN_REFINE_PROFILE", "best")
    if refine_profile not in ("best", "fast"):
        raise ValueError("refine_profile must be best or fast")
    if refine_profile == "fast" and upscale_texture:
        raise ValueError("fast refinement cannot add upscale-texture passes")
    if generate_beauty:
        if refine_profile != "fast":
            raise ValueError("generated beauty references need the fast refine profile")
        if beauty_dir is not None:
            raise ValueError("use saved or generated beauty references, not both")
        beauty_dir = Path(run).resolve() / "beauty"
    beauty_dir = beauty_dir if beauty_dir is not None else os.environ.get("HEROGEN_REFINE_BEAUTY_DIR") or None
    if beauty_dir is not None and not Path(beauty_dir).is_absolute():
        beauty_dir = Path(run) / beauty_dir
    beauty_refs = None
    beauty_selection = None
    if beauty_dir is not None and not generate_beauty:
        from . import beauty
        beauty_dir = Path(beauty_dir)
        beauty_refs = beauty.load(beauty_dir)
        beauty_selection = beauty_refs
        if beauty.selected_yaw(beauty_refs, 0) is None:
            beauty_refs = None
    run = Path(run).resolve()
    output = run / "03-refined.glb"
    pending = run / "03-refined.part.glb"
    output.unlink(missing_ok=True)
    pending.unlink(missing_ok=True)
    rec = StageRecord(run, STAGE, "texture refine")
    record_path = run / "stage-3f.json"
    prior = 0.0
    old_budget = None
    if cap_usd is not None and record_path.exists():
        old = json.loads(record_path.read_text())
        prior = float((old.get("budget") or {}).get("spent_usd_total", old.get("cost_usd") or 0.0))
        old_budget = old.get("budget")
    budget = Budget(client, cap_usd, prior, state=old_budget) if cap_usd is not None else None
    if budget is not None:
        prior = budget.state["spent_usd_total"]
    # Guards the record and budget state. Without spend limits there is no budget, but fast waves still share it.
    state_lock = budget.lock if budget is not None else threading.RLock()
    rec.data.update(profile=profile, views={}, cost_basis=COST_BASIS + "; replayed (cached) calls cost 0",
                    estimate_usd=cost_estimate(upscale_texture=upscale_texture, refine_profile=refine_profile,
                                               beauty=beauty_refs is not None or generate_beauty,
                                               references=generate_beauty),
                    estimate_max_samples_usd=cost_estimate(True, upscale_texture, refine_profile=refine_profile,
                                                          beauty=beauty_refs is not None or generate_beauty,
                                                          references=generate_beauty),
                    recipes=[recipe, "RF3H"] + (["RF5U"] if upscale_texture else []),
                    repaint_recipe=recipe, upscale_texture=upscale_texture, texture_size=4096, final_model="03-source.glb",
                    thresholds={})
    if refine_profile == "fast":
        from .fast import empty_progress
        rec.data["refine_profile"] = "fast"
        empty_progress(run)
    from .prompts import PROMPTS
    rec.data["prompt_sha256"] = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in PROMPTS.items()
                                  if side_beauty or (beauty_refs is not None and name in ("RF5B", "RF6B", "RF17B", "RF18B"))
                                  or name in ("RF2R", "RF2W", "RF3H", "RF4U", "RF5U")}
    if user_note:
        rec.data["user_note"] = user_note
        rec.data["user_note_sha256"] = hashlib.sha256(user_note.encode()).hexdigest()
    if beauty_selection is not None:
        rec.data["beauty_references"] = beauty_selection
    if beauty_refs is not None:
        rec.data.update(recipes=["RF6B"])
    start = len(client.log.calls)
    previous_guard = client.before_api_attempt
    previous_settle = client.after_api_attempt

    def guard(model):
        def check_cancelled():
            cancelled = _attempt_cancelled.get()
            if cancelled is not None and cancelled.is_set():
                raise CancelledError("lower attempt accepted")

        with budget.lock:
            check_cancelled()
            budget.reserve(model)
        acquired = False
        try:
            # A limiter can wait for another attempt to settle. Never hold the state lock while waiting.
            if previous_guard is not None:
                previous_guard(model)
                acquired = True
            with budget.lock:
                check_cancelled()  # selection may have cancelled this attempt during the limiter wait
                save(ok=False, status="running")
        except BaseException:
            with budget.lock:
                budget.settle(0)
                save(ok=False, status="running")
            if acquired and previous_settle is not None:
                previous_settle(0)
            raise

    def settle(cost):
        with budget.lock:
            budget.settle(cost)
            save(ok=False, status="running")
        if previous_settle is not None:
            previous_settle(cost)

    def save(**extra):
        with state_lock:
            return locked_save(**extra)

    def locked_save(**extra):
        rec.data.update(cost_usd=client.log.cost_usd(start), models=client.log.models(start),
                        calls=client.log.calls[start:])
        if budget is not None:
            rec.data["budget"] = {**budget.state, "cap_usd": cap_usd, "prior_usd": prior,
                                  "spent_usd_total": budget.spent()}
        return rec.finish(extra)

    if budget is not None:
        client.before_api_attempt = guard
        client.after_api_attempt = settle
    current_record = None
    generation = None

    def drain():
        # Generated references run beside setup; never record a result while their calls can still settle.
        if generation is not None:
            generation[0].shutdown(wait=True)

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
        artwork = {}
        if side_beauty:
            rec.data["side_beauty_orientation"] = side_orientation(run)
        use_side_art = side_beauty and rec.data["side_beauty_orientation"]["enabled"]
        for view_name in (("front", "right", "back", "left") if use_side_art else ()):
            view_path = run / "views" / f"{view_name}.png"
            with Image.open(view_path) as view_image:
                artwork[view_name] = view_image.convert("RGB")
        if artwork:
            rec.data["side_beauty"] = True
            from .prompts import ART_VARIANTS
            rec.data["recipes"] = [ART_VARIANTS[recipe], ART_VARIANTS["RF3H"]] + (
                [ART_VARIANTS["RF5U"]] if upscale_texture else [])
            rec.data["art_views"] = sorted(artwork)
        # All preparation is rebuilt from this invocation's source, never an earlier run's geometry or UVs.
        palette_manifest = os.environ.get("HEROGEN_REFINE_REGION_PALETTE", "")
        if palette_manifest.lower() in ("0", "false", "off"):
            palette_manifest = ""
        auto_palette = region_palette or palette_manifest.lower() in ("1", "true", "auto")
        setup_options = {"save_palette_masks": True} if auto_palette else {}
        if beauty_dir is not None:
            setup_options["save_head_mask"] = True
        if generate_beauty:
            from concurrent.futures import ThreadPoolExecutor
            from . import beauty
            pool = ThreadPoolExecutor(max_workers=1)
            generation = (pool, pool.submit(beauty.generate_fast, run, client, beauty_dir))
        from . import buildlog
        probe_started = time.monotonic()
        info = C.setup(source, work, client, timeout=timeout, **setup_options)
        buildlog.head_probe(info["head_centre"] is not None, time.monotonic() - probe_started)
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
        # Generated references may checkpoint this record from their own threads.
        with state_lock:
            rec.add(check("head_found", head_found, gate=False, note=head_note))
            if not head_found:
                rec.note(head_note)
                rec.note("face exclusion skipped: no head on the front render")
        save(ok=False, status="running")
        prep = C.prep(source, info, work, timeout=timeout)
        cams = C.cameras_for(info)
        if auto_palette:
            from .region_palette import prepare_manifest
            palette_manifest = prepare_manifest(run, prep, cams["body+0"], work / "region-palette-inputs")
        tex = I.Texture(prep, info)
        current = prep / "base.png"
        grade = None
        default_face_base = None
        if generation is not None:
            generated = generation[1].result()
            drain()
            rec.data["beauty_generation"] = {k: generated.get(k) for k in (
                "angles", "attempts_per_angle", "fallback", "selected", "status", "cost_usd", "seconds")}
            beauty_selection = beauty_refs = beauty.load(beauty_dir)
            if beauty.selected_yaw(beauty_refs, 0) is None:
                beauty_refs = None
            rec.data["beauty_references"] = beauty_selection
            if beauty_refs is not None:
                rec.data["prompt_sha256"].update({name: hashlib.sha256(PROMPTS[name].encode()).hexdigest()
                                                  for name in ("RF5B", "RF6B", "RF17B", "RF18B")})
                if not artwork:
                    rec.data["recipes"] = ["RF6B"]
            else:
                rec.note("no usable front beauty reference; standard recipes")
            save(ok=False, status="running")
        if beauty_refs is not None and refine_profile == "fast":
            import shutil
            default_face_base = work / "default-face-base.png"
            shutil.copyfile(current, default_face_base)
        if beauty_refs is not None:
            import numpy as np
            from .palette import apply
            front_yaw = beauty.selected_yaw(beauty_refs, 0)
            front = beauty_refs["views"][str(front_yaw)]
            rec.data["palette_grade"] = {"target": "beauty reference", "yaw": front_yaw}
            grade = _artwork_grade(run, prep, cams["body+0"], work / "palette",
                                   rec.data["palette_grade"], timeout,
                                   artwork=beauty_dir / front["image"],
                                   artwork_mask=beauty_dir / front["mask"])
            if grade is not None:
                with Image.open(current) as image:
                    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
                Image.fromarray((apply(base, grade) * 255).round().astype(np.uint8)).save(current)
                tex.T = apply(tex.T, grade)
                if refine_profile == "fast":
                    # Each progressive version rebuilds from the frozen graded base.
                    tex.base = apply(tex.base, grade)
                    rows, cols = np.divmod(tex.idx, tex.base.shape[1])
                    tex.base_cov = tex.base[rows, cols].copy()
            save(ok=False, status="running")
        sequence = [(name, False) for name in REFINE_ORDER] + [(name, True) for name in HAIR_ORDER]
        stopped = False
        if refine_profile == "fast":
            from .fast import waves
            current, stopped = waves(client, prep, current, tex, cams, work, rec.data, timeout,
                                     cover, head_found, recipe, save, state_lock, user_note=user_note,
                                     art_context=(run, info, artwork) if artwork else None,
                                     palette_manifest=palette_manifest,
                                     **({"beauty_context": (beauty_dir, beauty_refs, default_face_base, info)}
                                        if beauty_refs is not None else {}))
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
                    art, art_masks, framing = art_references(run, cams[name], cams["body+0"], info, artwork)
                    if framing:
                        current_record["art_framing"] = framing
                    view_cover, beauty_args = cover, {}
                    if beauty_refs is not None:
                        cam_angle = cams[name]["angle"]
                        if beauty.selected_yaw(beauty_refs, cam_angle) is None:
                            # A rear-quarter camera without a rear-facing reference stays on the
                            # standard recipe for this view; it must never take a face-bearing crop.
                            current_record.update(
                                note=f"no face-free beauty reference at {cam_angle:+g} yaw; standard recipe")
                        else:
                            view_cover, ref_record = beauty.reference(beauty_dir, beauty_refs, cam_angle,
                                                                       head=name.startswith("head"))
                            beauty_args = {"recipe_name": "RF6B", "reference_record": ref_record}
                    if beauty_refs is not None and name not in ("head+0", "body+0") and not any(
                            rec.data["views"].get(frontal, {}).get("accepted")
                            for frontal in ("head+0", "body+0")):
                        # Face-consistency guard: an oblique camera may only repaint the face once a
                        # frontal camera (head+0 or body+0) has an accepted repaint to stay coherent
                        # with; until then the face region keeps its base paint.
                        beauty_args["face_keep"] = True
                        current_record["face_consistency_guard"] = "no accepted frontal repaint; face keeps base paint"
                        rec.data["face_consistency_guard"] = True
                    current = _view(client, prep, current, tex, cams[name], hair, work / key,
                                    current_record, timeout, view_cover, face_exclusion=head_found,
                                    body_cam=cams["body+0"], repaint=recipe, user_note=user_note, art=art, art_masks=art_masks, **beauty_args)
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
                    from .closeups import nearest_view
                    close_art = tuple((v, artwork[v]) for v in (nearest_view(cam["angle"]),)
                                      if side_art_yaw(cam["angle"]) and v in artwork)
                    current = _view(client, prep, current, tex, cam, False, folder, current_record,
                                    timeout, cover, face_exclusion=head_found, closeup_reference=crop,
                                    user_note=user_note, art=close_art)
                except BudgetStop as exc:
                    current_record.update(accepted=False, reason=str(exc))
                    rec.data["stop_reason"] = str(exc)
                    rec.note(str(exc))
                    stopped = True
                save(ok=False, status="running")
        accepted = sum(v["accepted"] for v in rec.data["views"].values())
        rec.add(check("accepted_views", accepted, 1, gate=False))
        if refine_profile == "fast" and (accepted or output.is_file()):
            rec.data.update(outputs=["03-refined.glb"], final_model="03-refined.glb")
        elif accepted or grade is not None:
            from .backend import blender
            if I.EXPOSURE_SOLVE or I.CLOSEUP_FEATHER:
                gains = I.rebake_views(tex, work, rec.data["views"], body_cam=cams["body+0"])
                if gains and I.EXPOSURE_SOLVE:
                    rec.data["exposure_gains"] = gains
            final = tex.save(work / "texture.png")
            if manifest := palette_manifest:
                from .region_palette import export_texture
                final, grades = export_texture(manifest, prep, final, cams["body+0"],
                                               work / "region-palette", timeout=timeout)
                rec.data["region_palette"] = grades
            blender("export", {"blend": str(prep / "work.blend"), "texture": str(final), "out": str(pending)},
                    work / "blender.log", timeout=timeout)
            os.replace(pending, output)
            rec.data.update(outputs=["03-refined.glb"], final_model="03-refined.glb")
        else:
            rec.note("all views rejected; keeping 03-source.glb")
            if not stopped:
                buildlog.skipped("views_rejected")
        if stopped:
            buildlog.skipped("spend_cap")
        return save(status="cap_reached" if stopped else "complete")
    except BudgetStop as exc:
        drain()
        if refine_profile == "fast":
            empty_progress(run, done=True)
        rec.data["stop_reason"] = str(exc)
        rec.note(str(exc))
        from . import buildlog
        buildlog.skipped("spend_cap")
        return save(status="cap_reached")
    except Exception as exc:
        drain()
        if current_record is not None:
            current_record.update(accepted=False, reason=type(exc).__name__)
        save(ok=False, status="error", error=error_text(exc, run))
        raise
    finally:
        drain()
        if budget is not None:
            client.before_api_attempt = previous_guard
            client.after_api_attempt = previous_settle
        pending.unlink(missing_ok=True)

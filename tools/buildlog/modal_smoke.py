"""Smoke test of the live build log on a real Modal Volume: `modal run tools/buildlog/modal_smoke.py`.

What it checks, on a throwaway volume (`modal.Volume.ephemeral()`, gone when the run ends; a few CPU-minutes):

- a writer container appends events to a local file, as a stage subprocess does, while `worker.EventPump` (3 s)
  copies it to the volume and commits;
- a reader container, like the web container, reads it every second through the real `storage.ModalVolumeView`
  (`refresh`, then `reading()`) and `events.read`: every event must arrive within `MAX_LAG_S`, in order, with no
  line garbled or lost;
- the volume holds nothing but `events.jsonl` (no `.part` file is ever committed);
- delete while pumping: part way through, a third container deletes the run folder and marks the hero deleted;
  the pump's next write races that delete (its `still_current` still says yes), must take itself back out
  (`still_wanted`), and the folder must stay gone for the rest of the run.

Exit status 0 and a "smoke passed" line mean every check held; any failed check raises with what it measured.

The checks themselves (`write_events`, `read_events`, `check`) need no Modal and run on a local folder in
`tests/test_herogen_buildlog_modal_smoke.py`; this module imports without the `modal` package.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

# Run locally from the repository, the checkout root goes on the path. In a Modal container the script sits at
# /root/modal_smoke.py with `herogen` added to the image, so there is no checkout root to add.
_parents = Path(__file__).resolve().parents
ROOT = _parents[2] if len(_parents) > 2 and (_parents[2] / "herogen").is_dir() else None
if ROOT is not None and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from herogen.pipeline import events  # noqa: E402

MOUNT = "/data"
RUN = "users/u_smoke/runs/j_smoke"       # the run folder's path inside the volume
PUMP_S = 2.0                             # the service default (worker.PUMP_EVERY_S)
POLL_S = 1.0
MAX_LAG_S = 8.0                          # pump beat (2 s) + view refresh (2 s) + poll (1 s) + commit, up to ~3 s on Modal
EMIT_EVERY_S = 0.5
DELETE_AT_S = 40.0
SETTLE_S = 12.0                          # after the delete, by when the raced write must be gone


def write_events(volume, root: Path, local: Path, seconds: float, deleted, pump_s: float = PUMP_S,
                 emit_every_s: float = EMIT_EVERY_S) -> dict[str, Any]:
    """Append numbered events to ``local`` for ``seconds`` while an `EventPump` mirrors them to ``root/RUN``.
    ``deleted()`` says the hero was deleted (the store's `deleted_at`); `still_current` stays True to model the
    race where a beat checked before the delete landed."""
    from herogen.service.worker import EventPump

    run = root / RUN
    run.mkdir(parents=True, exist_ok=True)
    volume.commit()
    local.parent.mkdir(parents=True, exist_ok=True)
    pump = EventPump(local, run / events.FILE, volume, lambda: True, every_s=pump_s,
                     still_wanted=lambda: not deleted(), stop=root / "users")
    n = 0
    t0 = time.monotonic()
    with pump:
        while time.monotonic() - t0 < seconds:
            events.emit("local", file=local, step="4", a=1, what="smoke", n=n)
            n += 1
            time.sleep(emit_every_s)
    return {"emitted": n, "pump_stopped": pump.stopped}


def delete_run(volume, root: Path) -> list[str]:
    """Delete the run folder as the web container does (`Storage.delete_job`), returning what the volume held."""
    volume.reload()
    held = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
    shutil.rmtree(root / RUN, ignore_errors=True)
    volume.commit()
    return held


def read_events(view, root: Path, seconds: float, deleted, poll_s: float = POLL_S) -> dict[str, Any]:
    """Poll ``root/RUN/events.jsonl`` like the endpoint does; record each event's lag and what was on disk."""
    path = root / RUN / events.FILE
    offset, seen, lags, bad, after_delete = 0, [], [], 0, []
    t0, start_ms = time.monotonic(), time.time() * 1000
    # Only events written after this reader started say how fast the log travels; earlier ones were waiting.
    while time.monotonic() - t0 < seconds:
        view.refresh()
        with view.reading():
            exists = path.exists()
            if exists:
                raw = path.read_bytes()
                bad += sum(1 for line in raw[offset:].split(b"\n")[:-1] if not _valid(line))
                page, offset, _more = events.read(path, offset, limit=1000)
            else:
                page = []
        now_ms = time.time() * 1000
        for _end, ev in page:
            seen.append(ev["n"])
            if ev["t"] >= start_ms:
                lags.append((now_ms - ev["t"]) / 1000)
        if deleted():
            after_delete.append((round(time.monotonic() - t0, 1), exists))
        time.sleep(poll_s)
    return {"seen": seen, "lags": lags, "bad": bad, "after_delete": after_delete}


def _valid(line: bytes) -> bool:
    try:
        events.validate(json.loads(line))
        return True
    except ValueError:
        return False


def check(written: dict, read: dict, held: list[str], left: list[str], *, max_lag_s: float = MAX_LAG_S,
          settle_s: float = SETTLE_S, raced: bool = True) -> dict[str, Any]:
    """Measure every check, print the measures, then raise one AssertionError naming every check that failed;
    return the measures when all pass. ``raced``: the writer's view of the volume still held the folder after the
    delete (separate containers), so its next write must have taken itself back and stopped the pump; one shared
    folder (the local test) has no such race."""
    seen, lags = read["seen"], sorted(read["lags"])
    first = read["after_delete"][0][0] if read["after_delete"] else 0.0
    gone = [exists for t, exists in read["after_delete"] if t >= first + settle_s]
    pick = lambda q: round(lags[min(len(lags) - 1, int(q * len(lags)))], 2) if lags else None  # noqa: E731
    measures = {"events": len(seen), "emitted": written["emitted"], "garbled": read["bad"],
                "lag_p50_s": pick(0.5), "lag_p95_s": pick(0.95), "lag_max_s": round(lags[-1], 2) if lags else None,
                "held": held, "left": left, "pump_stopped": written["pump_stopped"],
                "came_back": any(gone), "settle_samples": len(gone)}
    print("smoke measures:", json.dumps(measures))
    failures = []
    if read["bad"]:
        failures.append(f"{read['bad']} garbled lines")
    if not seen or not lags:
        failures.append("the reader saw no events")
    elif seen != sorted(set(seen)):
        failures.append("events arrived out of order or twice")
    elif seen != list(range(len(seen))):
        failures.append(f"events were lost: saw {len(seen)}, not a gapless run from 0")
    if lags and lags[-1] > max_lag_s:
        failures.append(f"slowest event took {lags[-1]:.1f} s to arrive (limit {max_lag_s} s)")
    if held != [f"{RUN}/{events.FILE}"]:
        failures.append(f"the volume held more than the log: {held}")
    if not gone:
        failures.append("the run ended too soon after the delete to tell")
    elif any(gone):
        failures.append("the deleted run folder's log came back")
    if raced and not written["pump_stopped"]:
        failures.append("the pump kept writing after the delete")
    if left:
        failures.append(f"files left on the volume after the delete: {left}")
    assert not failures, "; ".join(failures)
    return measures


try:
    import modal
except ImportError:      # the checks above work without it; only `modal run` needs it
    modal = None

if modal is not None:
    app = modal.App("herogen-buildlog-smoke")
    image = modal.Image.debian_slim(python_version="3.12").add_local_python_source("herogen")

    @app.function(image=image, timeout=600)
    def writer(seconds: float, flags) -> dict:
        import os

        volume = modal.Volume.from_id(os.environ["SMOKE_VOLUME_ID"])
        return write_events(volume, Path(MOUNT), Path("/tmp/smoke") / events.FILE, seconds,
                            lambda: flags.get("deleted", False))

    @app.function(image=image, timeout=600)
    def reader(seconds: float, flags) -> dict:
        import os

        from herogen.service.storage import ModalVolumeView

        volume = modal.Volume.from_id(os.environ["SMOKE_VOLUME_ID"])
        return read_events(ModalVolumeView(volume), Path(MOUNT), seconds, lambda: flags.get("deleted", False))

    @app.function(image=image, timeout=120)
    def deleter(flags) -> list:
        import os

        volume = modal.Volume.from_id(os.environ["SMOKE_VOLUME_ID"])
        flags["deleted"] = True          # the store marks the hero deleted before its files go
        return delete_run(volume, Path(MOUNT))

    @app.function(image=image, timeout=120)
    def leftovers() -> list:
        import os

        modal.Volume.from_id(os.environ["SMOKE_VOLUME_ID"]).reload()
        return sorted(str(p.relative_to(MOUNT)) for p in Path(MOUNT).rglob("*") if p.is_file())

    @app.local_entrypoint()
    def main(seconds: float = 60.0) -> None:
        with modal.Volume.ephemeral() as volume, modal.Dict.ephemeral() as flags:
            opts = {"volumes": {MOUNT: volume}, "env": {"SMOKE_VOLUME_ID": volume.object_id}}
            w = writer.with_options(**opts).spawn(seconds, flags)
            time.sleep(2.0)
            r = reader.with_options(**opts).spawn(seconds + SETTLE_S, flags)
            time.sleep(DELETE_AT_S)
            held = deleter.with_options(**opts).remote(flags)
            written, read = w.get(), r.get()
            left = leftovers.with_options(**opts).remote()
            summary = check(written, read, held, left)
            print("smoke passed:", json.dumps(summary))

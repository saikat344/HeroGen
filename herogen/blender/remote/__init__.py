"""Remote GPU runners for herogen/blender (SAM 3D Objects, SAM 3D Body on Modal) and their record/replay client.

`objects.py` and `body.py` are self-contained Modal apps. The record/replay client never starts Modal in
replay mode; explicit `prewarm` calls can request warm capacity ahead of live inference.
"""
from __future__ import annotations

import logging
import os
import threading
import time

_APPS = {"objects": "herogen-sam3d-objects", "body": "herogen-sam3d-body"}
_last_spawn: dict[str, float] = {}
_spawn_lock = threading.Lock()
_log = logging.getLogger(__name__)


def prewarm(kinds=("objects", "body")) -> None:
    """Spawn model warmups without waiting for model loading; failures are best effort.

    Each app is attempted at most once per process per 60 seconds, including failed attempts.
    Set HEROGEN_PREWARM=0 to disable warmups. No pipeline or service call is made implicitly.
    """
    failed = False
    try:
        if os.environ.get("HEROGEN_PREWARM") == "0":
            return
        for kind in kinds:
            try:
                app = os.environ.get(f"HEROGEN_{kind.upper()}_APP", _APPS[kind])
                with _spawn_lock:
                    now = time.monotonic()
                    previous = _last_spawn.get(app)
                    if previous is not None and now - previous < 60:
                        continue
                    _last_spawn[app] = now
                import modal

                modal.Cls.from_name(app, "Model")().warm.spawn()
            except Exception:
                failed = True
    except Exception:
        failed = True
    if failed:
        _log.warning("Remote model prewarm could not be submitted")

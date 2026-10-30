"""CLI: `python3 -m herogen.service dev [...]` (the whole service on 127.0.0.1 with local stand-ins) and
`python3 -m herogen.service admin --data <dir> <command>` (admin commands against a dev store).

Dev mode keeps everything under `--data` (default `service-data/`, git-ignored by its own .gitignore): the
metadata database, the data folder, a generated signing key (mode 0600) and `outbox.jsonl`, where sign-in emails
are written instead of being sent; each link is also printed. `--runner fake` (default) makes every stage
instantly with synthetic files; `--runner local` runs the real stage CLIs on this machine (Blender and
MODEL_API_KEY needed). On first start it allowlists dev@example.com and prints an invite code.
"""

from __future__ import annotations

import argparse
import logging
import os
import secrets
import sys
import threading
import time
from pathlib import Path


def _dev_key(data: Path) -> bytes:
    env = os.environ.get("HEROGEN_SESSION_KEY")
    if env:
        return env.encode()
    p = data / "dev-session-key"
    if not p.exists():
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_urlsafe(48))
    return p.read_text().strip().encode()


def dev_services(data: Path, port: int, runner_kind: str = "fake", delay: float = 0.0, echo: bool = True):
    from herogen.service.auth import Signer
    from herogen.service.config import from_env
    from herogen.service.emailer import FileSender
    from herogen.service.fakes import FakeRunner
    from herogen.service.runner import CliRunner
    from herogen.service.storage import Storage
    from herogen.service.store import Store
    from herogen.service.uploads import Limits, sanitize
    from herogen.service.web import Services
    from herogen.service.worker import ThreadBackend

    data.mkdir(parents=True, exist_ok=True)
    (data / ".gitignore").write_text("*\n")
    env = {k: v for k, v in os.environ.items() if k.startswith("HEROGEN_SERVICE_") and k not in
           ("HEROGEN_SERVICE_MODE", "HEROGEN_SERVICE_BASE_URL")}
    settings = from_env(env, mode="dev", base_url=f"http://127.0.0.1:{port}", data_dir=data / "data",
                        meta_path=data / "meta.db", outbox_path=data / "outbox.jsonl", signing_key=_dev_key(data),
                        email_provider="file", terms_version=os.environ.get("HEROGEN_SERVICE_TERMS_VERSION", "dev"),
                        allowed_hosts=("127.0.0.1", "localhost"))
    store = Store(settings.meta_path)
    storage = Storage(settings.data_dir)
    if runner_kind == "fake":
        runner = FakeRunner(delay_s=delay)
    elif runner_kind == "local":
        runner = CliRunner(model_api_key=os.environ.get("MODEL_API_KEY", ""))
    else:
        raise SystemExit(f"unknown runner {runner_kind!r}")
    backend = ThreadBackend(store, settings, storage, runner)
    limits = Limits(settings.upload_max_bytes, settings.upload_max_pixels, settings.upload_max_side,
                    settings.upload_min_side)

    class EchoSender(FileSender):
        def send(self, message):
            super().send(message)
            if echo:
                link = next((w for w in message.text.split() if "/signin#t=" in w), "")
                print(f"[dev] sign-in link for {message.to}: {link}", file=sys.stderr, flush=True)

    svc = Services(settings=settings, store=store, signer=Signer(settings.signing_key), storage=storage,
                   sender=EchoSender(settings.outbox_path), sanitize=lambda b: sanitize(b, limits),
                   submit=backend.submit)
    return svc, backend


def _first_start(svc) -> None:
    from herogen.service.admin import run as admin

    if svc.store.allow_list() or svc.store.invite_list():
        return
    svc.store.allow_add("dev@example.com", "dev mode")
    print("[dev] allowlisted dev@example.com", file=sys.stderr)
    print("[dev] invite code: " + admin("invite --count 1 --uses 5 --note dev", svc.store, svc.signer).split()[0],
          file=sys.stderr, flush=True)


def _sweeper(svc, backend, stop: threading.Event) -> None:
    from herogen.service.worker import purge, reconcile

    last_purge = 0.0
    while not stop.wait(15.0):
        try:
            reconcile(svc.store, svc.settings, backend.submit)
            if time.time() - last_purge > 3600:
                purge(svc.store, svc.settings, svc.storage)
                last_purge = time.time()
        except Exception:
            logging.getLogger("herogen.service").exception("sweep failed")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m herogen.service")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dev", help="run the service locally with stand-ins")
    d.add_argument("--port", type=int, default=8800)
    d.add_argument("--data", type=Path, default=Path("service-data"))
    d.add_argument("--runner", choices=("fake", "local"), default="fake")
    d.add_argument("--delay", type=float, default=0.3, help="fake runner: seconds per stage")
    adm = sub.add_parser("admin", help="admin commands against a dev store")
    adm.add_argument("--data", type=Path, default=Path("service-data"))
    adm.add_argument("args", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.cmd == "admin":
        from herogen.service.admin import run as admin
        from herogen.service.auth import Signer
        from herogen.service.store import Store

        if not a.args:
            ap.error("give an admin command, e.g. `invite --count 3`")
        store = Store(a.data / "meta.db")
        print(admin(a.args, store, Signer(_dev_key(a.data))))
        return 0
    try:
        import uvicorn
    except ImportError:
        print("dev mode needs the dev dependencies: pip install -r herogen/service/requirements-dev.txt",
              file=sys.stderr)
        return 1
    from herogen.service.web import create_app

    svc, backend = dev_services(a.data, a.port, a.runner, a.delay)
    _first_start(svc)
    stop = threading.Event()
    threading.Thread(target=_sweeper, args=(svc, backend, stop), daemon=True).start()
    print(f"[dev] HeroGen service on {svc.settings.base_url} (runner: {a.runner}; emails in "
          f"{svc.settings.outbox_path})", file=sys.stderr, flush=True)
    try:
        uvicorn.run(create_app(svc), host="127.0.0.1", port=a.port, proxy_headers=False, server_header=False,
                    log_level="warning")
    finally:
        stop.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

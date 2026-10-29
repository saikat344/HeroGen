"""Sending sign-in email through one interface.

- `FileSender` (dev mode and tests): appends each message as one JSON line to a local file; nothing leaves the
  machine.
- `ResendSender`: Resend's HTTP API (`POST https://api.resend.com/emails`), standard library only. The API key is
  sent only in the Authorization header and never logged; provider errors are reduced to a status code.

A message holds no tracking pixels or links other than the sign-in link itself.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

RESEND_URL = "https://api.resend.com/emails"
TIMEOUT_S = 15.0


class EmailError(RuntimeError):
    """The provider did not accept a message. The text is safe to log."""


@dataclass(frozen=True)
class Message:
    to: str
    subject: str
    text: str


class EmailSender(Protocol):
    def send(self, message: Message) -> None: ...


class FileSender:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.lock = threading.Lock()

    def send(self, message: Message) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"to": message.to, "subject": message.subject, "text": message.text}) + "\n"
        with self.lock:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as f:
                f.write(line)

    def messages(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(x) for x in self.path.read_text(encoding="utf-8").splitlines() if x.strip()]


class ResendSender:
    def __init__(self, api_key: str, sender: str, url: str = RESEND_URL, opener=urllib.request.urlopen) -> None:
        if not api_key:
            raise ValueError("ResendSender needs an API key")
        self._key = api_key
        self.sender = sender
        self.url = url
        self.opener = opener

    def send(self, message: Message) -> None:
        body = json.dumps({"from": self.sender, "to": [message.to], "subject": message.subject,
                           "text": message.text}).encode()
        req = urllib.request.Request(self.url, data=body, method="POST", headers={
            "Authorization": f"Bearer {self._key}", "Content-Type": "application/json",
            "User-Agent": "herogen-service"})
        try:
            with self.opener(req, timeout=TIMEOUT_S) as resp:
                status = getattr(resp, "status", 200)
        except urllib.error.HTTPError as e:
            raise EmailError(f"email provider refused the message (HTTP {e.code})") from None
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            raise EmailError(f"email provider unreachable ({type(e).__name__})") from None
        if not 200 <= status < 300:
            raise EmailError(f"email provider refused the message (HTTP {status})")


def signin_message(to: str, link: str, minutes: int) -> Message:
    text = (
        "Hello,\n\n"
        "Use this link to sign in to HeroGen:\n\n"
        f"{link}\n\n"
        f"It works once and expires in {minutes} minutes. Open it in the browser where you want to be signed in.\n\n"
        "If you did not ask to sign in, you can ignore this email; nobody can sign in without the link.\n"
    )
    return Message(to=to, subject="Your HeroGen sign-in link", text=text)


def make_sender(provider: str, *, api_key: str = "", sender: str = "", outbox: Path | None = None) -> EmailSender:
    if provider == "file":
        return FileSender(outbox or Path("outbox.jsonl"))
    if provider == "resend":
        return ResendSender(api_key, sender)
    raise ValueError(f"unknown email provider {provider!r}")

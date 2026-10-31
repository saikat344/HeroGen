"""Service settings, read once from the environment (see docs/ARCHITECTURE.md, "Service").

Secrets come only from the environment, which on Modal is filled from Modal secrets:
`HEROGEN_SESSION_KEY` (signs session ids, CSRF tokens, sign-in links and invite codes), `HEROGEN_EMAIL_API_KEY`
(the email provider) and `MODEL_API_KEY` (the Meta Model API, read by herogen.pipeline in the stage workers only).
Every other setting is a plain `HEROGEN_SERVICE_*` variable with the default below.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

MODES = ("dev", "production")
EMAIL_PROVIDERS = ("file", "resend")
PROFILES = ("roblox-ingame", "roblox-avatar", "best")
KEY_MIN_BYTES = 32
_EMAIL_FROM = re.compile(r"^[^<>@\s]+(?: [^<>@\s]+)* <[^<>@\s]+@[^<>@\s]+\.[^<>@\s]+>$|^[^<>@\s]+@[^<>@\s]+\.[^<>@\s]+$")


class ConfigError(ValueError):
    """A setting is missing or invalid. The message names the variable, never its value."""


@dataclass(frozen=True)
class Settings:
    mode: str = "dev"
    base_url: str = "http://127.0.0.1:8800"
    data_dir: Path = Path("service-data/data")
    meta_path: Path = Path("service-data/meta.db")
    signing_key: bytes = field(default=b"", repr=False)
    # Email
    email_provider: str = "file"
    email_from: str = "HeroGen <signin@example.com>"
    email_api_key: str = field(default="", repr=False)
    outbox_path: Path = Path("service-data/outbox.jsonl")
    # Sign-in and sessions
    link_ttl_s: int = 15 * 60
    session_ttl_s: int = 30 * 24 * 3600
    session_idle_s: int = 7 * 24 * 3600
    terms_version: str = ""
    legal_dir: Path | None = None
    # Quotas and limits
    jobs_per_day: int = 5
    jobs_total: int = 50
    max_running_jobs: int = 4
    heldout_concurrency: int = 2      # held-out view runs at once, service-wide (they yield to the jobs' own work)
    max_attempts: int = 2
    user_storage_max_bytes: int = 2 * 1024 ** 3
    upload_max_bytes: int = 12 * 1024 ** 2
    upload_max_pixels: int = 40_000_000
    upload_max_side: int = 2048
    upload_min_side: int = 256
    prompt_max_chars: int = 300
    name_max_chars: int = 60
    # Rate limits: (count, window seconds)
    rate_auth_ip: tuple[int, int] = (20, 3600)
    rate_auth_email: tuple[int, int] = (5, 3600)
    rate_link_ip: tuple[int, int] = (60, 3600)
    rate_jobs_user: tuple[int, int] = (20, 3600)
    rate_api_ip: tuple[int, int] = (600, 60)
    # Retention (days); recorded in the privacy policy
    retention_days: int = 30
    audit_retention_days: int = 365
    # Requests
    trusted_ip_header: str = ""
    allowed_hosts: tuple[str, ...] = ()
    # Pipeline
    default_profile: str = "roblox-ingame"
    profiles: tuple[str, ...] = PROFILES
    stage3_args: str = "--mesh-source sam3d-objects --joints sam3d-body --remote live"
    stage_timeout_s: int = 3600
    blender_timeout_s: int = 3 * 3600
    code_commit: str = ""

    @property
    def secure_cookies(self) -> bool:
        return self.base_url.startswith("https://")

    @property
    def origin(self) -> str:
        m = re.match(r"^(https?://[^/]+)", self.base_url)
        return m.group(1) if m else self.base_url

    @property
    def host(self) -> str:
        return self.origin.split("://", 1)[1]

    def hosts(self) -> tuple[str, ...]:
        return self.allowed_hosts or (self.host.split(":")[0],)

    def public(self) -> dict[str, Any]:
        """The limits a signed-in user may see."""
        from herogen.service.uploads import prompt_limit

        return {"jobs_per_day": self.jobs_per_day, "jobs_total": self.jobs_total,
                "upload_max_bytes": self.upload_max_bytes, "upload_min_side": self.upload_min_side,
                "prompt_max_chars": prompt_limit(self.prompt_max_chars), "name_max_chars": self.name_max_chars,
                "retention_days": self.retention_days, "profiles": list(self.profiles),
                "default_profile": self.default_profile, "terms_version": self.terms_version}

    def with_(self, **kw: Any) -> Settings:
        return replace(self, **kw)


ENV = {
    "mode": "HEROGEN_SERVICE_MODE",
    "base_url": "HEROGEN_SERVICE_BASE_URL",
    "data_dir": "HEROGEN_SERVICE_DATA",
    "meta_path": "HEROGEN_SERVICE_META",
    "signing_key": "HEROGEN_SESSION_KEY",
    "email_provider": "HEROGEN_SERVICE_EMAIL_PROVIDER",
    "email_from": "HEROGEN_SERVICE_EMAIL_FROM",
    "email_api_key": "HEROGEN_EMAIL_API_KEY",
    "outbox_path": "HEROGEN_SERVICE_OUTBOX",
    "link_ttl_s": "HEROGEN_SERVICE_LINK_TTL",
    "session_ttl_s": "HEROGEN_SERVICE_SESSION_TTL",
    "session_idle_s": "HEROGEN_SERVICE_SESSION_IDLE",
    "terms_version": "HEROGEN_SERVICE_TERMS_VERSION",
    "legal_dir": "HEROGEN_SERVICE_LEGAL_DIR",
    "jobs_per_day": "HEROGEN_SERVICE_JOBS_PER_DAY",
    "jobs_total": "HEROGEN_SERVICE_JOBS_TOTAL",
    "max_running_jobs": "HEROGEN_SERVICE_MAX_RUNNING",
    "heldout_concurrency": "HEROGEN_SERVICE_HELDOUT_MAX",
    "max_attempts": "HEROGEN_SERVICE_MAX_ATTEMPTS",
    "user_storage_max_bytes": "HEROGEN_SERVICE_USER_STORAGE",
    "upload_max_bytes": "HEROGEN_SERVICE_UPLOAD_MAX_BYTES",
    "upload_max_pixels": "HEROGEN_SERVICE_UPLOAD_MAX_PIXELS",
    "upload_max_side": "HEROGEN_SERVICE_UPLOAD_MAX_SIDE",
    "upload_min_side": "HEROGEN_SERVICE_UPLOAD_MIN_SIDE",
    "prompt_max_chars": "HEROGEN_SERVICE_PROMPT_MAX",
    "name_max_chars": "HEROGEN_SERVICE_NAME_MAX",
    "rate_auth_ip": "HEROGEN_SERVICE_RATE_AUTH_IP",
    "rate_auth_email": "HEROGEN_SERVICE_RATE_AUTH_EMAIL",
    "rate_link_ip": "HEROGEN_SERVICE_RATE_LINK_IP",
    "rate_jobs_user": "HEROGEN_SERVICE_RATE_JOBS_USER",
    "rate_api_ip": "HEROGEN_SERVICE_RATE_API_IP",
    "retention_days": "HEROGEN_SERVICE_RETENTION_DAYS",
    "audit_retention_days": "HEROGEN_SERVICE_AUDIT_RETENTION_DAYS",
    "trusted_ip_header": "HEROGEN_SERVICE_IP_HEADER",
    "allowed_hosts": "HEROGEN_SERVICE_HOSTS",
    "default_profile": "HEROGEN_SERVICE_PROFILE",
    "profiles": "HEROGEN_SERVICE_PROFILES",
    "stage3_args": "HEROGEN_SERVICE_STAGE3_ARGS",
    "stage_timeout_s": "HEROGEN_SERVICE_STAGE_TIMEOUT",
    "blender_timeout_s": "HEROGEN_SERVICE_BLENDER_TIMEOUT",
    "code_commit": "HEROGEN_SERVICE_COMMIT",
}
SECRET_FIELDS = ("signing_key", "email_api_key")


def _parse(name: str, raw: str, default: Any) -> Any:
    var = ENV[name]
    try:
        if name == "signing_key":
            return raw.encode()
        if isinstance(default, bool):
            return raw.strip().lower() in ("1", "true", "yes")
        if isinstance(default, int):
            return int(raw)
        if isinstance(default, tuple) and default and isinstance(default[0], int):
            a, b = raw.split("/", 1)
            return (int(a), int(b))
        if isinstance(default, tuple) or name == "allowed_hosts":
            return tuple(x.strip() for x in raw.split(",") if x.strip())
        if name in ("data_dir", "meta_path", "outbox_path", "legal_dir"):
            return Path(raw)
        return raw.strip()
    except ValueError:
        raise ConfigError(f"{var} is not valid") from None


def from_env(env: Mapping[str, str] | None = None, require_key: bool = True, **overrides: Any) -> Settings:
    """Settings from `env` (default os.environ), then `overrides`, then validated. Workers, which never sign
    anything, pass `require_key=False`."""
    env = os.environ if env is None else env
    defaults = Settings()
    values: dict[str, Any] = {}
    for f in fields(Settings):
        raw = env.get(ENV[f.name])
        if raw is not None and raw != "":
            values[f.name] = _parse(f.name, raw, getattr(defaults, f.name))
    values.update(overrides)
    s = Settings(**values)
    validate(s, require_key)
    return s


def validate(s: Settings, require_key: bool = True) -> None:
    if s.mode not in MODES:
        raise ConfigError(f"{ENV['mode']} must be one of {', '.join(MODES)}")
    if require_key and len(s.signing_key) < KEY_MIN_BYTES:
        raise ConfigError(f"{ENV['signing_key']} must be set to at least {KEY_MIN_BYTES} random bytes "
                          "(for example `python3 -c \"import secrets; print(secrets.token_urlsafe(48))\"`)")
    if not re.match(r"^https?://[A-Za-z0-9.-]+(:\d+)?/?$", s.base_url):
        raise ConfigError(f"{ENV['base_url']} must be an origin such as https://hero.example.com")
    if s.email_provider not in EMAIL_PROVIDERS:
        raise ConfigError(f"{ENV['email_provider']} must be one of {', '.join(EMAIL_PROVIDERS)}")
    if not _EMAIL_FROM.match(s.email_from):
        raise ConfigError(f"{ENV['email_from']} must look like `HeroGen <signin@your-domain>`")
    if s.default_profile not in s.profiles or any(p not in PROFILES for p in s.profiles):
        raise ConfigError(f"{ENV['profiles']} and {ENV['default_profile']} must name profiles in {PROFILES}")
    for name in ("link_ttl_s", "session_ttl_s", "session_idle_s", "jobs_per_day", "jobs_total", "max_running_jobs",
                 "max_attempts", "heldout_concurrency", "upload_max_bytes", "upload_max_pixels", "upload_max_side",
                 "upload_min_side",
                 "prompt_max_chars", "retention_days", "audit_retention_days", "stage_timeout_s",
                 "blender_timeout_s"):
        if getattr(s, name) <= 0:
            raise ConfigError(f"{ENV[name]} must be positive")
    if s.upload_min_side > s.upload_max_side:
        raise ConfigError(f"{ENV['upload_min_side']} must not exceed {ENV['upload_max_side']}")
    if s.mode == "production":
        if not s.base_url.startswith("https://"):
            raise ConfigError(f"{ENV['base_url']} must use https in production")
        if s.email_provider == "file":
            raise ConfigError(f"{ENV['email_provider']} must name a real provider in production")
        if require_key and not s.email_api_key:
            raise ConfigError(f"{ENV['email_api_key']} must be set in production")
        if "example.com" in s.email_from:
            raise ConfigError(f"{ENV['email_from']} must use your own sending domain in production")


def env_for_deploy(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Non-secret service settings and pipeline prompt selectors to bake into the Modal image at deploy time."""
    env = os.environ if env is None else env
    secret_vars = {ENV[f] for f in SECRET_FIELDS}
    selectors = {"HEROGEN_PROMPT_SKETCH_VERSION", "HEROGEN_SAME_CHARACTER_VERSION"}
    return {k: v for k, v in env.items() if (k.startswith("HEROGEN_SERVICE_") or k in selectors)
            and k not in secret_vars}

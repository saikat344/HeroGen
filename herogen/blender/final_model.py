"""Select the best-quality display/download mesh without changing the R15 export's body."""
import json
from pathlib import Path


def run_profile(run: Path) -> str:
    for stage in ("4", "3f"):
        try:
            record = json.loads((run / f"stage-{stage}.json").read_text())
            profile = record.get("profile")
            name = profile.get("name") if isinstance(profile, dict) else profile
            if name:
                return name
        except (OSError, ValueError, AttributeError):
            continue
    return "best"


def final_model(run: Path, profile: str | None = None) -> Path | None:
    """Best: refined, then SAM source, then sectioned high. Roblox profiles always use sectioned high."""
    run = Path(run)
    names = ["04-body/high.glb"]
    if (profile or run_profile(run)) == "best":
        names = ["03-refined.glb", "03-source.glb", *names]
    for name in names:
        path = run / name
        try:
            if path.is_file() and not path.is_symlink() and path.resolve() == run.resolve() / name:
                return path
        except OSError:
            continue
    return None

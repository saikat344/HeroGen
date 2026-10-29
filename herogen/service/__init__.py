"""The hosted HeroGen service: invite-only sign-in by emailed link, per-user quotas, sketch or description in,
pipeline run on Modal, GLB and Roblox bundle out. See docs/ARCHITECTURE.md, "Service".

Only `config`, `auth`, `store`, `snapshot`, `storage`, `runner`, `worker`, `uploads`, `bundle`, `labapi`,
`emailer`, `admin` and `fakes` are imported outside Modal; they need the standard library (Pillow and numpy for
uploads and fakes). `web` needs Starlette and python-multipart (`requirements.txt`); `modal_app` needs Modal.
"""

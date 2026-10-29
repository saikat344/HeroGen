"""Remote GPU runners for herogen/blender (SAM 3D Objects, SAM 3D Body on Modal) and their record/replay client.

`objects.py` and `body.py` are self-contained Modal apps; only `client.py` is imported by the stages, and in
replay mode (the default, and the only mode tests use) it never starts Modal.
"""

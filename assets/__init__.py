"""Level 1 asset pipeline: real-CT segmentations to meshes attached to AWR entities.

``assets.labels`` is standard library only, so the core and the viewer server can read the
label map. Mesh building (``assets.meshes``) needs the ``assets`` extra (ADR-L1-003).
"""

"""Level 1 viewer: a local server that runs the CommandEngine, and a browser page that draws.

The engine is the only writer of scene state. The page reads ``/state`` and applies it to the
meshes in ``scene.glb``; every change it asks for goes through ``/command`` as text, exactly as
a person would type it. Standard library only on the Python side.
"""

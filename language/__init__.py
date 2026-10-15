"""Level 1 language layer: free text to a validated AWR program, never to geometry (ADR 0007).

``language.program`` (standard library only) defines the closed program vocabulary, its JSON
schema over the ontology's entity ids, validation, and the translation of a program into the
canonical commands the deterministic engine already understands. ``language.claude`` asks the
Claude API for a program under that schema. ``language.layer`` decides which path a request
takes: the free rule parser first, Claude only when it is enabled and the parser cannot help.
"""

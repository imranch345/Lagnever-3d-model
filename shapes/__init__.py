"""Level 2a: statistical shape models of thorax entities, built from Level 1 subjects.

``shapes.align`` puts every subject in the reference subject's frame (rigid, no scaling).
``shapes.correspond`` deforms one template mesh per entity onto each subject, so that vertex
``i`` means the same place everywhere. ``shapes.ssm`` fits a principal-component model over
those corresponded shapes and samples from it. ``shapes.build`` runs the whole thing with a
held-out evaluation. Needs the ``assets`` extra plus ``rtree``.
"""

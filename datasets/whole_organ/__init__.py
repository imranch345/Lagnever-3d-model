"""Step 7 whole-organ synthetic generator.

The organ is built as a whole from global parameters and then segmented, so entity
relationships are measured from the finished shape rather than asserted from the
ontology. This is the reverse of the Step 6 generator, which built entities separately
and assembled them, and it is what makes relationship graphs vary between scenes.

Tier two: this package evaluates occupancy fields over point arrays and depends on
NumPy. See ADR 0010.
"""

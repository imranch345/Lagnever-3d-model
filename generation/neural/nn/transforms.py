"""Step 10: similarity transforms, composition, and the parent-relative convention.

Every convention here is stated explicitly, because the Step 10 target is defined by
composition and an implicit convention would make the composition silently wrong.

The transform
-------------

A frame is the existing 12-number contract, unchanged::

    frame[0:3]    translation t, in the parent's coordinates
    frame[3:6]    log-scale s, per axis
    frame[6:12]   6D rotation, two vectors from which R is built by Gram-Schmidt

It denotes the **similarity transform that maps child-local coordinates into the
parent's coordinates**::

    p_parent = R @ (exp(s) * p_child) + t

which is the forward of the existing ``points_to_local``, so the decoder's meaning of a
frame is unchanged. Written as a map, ``T_parent_child``.

Conventions, stated once
------------------------

=========================  ==================================================
handedness                 right-handed
axes                       x lateral, y superior, z anterior, as the corpus builds them
rotation matrix rows       the basis vectors b1, b2, b3 = b1 x b2
rotation representation    6D continuous (Zhou et al.), the existing contract
scale                      per-axis, stored as a natural logarithm
composition order          ``T_a_c = T_a_b @ T_b_c``, parent on the left
direction                  a frame maps **child into parent**, never parent into child
units                      scene units; the organ occupies roughly [-1, 1] on each axis
root convention            the root's frame is expressed in scene coordinates
=========================  ==================================================

Why 6D rather than a quaternion
-------------------------------

It is what the decoder, the metrics and every checkpoint already use, so keeping it means
Step 10 changes the target and not the representation at the same time. It is continuous,
which quaternions and Euler angles are not, and composition and inversion are ordinary
matrix operations. The Step 10 brief asks for a representation supporting composition,
inversion and an unambiguous error, and this one does without introducing a second
convention to keep in step with the first.

Closure: why the parent contributes only an isotropic scale
-----------------------------------------------------------

Twelve-number frames are **not closed under composition** when a rotated parent carries an
anisotropic scale. ``R_p S_p R_c S_c`` has shear, no ``R diag(s)`` represents it, and the
projection back to a frame silently discards the difference. That matters here because the
Step 10 target is *defined* by composition: the error would sit under every arm as a
target-definition floor that no model could beat, and would be indistinguishable from model
error.

It is invisible on the Step 8/9 corpus only because every rotation there is exactly the
identity, which makes the linear parts diagonal and diagonal matrices commute. Step 10
Change 2 replaces that constant with a real rotation, so the defect would appear exactly
when the rotation target became meaningful. Measured on ``test_seen`` with the corpus's own
anisotropic scales and injected rotations, the round trip ``compose(p, relative(p, c))``
against ``c`` costs:

===============  ==============  ==================
parent rotation  position error  rotation error
===============  ==============  ==================
0 degrees        3e-17           0.000 degrees
30 degrees       7.3e-03         0.97 degrees
90 degrees       1.4e-02         5.24 degrees
===============  ==============  ==================

against a placement-blind floor of 0.1605, so up to 9% of the floor and several degrees,
purely from the choice of target.

The fix is to let the parent contribute rotation and an **isotropic** scale. Then
``R_p s_p R_c S_c = (R_p R_c)(s_p S_c)`` is again a rotation times a diagonal, composition
is exact to machine precision at any rotation, and the child keeps its full anisotropic
scale. :data:`PARENT_CONVENTIONS` lists the three settings and
:func:`as_parent` applies them; ``isotropic`` is the default because ``rigid`` also closes
exactly but stops a parent's size from propagating to its children at all.
"""

from __future__ import annotations

import torch

from generation.neural.nn.geometry import frame_rotation

__all__ = [
    "IDENTITY_FRAME",
    "PARENT_CONVENTIONS",
    "DEFAULT_PARENT_CONVENTION",
    "identity_frame",
    "as_parent",
    "similarity_residual",
    "frame_to_matrix",
    "matrix_to_frame",
    "compose",
    "invert",
    "relative",
    "compose_hierarchy",
    "resolve_parents",
    "rotation_angle",
    "rotation_chordal",
]


def identity_frame(
    *shape: int, device: torch.device | None = None, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """The frame that maps a child onto its parent unchanged."""
    frame = torch.zeros(*shape, 12, device=device, dtype=dtype)
    frame[..., 6] = 1.0
    frame[..., 10] = 1.0
    return frame


#: The identity as a plain 12-vector, for tests and defaults.
IDENTITY_FRAME: tuple[float, ...] = (
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
)


def frame_to_matrix(frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a frame into its linear part ``R @ diag(exp(s))`` and its translation.

    Returning the linear part already scaled is what makes composition associative for
    anisotropic scales: the product of two such matrices is exactly the composed map, and
    nothing is lost by pretending the result factors back into a rotation and a diagonal.
    """
    rotation = frame_rotation(frame)
    scale = frame[..., 3:6].exp().clamp_min(1e-6)
    # ``frame_rotation`` returns R with its **rows** as the basis vectors, and the decoder
    # computes local = (p - t) @ R / s. Inverting that gives p = t + R @ diag(s) @ local,
    # so the linear part scales R's columns. Transposing here would invert the rotation
    # and every composition below would be silently wrong.
    linear = rotation * scale.unsqueeze(-2)
    return linear, frame[..., 0:3]


def matrix_to_frame(linear: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Recover a frame from a linear part and a translation.

    ``linear`` is ``R @ diag(s)``, so its column *j* is ``R[:, j] * s[j]``: the scale is the
    per-column norm and the rotation is the column-normalised matrix. That is exact, cheap
    and differentiable everywhere, where a QR would be exact but carries an unstable
    gradient through a function that sits in the loss path.

    It assumes the input really is a rotation times a diagonal, which every composition
    under :data:`DEFAULT_PARENT_CONVENTION` guarantees. :func:`similarity_residual` measures
    the assumption rather than trusting it; the ``full`` convention is the one that breaks
    it, and the module docstring says by how much.
    """
    scale = linear.norm(dim=-2).clamp_min(1e-6)
    orthonormal = linear / scale.unsqueeze(-2)
    # Rows are the basis vectors the frame stores, and only the first two are kept: the
    # third is recovered as b1 x b2 by ``frame_rotation``.
    return torch.cat(
        [translation, scale.log(), orthonormal[..., 0, :], orthonormal[..., 1, :]], dim=-1
    )


#: How much of the parent's transform propagates to its children.
#:
#: ``isotropic`` and ``rigid`` keep composition exactly closed; ``full`` does not, and is
#: kept only so the cost of the naive choice can be measured rather than asserted.
PARENT_CONVENTIONS: tuple[str, ...] = ("isotropic", "rigid", "full")

#: Rotation and mean scale: closed under composition, and a parent's size still reaches its
#: children.
DEFAULT_PARENT_CONVENTION = "isotropic"


def as_parent(frame: torch.Tensor, convention: str = DEFAULT_PARENT_CONVENTION) -> torch.Tensor:
    """The part of a parent's frame that its children are placed relative to."""
    if convention == "full":
        return frame
    if convention == "isotropic":
        # The geometric mean of the per-axis scales, which is the mean of the logs.
        mean_log_scale = frame[..., 3:6].mean(dim=-1, keepdim=True)
        return torch.cat(
            [frame[..., 0:3], mean_log_scale.expand_as(frame[..., 3:6]), frame[..., 6:12]],
            dim=-1,
        )
    if convention == "rigid":
        return torch.cat(
            [frame[..., 0:3], torch.zeros_like(frame[..., 3:6]), frame[..., 6:12]], dim=-1
        )
    raise ValueError(
        f"Unknown parent convention {convention!r}; expected one of {PARENT_CONVENTIONS}."
    )


def similarity_residual(linear: torch.Tensor) -> torch.Tensor:
    """How far a linear part is from being a rotation times a diagonal.

    Zero exactly when the columns are orthogonal, so this is the shear that
    :func:`matrix_to_frame` would discard. Reported in the Step 10 integrity checks so the
    closure property is verified on the data rather than argued from the algebra.
    """
    gram = linear.transpose(-1, -2) @ linear
    off = gram - torch.diag_embed(torch.diagonal(gram, dim1=-2, dim2=-1))
    return off.abs().amax(dim=(-1, -2))


def compose(
    parent: torch.Tensor,
    child: torch.Tensor,
    convention: str = DEFAULT_PARENT_CONVENTION,
) -> torch.Tensor:
    """``T_a_c = T_a_b @ T_b_c``: the child's frame expressed in the grandparent's space.

    Args:
        parent: ``[..., 12]`` mapping b into a.
        child: ``[..., 12]`` mapping c into b.
        convention: how much of the parent propagates; see :data:`PARENT_CONVENTIONS`.

    Returns a frame mapping c into a. Exact under the closed conventions, so
    ``compose(p, relative(p, c)) == c`` to machine precision. Pass the *same* convention to
    both or the round trip does not hold.

    Not a group operation: the *parent* is reduced by :func:`as_parent` and the child is
    not, so the identity is neutral on the left but not on the right
    (``compose(f, identity)`` returns ``as_parent(f)``, which differs from ``f`` whenever
    ``f`` has an anisotropic scale). That asymmetry is the point — it is what keeps the
    composition closed — and it is consistent between :func:`relative` and
    :func:`compose_hierarchy`, which is what the exactness above depends on.

    """
    parent_linear, parent_translation = frame_to_matrix(as_parent(parent, convention))
    child_linear, child_translation = frame_to_matrix(child)
    linear = parent_linear @ child_linear
    translation = (
        torch.einsum("...ij,...j->...i", parent_linear, child_translation) + parent_translation
    )
    return matrix_to_frame(linear, translation)


def invert(frame: torch.Tensor) -> torch.Tensor:
    """The frame mapping the parent back into the child.

    Note that the inverse of a rotation times an anisotropic diagonal is a diagonal times a
    rotation, which is not itself of that form, so this is exact only for an isotropic or
    unit scale. It is applied to parents, which under the closed conventions are exactly
    that; :func:`relative` is where that matters.
    """
    linear, translation = frame_to_matrix(frame)
    inverse = torch.linalg.inv(linear)
    return matrix_to_frame(inverse, -torch.einsum("...ij,...j->...i", inverse, translation))


def relative(
    parent: torch.Tensor,
    child: torch.Tensor,
    convention: str = DEFAULT_PARENT_CONVENTION,
) -> torch.Tensor:
    """``T_b_c`` from two frames given in the same space: ``inverse(T_a_b) @ T_a_c``.

    This is how a parent-relative **target** is built from the corpus's global frames, and
    it is the exact inverse of :func:`compose` under the closed conventions: the child keeps
    its own anisotropic scale, and only the parent is reduced.
    """
    return compose(invert(as_parent(parent, convention)), child, convention="full")


def resolve_parents(
    parent_index: torch.Tensor, present: torch.Tensor, order: torch.Tensor
) -> torch.Tensor:
    """Re-parent each entity to its nearest **present** ancestor, per scene.

    Args:
        parent_index: ``[N]`` static parent slot per entity, -1 for a root.
        present: ``[B, N]`` presence mask, the same one the model is scored under.
        order: ``[N]`` parents before children.

    Returns ``[B, N]`` parent slots, -1 where no ancestor is present.

    Levels of detail expose entities in groups, and a coarse level can expose a child while
    hiding its parent: on this corpus the aorta is present in scenes where the aortic valve
    is not, for 8% of non-root instances. Composing against an absent parent would use a
    frame that is not in the scene at all. Walking up to the nearest present ancestor keeps
    the chain inside the scene, and uses only the presence mask, which is an input the model
    already sees, so it leaks nothing about placement.

    """
    batch = present.shape[0]
    resolved = parent_index.unsqueeze(0).expand(batch, -1).clone()
    # Only parented slots do any work, and on this corpus that is ten of sixty-four. Roots
    # and padding are left as they are, which is what the table already says.
    for slot in order.tolist():
        parent = int(parent_index[slot])
        if parent < 0:
            continue
        # ``parent`` precedes ``slot`` in ``order``, so its own resolution is already final
        # and one step of substitution is enough to skip a whole chain of absent ancestors.
        missing = ~present[:, parent]
        resolved[missing, slot] = resolved[missing, parent]
    return torch.where(present, resolved, torch.full_like(resolved, -1))


def compose_hierarchy(
    local: torch.Tensor,
    parent_index: torch.Tensor,
    order: torch.Tensor,
    convention: str = DEFAULT_PARENT_CONVENTION,
) -> torch.Tensor:
    """Compose local frames into global ones by walking the hierarchy.

    Args:
        local: ``[B, N, 12]`` each entity's frame in its parent's coordinates.
        parent_index: ``[N]`` static parent slots, or ``[B, N]`` from
            :func:`resolve_parents` when levels of detail hide some parents.
        order: ``[N]`` slot order with every parent before its children, from
            :func:`datasets.whole_organ.hierarchy.topological_order`.
        convention: see :data:`PARENT_CONVENTIONS`.

    Returns ``[B, N, 12]`` global frames. Roots pass through unchanged, which is the
    convention: a root's frame is already in scene coordinates.

    The walk composes each child against its parent's **already global** frame, so a depth-3
    chain is three exact compositions rather than one approximation of three.

    """
    frames = local.clone()
    per_scene = parent_index.dim() == 2
    static = parent_index.amax(dim=0) if per_scene else parent_index
    for slot in order.tolist():
        if int(static[slot]) < 0:
            # No scene gives this slot a parent, so composition is the identity here.
            continue
        if per_scene:
            column = parent_index[:, slot]
            has = column >= 0
            if not bool(has.any()):
                continue
            # ``gather`` needs a valid index everywhere, so clamp and mask afterwards.
            safe = column.clamp_min(0)
            composed = compose(
                frames[torch.arange(frames.shape[0], device=frames.device), safe],
                local[:, slot],
                convention,
            )
            frames[:, slot] = torch.where(has.unsqueeze(-1), composed, local[:, slot])
        else:
            parent = int(parent_index[slot])
            if parent < 0:
                continue
            frames[:, slot] = compose(frames[:, parent], local[:, slot], convention)
    return frames


def rotation_angle(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Geodesic angle in radians between the rotations of two frames.

    The only unambiguous distance on rotations, and the one every Step 10 rotation metric
    is built from.
    """
    a = frame_rotation(first)
    b = frame_rotation(second)
    relative_rotation = torch.matmul(a, b.transpose(-1, -2))
    trace = relative_rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.arccos(cosine)


def rotation_chordal(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Squared Frobenius distance between two rotations, normalised to ``[0, 1]``.

    A monotone function of :func:`rotation_angle` (it is ``(1 - cos t) / 2`` averaged over
    the three axes) and therefore an equivalent ordering, but smooth and bounded where
    ``arccos`` has an unbounded derivative at zero. That matters for Change 2: the geodesic
    angle is the right thing to *report* and the wrong thing to *descend*, because a model
    that is nearly correct gets an arbitrarily large gradient.

    The same singularity is why :func:`rotation_angle` returns about 3e-8 rather than 0 for
    a rotation against itself in float64: ``arccos`` amplifies the square root of the
    floating-point epsilon. Harmless in a metric, which is all it is used for.
    """
    difference = frame_rotation(first) - frame_rotation(second)
    return difference.pow(2).sum(dim=(-1, -2)) / 8.0

"""Tensor and data contracts for the proposed Lagnav neural architecture.

STATUS: PROPOSED. Nothing here is trained, and no deep-learning framework is
imported. This module fixes *what every representation means* before any model
exists, because the expensive mistakes in a project like this are shape and
semantics mistakes, not layer choices.

Three ideas carry the module:

* A **named dimension** (``B``, ``N_ENT``, ``K_GEO``, ...) has one meaning across
  the whole architecture. A tensor never says ``[B, N, D]`` without those names
  resolving to documented quantities.
* A :class:`TensorSpec` records semantics, dtype, normalisation, padding value
  and which mask governs it. A spec is a contract, not a comment.
* A :class:`Contract` groups the specs that travel together (one AWR batch, one
  latent bundle) and can validate a concrete payload.

Arrays are represented by :class:`PlainArray`, a flat tuple plus a shape. That is
deliberately framework-free: Step 6 will map these contracts onto real tensors in
whichever framework is chosen, and the choice is not being made here.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from awr.errors import ContractError

__all__ = [
    "DType",
    "Dim",
    "DIMENSIONS",
    "dim",
    "PlainArray",
    "TensorSpec",
    "Contract",
    "ContractRegistry",
    "CONTRACTS",
]


class DType(StrEnum):
    """Element type of a declared tensor."""

    FLOAT32 = "float32"
    FLOAT16 = "float16"
    INT32 = "int32"
    INT64 = "int64"
    BOOL = "bool"

    @property
    def is_floating(self) -> bool:
        """Whether the dtype holds real numbers."""
        return self in (DType.FLOAT32, DType.FLOAT16)

    @property
    def is_integral(self) -> bool:
        """Whether the dtype holds integers."""
        return self in (DType.INT32, DType.INT64)


@dataclass(frozen=True, slots=True)
class Dim:
    """One named dimension of the architecture.

    ``size`` is the value proposed for the first heart prototype. ``None`` means
    the dimension is genuinely dynamic (batch size, number of query points) and
    is bound per call. ``padded`` records that the axis is padded to a maximum and
    therefore requires a mask.
    """

    name: str
    description: str
    size: int | None = None
    padded: bool = False
    mask: str | None = None

    def __post_init__(self) -> None:
        if self.padded and self.mask is None:
            raise ContractError(
                f"Dimension {self.name!r} is padded and must name the mask that governs it."
            )
        if self.size is not None and self.size <= 0:
            raise ContractError(f"Dimension {self.name!r} has a non-positive size {self.size}.")


DIMENSIONS: tuple[Dim, ...] = (
    Dim("B", "Batch size: number of independent anatomical scenes in the batch."),
    Dim(
        "N_ENT",
        "Entity slots per scene. Heart Ontology v0.1 uses 42; padded to this maximum so "
        "later ontologies fit without a contract change.",
        size=64,
        padded=True,
        mask="entity_mask",
    ),
    Dim(
        "E_REL",
        "Relationship edge slots per scene, across all three graphs. Heart v0.1 declares "
        "130 edges including hierarchy-derived partonomy.",
        size=256,
        padded=True,
        mask="edge_mask",
    ),
    Dim("N_GRAPH", "Number of relationship graphs: structure, spatial, functional.", size=3),
    Dim(
        "K_GEO",
        "Geometry tokens per entity. Token order is coarse to fine, so a level of detail "
        "selects a prefix (see generation/neural/three_d_latent.py).",
        size=32,
    ),
    Dim("D_SCENE", "Width of the scene (global) latent.", size=256),
    Dim("D_ENT", "Width of an entity latent.", size=256),
    Dim("D_REL", "Width of a relation embedding.", size=64),
    Dim("D_GEO", "Width of a geometry token.", size=64),
    Dim("D_TEXT", "Width of the language encoder's hidden states.", size=384),
    Dim(
        "T_TOK",
        "Text token slots per utterance.",
        size=128,
        padded=True,
        mask="text_mask",
    ),
    Dim("N_STATE", "Channels of the explicit entity state vector; layout is documented.", size=9),
    Dim(
        "N_FRAME",
        "Channels of an entity pose: 3 translation, 3 log-scale, 3+3 rotation basis.",
        size=12,
    ),
    Dim("P_QUERY", "Query points sampled per scene for field supervision."),
    Dim("N_PART", "Part-label classes: N_ENT entities plus one background class.", size=65),
    Dim("N_VIEW", "Rendered views per scene. Only used by the optional image stage.", size=8),
    Dim("N_PHASE", "Cardiac-cycle phases declared in configs/heart.yaml.", size=5),
    Dim("N_HEAD", "Attention heads in the anatomical graph encoder.", size=8),
    Dim("3D", "Cartesian coordinate axes of scene space.", size=3),
    Dim("D_ALIGN", "Width of the shared multimodal alignment space.", size=256),
    Dim(
        "N_CODEBOOK",
        "Entities in the ontology codebook. Heart Ontology v0.1 declares 42.",
        size=42,
    ),
    Dim(
        "L_OP",
        "Operation slots in one AWR program. A single utterance maps to a short program.",
        size=16,
        padded=True,
        mask="program_mask",
    ),
    Dim(
        "A_ENT",
        "Entity arguments per operation. Group references are kept as the group entity, so "
        "programs stay short: 'show the four chambers' is one argument, not four.",
        size=4,
    ),
    Dim(
        "A_SCALAR",
        "Scalar arguments per operation, such as an opacity or a level of detail.",
        size=4,
    ),
)
"""Every named dimension used by the architecture. One meaning each, repository-wide."""

_DIM_INDEX: Mapping[str, Dim] = {d.name: d for d in DIMENSIONS}


def dim(name: str) -> Dim:
    """Return a named dimension, or raise listing the declared names."""
    try:
        return _DIM_INDEX[name]
    except KeyError:
        raise ContractError(
            f"Unknown dimension {name!r}. Declared dimensions: {', '.join(sorted(_DIM_INDEX))}."
        ) from None


def default_bindings() -> dict[str, int]:
    """Prototype sizes for every dimension that has one."""
    return {d.name: d.size for d in DIMENSIONS if d.size is not None}


@dataclass(frozen=True, slots=True)
class PlainArray:
    """A framework-free array: flat row-major values plus a shape.

    Used so contracts can be validated, and a whole batch assembled from an AWR
    scene, without committing to a tensor library. Step 6 replaces this with real
    tensors; the specs it satisfies do not change.
    """

    shape: tuple[int, ...]
    dtype: DType
    data: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        expected = 1
        for extent in self.shape:
            if extent < 0:
                raise ContractError(f"Negative extent in shape {self.shape}.")
            expected *= extent
        if len(self.data) != expected:
            raise ContractError(
                f"Array with shape {self.shape} needs {expected} values, got {len(self.data)}."
            )

    @property
    def rank(self) -> int:
        """Number of axes."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """Total number of elements."""
        return len(self.data)

    @classmethod
    def full(cls, shape: Sequence[int], value: Any, dtype: DType) -> PlainArray:
        """Build an array filled with one value."""
        total = 1
        for extent in shape:
            total *= extent
        return cls(tuple(shape), dtype, tuple([value] * total))

    @classmethod
    def zeros(cls, shape: Sequence[int], dtype: DType = DType.FLOAT32) -> PlainArray:
        """Build a zero-filled array."""
        zero: Any = False if dtype is DType.BOOL else (0.0 if dtype.is_floating else 0)
        return cls.full(shape, zero, dtype)

    @classmethod
    def from_nested(cls, values: Any, dtype: DType) -> PlainArray:
        """Build an array from nested lists, checking that it is rectangular."""
        shape: list[int] = []
        probe = values
        while isinstance(probe, list | tuple):
            shape.append(len(probe))
            probe = probe[0] if probe else None
            if probe is None:
                break
        flat: list[Any] = []

        def _walk(node: Any, depth: int) -> None:
            if depth == len(shape):
                flat.append(node)
                return
            if not isinstance(node, list | tuple):
                raise ContractError(f"Ragged nested input at depth {depth}: {node!r}.")
            if len(node) != shape[depth]:
                raise ContractError(
                    f"Ragged nested input at depth {depth}: expected {shape[depth]} items, "
                    f"got {len(node)}."
                )
            for child in node:
                _walk(child, depth + 1)

        _walk(values, 0)
        return cls(tuple(shape), dtype, tuple(flat))

    def at(self, *indices: int) -> Any:
        """Read one element by full index."""
        if len(indices) != self.rank:
            raise ContractError(f"Expected {self.rank} indices, got {len(indices)}.")
        offset = 0
        for axis, index in enumerate(indices):
            extent = self.shape[axis]
            if not -extent <= index < extent:
                raise ContractError(f"Index {index} out of range for axis {axis} of {extent}.")
            offset = offset * extent + (index % extent)
        return self.data[offset]

    def row(self, *indices: int) -> tuple[Any, ...]:
        """Read the trailing axis at a given leading index."""
        if len(indices) != self.rank - 1:
            raise ContractError(f"Expected {self.rank - 1} leading indices, got {len(indices)}.")
        stride = self.shape[-1]
        offset = 0
        for axis, index in enumerate(indices):
            offset = offset * self.shape[axis] + index
        start = offset * stride
        return self.data[start : start + stride]


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """The declared meaning and layout of one tensor.

    Every field exists because leaving it implicit has a cost: ``normalization``
    prevents two stages disagreeing about units, ``mask`` prevents padded slots
    being read as data, and ``padding_value`` prevents a padded index being
    mistaken for entity 0.
    """

    name: str
    dims: tuple[str, ...]
    dtype: DType
    semantics: str
    normalization: str = "none"
    mask: str | None = None
    padding_value: Any = None
    value_range: tuple[float, float] | None = None
    channels: tuple[str, ...] = ()
    notes: str | None = None

    def __post_init__(self) -> None:
        for name in self.dims:
            dim(name)  # raises on an undeclared dimension
        if self.channels and dim(self.dims[-1]).size not in (None, len(self.channels)):
            raise ContractError(
                f"{self.name}: {len(self.channels)} channel labels do not match the size of "
                f"dimension {self.dims[-1]!r}."
            )

    @property
    def rank(self) -> int:
        """Number of axes."""
        return len(self.dims)

    def resolve(self, bindings: Mapping[str, int] | None = None) -> tuple[int, ...]:
        """Concrete shape under a set of dimension bindings."""
        merged = {**default_bindings(), **(bindings or {})}
        shape: list[int] = []
        for name in self.dims:
            if name not in merged:
                raise ContractError(
                    f"{self.name}: dimension {name!r} is dynamic and must be bound explicitly."
                )
            shape.append(merged[name])
        return tuple(shape)

    def validate(
        self, array: PlainArray, bindings: Mapping[str, int] | None = None
    ) -> dict[str, int]:
        """Check an array against this spec and return the bindings it implies.

        Dimensions already present in ``bindings`` must match; dimensions absent
        from it are inferred from the array, which is how one batch's ``B`` and
        ``P_QUERY`` get pinned consistently across every tensor in a contract.
        """
        if array.dtype is not self.dtype:
            raise ContractError(
                f"{self.name}: expected dtype {self.dtype}, got {array.dtype}."
            )
        if array.rank != self.rank:
            raise ContractError(
                f"{self.name}: expected rank {self.rank} for dims {self.dims}, "
                f"got shape {array.shape}."
            )
        resolved = dict(bindings or {})
        for name, extent in zip(self.dims, array.shape, strict=True):
            declared = dim(name)
            if name in resolved and resolved[name] != extent:
                raise ContractError(
                    f"{self.name}: dimension {name!r} is {extent} here but {resolved[name]} "
                    "elsewhere in the same batch."
                )
            if declared.size is not None and extent != declared.size and name not in resolved:
                raise ContractError(
                    f"{self.name}: dimension {name!r} is declared as {declared.size} but the "
                    f"array has {extent}. Bind it explicitly if this batch overrides it."
                )
            resolved[name] = extent
        if self.value_range is not None and self.dtype.is_floating:
            low, high = self.value_range
            for value in array.data:
                if not low <= float(value) <= high:
                    raise ContractError(
                        f"{self.name}: value {value!r} is outside the declared range "
                        f"[{low}, {high}]."
                    )
        return resolved

    def describe(self) -> str:
        """One-line human-readable description."""
        shape = "[" + ", ".join(self.dims) + "]"
        return f"{self.name} {shape} {self.dtype}: {self.semantics}"


@dataclass(frozen=True, slots=True)
class Contract:
    """A named group of tensors that travel together."""

    name: str
    purpose: str
    specs: tuple[TensorSpec, ...]
    status: str = "PROPOSED"

    def __post_init__(self) -> None:
        names = [spec.name for spec in self.specs]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ContractError(f"{self.name}: duplicate tensor names {sorted(duplicates)}.")
        available = set(names)
        for spec in self.specs:
            if spec.mask is not None and spec.mask not in available:
                raise ContractError(
                    f"{self.name}: {spec.name} names mask {spec.mask!r}, which the contract "
                    "does not declare."
                )

    def __iter__(self) -> Iterator[TensorSpec]:
        return iter(self.specs)

    def __len__(self) -> int:
        return len(self.specs)

    def spec(self, name: str) -> TensorSpec:
        """Return one tensor spec by name."""
        for candidate in self.specs:
            if candidate.name == name:
                return candidate
        raise ContractError(
            f"{self.name} declares no tensor {name!r}. Declared: "
            f"{', '.join(s.name for s in self.specs)}."
        )

    def names(self) -> tuple[str, ...]:
        """Names of every declared tensor, in declaration order."""
        return tuple(spec.name for spec in self.specs)

    def validate_payload(
        self,
        payload: Mapping[str, PlainArray],
        *,
        bindings: Mapping[str, int] | None = None,
        require_all: bool = True,
    ) -> dict[str, int]:
        """Validate a payload against the whole contract.

        Returns the dimension bindings the payload implies, so a caller can check
        that, for example, every tensor in the batch agreed on ``B``.
        """
        unknown = sorted(set(payload) - set(self.names()))
        if unknown:
            raise ContractError(f"{self.name}: payload has undeclared tensors {unknown}.")
        if require_all:
            missing = sorted(set(self.names()) - set(payload))
            if missing:
                raise ContractError(f"{self.name}: payload is missing tensors {missing}.")
        resolved = dict(bindings or {})
        for spec in self.specs:
            if spec.name in payload:
                resolved = spec.validate(payload[spec.name], resolved)
        return resolved

    def table(self) -> str:
        """Markdown table of the contract, for the design document."""
        lines = [
            "| tensor | shape | dtype | normalization | mask | meaning |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for spec in self.specs:
            shape = "[" + ", ".join(spec.dims) + "]"
            lines.append(
                f"| `{spec.name}` | `{shape}` | {spec.dtype} | {spec.normalization} | "
                f"{spec.mask or '-'} | {spec.semantics} |"
            )
        return "\n".join(lines)


class ContractRegistry:
    """All declared contracts, addressable by name."""

    def __init__(self, contracts: Iterable[Contract] = ()) -> None:
        self._contracts: dict[str, Contract] = {}
        for contract in contracts:
            self.register(contract)

    def register(self, contract: Contract) -> Contract:
        """Add a contract, rejecting a duplicate name."""
        if contract.name in self._contracts:
            raise ContractError(f"Contract {contract.name!r} is already registered.")
        self._contracts[contract.name] = contract
        return contract

    def get(self, name: str) -> Contract:
        """Return a contract by name."""
        try:
            return self._contracts[name]
        except KeyError:
            raise ContractError(
                f"Unknown contract {name!r}. Registered: {', '.join(sorted(self._contracts))}."
            ) from None

    def names(self) -> tuple[str, ...]:
        """Names of all registered contracts, sorted."""
        return tuple(sorted(self._contracts))

    def __len__(self) -> int:
        return len(self._contracts)

    def __iter__(self) -> Iterator[Contract]:
        return iter(self._contracts.values())

    def all_tensor_specs(self) -> tuple[TensorSpec, ...]:
        """Every tensor spec across every contract."""
        return tuple(spec for contract in self for spec in contract)


CONTRACTS = ContractRegistry()
"""Registry populated by the architecture modules as they are imported."""


@dataclass(frozen=True, slots=True)
class ChannelLayout:
    """Documented channel layout of a packed feature vector."""

    name: str
    channels: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def width(self) -> int:
        """Number of channels."""
        return len(self.channels)

    def index(self, channel: str) -> int:
        """Position of a named channel."""
        for position, (name, _) in enumerate(self.channels):
            if name == channel:
                return position
        raise ContractError(
            f"{self.name} has no channel {channel!r}. Channels: "
            f"{', '.join(name for name, _ in self.channels)}."
        )

    def table(self) -> str:
        """Markdown table of the layout."""
        lines = ["| index | channel | meaning |", "| --- | --- | --- |"]
        for position, (name, meaning) in enumerate(self.channels):
            lines.append(f"| {position} | `{name}` | {meaning} |")
        return "\n".join(lines)

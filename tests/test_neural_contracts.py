"""Tensor and data contracts: named dimensions, validation, and payload checking."""

from __future__ import annotations

import pytest

from awr.errors import ContractError
from generation.neural.contracts import (
    CONTRACTS,
    DIMENSIONS,
    ChannelLayout,
    Contract,
    Dim,
    DType,
    PlainArray,
    TensorSpec,
    default_bindings,
    dim,
)


def test_every_dimension_is_documented() -> None:
    """No dimension is a bare letter; each has a meaning."""
    for declared in DIMENSIONS:
        assert declared.description.strip()
        assert len(declared.description) > 20, declared.name


def test_dimension_names_are_unique() -> None:
    """One meaning per name, repository-wide."""
    names = [declared.name for declared in DIMENSIONS]
    assert len(names) == len(set(names))


def test_padded_dimensions_declare_a_mask() -> None:
    """A padded axis without a mask would let padding be read as data."""
    for declared in DIMENSIONS:
        if declared.padded:
            assert declared.mask, declared.name


def test_padded_dimension_without_mask_is_rejected() -> None:
    """The rule is enforced, not just documented."""
    with pytest.raises(ContractError, match="must name the mask"):
        Dim("X", "a padded axis with no mask", size=4, padded=True)


def test_unknown_dimension_lists_the_declared_ones() -> None:
    """A typo in a dimension name fails with the alternatives."""
    with pytest.raises(ContractError, match="Declared dimensions"):
        dim("D_MYSTERY")


def test_dynamic_dimensions_have_no_default() -> None:
    """Batch size and query count are bound per call, never assumed."""
    bindings = default_bindings()
    assert "B" not in bindings
    assert "P_QUERY" not in bindings
    assert bindings["N_ENT"] == 64


def test_plain_array_shape_and_indexing() -> None:
    """The framework-free array indexes correctly."""
    array = PlainArray.from_nested([[1, 2, 3], [4, 5, 6]], DType.INT32)
    assert array.shape == (2, 3)
    assert array.rank == 2
    assert array.at(1, 2) == 6
    assert array.row(1) == (4, 5, 6)


def test_ragged_input_is_rejected() -> None:
    """A ragged nested list is an error, not a silent truncation."""
    with pytest.raises(ContractError, match="Ragged"):
        PlainArray.from_nested([[1, 2], [3]], DType.INT32)


def test_array_data_length_must_match_shape() -> None:
    """Shape and data length cannot disagree."""
    with pytest.raises(ContractError, match="needs 6 values"):
        PlainArray((2, 3), DType.INT32, (1, 2, 3))


def test_tensor_spec_resolves_and_validates() -> None:
    """A spec resolves to a shape and validates an array against it."""
    spec = TensorSpec("z", ("B", "N_ENT", "D_ENT"), DType.FLOAT32, "test tensor")
    assert spec.resolve({"B": 4}) == (4, 64, 256)
    bindings = spec.validate(PlainArray.zeros((4, 64, 256)))
    assert bindings["B"] == 4


def test_dtype_mismatch_is_caught() -> None:
    """A float tensor cannot be satisfied by integers."""
    spec = TensorSpec("z", ("B", "D_ENT"), DType.FLOAT32, "test tensor")
    with pytest.raises(ContractError, match="expected dtype"):
        spec.validate(PlainArray.zeros((2, 256), DType.INT32))


def test_rank_mismatch_is_caught() -> None:
    """Rank errors are reported with the declared dimensions."""
    spec = TensorSpec("z", ("B", "N_ENT", "D_ENT"), DType.FLOAT32, "test tensor")
    with pytest.raises(ContractError, match="expected rank 3"):
        spec.validate(PlainArray.zeros((2, 256)))


def test_a_dimension_cannot_mean_two_things_in_one_batch() -> None:
    """The check that catches the most expensive class of shape bug."""
    first = TensorSpec("a", ("B", "N_ENT"), DType.INT32, "first tensor")
    second = TensorSpec("b", ("B", "N_ENT"), DType.INT32, "second tensor")
    bindings = first.validate(PlainArray.zeros((4, 64), DType.INT32))
    with pytest.raises(ContractError, match="is 8 here but 4 elsewhere"):
        second.validate(PlainArray.zeros((8, 64), DType.INT32), bindings)


def test_value_range_is_enforced() -> None:
    """A declared range is checked, so normalisation errors surface early."""
    spec = TensorSpec(
        "state", ("B", "N_STATE"), DType.FLOAT32, "state", value_range=(0.0, 1.0)
    )
    with pytest.raises(ContractError, match="outside the declared range"):
        spec.validate(PlainArray.full((2, 9), 5.0, DType.FLOAT32))


def test_contract_requires_its_masks() -> None:
    """A tensor cannot name a mask the contract does not declare."""
    with pytest.raises(ContractError, match="names mask"):
        Contract(
            name="broken",
            purpose="test",
            specs=(
                TensorSpec("x", ("B", "N_ENT"), DType.INT32, "masked", mask="absent_mask"),
            ),
        )


def test_contract_rejects_duplicate_tensor_names() -> None:
    """One name per tensor within a contract."""
    spec = TensorSpec("x", ("B",), DType.INT32, "duplicated")
    with pytest.raises(ContractError, match="duplicate tensor names"):
        Contract(name="broken", purpose="test", specs=(spec, spec))


def test_payload_validation_rejects_extra_and_missing_tensors() -> None:
    """A payload must match the contract exactly."""
    contract = CONTRACTS.get("awr_batch")
    with pytest.raises(ContractError, match="missing tensors"):
        contract.validate_payload({}, bindings={"B": 1})
    with pytest.raises(ContractError, match="undeclared tensors"):
        contract.validate_payload(
            {"nonsense": PlainArray.zeros((1, 1))}, bindings={"B": 1}, require_all=False
        )


def test_registered_contracts_are_addressable() -> None:
    """Every architecture contract is registered under a stable name."""
    expected = {
        "awr_batch",
        "anatomical_latent",
        "graph_encoder_io",
        "awr_program",
        "language_io",
        "geometry_decoding",
        "multimodal_alignment",
        "scene_context",
    }
    assert expected <= set(CONTRACTS.names())
    with pytest.raises(ContractError, match="Unknown contract"):
        CONTRACTS.get("not_a_contract")


def test_every_spec_has_semantics_and_every_padded_tensor_has_a_mask() -> None:
    """No tensor in the architecture is undocumented or unmasked by accident."""
    for spec in CONTRACTS.all_tensor_specs():
        assert spec.semantics.strip(), spec.name
        padded = any(dim(name).padded for name in spec.dims)
        if padded and spec.dtype is not DType.BOOL:
            assert spec.mask is not None, spec.name


def test_channel_layout_indexes_by_name() -> None:
    """Packed feature vectors are addressed by channel name, not by number."""
    layout = ChannelLayout("demo", (("a", "first"), ("b", "second")))
    assert layout.width == 2
    assert layout.index("b") == 1
    with pytest.raises(ContractError, match="has no channel"):
        layout.index("c")


def test_contract_renders_a_table() -> None:
    """Contracts can document themselves, so the design document cannot drift."""
    table = CONTRACTS.get("awr_batch").table()
    assert "| tensor |" in table
    assert "entity_ids" in table

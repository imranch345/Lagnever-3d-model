"""Level 1: the thorax domain, built by including Heart Ontology v0.1 unchanged.

Three things have to hold for the composite to be safe:

* **the heart is untouched.** The research steps load ``ontology/heart`` and their neural
  features are sized from its vocabulary; including it elsewhere must not change it.
* **inclusion is strict.** An include is validated on its own, pinned by id and version, and
  cannot be used to smuggle in duplicates or cycles.
* **the existing engine works on it.** The command layer was written for the heart; the lungs
  must be reachable through the same commands without special cases.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from awr.config import load_domain_config
from awr.errors import OntologyError
from awr.ontology import AnatomyOntology, load_ontology
from awr.schema import AnatomyType, Laterality
from awr.validation import validate_ontology
from reasoning.command_engine import CommandEngine

HEART = Path("ontology/heart")
THORAX = Path("ontology/thorax")

RESPIRATORY = (
    "respiratory.lungs",
    "respiratory.airways",
    "respiratory.left_lung",
    "respiratory.right_lung",
    "respiratory.lung_upper_lobe_left",
    "respiratory.lung_lower_lobe_left",
    "respiratory.lung_upper_lobe_right",
    "respiratory.lung_middle_lobe_right",
    "respiratory.lung_lower_lobe_right",
    "respiratory.trachea",
)


@pytest.fixture(scope="module")
def heart() -> AnatomyOntology:
    """Heart Ontology v0.1 on its own, as the research steps load it."""
    return load_ontology(HEART, expected_version="0.1.0")


@pytest.fixture(scope="module")
def thorax() -> AnatomyOntology:
    """The Level 1 composite."""
    return load_ontology(THORAX, expected_version="0.1.0")


class TestTheHeartIsUntouched:
    """Including the heart must not change what the research steps load."""

    def test_the_heart_alone_is_as_before(self, heart: AnatomyOntology) -> None:
        """42 entities, rooted at the heart, with no includes of its own."""
        assert heart.ontology_id == "lagnav.heart"
        assert len(heart) == 42
        assert heart.root_id == "heart"
        assert heart.includes == ()
        assert heart.relationships.counts() == {"structure": 70, "spatial": 34, "functional": 26}

    def test_the_anatomy_type_vocabulary_is_frozen(self) -> None:
        """features.py sizes a neural input from this enum; ADR-L1-001 freezes it."""
        assert len(AnatomyType) == 14

    def test_every_heart_entity_is_identical_inside_the_thorax(
        self, heart: AnatomyOntology, thorax: AnatomyOntology
    ) -> None:
        """Same ids, same definitions; only the heart root gains a parent."""
        for entity_id in heart.ids():
            assert thorax.get(entity_id) == heart.get(entity_id)
            if entity_id != heart.root_id:
                assert thorax.parent(entity_id) == heart.parent(entity_id)
        assert thorax.parent("heart") == "thorax"

    def test_every_heart_edge_survives(
        self, heart: AnatomyOntology, thorax: AnatomyOntology
    ) -> None:
        """No edge of the included ontology is dropped or altered."""
        def edges(ontology: AnatomyOntology) -> set[tuple[str, str, str]]:
            return {(e.subject, e.relation, e.object) for e in ontology.relationships.all_edges()}

        assert edges(heart) <= edges(thorax)


class TestTheThoraxOntology:
    """What Level 1 adds."""

    def test_it_loads_as_a_validated_draft(self, thorax: AnatomyOntology) -> None:
        """Root, size, the recorded include, and no validation errors."""
        assert thorax.ontology_id == "lagnav.thorax"
        assert thorax.root_id == "thorax"
        assert len(thorax) == 42 + 1 + len(RESPIRATORY)
        assert thorax.includes == ("lagnav.heart@0.1.0",)
        assert thorax.provenance.medically_validated is False
        assert validate_ontology(thorax).errors == ()

    def test_respiratory_entities_are_present(self, thorax: AnatomyOntology) -> None:
        """Every structure the Level 1 data can show has an entity."""
        for entity_id in RESPIRATORY:
            assert entity_id in thorax

    def test_laterality_matches_the_name(self, thorax: AnatomyOntology) -> None:
        """A left lobe is left, a right lobe is right: the viewer's side test relies on it."""
        for entity_id in RESPIRATORY:
            entity = thorax.get(entity_id)
            if entity_id.endswith("_left") or entity_id == "respiratory.left_lung":
                assert entity.laterality is Laterality.LEFT
            elif entity_id.endswith("_right") or entity_id == "respiratory.right_lung":
                assert entity.laterality is Laterality.RIGHT

    def test_lungs_mirror_the_heart_shell_convention(self, thorax: AnatomyOntology) -> None:
        """A lung is a shell at LOD 0 only; its lobes appear from LOD 1."""
        for lung in ("respiratory.left_lung", "respiratory.right_lung"):
            policy = thorax.get(lung).lod_policy
            assert (policy.min_lod, policy.max_lod) == (0, 0)
            for lobe in thorax.children(lung):
                assert thorax.get(lobe).lod_policy.min_lod == 1
        assert len(thorax.children("respiratory.left_lung")) == 2
        assert len(thorax.children("respiratory.right_lung")) == 3

    def test_respiratory_entities_use_existing_types(self, thorax: AnatomyOntology) -> None:
        """No new AnatomyType; the specific kind lives in semantic_role."""
        roles = {thorax.get(e).semantic_role for e in RESPIRATORY if not thorax.get(e).is_group}
        assert roles == {"lung", "lung_lobe", "airway"}

    def test_there_is_no_heart_lung_functional_edge_yet(self, thorax: AnatomyOntology) -> None:
        """Pulmonary circulation is deferred until the flow animation is tested against it."""
        functional = [
            e for e in thorax.relationships.all_edges()
            if str(thorax.registry.get(e.relation).graph) == "functional"
            and (e.subject.startswith("respiratory.") or e.object.startswith("respiratory."))
        ]
        assert functional == []


def _write(directory: Path, ontology_id: str, *, includes: list[Any] | None = None,
           entities: list[str], root: str, version: str = "0.1.0") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    common = {"ontology_id": ontology_id, "version": version}
    anatomy: dict[str, Any] = {
        **common,
        "entities": [
            {"id": e, "canonical_name": e, "anatomy_type": "organ", "renderable": True}
            for e in entities
        ],
    }
    if includes is not None:
        anatomy["includes"] = includes
    children = [e for e in entities if e != root]
    nodes = [{"parent": root, "children": children}] if children else []
    (directory / "anatomy.json").write_text(json.dumps(anatomy), encoding="utf-8")
    (directory / "hierarchy.json").write_text(
        json.dumps({**common, "root": root, "nodes": nodes}), encoding="utf-8"
    )
    (directory / "relationships.json").write_text(
        json.dumps({**common, "edges": []}), encoding="utf-8"
    )


class TestInclusionIsStrict:
    """An include is pinned, validated on its own, and cannot hide errors."""

    def _base(self, tmp_path: Path) -> Path:
        _write(tmp_path / "base", "t.base", entities=["b"], root="b")
        return tmp_path

    def _top(self, tmp_path: Path, spec: dict[str, Any], entities: list[str]) -> Path:
        top = tmp_path / "top"
        _write(top, "t.top", includes=[spec], entities=["top", *entities], root="top")
        hierarchy = json.loads((top / "hierarchy.json").read_text(encoding="utf-8"))
        hierarchy["nodes"] = [{"parent": "top", "children": ["b", *entities]}]
        (top / "hierarchy.json").write_text(json.dumps(hierarchy), encoding="utf-8")
        return top

    def test_a_valid_include_composes(self, tmp_path: Path) -> None:
        """The included root is attached where the includer's tree puts it."""
        root = self._base(tmp_path)
        spec = {"ontology_id": "t.base", "version": "0.1.0", "path": "../base"}
        loaded = load_ontology(self._top(root, spec, []))
        assert loaded.parent("b") == "top"
        assert loaded.includes == ("t.base@0.1.0",)

    @pytest.mark.parametrize(
        "spec",
        [
            {"ontology_id": "t.base", "version": "9.9.9", "path": "../base"},
            {"ontology_id": "t.other", "version": "0.1.0", "path": "../base"},
            {"ontology_id": "t.base", "version": "0.1.0", "path": "../missing"},
            {"ontology_id": "t.base", "path": "../base"},
        ],
    )
    def test_a_mismatched_or_broken_include_is_refused(
        self, tmp_path: Path, spec: dict[str, Any]
    ) -> None:
        """Wrong version, wrong id, missing directory or missing field all raise."""
        root = self._base(tmp_path)
        with pytest.raises(OntologyError):
            load_ontology(self._top(root, spec, []))

    def test_an_includer_cannot_redefine_an_included_entity(self, tmp_path: Path) -> None:
        """Duplicates across the boundary are caught, not silently overridden."""
        root = self._base(tmp_path)
        spec = {"ontology_id": "t.base", "version": "0.1.0", "path": "../base"}
        top = self._top(root, spec, [])
        anatomy = json.loads((top / "anatomy.json").read_text(encoding="utf-8"))
        anatomy["entities"].append({"id": "b", "canonical_name": "b", "anatomy_type": "organ"})
        (top / "anatomy.json").write_text(json.dumps(anatomy), encoding="utf-8")
        with pytest.raises(OntologyError, match="Duplicate entity id 'b'"):
            load_ontology(top)

    def test_an_include_cycle_is_refused(self, tmp_path: Path) -> None:
        """A includes B includes A must fail rather than recurse forever."""
        _write(tmp_path / "a", "t.a", entities=["a"], root="a",
               includes=[{"ontology_id": "t.b", "version": "0.1.0", "path": "../b"}])
        _write(tmp_path / "b", "t.b", entities=["bb"], root="bb",
               includes=[{"ontology_id": "t.a", "version": "0.1.0", "path": "../a"}])
        with pytest.raises(OntologyError, match="cycle"):
            load_ontology(tmp_path / "a")


class TestTheEngineDrivesTheLungs:
    """The existing command layer, unchanged, on the thorax domain."""

    @pytest.fixture()
    def engine(self) -> CommandEngine:
        """A thorax scene built at its default level of detail."""
        engine = CommandEngine(config=load_domain_config(domain="thorax"))
        engine.execute("Generate a human heart")
        return engine

    def test_the_domain_config_points_at_the_thorax(self) -> None:
        """configs/thorax.yaml loads and names the composite ontology."""
        config = load_domain_config(domain="thorax")
        assert config.domain.id == "thorax"
        assert config.domain.ontology_version == "0.1.0"

    def test_lung_commands_change_only_lung_state(self, engine: CommandEngine) -> None:
        """Hiding a lung leaves every heart entity exactly as it was."""
        heart_before = {
            i: copy.deepcopy(engine.scene.get(i).state)
            for i in engine.scene.ids()
            if i.startswith("heart")
        }
        result = engine.execute("hide the left lung")
        assert result.ok
        heart_after = {
            i: copy.deepcopy(engine.scene.get(i).state)
            for i in engine.scene.ids()
            if i.startswith("heart")
        }
        assert heart_after == heart_before

    @pytest.mark.parametrize(
        "command",
        [
            "show the lungs",
            "make the left upper lobe transparent",
            "isolate the trachea",
            "explain the right middle lobe",
        ],
    )
    def test_respiratory_commands_resolve(self, engine: CommandEngine, command: str) -> None:
        """Names, synonyms and groups resolve through the composite ontology."""
        assert engine.execute(command).ok


def _visible(engine: CommandEngine) -> set[str]:
    return {i for i in engine.scene.ids() if engine.scene.get(i).state.visibility}


class TestShellOrPartsNeverBoth:
    """With real geometry a shell and its parts overlap, so one stands for the organ at a time."""

    @pytest.fixture()
    def engine(self) -> CommandEngine:
        """A thorax scene at LOD 1."""
        engine = CommandEngine(config=load_domain_config(domain="thorax"))
        engine.execute("Generate a human heart")
        return engine

    SHELLS = {
        "heart": "heart.left_ventricle",
        "respiratory.left_lung": "respiratory.lung_upper_lobe_left",
        "respiratory.right_lung": "respiratory.lung_middle_lobe_right",
    }

    def _assert_never_both(self, engine: CommandEngine) -> None:
        visible = _visible(engine)
        for shell, part in self.SHELLS.items():
            assert not (shell in visible and part in visible), (shell, part)

    @pytest.mark.parametrize(
        "commands",
        [
            ["show the lungs"],
            ["show the heart"],
            ["isolate the lungs"],
            ["show everything"],
            ["show less detail", "show the lungs"],
            ["show less detail", "show everything"],
            ["hide the heart", "show the heart"],
        ],
    )
    def test_no_command_shows_a_shell_with_its_parts(
        self, engine: CommandEngine, commands: list[str]
    ) -> None:
        """At any LOD, after any of these, a shell and its parts are never both visible."""
        for command in commands:
            assert engine.execute(command).ok, command
        self._assert_never_both(engine)

    def test_naming_an_organ_above_its_shell_level_means_its_parts(
        self, engine: CommandEngine
    ) -> None:
        """At LOD 1, 'hide the heart' hides the chambers and vessels."""
        engine.execute("hide the heart")
        visible = _visible(engine)
        assert not any(i.startswith("heart") for i in visible)
        assert "respiratory.lung_upper_lobe_left" in visible

    def test_at_the_shell_level_the_organ_is_its_shell(self, engine: CommandEngine) -> None:
        """At LOD 0, the lungs are two shells, not five lobes."""
        engine.execute("show less detail")
        engine.execute("isolate the lungs")
        assert _visible(engine) == {"respiratory.left_lung", "respiratory.right_lung"}

    def test_explicit_requests_outside_the_lod_still_win(self, engine: CommandEngine) -> None:
        """Unchanged behaviour: 'show the valves' at LOD 1 shows the valves."""
        engine.execute("show the valves")
        assert "heart.mitral_valve" in _visible(engine)

    def test_explain_still_targets_the_organ_itself(self, engine: CommandEngine) -> None:
        """Only drawing operations resolve shells to parts; explanation does not."""
        response = engine.execute("explain the heart")
        assert response.ok and "heart" in response.message.lower()


class TestGeneratingTheThorax:
    """'Generate the heart and lungs' is understood, and refused where there are no lungs."""

    @pytest.mark.parametrize(
        "text",
        ["Generate the heart and lungs", "build the thorax", "create the chest",
         "generate the lungs and heart"],
    )
    def test_the_thorax_domain_builds_on_request(self, text: str) -> None:
        """All 53 entities, as with the heart phrase."""
        engine = CommandEngine(config=load_domain_config(domain="thorax"))
        response = engine.execute(text)
        assert response.ok, response.message
        assert len(engine.scene.ids()) == 53

    def test_the_heart_domain_says_it_has_no_lungs(self) -> None:
        """Rather than silently building a heart and calling it the lungs."""
        engine = CommandEngine(config=load_domain_config(domain="heart"))
        response = engine.execute("generate the heart and lungs")
        assert not response.ok
        assert "no lungs" in response.message
        assert engine.execute("generate a human heart").ok

"""Unit tests for EntityRegistry persistent state machine."""

import pytest

from methods.reward_forcing.teleport.registry import EntityRegistry


class TestBasicLifecycle:
    def test_empty_registry(self):
        r = EntityRegistry()
        assert len(r) == 0
        assert r.get_absent() == []
        assert r.get_visible() == []
        assert r.snapshot() == {}

    def test_register_adds_never_seen(self):
        r = EntityRegistry()
        r.register_prompt_entities(["dog", "cat"])
        snap = r.snapshot()
        assert snap == {"dog": "never_seen", "cat": "never_seen"}

    def test_register_idempotent(self):
        r = EntityRegistry()
        r.register_prompt_entities(["dog"])
        r.register_prompt_entities(["dog", "cat"])
        snap = r.snapshot()
        assert snap == {"dog": "never_seen", "cat": "never_seen"}

    def test_reset_wipes_state(self):
        r = EntityRegistry()
        r.register_prompt_entities(["dog"])
        r.update_from_observation(visible=["dog"], absent=[], partial=[])
        r.reset()
        assert r.snapshot() == {}


class TestStateTransitions:
    def test_never_seen_to_visible(self):
        r = EntityRegistry()
        r.register_prompt_entities(["dog"])
        r.update_from_observation(visible=["dog"], absent=[], partial=[])
        assert r.snapshot()["dog"] == "visible"

    def test_partial_counts_as_visible(self):
        r = EntityRegistry()
        r.register_prompt_entities(["dog"])
        r.update_from_observation(visible=[], absent=[], partial=["dog"])
        assert r.snapshot()["dog"] == "visible"

    def test_never_seen_absent_stays_never_seen(self):
        """An entity that's never been visible cannot lock to absent yet."""
        r = EntityRegistry()
        r.register_prompt_entities(["dog"])
        r.update_from_observation(visible=[], absent=["dog"], partial=[])
        assert r.snapshot()["dog"] == "never_seen"
        assert r.get_absent() == []

    def test_visible_to_absent_locks(self):
        r = EntityRegistry()
        r.register_prompt_entities(["dog"])
        r.update_from_observation(visible=["dog"], absent=[], partial=[])
        r.update_from_observation(visible=[], absent=["dog"], partial=[])
        assert r.snapshot()["dog"] == "absent"
        assert r.get_absent() == ["dog"]

    def test_absent_lock_is_permanent_against_visible(self):
        r = EntityRegistry()
        r.register_prompt_entities(["dog"])
        # visible → absent (locked)
        r.update_from_observation(visible=["dog"], absent=[], partial=[])
        r.update_from_observation(visible=[], absent=["dog"], partial=[])
        # VLM later says visible again — must stay absent
        r.update_from_observation(visible=["dog"], absent=[], partial=[])
        assert r.snapshot()["dog"] == "absent"

    def test_absent_lock_is_permanent_against_partial(self):
        r = EntityRegistry()
        r.register_prompt_entities(["dog"])
        r.update_from_observation(visible=["dog"], absent=[], partial=[])
        r.update_from_observation(visible=[], absent=["dog"], partial=[])
        r.update_from_observation(visible=[], absent=[], partial=["dog"])
        assert r.snapshot()["dog"] == "absent"


class TestNamedEntityOnly:
    def test_untracked_entity_ignored(self):
        """VLM mentions an entity that was never in prompt_hint → ignored."""
        r = EntityRegistry()
        r.register_prompt_entities(["dog"])
        # VLM saw 'tree' but tree was never in prompt-hint
        r.update_from_observation(visible=["dog", "tree"], absent=[], partial=[])
        snap = r.snapshot()
        assert "tree" not in snap
        assert snap == {"dog": "visible"}

    def test_untracked_absent_ignored(self):
        r = EntityRegistry()
        r.register_prompt_entities(["dog"])
        r.update_from_observation(visible=["dog"], absent=["nonexistent"], partial=[])
        assert r.snapshot() == {"dog": "visible"}


class TestMultiEntity:
    def test_independent_state_machines(self):
        r = EntityRegistry()
        r.register_prompt_entities(["dog", "cat", "ball"])
        r.update_from_observation(
            visible=["dog", "cat"], absent=[], partial=["ball"],
        )
        # All three transition to visible
        snap = r.snapshot()
        assert snap == {"dog": "visible", "cat": "visible", "ball": "visible"}

        # cat leaves; dog and ball stay
        r.update_from_observation(
            visible=["dog"], absent=["cat"], partial=["ball"],
        )
        snap = r.snapshot()
        assert snap == {"dog": "visible", "cat": "absent", "ball": "visible"}
        assert r.get_absent() == ["cat"]
        assert set(r.get_visible()) == {"dog", "ball"}

    def test_get_absent_sorted(self):
        r = EntityRegistry()
        r.register_prompt_entities(["zebra", "apple", "mango"])
        for e in ["zebra", "apple", "mango"]:
            r.update_from_observation(visible=[e], absent=[], partial=[])
            r.update_from_observation(visible=[], absent=[e], partial=[])
        assert r.get_absent() == ["apple", "mango", "zebra"]  # sorted


class TestSecondPromptIntroducesNewEntity:
    """User's exact scenario: segment A ['dog'], segment B ['dog', 'cat'].
    cat is new in segment B — it must be added and tracked."""

    def test_new_entity_introduced_later(self):
        r = EntityRegistry()
        # segment A
        r.register_prompt_entities(["dog"])
        r.update_from_observation(visible=["dog"], absent=[], partial=[])
        assert r.snapshot() == {"dog": "visible"}

        # segment B introduces 'cat'
        r.register_prompt_entities(["dog", "cat"])
        assert r.snapshot() == {"dog": "visible", "cat": "never_seen"}

        # cat shows up in frame
        r.update_from_observation(visible=["dog", "cat"], absent=[], partial=[])
        assert r.snapshot() == {"dog": "visible", "cat": "visible"}

        # cat leaves
        r.update_from_observation(visible=["dog"], absent=["cat"], partial=[])
        assert r.snapshot() == {"dog": "visible", "cat": "absent"}
        assert r.get_absent() == ["cat"]

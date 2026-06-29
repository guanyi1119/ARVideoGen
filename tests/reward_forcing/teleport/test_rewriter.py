"""Unit tests for PromptRewriter — pure rule-based, no VLM / GPU needed."""

import pytest

from methods.reward_forcing.teleport.rewriter import (
    PromptRewriter,
    build_prompt_rewriter,
)


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestFactory:
    """build_prompt_rewriter dispatch logic."""

    def test_factory_disabled(self):
        """None / {} / mode=off → None."""
        assert build_prompt_rewriter(None) is None
        assert build_prompt_rewriter({}) is None
        assert build_prompt_rewriter({"mode": "off"}) is None

    def test_factory_unknown_mode(self):
        """Unknown mode → ValueError with helpful message."""
        with pytest.raises(ValueError, match="Unknown.*rewrite mode"):
            build_prompt_rewriter({"mode": "foo"})

    def test_factory_constructs(self):
        """mode=conservative → PromptRewriter instance."""
        rw = build_prompt_rewriter({"mode": "conservative"})
        assert isinstance(rw, PromptRewriter)


# ---------------------------------------------------------------------------
# Rewrite — conservative mode
# ---------------------------------------------------------------------------


class TestRewriteConservative:
    """rewrite() behaviour in conservative mode."""

    def test_off_mode_passthrough(self):
        """mode=off → original prompt returned as-is regardless of visibility."""
        rw = PromptRewriter(mode="off")
        vis = {"visible": [], "absent": ["dog"], "partial": []}
        result = rw.rewrite("A sunny park.", vis)
        assert result == "A sunny park."

    def test_empty_visibility_returns_original(self):
        """All three lists empty → original prompt returned without extra whitespace."""
        rw = PromptRewriter(mode="conservative")
        vis = {"visible": [], "absent": [], "partial": []}
        result = rw.rewrite("A scene.", vis)
        assert result == "A scene."

    def test_absent_single_entity(self):
        """absent entity → output contains entity name and 'no longer in the scene'."""
        rw = PromptRewriter(mode="conservative")
        vis = {"visible": [], "absent": ["dog"], "partial": []}
        result = rw.rewrite("A scene.", vis)
        assert "A scene." in result
        assert "dog" in result
        assert "no longer in the scene" in result

    def test_partial_single_entity(self):
        """partial entity → output contains entity name and 'partially'."""
        rw = PromptRewriter(mode="conservative")
        vis = {"visible": [], "absent": [], "partial": ["dog"]}
        result = rw.rewrite("A scene.", vis)
        assert "dog" in result
        assert "partially" in result

    def test_absent_and_partial(self):
        """absent + partial → both suffixes present, absent before partial."""
        rw = PromptRewriter(mode="conservative")
        vis = {"visible": [], "absent": ["dog"], "partial": ["cat"]}
        result = rw.rewrite("A scene.", vis)
        # Both suffixes appear
        assert "no longer in the scene" in result
        assert "partially" in result
        # absent suffix appears before partial suffix
        absent_pos = result.index("no longer in the scene")
        partial_pos = result.index("partially")
        assert absent_pos < partial_pos


# ---------------------------------------------------------------------------
# Rewrite — aggressive mode
# ---------------------------------------------------------------------------


class TestRewriteAggressive:
    """rewrite() behaviour in aggressive mode."""

    def test_aggressive_mentions_edge(self):
        """aggressive mode + absent → output MUST contain 'edge'."""
        rw = PromptRewriter(mode="aggressive")
        vis = {"visible": [], "absent": ["dog"], "partial": []}
        result = rw.rewrite("A scene.", vis)
        assert "edge" in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Additional coverage for boundary conditions."""

    def test_custom_templates_override(self):
        """Injected templates → custom suffix text appears in output."""
        rw = PromptRewriter(
            mode="conservative",
            templates={
                "conservative": {"absent_suffix": "GONE: {entities}"}
            },
        )
        vis = {"visible": [], "absent": ["dog"], "partial": []}
        result = rw.rewrite("A scene.", vis)
        assert "GONE: dog" in result

    def test_multi_entity_join(self):
        """Multiple entities in absent → comma-joined in output."""
        rw = PromptRewriter(mode="conservative")
        vis = {"visible": [], "absent": ["dog", "cat", "bird"], "partial": []}
        result = rw.rewrite("A scene.", vis)
        assert "dog, cat, bird" in result

    def test_visible_entities_not_in_output(self):
        """visible entities → NOT mentioned in suffix."""
        rw = PromptRewriter(mode="conservative")
        vis = {
            "visible": ["car"],
            "absent": ["dog"],
            "partial": [],
        }
        result = rw.rewrite("A scene with a car.", vis)
        # car should appear in the original prompt, not injected by us
        assert "car" in result  # from original prompt
        assert "Note:" in result  # suffix present
        # Verify car is NOT in the suffix part (after "Note:")
        suffix_part = result[result.index("Note:") :]
        assert "car" not in suffix_part

    def test_unknown_mode_raises(self):
        """Unknown mode in constructor → ValueError."""
        with pytest.raises(ValueError, match="Unknown.*rewrite mode"):
            PromptRewriter(mode="bogus")

    def test_templates_partial_merge(self):
        """Partial template override → missing keys filled from defaults."""
        rw = PromptRewriter(
            mode="aggressive",
            templates={
                "aggressive": {
                    "absent_suffix": "REMOVED: {entities}"
                }
            },
        )
        vis = {"visible": [], "absent": ["dog"], "partial": ["cat"]}
        result = rw.rewrite("A scene.", vis)
        # Custom absent_suffix used
        assert "REMOVED: dog" in result
        # Default partial_suffix still works (aggressive has 'edge' in it)
        assert "edge" in result
        assert "cat" in result

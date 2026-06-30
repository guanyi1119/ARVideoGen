"""Unit tests for TeleportVLM — all VLM calls mocked, no real model loaded."""

import pytest
import torch

from methods.reward_forcing.teleport.vlm import (
    QwenTeleportVLM,
    TeleportVLM,
    build_teleport_vlm,
)

# Reusable fake frame tensor (does not require GPU)
FAKE_FRAME = torch.zeros(3, 224, 224)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vlm() -> QwenTeleportVLM:
    """Build a QwenTeleportVLM that will never load the real model."""
    return QwenTeleportVLM(model_path="/tmp/fake", device="cpu")


def _patch_call_vlm(vlm: QwenTeleportVLM, monkeypatch, return_text: str):
    """Replace ``_call_vlm`` so it returns *return_text* without touching
    the real QwenPromptExpander."""

    def _fake_call(frame_rgb, entity_list):
        return return_text

    monkeypatch.setattr(vlm, "_call_vlm", _fake_call)
    # Also skip _load_model so we don't even try to import qwen
    monkeypatch.setattr(vlm, "_load_model", lambda: None)
    # And fake is_available for tests that check it
    monkeypatch.setattr(vlm, "_expander", True)  # truthy


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestFactory:
    """build_teleport_vlm dispatch logic."""

    def test_factory_disabled(self):
        """None / {} / disabled → None."""
        assert build_teleport_vlm(None) is None
        assert build_teleport_vlm({}) is None
        assert build_teleport_vlm({"type": "disabled"}) is None

    def test_factory_unknown_type(self):
        """Unknown type → ValueError with helpful message."""
        with pytest.raises(ValueError, match="Unknown.*qwen_vl"):
            build_teleport_vlm({"type": "foo"})

    def test_factory_qwen_constructs(self):
        """qwen_vl_3b → QwenTeleportVLM instance; not loaded yet."""
        vlm = build_teleport_vlm(
            {"type": "qwen_vl_3b", "model_path": "/tmp/fake"}
        )
        assert isinstance(vlm, QwenTeleportVLM)
        # Lazy-load → model not loaded, is_available() is False
        assert vlm.is_available() is False


# ---------------------------------------------------------------------------
# Response parsing tests
# ---------------------------------------------------------------------------


class TestParseResponse:
    """check_visibility JSON parsing and fail-safe behaviour."""

    def test_parse_valid_json(self, monkeypatch):
        """Valid JSON → correctly parsed three-way dict."""
        vlm = _make_vlm()
        _patch_call_vlm(
            vlm,
            monkeypatch,
            '{"visible": ["dog"], "absent": ["ball"], "partial": []}',
        )
        result = vlm.check_visibility(FAKE_FRAME, ["dog", "ball"])
        assert result == {"visible": ["dog"], "absent": ["ball"], "partial": []}

    def test_parse_invalid_json_fails_safe(self, monkeypatch):
        """Non-JSON response → empty safe dict, no exception."""
        vlm = _make_vlm()
        _patch_call_vlm(vlm, monkeypatch, "this is not json at all")
        result = vlm.check_visibility(FAKE_FRAME, ["dog"])
        assert result == {"visible": [], "absent": [], "partial": []}

    def test_parse_extra_text_around_json(self, monkeypatch):
        """JSON buried in extra text → still extracted and parsed."""
        vlm = _make_vlm()
        _patch_call_vlm(
            vlm,
            monkeypatch,
            'Sure! Here: {"visible":["dog"],"absent":[],"partial":[]} hope that helps.',
        )
        result = vlm.check_visibility(FAKE_FRAME, ["dog"])
        assert result == {"visible": ["dog"], "absent": [], "partial": []}

    def test_parse_schema_violation(self, monkeypatch):
        """Schema violation (visible is str, not list) → fail-safe empty."""
        vlm = _make_vlm()
        _patch_call_vlm(
            vlm,
            monkeypatch,
            '{"visible": "dog", "absent": [], "partial": []}',
        )
        result = vlm.check_visibility(FAKE_FRAME, ["dog"])
        assert result == {"visible": [], "absent": [], "partial": []}

    def test_parse_out_of_vocab_filtered(self, monkeypatch):
        """Entity not in entity_list → filtered out; valid ones kept."""
        vlm = _make_vlm()
        _patch_call_vlm(
            vlm,
            monkeypatch,
            '{"visible": ["cat", "dog"], "absent": [], "partial": []}',
        )
        result = vlm.check_visibility(FAKE_FRAME, ["dog"])
        assert result == {"visible": ["dog"], "absent": [], "partial": []}


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Additional coverage for edge conditions."""

    def test_empty_entity_list(self, monkeypatch):
        """Empty entity_list → empty dict immediately (no VLM call)."""
        vlm = _make_vlm()
        # Don't even patch _call_vlm — it should short-circuit
        monkeypatch.setattr(vlm, "_load_model", lambda: None)
        monkeypatch.setattr(vlm, "_expander", True)
        result = vlm.check_visibility(FAKE_FRAME, [])
        assert result == {"visible": [], "absent": [], "partial": []}

    def test_no_json_block_at_all(self, monkeypatch):
        """Response with zero braces → empty dict."""
        vlm = _make_vlm()
        _patch_call_vlm(vlm, monkeypatch, "no braces here at all")
        result = vlm.check_visibility(FAKE_FRAME, ["dog"])
        assert result == {"visible": [], "absent": [], "partial": []}


# ---------------------------------------------------------------------------
# analyze() tests — open-vocab + visibility in one VLM call
# ---------------------------------------------------------------------------


class TestAnalyze:
    """analyze() open-vocab + visibility in one VLM call."""

    def test_analyze_valid_response(self, monkeypatch):
        vlm = _make_vlm()
        raw = (
            '{"in_frame": ["dog", "tree"], '
            '"visibility": {"visible": ["dog"], "absent": ["cat"], '
            '"partial": ["tree"]}}'
        )
        # Patch _call_vlm_analyze + _load_model
        monkeypatch.setattr(vlm, "_call_vlm_analyze", lambda f, h: raw)
        monkeypatch.setattr(vlm, "_load_model", lambda: None)
        monkeypatch.setattr(vlm, "_expander", True)

        result = vlm.analyze(FAKE_FRAME, ["cat"])
        assert result["in_frame"] == ["dog", "tree"]
        assert result["visibility"]["visible"] == ["dog"]
        assert result["visibility"]["absent"] == ["cat"]
        assert result["visibility"]["partial"] == ["tree"]

    def test_analyze_invalid_json_fails_safe(self, monkeypatch):
        vlm = _make_vlm()
        monkeypatch.setattr(vlm, "_call_vlm_analyze", lambda f, h: "garbage")
        monkeypatch.setattr(vlm, "_load_model", lambda: None)
        monkeypatch.setattr(vlm, "_expander", True)

        result = vlm.analyze(FAKE_FRAME, ["dog"])
        assert result == {
            "in_frame": [],
            "visibility": {"visible": [], "absent": [], "partial": []},
        }

    def test_analyze_missing_in_frame_key_fails_safe(self, monkeypatch):
        vlm = _make_vlm()
        raw = '{"visibility": {"visible":["dog"], "absent":[], "partial":[]}}'
        monkeypatch.setattr(vlm, "_call_vlm_analyze", lambda f, h: raw)
        monkeypatch.setattr(vlm, "_load_model", lambda: None)
        monkeypatch.setattr(vlm, "_expander", True)

        result = vlm.analyze(FAKE_FRAME, ["dog"])
        assert result["in_frame"] == []
        assert result["visibility"] == {"visible": [], "absent": [], "partial": []}

    def test_analyze_visibility_oov_filtered_by_union(self, monkeypatch):
        """Entities NOT in (in_frame ∪ prompt_hint) must be filtered out."""
        vlm = _make_vlm()
        # in_frame contains "dog"; prompt_hint contains "cat";
        # visibility includes "horse" which is OOV → must be filtered.
        raw = (
            '{"in_frame": ["dog"], '
            '"visibility": {"visible": ["dog", "horse"], '
            '"absent": ["cat"], "partial": []}}'
        )
        monkeypatch.setattr(vlm, "_call_vlm_analyze", lambda f, h: raw)
        monkeypatch.setattr(vlm, "_load_model", lambda: None)
        monkeypatch.setattr(vlm, "_expander", True)

        result = vlm.analyze(FAKE_FRAME, ["cat"])
        assert result["in_frame"] == ["dog"]
        assert "horse" not in result["visibility"]["visible"]
        assert result["visibility"]["visible"] == ["dog"]
        assert result["visibility"]["absent"] == ["cat"]

    def test_analyze_in_frame_open_vocab_not_filtered(self, monkeypatch):
        """in_frame is open-vocabulary; entities NOT in prompt_hint are kept."""
        vlm = _make_vlm()
        raw = (
            '{"in_frame": ["dog", "completely_novel_thing"], '
            '"visibility": {"visible": ["dog", "completely_novel_thing"], '
            '"absent": [], "partial": []}}'
        )
        monkeypatch.setattr(vlm, "_call_vlm_analyze", lambda f, h: raw)
        monkeypatch.setattr(vlm, "_load_model", lambda: None)
        monkeypatch.setattr(vlm, "_expander", True)

        result = vlm.analyze(FAKE_FRAME, [])  # empty prompt hint
        # in_frame open-vocab → both kept
        assert set(result["in_frame"]) == {"dog", "completely_novel_thing"}
        # visibility universe = in_frame ∪ prompt_hint = in_frame only,
        # so both visibility entries are in-universe → kept
        assert set(result["visibility"]["visible"]) == {"dog", "completely_novel_thing"}

    def test_analyze_extra_text_around_json(self, monkeypatch):
        vlm = _make_vlm()
        raw = (
            'Sure! Here is the result: '
            '{"in_frame": ["dog"], '
            '"visibility": {"visible": ["dog"], "absent": [], "partial": []}} '
            'I hope this helps.'
        )
        monkeypatch.setattr(vlm, "_call_vlm_analyze", lambda f, h: raw)
        monkeypatch.setattr(vlm, "_load_model", lambda: None)
        monkeypatch.setattr(vlm, "_expander", True)

        result = vlm.analyze(FAKE_FRAME, [])
        assert result["in_frame"] == ["dog"]
        assert result["visibility"]["visible"] == ["dog"]

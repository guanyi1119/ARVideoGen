"""Tests for teleport debug-print helper."""
import os
import sys
from unittest.mock import patch

import pytest
import torch

from methods.reward_forcing.teleport._debug import is_teleport_debug, tdprint
from methods.reward_forcing.teleport.hook import TeleportSwitchHook

# Reuse helpers from test_hook
from .test_hook import (
    _make_mock_vlm,
    _make_mock_rewriter,
    _make_mock_text_encoder,
    _make_mock_frame_decoder,
    FAKE_PROMPTS,
)


# ---------------------------------------------------------------------------
# is_teleport_debug() detection
# ---------------------------------------------------------------------------


def test_is_teleport_debug_env_off(monkeypatch):
    """Without env var or source DEBUG -> False."""
    monkeypatch.delenv("TELEPORT_DEBUG", raising=False)
    # Ensure core.misc.debug_option.DEBUG is False
    fake_module = type(sys)("core.misc.debug_option")
    fake_module.DEBUG = False
    with patch.dict(sys.modules, {"core.misc.debug_option": fake_module}):
        assert is_teleport_debug() is False


def test_is_teleport_debug_env_truthy(monkeypatch):
    """TELEPORT_DEBUG=1 -> True."""
    for val in ("1", "TRUE", "yes", "on"):
        monkeypatch.setenv("TELEPORT_DEBUG", val)
        assert is_teleport_debug() is True, f"TELEPORT_DEBUG={val!r} should be truthy"


def test_is_teleport_debug_env_falsey(monkeypatch):
    """TELEPORT_DEBUG=0/false -> False (assuming source DEBUG=False)."""
    fake_module = type(sys)("core.misc.debug_option")
    fake_module.DEBUG = False
    with patch.dict(sys.modules, {"core.misc.debug_option": fake_module}):
        for val in ("0", ""):
            monkeypatch.setenv("TELEPORT_DEBUG", val)
            assert is_teleport_debug() is False, f"TELEPORT_DEBUG={val!r} should be falsey"


def test_is_teleport_debug_source_constant(monkeypatch):
    """Source-level DEBUG=True -> True (env var off)."""
    fake_module = type(sys)("core.misc.debug_option")
    fake_module.DEBUG = True
    with patch.dict(sys.modules, {"core.misc.debug_option": fake_module}):
        monkeypatch.delenv("TELEPORT_DEBUG", raising=False)
        assert is_teleport_debug() is True


def test_is_teleport_debug_no_stub_package(monkeypatch):
    """When core.misc.debug_option cannot be imported, falls back to env var."""
    monkeypatch.setenv("TELEPORT_DEBUG", "1")
    # Remove the fake module so import fails
    if "core.misc.debug_option" in sys.modules:
        del sys.modules["core.misc.debug_option"]
    try:
        assert is_teleport_debug() is True
    finally:
        # Restore (the test_hook import above may have set it)
        pass


# ---------------------------------------------------------------------------
# tdprint() output
# ---------------------------------------------------------------------------


def test_tdprint_silent_when_off(monkeypatch, capsys):
    """tdprint produces NO output when debug is off."""
    fake_module = type(sys)("core.misc.debug_option")
    fake_module.DEBUG = False
    with patch.dict(sys.modules, {"core.misc.debug_option": fake_module}):
        monkeypatch.delenv("TELEPORT_DEBUG", raising=False)
        tdprint("should not appear")
    captured = capsys.readouterr()
    assert captured.out == ""


def test_tdprint_prefixed_when_on(monkeypatch, capsys):
    """tdprint outputs with [TeleportDebug] prefix when debug is on."""
    monkeypatch.setenv("TELEPORT_DEBUG", "1")
    tdprint("hello world")
    captured = capsys.readouterr()
    assert "[TeleportDebug]" in captured.out
    assert "hello world" in captured.out


# ---------------------------------------------------------------------------
# Hook integration: maybe_rewrite prints VLM raw + parsed + rewritten
# ---------------------------------------------------------------------------


def test_maybe_rewrite_emits_debug_lines(monkeypatch, capsys):
    """When TELEPORT_DEBUG=1, maybe_rewrite emits raw VLM + visibility + rewritten."""
    monkeypatch.setenv("TELEPORT_DEBUG", "1")

    vlm = _make_mock_vlm()
    vlm.analyze.return_value = {
        "in_frame": ["dog"],
        "visibility": {"visible": ["dog"], "absent": ["cat"], "partial": []},
    }
    rewriter = _make_mock_rewriter()
    rewriter.rewrite.return_value = "REWRITTEN_FOR_DEBUG"

    hook = TeleportSwitchHook(
        vlm=vlm,
        rewriter=rewriter,
        text_encoder_fn=_make_mock_text_encoder(),
        frame_decoder_fn=_make_mock_frame_decoder(),
        entity_extractor_fn=lambda p: ["cat"],
    )

    output_latent = torch.zeros(1, 21, 16, 40, 40)
    hook.maybe_rewrite(output_latent, 21, FAKE_PROMPTS)

    out = capsys.readouterr().out
    assert "[TeleportDebug]" in out
    assert "in_frame=['dog']" in out
    assert "REWRITTEN_FOR_DEBUG" in out
    # visibility should be printed
    assert "absent" in out and "cat" in out


def test_maybe_rewrite_silent_when_debug_off(monkeypatch, capsys):
    """Without TELEPORT_DEBUG, maybe_rewrite produces no [TeleportDebug] lines."""
    monkeypatch.delenv("TELEPORT_DEBUG", raising=False)
    fake_module = type(sys)("core.misc.debug_option")
    fake_module.DEBUG = False
    with patch.dict(sys.modules, {"core.misc.debug_option": fake_module}):
        vlm = _make_mock_vlm()
        vlm.analyze.return_value = {
            "in_frame": ["dog"],
            "visibility": {"visible": ["dog"], "absent": [], "partial": []},
        }
        rewriter = _make_mock_rewriter()
        rewriter.rewrite.return_value = "REWRITTEN"

        hook = TeleportSwitchHook(
            vlm=vlm,
            rewriter=rewriter,
            text_encoder_fn=_make_mock_text_encoder(),
            frame_decoder_fn=_make_mock_frame_decoder(),
            entity_extractor_fn=lambda p: ["dog"],
        )
        output_latent = torch.zeros(1, 21, 16, 40, 40)
        hook.maybe_rewrite(output_latent, 21, FAKE_PROMPTS)

    out = capsys.readouterr().out
    assert "[TeleportDebug]" not in out


def test_maybe_rewrite_exception_emits_debug(monkeypatch, capsys):
    """When TELEPORT_DEBUG=1 and VLM crashes, exception traceback is printed."""
    monkeypatch.setenv("TELEPORT_DEBUG", "1")

    vlm = _make_mock_vlm()
    vlm.analyze.side_effect = RuntimeError("VLM crash in test")

    hook = TeleportSwitchHook(
        vlm=vlm,
        rewriter=_make_mock_rewriter(),
        text_encoder_fn=_make_mock_text_encoder(),
        frame_decoder_fn=_make_mock_frame_decoder(),
        entity_extractor_fn=lambda p: ["dog"],
    )

    output_latent = torch.zeros(1, 21, 16, 40, 40)
    hook.maybe_rewrite(output_latent, 21, FAKE_PROMPTS)

    out = capsys.readouterr().out
    assert "[TeleportDebug]" in out
    assert "FELL BACK" in out
    assert "VLM crash in test" in out


def test_maybe_rewrite_exception_silent_when_off(monkeypatch, capsys):
    """Without TELEPORT_DEBUG, exception path produces no [TeleportDebug] lines."""
    monkeypatch.delenv("TELEPORT_DEBUG", raising=False)
    fake_module = type(sys)("core.misc.debug_option")
    fake_module.DEBUG = False
    with patch.dict(sys.modules, {"core.misc.debug_option": fake_module}):
        vlm = _make_mock_vlm()
        vlm.analyze.side_effect = RuntimeError("VLM crash")

        hook = TeleportSwitchHook(
            vlm=vlm,
            rewriter=_make_mock_rewriter(),
            text_encoder_fn=_make_mock_text_encoder(),
            frame_decoder_fn=_make_mock_frame_decoder(),
            entity_extractor_fn=lambda p: ["dog"],
        )
        output_latent = torch.zeros(1, 21, 16, 40, 40)
        hook.maybe_rewrite(output_latent, 21, FAKE_PROMPTS)

    out = capsys.readouterr().out
    assert "[TeleportDebug]" not in out


def test_maybe_rewrite_empty_universe_emits_debug(monkeypatch, capsys):
    """When TELEPORT_DEBUG=1 and universe is empty, prints fallback message."""
    monkeypatch.setenv("TELEPORT_DEBUG", "1")

    vlm = _make_mock_vlm()
    vlm.analyze.return_value = {
        "in_frame": [],
        "visibility": {"visible": [], "absent": [], "partial": []},
    }

    hook = TeleportSwitchHook(
        vlm=vlm,
        rewriter=_make_mock_rewriter(),
        text_encoder_fn=_make_mock_text_encoder(),
        frame_decoder_fn=_make_mock_frame_decoder(),
        entity_extractor_fn=lambda p: [],  # empty prompt hint
    )

    output_latent = torch.zeros(1, 21, 16, 40, 40)
    hook.maybe_rewrite(output_latent, 21, ["A blank scene"])

    out = capsys.readouterr().out
    assert "[TeleportDebug]" in out
    assert "empty universe" in out

"""Unit tests for TeleportSwitchHook — all VLM / rewriter calls mocked, no GPU."""

from typing import Dict, List
from unittest.mock import Mock, patch

import pytest
import torch

from methods.reward_forcing.teleport.hook import (
    TeleportSwitchHook,
    _default_entity_extract,
    build_teleport_switch_hook,
)
from methods.reward_forcing.teleport.rewriter import PromptRewriter
from methods.reward_forcing.teleport.vlm import QwenTeleportVLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_PROMPTS = ["A dog runs in the park"]
FAKE_COND = {"context": torch.zeros(1, 512)}  # minimal conditional_dict shape


def _make_mock_vlm(available=True):
    """Return a mock TeleportVLM whose is_available() returns *available*."""
    vlm = Mock()
    vlm.is_available.return_value = available
    vlm.analyze.return_value = {
        "in_frame": [],
        "visibility": {"visible": [], "absent": [], "partial": []},
    }
    return vlm


def _make_mock_rewriter():
    """Return a mock PromptRewriter that passes through by default."""
    rw = Mock(spec=PromptRewriter)
    rw.rewrite.return_value = FAKE_PROMPTS[0]
    return rw


def _make_mock_text_encoder(returns=None):
    """Return a callable mock that records calls and returns *returns*."""
    if returns is None:
        returns = FAKE_COND
    return Mock(return_value=returns)


def _make_mock_frame_decoder():
    """Return a callable mock that returns a [1, 1, 3, 256, 256] tensor."""
    return Mock(return_value=torch.ones(1, 1, 3, 256, 256))


def _make_hook(
    *,
    vlm=None,
    rewriter=None,
    text_encoder_fn=None,
    frame_decoder_fn=None,
    entity_extractor_fn=None,
    with_registry: bool = False,
):
    """Convenience builder with sensible defaults."""
    return TeleportSwitchHook(
        vlm=vlm or _make_mock_vlm(),
        rewriter=rewriter or _make_mock_rewriter(),
        text_encoder_fn=text_encoder_fn or _make_mock_text_encoder(),
        frame_decoder_fn=frame_decoder_fn or _make_mock_frame_decoder(),
        entity_extractor_fn=entity_extractor_fn,
        registry_enabled=with_registry,
    )


# ---------------------------------------------------------------------------
# 1. test_factory_disabled
# ---------------------------------------------------------------------------


class TestFactoryDisabled:
    """build_teleport_switch_hook returns None when config is missing or disabled."""

    def test_none_cfg(self):
        assert build_teleport_switch_hook(None, Mock(), Mock()) is None

    def test_disabled_explicit(self):
        assert (
            build_teleport_switch_hook({"enabled": False}, Mock(), Mock()) is None
        )


# ---------------------------------------------------------------------------
# 2. test_factory_missing_vlm
# ---------------------------------------------------------------------------


class TestFactoryMissingSubComponents:
    """build_teleport_switch_hook returns None when vlm/rewriter are disabled."""

    def test_missing_vlm(self):
        """vlm config empty → vlm=None → overall None."""
        cfg = {
            "enabled": True,
            "vlm": {},
            "rewriter": {"mode": "conservative"},
        }
        assert build_teleport_switch_hook(cfg, Mock(), Mock()) is None

    def test_missing_rewriter(self):
        """rewriter config mode=off → rewriter=None → overall None."""
        cfg = {
            "enabled": True,
            "vlm": {"type": "qwen_vl_3b", "model_path": "/tmp/fake"},
            "rewriter": {"mode": "off"},
        }
        assert build_teleport_switch_hook(cfg, Mock(), Mock()) is None


# ---------------------------------------------------------------------------
# 3. test_factory_constructs
# ---------------------------------------------------------------------------


class TestFactoryConstructs:
    """build_teleport_switch_hook returns a TeleportSwitchHook when fully configured."""

    def test_constructs(self):
        cfg = {
            "enabled": True,
            "vlm": {"type": "qwen_vl_3b", "model_path": "/tmp/fake"},
            "rewriter": {"mode": "conservative"},
        }
        hook = build_teleport_switch_hook(cfg, Mock(), Mock())
        assert isinstance(hook, TeleportSwitchHook)


# ---------------------------------------------------------------------------
# 4. test_maybe_rewrite_zero_frame
# ---------------------------------------------------------------------------


class TestMaybeRewriteZeroFrame:
    """current_start_frame=0 → skip VLM, skip decoder, encode original prompt."""

    def test_zero_frame_skips_vlm(self):
        text_enc = _make_mock_text_encoder()
        vlm = _make_mock_vlm()
        hook = _make_hook(vlm=vlm, text_encoder_fn=text_enc)

        output_latent = torch.zeros(1, 21, 16, 40, 40)  # [B, F, C, H, W]
        result = hook.maybe_rewrite(output_latent, 0, FAKE_PROMPTS)

        # VLM should NOT be called
        vlm.analyze.assert_not_called()
        # text_encoder should be called with original prompts
        text_enc.assert_called_once_with(FAKE_PROMPTS)
        assert result is not None


# ---------------------------------------------------------------------------
# 5. test_maybe_rewrite_normal_path
# ---------------------------------------------------------------------------


class TestMaybeRewriteNormalPath:
    """Full happy path: extract frame → VLM → rewriter → re-encode."""

    def test_normal_path(self):
        text_enc = _make_mock_text_encoder()
        vlm = _make_mock_vlm()
        vlm.analyze.return_value = {
            "in_frame": [],
            "visibility": {
                "visible": [],
                "absent": ["dog"],
                "partial": [],
            },
        }
        rewriter = _make_mock_rewriter()
        rewriter.rewrite.return_value = "A dog runs in the park. Note: dog is absent."

        hook = _make_hook(
            vlm=vlm, rewriter=rewriter, text_encoder_fn=text_enc
        )

        output_latent = torch.zeros(1, 21, 16, 40, 40)
        result = hook.maybe_rewrite(output_latent, 21, FAKE_PROMPTS)

        # VLM was called with a frame tensor and entity list
        vlm.analyze.assert_called_once()
        call_args = vlm.analyze.call_args[0]
        frame_arg = call_args[0]  # should be [3, H, W]
        entity_arg = call_args[1]
        assert isinstance(frame_arg, torch.Tensor)
        assert frame_arg.dim() == 3  # [3, H, W]
        assert "dog" in entity_arg

        # rewriter was called
        rewriter.rewrite.assert_called_once()

        # text_encoder was called with rewritten prompt (broadcast to batch)
        text_enc.assert_called_once()
        encoded_prompts = text_enc.call_args[0][0]
        assert "absent" in encoded_prompts[0].lower() or "dog" in encoded_prompts[0].lower()

        assert result is not None


# ---------------------------------------------------------------------------
# 6. test_maybe_rewrite_exception_recovers
# ---------------------------------------------------------------------------


class TestMaybeRewriteExceptionRecovers:
    """Any internal exception → graceful degradation to original prompt encode."""

    def test_decoder_exception(self):
        text_enc = _make_mock_text_encoder()
        frame_dec = Mock(side_effect=RuntimeError("VAE decode failed"))

        hook = _make_hook(
            text_encoder_fn=text_enc, frame_decoder_fn=frame_dec
        )

        output_latent = torch.zeros(1, 21, 16, 40, 40)
        result = hook.maybe_rewrite(output_latent, 21, FAKE_PROMPTS)

        # Must not raise — fallback to original encode
        text_enc.assert_called_once_with(FAKE_PROMPTS)
        assert result is not None

    def test_vlm_exception(self):
        text_enc = _make_mock_text_encoder()
        vlm = _make_mock_vlm()
        vlm.analyze.side_effect = RuntimeError("VLM crash")

        hook = _make_hook(vlm=vlm, text_encoder_fn=text_enc)

        output_latent = torch.zeros(1, 21, 16, 40, 40)
        result = hook.maybe_rewrite(output_latent, 21, FAKE_PROMPTS)

        # Must not raise — fallback to original encode
        text_enc.assert_called_once_with(FAKE_PROMPTS)
        assert result is not None


# ---------------------------------------------------------------------------
# 7. test_default_entity_extract
# ---------------------------------------------------------------------------


class TestDefaultEntityExtract:
    """_default_entity_extract placeholder behaviour."""

    def test_extracts_nouns(self):
        result = _default_entity_extract("A dog runs in the park")
        assert len(result) >= 1
        assert "dog" in result
        assert "park" in result
        # All words should be ≥ 3 chars
        assert all(len(w) >= 3 for w in result)

    def test_empty_string(self):
        result = _default_entity_extract("")
        assert result == []

    def test_short_words_filtered(self):
        result = _default_entity_extract("A dog")
        # "A" has len 1, filtered; "dog" has len 3, kept
        assert result == ["dog"]

    def test_deduplication(self):
        result = _default_entity_extract("dog dog dog")
        assert result == ["dog"]


# ---------------------------------------------------------------------------
# 8. test_maybe_rewrite_lazy_load_bug
# ---------------------------------------------------------------------------


class TestMaybeRewriteLazyLoadBug:
    """Regression test for the is_available()-as-gate bug.

    Before the fix: ``maybe_rewrite`` short-circuits on the FIRST call because
    ``is_available()`` returns False until ``check_visibility`` triggers
    ``_load_model`` — but ``check_visibility`` is never reached because of the
    short-circuit.  The result: lazy-load never happens, rewrite never fires.

    After the fix: ``maybe_rewrite`` does NOT call ``is_available()`` as a gate.
    It calls ``check_visibility`` directly, which triggers the lazy load.
    """

    def test_first_call_does_not_short_circuit_on_is_available(self):
        from methods.reward_forcing.teleport.vlm import QwenTeleportVLM

        # Use a real QwenTeleportVLM (not a Mock) so we exercise the
        # lazy-load contract.  Replace _call_vlm to avoid actually loading.
        vlm = QwenTeleportVLM(model_path="/tmp/fake", device="cpu")
        # Sanity: pre-call, expander is None and is_available is False.
        assert vlm._expander is None
        assert vlm.is_available() is False

        # Patch _call_vlm so check_visibility can run end-to-end.
        # We do NOT patch is_available — that's exactly what we're
        # regression-testing.
        calls = []

        def fake_call_vlm_analyze(self, frame_rgb, prompt_hint):
            calls.append(("call_vlm_analyze", prompt_hint))
            return ('{"in_frame": ["dog"], '
                    '"visibility": {"visible": ["dog"], "absent": [], "partial": []}}')

        # Also patch _load_model: we want to verify it WAS attempted,
        # but we don't want the real Qwen load.
        load_attempts = []

        def fake_load_model(self):
            load_attempts.append("load")
            # Simulate successful load by setting _expander to a sentinel.
            self._expander = object()

        rewriter = _make_mock_rewriter()
        rewriter.rewrite.return_value = "REWRITTEN_PROMPT"
        text_enc = _make_mock_text_encoder()
        frame_dec = _make_mock_frame_decoder()

        with patch.object(QwenTeleportVLM, "_call_vlm_analyze", fake_call_vlm_analyze), \
             patch.object(QwenTeleportVLM, "_load_model", fake_load_model):
            hook = TeleportSwitchHook(
                vlm=vlm,
                rewriter=rewriter,
                text_encoder_fn=text_enc,
                frame_decoder_fn=frame_dec,
                entity_extractor_fn=lambda p: ["dog"],
            )
            output_latent = torch.zeros(1, 21, 16, 40, 40)
            hook.maybe_rewrite(output_latent, 21, ["A dog runs"])

        # The lazy-load MUST have been attempted on the first call.
        assert len(load_attempts) == 1, (
            f"_load_model was NOT called on first maybe_rewrite — "
            f"this is the is_available()-as-gate bug. "
            f"load_attempts={load_attempts}"
        )
        # check_visibility's underlying _call_vlm MUST have run.
        assert len(calls) == 1, (
            f"_call_vlm was NOT called — short-circuit bug. calls={calls}"
        )
        # text_encoder MUST have received the REWRITTEN prompt, not the
        # original.
        encoded = text_enc.call_args[0][0]
        assert encoded[0] == "REWRITTEN_PROMPT", (
            f"text_encoder received original prompt instead of rewritten — "
            f"hook short-circuited. encoded={encoded!r}"
        )


# ---------------------------------------------------------------------------
# 9. test_trigger_field
# ---------------------------------------------------------------------------


class TestTriggerField:
    """trigger='switch'/'chunk' parsing and per_chunk property."""

    def test_default_trigger_is_switch(self):
        """Default trigger='switch' → per_chunk False."""
        hook = _make_hook()
        assert hook.per_chunk is False

    def test_trigger_chunk_sets_per_chunk_true(self):
        """trigger='chunk' → per_chunk True."""
        hook = TeleportSwitchHook(
            vlm=_make_mock_vlm(),
            rewriter=_make_mock_rewriter(),
            text_encoder_fn=_make_mock_text_encoder(),
            frame_decoder_fn=_make_mock_frame_decoder(),
            trigger="chunk",
        )
        assert hook.per_chunk is True

    def test_trigger_switch_explicit(self):
        """trigger='switch' → per_chunk False."""
        hook = TeleportSwitchHook(
            vlm=_make_mock_vlm(),
            rewriter=_make_mock_rewriter(),
            text_encoder_fn=_make_mock_text_encoder(),
            frame_decoder_fn=_make_mock_frame_decoder(),
            trigger="switch",
        )
        assert hook.per_chunk is False

    def test_trigger_unknown_raises(self):
        """Unknown trigger → ValueError."""
        with pytest.raises(ValueError, match="Unknown trigger"):
            TeleportSwitchHook(
                vlm=_make_mock_vlm(),
                rewriter=_make_mock_rewriter(),
                text_encoder_fn=_make_mock_text_encoder(),
                frame_decoder_fn=_make_mock_frame_decoder(),
                trigger="every_frame",
            )

    def test_factory_propagates_trigger_chunk(self):
        """build_teleport_switch_hook reads trigger from cfg and propagates."""
        cfg = {
            "enabled": True,
            "trigger": "chunk",
            "vlm": {"type": "qwen_vl_3b", "model_path": "/tmp/fake"},
            "rewriter": {"mode": "conservative"},
        }
        hook = build_teleport_switch_hook(cfg, Mock(), Mock())
        assert hook is not None
        assert hook.per_chunk is True

    def test_factory_default_trigger_switch_backward_compat(self):
        """build_teleport_switch_hook without trigger field defaults to switch."""
        cfg = {
            "enabled": True,
            "vlm": {"type": "qwen_vl_3b", "model_path": "/tmp/fake"},
            "rewriter": {"mode": "conservative"},
        }
        hook = build_teleport_switch_hook(cfg, Mock(), Mock())
        assert hook is not None
        assert hook.per_chunk is False


# ---------------------------------------------------------------------------
# 10. test_entity_source_union
# ---------------------------------------------------------------------------


class TestEntitySourceUnion:
    """analyze() is called once with prompt-hint; rewrite uses returned visibility."""

    def test_analyze_called_with_prompt_hint(self):
        text_enc = _make_mock_text_encoder()
        vlm = _make_mock_vlm()
        vlm.analyze.return_value = {
            "in_frame": ["car"],
            "visibility": {
                "visible": ["car"],
                "absent": ["dog"],
                "partial": [],
            },
        }
        rewriter = _make_mock_rewriter()
        rewriter.rewrite.return_value = "REWRITTEN"

        hook = _make_hook(
            vlm=vlm, rewriter=rewriter, text_encoder_fn=text_enc,
            entity_extractor_fn=lambda p: ["dog"],  # naive prompt hint
        )
        output_latent = torch.zeros(1, 21, 16, 40, 40)
        hook.maybe_rewrite(output_latent, 21, ["A dog runs"])

        # analyze was called exactly once with prompt hint
        vlm.analyze.assert_called_once()
        call_args = vlm.analyze.call_args[0]
        assert call_args[1] == ["dog"]  # prompt_hint
        # check_visibility must NOT be called (we're using analyze)
        vlm.check_visibility.assert_not_called()
        # rewriter received the visibility from analyze
        rewriter.rewrite.assert_called_once()
        vis_arg = rewriter.rewrite.call_args[0][1]
        assert vis_arg == {
            "visible": ["car"],
            "absent": ["dog"],
            "partial": [],
        }

    def test_empty_universe_skips_rewrite(self):
        """When in_frame=[] AND prompt_hint=[] → skip rewrite, use original."""
        text_enc = _make_mock_text_encoder()
        vlm = _make_mock_vlm()
        vlm.analyze.return_value = {
            "in_frame": [],
            "visibility": {"visible": [], "absent": [], "partial": []},
        }
        rewriter = _make_mock_rewriter()

        hook = _make_hook(
            vlm=vlm, rewriter=rewriter, text_encoder_fn=text_enc,
            entity_extractor_fn=lambda p: [],  # empty prompt hint
        )
        output_latent = torch.zeros(1, 21, 16, 40, 40)
        hook.maybe_rewrite(output_latent, 21, ["A blank scene"])

        # analyze was called (frame was decoded), but rewriter was NOT
        vlm.analyze.assert_called_once()
        rewriter.rewrite.assert_not_called()
        # text_encoder was called with original prompts
        text_enc.assert_called_once_with(["A blank scene"])

    def test_frame_entities_rescue_empty_prompt_hint(self):
        """prompt_hint=[] but in_frame=[X] → still rewrite using frame entities."""
        text_enc = _make_mock_text_encoder()
        vlm = _make_mock_vlm()
        vlm.analyze.return_value = {
            "in_frame": ["dog"],
            "visibility": {"visible": ["dog"], "absent": [], "partial": []},
        }
        rewriter = _make_mock_rewriter()
        rewriter.rewrite.return_value = "REWRITTEN_FROM_FRAME"

        hook = _make_hook(
            vlm=vlm, rewriter=rewriter, text_encoder_fn=text_enc,
            entity_extractor_fn=lambda p: [],  # empty prompt hint
        )
        output_latent = torch.zeros(1, 21, 16, 40, 40)
        hook.maybe_rewrite(output_latent, 21, ["???"])

        # rewriter IS called because in_frame is non-empty
        rewriter.rewrite.assert_called_once()
        text_enc.assert_called_once()
        encoded = text_enc.call_args[0][0]
        assert encoded[0] == "REWRITTEN_FROM_FRAME"


# ---------------------------------------------------------------------------
# 11. TestRegistryIntegration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    """maybe_rewrite uses registry to persist absent across chunks."""

    def test_registry_locks_after_visible_to_absent(self):
        from methods.reward_forcing.teleport.registry import EntityRegistry
        registry = EntityRegistry()

        text_enc = _make_mock_text_encoder()
        vlm = _make_mock_vlm()
        rewriter = _make_mock_rewriter()
        rewriter.rewrite.side_effect = lambda p, v: f"REWRITTEN_absent={v['absent']}"

        hook = TeleportSwitchHook(
            vlm=vlm,
            rewriter=rewriter,
            text_encoder_fn=text_enc,
            frame_decoder_fn=_make_mock_frame_decoder(),
            entity_extractor_fn=lambda p: ["dog"],
            entity_registry=registry,
        )
        output_latent = torch.zeros(1, 21, 16, 40, 40)

        # Verify hook was constructed with the injected registry
        assert hook._registry is not None, "hook._registry is None!"
        assert hook._registry is registry, (
            f"hook._registry is not the injected registry: "
            f"{type(hook._registry)} vs {type(registry)}"
        )

        # Chunk 1: VLM sees dog
        vlm.analyze.return_value = {
            "in_frame": ["dog"],
            "visibility": {"visible": ["dog"], "absent": [], "partial": []},
        }
        hook.maybe_rewrite(output_latent, 21, ["A dog runs"])
        assert registry.snapshot()["dog"] == "visible"

        # Chunk 2: VLM says dog absent → locked
        vlm.analyze.return_value = {
            "in_frame": [],
            "visibility": {"visible": [], "absent": ["dog"], "partial": []},
        }
        hook.maybe_rewrite(output_latent, 42, ["A dog runs"])
        assert registry.snapshot()["dog"] == "absent"
        # rewriter should have been called with absent=['dog']
        last_call_v = rewriter.rewrite.call_args[0][1]
        assert last_call_v["absent"] == ["dog"]

        # Chunk 3: VLM says dog visible again — registry must keep absent
        vlm.analyze.return_value = {
            "in_frame": ["dog"],
            "visibility": {"visible": ["dog"], "absent": [], "partial": []},
        }
        hook.maybe_rewrite(output_latent, 63, ["A dog runs"])
        assert registry.snapshot()["dog"] == "absent"
        last_call_v = rewriter.rewrite.call_args[0][1]
        assert last_call_v["absent"] == ["dog"]  # still suppressed

    def test_reset_clears_registry(self):
        from methods.reward_forcing.teleport.registry import EntityRegistry
        registry = EntityRegistry()

        vlm = _make_mock_vlm()
        rewriter = _make_mock_rewriter()
        hook = TeleportSwitchHook(
            vlm=vlm,
            rewriter=rewriter,
            text_encoder_fn=_make_mock_text_encoder(),
            frame_decoder_fn=_make_mock_frame_decoder(),
            entity_extractor_fn=lambda p: ["dog"],
            entity_registry=registry,
        )
        vlm.analyze.return_value = {
            "in_frame": ["dog"],
            "visibility": {"visible": ["dog"], "absent": [], "partial": []},
        }
        output_latent = torch.zeros(1, 21, 16, 40, 40)
        hook.maybe_rewrite(output_latent, 21, ["A dog runs"])
        assert len(registry) == 1

        hook.reset()
        assert len(registry) == 0

    def test_registry_default_constructed(self):
        """When entity_registry=None (default), hook builds its own."""
        hook = _make_hook(with_registry=True)
        # By default a registry is auto-constructed
        assert hook._registry is not None

    def test_registry_disabled_via_factory_flag(self):
        """build_teleport_switch_hook with registry_enabled=False → no registry."""
        cfg = {
            "enabled": True,
            "registry_enabled": False,
            "vlm": {"type": "qwen_vl_3b", "model_path": "/tmp/fake"},
            "rewriter": {"mode": "conservative"},
        }
        hook = build_teleport_switch_hook(cfg, Mock(), Mock())
        assert hook is not None
        assert hook._registry is None

    def test_open_vocab_in_frame_not_tracked(self):
        """VLM sees 'tree' but it's not in prompt_hint → not in registry."""
        from methods.reward_forcing.teleport.registry import EntityRegistry
        registry = EntityRegistry()
        vlm = _make_mock_vlm()
        vlm.analyze.return_value = {
            "in_frame": ["dog", "tree"],
            "visibility": {
                "visible": ["dog", "tree"], "absent": [], "partial": [],
            },
        }
        hook = TeleportSwitchHook(
            vlm=vlm,
            rewriter=_make_mock_rewriter(),
            text_encoder_fn=_make_mock_text_encoder(),
            frame_decoder_fn=_make_mock_frame_decoder(),
            entity_extractor_fn=lambda p: ["dog"],  # prompt only mentions dog
            entity_registry=registry,
        )
        output_latent = torch.zeros(1, 21, 16, 40, 40)
        hook.maybe_rewrite(output_latent, 21, ["A dog runs"])
        # tree should not enter registry
        assert "tree" not in registry.snapshot()
        assert registry.snapshot() == {"dog": "visible"}

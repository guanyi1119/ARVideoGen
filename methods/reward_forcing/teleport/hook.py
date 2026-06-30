"""
TeleportSwitchHook: coordinates VLM visibility classification + PromptRewriter
rewriting + text encoding at the prompt-switch boundary during inference.

Pluggable into ``SwitchCausalInferencePipeline`` — injected in ``__init__``,
invoked in ``inference()`` at the switch point.  When disabled the pipeline
behaves identically to the unmodified version (bit-equivalent).
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

import torch

from .rewriter import PromptRewriter, build_prompt_rewriter
from .vlm import TeleportVLM, build_teleport_vlm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ConditionalDict = Dict  # opaque dict returned by text_encoder(text_prompts=...)


# ---------------------------------------------------------------------------
# Default entity extractor (placeholder — replaced by T16 / C-light)
# ---------------------------------------------------------------------------


def _default_entity_extract(prompt: str) -> List[str]:
    """Placeholder entity extraction: simple space-tokenisation + noun candidate
    heuristic.

    **In production the caller should inject ``entity_extractor_fn``** (T16 /
    Stage C-light provides the real implementation).  This default exists so
    integration tests can run without an external NLP pipeline.

    Heuristic: returns every space-delimited token of length ≥ 3, lower-cased
    and de-duplicated.
    """
    tokens = prompt.split()
    candidates = [t.lower().strip(".,!?;:\"'()[]{}") for t in tokens]
    seen: set[str] = set()
    result: List[str] = []
    for c in candidates:
        if len(c) >= 3 and c not in seen:
            seen.add(c)
            result.append(c)
    return result


# ---------------------------------------------------------------------------
# TeleportSwitchHook
# ---------------------------------------------------------------------------


class TeleportSwitchHook:
    """Glue class coordinating VLM → rewriter → text-encoder at switch time.

    Constructed without strong references to ``text_encoder`` / ``vae`` —
    instead receives **callables** (``text_encoder_fn``, ``frame_decoder_fn``)
    that the upstream pipeline injects, avoiding any dependency on the
    specific wrapper classes.

    Parameters:
        vlm:
            ``TeleportVLM`` instance for visibility classification.
        rewriter:
            ``PromptRewriter`` instance for suffix-based prompt rewriting.
        text_encoder_fn:
            ``Callable(List[str]) -> conditional_dict`` — injected by the
            pipeline (e.g. ``self.text_encoder(text_prompts=prompts)``).
        frame_decoder_fn:
            ``Callable(latent_BFCHW) -> rgb_BTCHW`` in **[0, 1]** — injected
            by the pipeline (decodes a VAE latent to pixel space).
        entity_extractor_fn:
            Optional ``Callable(str) -> List[str]``.  When ``None`` (default)
            the built-in ``_default_entity_extract`` is used — a placeholder
            that should be replaced by the Stage C-light extractor (T16).
    """

    def __init__(
        self,
        vlm: TeleportVLM,
        rewriter: PromptRewriter,
        text_encoder_fn: Callable[[List[str]], ConditionalDict],
        frame_decoder_fn: Callable[[torch.Tensor], torch.Tensor],
        entity_extractor_fn: Optional[Callable[[str], List[str]]] = None,
        trigger: str = "switch",
    ) -> None:
        if trigger not in ("switch", "chunk"):
            raise ValueError(
                f"Unknown trigger {trigger!r}. "
                "Supported: ['switch', 'chunk']"
            )
        self._trigger = trigger
        self.vlm = vlm
        self.rewriter = rewriter
        self.text_encoder_fn = text_encoder_fn
        self.frame_decoder_fn = frame_decoder_fn
        self.entity_extractor_fn = entity_extractor_fn or _default_entity_extract

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def per_chunk(self) -> bool:
        """Whether maybe_rewrite should be invoked on every chunk boundary
        (in addition to the switch boundary).

        True when ``trigger == "chunk"``; False when ``trigger == "switch"``.
        """
        return self._trigger == "chunk"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def maybe_rewrite(
        self,
        output_latent: torch.Tensor,  # [B, F_so_far, C, H, W]
        current_start_frame: int,
        text_prompts_second: List[str],
    ) -> ConditionalDict:
        """Called once at the switch boundary: extract last frame → VLM →
        rewriter → re-encode.

        Parameters:
            output_latent:
                Accumulated output frames so far (VAE latent space),
                shape ``[B, F_so_far, C, H, W]``.
            current_start_frame:
                Current frame index in the autoregressive loop.
            text_prompts_second:
                Original next-segment prompts (before rewriting).

        Returns:
            ``conditional_dict`` for the rewritten (or original) prompt.

        **Never raises** — any internal error degrades gracefully: the
        original ``text_prompts_second`` are re-encoded and returned.
        """
        try:
            return self._maybe_rewrite_impl(
                output_latent, current_start_frame, text_prompts_second
            )
        except Exception:
            logger.warning(
                "TeleportSwitchHook: maybe_rewrite failed, falling back to "
                "original prompt encoding.",
                exc_info=True,
            )
            return self.text_encoder_fn(text_prompts_second)

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _maybe_rewrite_impl(
        self,
        output_latent: torch.Tensor,
        current_start_frame: int,
        text_prompts_second: List[str],
    ) -> ConditionalDict:
        """Guarded implementation — all tensor ops run under ``torch.no_grad()``.

        Note: we deliberately do NOT pre-check ``self.vlm.is_available()`` here
        because the VLM is lazy-loaded on the first ``check_visibility`` call.
        If the load fails (e.g. model file missing), ``check_visibility`` raises
        and the outer ``maybe_rewrite`` ``try/except`` falls back to encoding the
        original prompt.
        """

        # 1. No history yet (first frame block) → nothing to analyse
        if current_start_frame <= 0:
            logger.debug(
                "TeleportSwitchHook: current_start_frame=%d, no history to "
                "analyse, skipping rewrite.",
                current_start_frame,
            )
            return self.text_encoder_fn(text_prompts_second)

        # 2. Extract the last frame from accumulated latents
        #    output_latent: [B, F_so_far, C, H, W]
        #    Take frame at index current_start_frame-1 (single frame)
        single_frame_latent = output_latent[
            :, current_start_frame - 1 : current_start_frame
        ]  # [B, 1, C, H, W]

        # Decode to pixel space [B, 1, 3, H', W'] in [0, 1]
        pixels = self.frame_decoder_fn(single_frame_latent)
        # Take first batch item and first frame → [3, H', W']
        frame_rgb = pixels[0, 0]  # [3, H', W']

        # 3. Extract entities from the prompt
        prompt_str = text_prompts_second[0]
        entities = self.entity_extractor_fn(prompt_str)

        if not entities:
            logger.debug(
                "TeleportSwitchHook: no entities extracted from prompt, "
                "skipping rewrite."
            )
            return self.text_encoder_fn(text_prompts_second)

        # 4. VLM visibility check
        visibility = self.vlm.check_visibility(frame_rgb, entities)

        # 5. Rewrite prompt
        rewritten = self.rewriter.rewrite(prompt_str, visibility)

        # 6. Re-encode (broadcast to batch)
        return self.text_encoder_fn([rewritten] * len(text_prompts_second))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_teleport_switch_hook(
    cfg: Optional[dict],
    text_encoder_fn: Callable[[List[str]], ConditionalDict],
    frame_decoder_fn: Callable[[torch.Tensor], torch.Tensor],
) -> Optional[TeleportSwitchHook]:
    """Construct a ``TeleportSwitchHook`` from a config dict.

    Parameters:
        cfg:
            Config dict with keys:
            - ``enabled`` (bool): ``False`` → return ``None``.
            - ``vlm`` (dict): forwarded to ``build_teleport_vlm``.
            - ``rewriter`` (dict): forwarded to ``build_prompt_rewriter``.
            - ``entity_extractor`` (str, optional): reserved for T16.

            When *cfg* is ``None`` or ``{"enabled": False}``, returns ``None``.
        text_encoder_fn:
            Callable for text encoding (injected by pipeline).
        frame_decoder_fn:
            Callable for VAE latent → pixel decoding (injected by pipeline).

    Returns:
        ``TeleportSwitchHook`` instance, or ``None`` if any sub-component is
        unavailable.
    """
    if cfg is None or not cfg.get("enabled", False):
        return None

    vlm = build_teleport_vlm(cfg.get("vlm"))
    rewriter = build_prompt_rewriter(cfg.get("rewriter"))

    if vlm is None or rewriter is None:
        logger.warning(
            "TeleportSwitchHook: vlm=%s rewriter=%s — at least one is "
            "None, returning None (disabled).",
            vlm,
            rewriter,
        )
        return None

    trigger = cfg.get("trigger", "switch")

    return TeleportSwitchHook(
        vlm=vlm,
        rewriter=rewriter,
        text_encoder_fn=text_encoder_fn,
        frame_decoder_fn=frame_decoder_fn,
        entity_extractor_fn=None,  # placeholder; T16 provides real impl
        trigger=trigger,
    )

"""
TeleportVLM: abstract interface + Qwen VL implementation for entity visibility
classification in generated video frames.

Used by the teleport-mitigation pipeline (prompt_rewrite / Stage C-light) to detect
whether named entities are visible, absent, or partially visible in a frame.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class TeleportVLM(ABC):
    """Abstract interface for frame-level entity visibility classification.

    All implementations must accept a single RGB frame tensor and a list of
    entity names, and return a three-way classification dict.

    Frame tensor convention (documented, not enforced):
        ``frame_rgb`` shape ``[3, H, W]`` or ``[1, 3, H, W]``, values in **[0, 1]**.

    Subclasses implement the actual VLM backend (local, cloud, etc.).
    This interface is designed to be **pluggable** — callers should only depend
    on ``TeleportVLM``, never on a concrete subclass.
    """

    @abstractmethod
    def check_visibility(
        self, frame_rgb: torch.Tensor, entity_list: List[str]
    ) -> Dict[str, List[str]]:
        """Return a three-way classification for every entity in *entity_list*.

        Returns:
            ``{"visible": [...], "absent": [...], "partial": [...]}``

            - Every entity in *entity_list* appears in **exactly one** list.
            - Entities NOT in *entity_list* (hallucinated by the VLM) are **filtered out**.
            - If parsing fails for any reason the method returns three empty lists
              (fail-safe, never raises).
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Whether the underlying VLM model has been successfully loaded."""
        ...


class QwenTeleportVLM(TeleportVLM):
    """TeleportVLM backed by a local Qwen2.5-VL model via ``QwenPromptExpander``.

    The model is **lazy-loaded** on the first call to ``check_visibility`` so
    that construction is cheap and the factory never triggers a GPU allocation.

    Parameters:
        model_path:
            Path or HuggingFace model id understood by ``QwenPromptExpander``.
        device:
            PyTorch device string (e.g. ``"cuda"``, ``"cuda:0"``, ``"cpu"``).
        max_new_tokens:
            Maximum tokens the VLM may generate per request.
    """

    # ------------------------------------------------------------------
    # Fixed system prompt — never change without updating the plan.
    # ------------------------------------------------------------------
    _SYSTEM_PROMPT: str = (
        "You are an object visibility classifier. Look at the image and the "
        "entity list provided in the user message. For each entity, decide "
        'whether it is:\n'
        '- "visible": clearly visible in the image\n'
        '- "absent": not visible at all\n'
        '- "partial": only partially visible (cropped at the edge, heavily '
        "occluded)\n\n"
        "Respond with ONLY a JSON object matching this exact schema, with no "
        "additional text, no markdown, no code fences:\n"
        '{"visible": [<entity names>], "absent": [<entity names>], '
        '"partial": [<entity names>]}\n\n'
        "Every entity from the list must appear in exactly one of the three "
        "arrays. Do not invent entities that are not in the list."
    )

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_new_tokens: int = 256,
    ) -> None:
        self._model_path = model_path
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._expander: Optional[object] = None  # QwenPromptExpander, lazy

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Instantiate the underlying QwenPromptExpander (lazy, called once)."""
        if self._expander is not None:
            return
        # Import locally so the module is importable even when
        # qwen-vl dependencies are absent (e.g. CPU-only CI).
        from wan.utils.prompt_extend import QwenPromptExpander

        logger.info(
            "QwenTeleportVLM: loading model from %s (device=%s)",
            self._model_path,
            self._device,
        )
        self._expander = QwenPromptExpander(
            model_name=self._model_path,
            is_vl=True,
            device=self._device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return self._expander is not None

    def check_visibility(
        self, frame_rgb: torch.Tensor, entity_list: List[str]
    ) -> Dict[str, List[str]]:
        """See ``TeleportVLM.check_visibility``."""
        if not entity_list:
            return {"visible": [], "absent": [], "partial": []}

        self._load_model()

        raw_response = self._call_vlm(frame_rgb, entity_list)
        return self._parse_response(raw_response, entity_list)

    # ------------------------------------------------------------------
    # VLM call (testable mock point)
    # ------------------------------------------------------------------

    def _call_vlm(
        self, frame_rgb: torch.Tensor, entity_list: List[str]
    ) -> str:
        """Convert frame → PIL, construct prompt, invoke VLM, return raw text.

        This is the **primary mock point** for unit tests — monkey-patch this
        method to avoid loading the real model.
        """
        pil_image = self._tensor_to_pil(frame_rgb)

        user_prompt = f"Entity list: {entity_list}. Classify each."

        result = self._expander.extend_with_img(
            prompt=user_prompt,
            system_prompt=self._SYSTEM_PROMPT,
            image=pil_image,
        )
        return result.prompt  # raw string from the VLM

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """Convert a [3,H,W] or [1,3,H,W] float [0,1] tensor to PIL RGB."""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        # CHW → HWC, [0,1] → [0,255]
        arr = (tensor.permute(1, 2, 0).clamp(0.0, 1.0) * 255.0).to(
            torch.uint8
        )
        return Image.fromarray(arr.cpu().numpy(), mode="RGB")

    @staticmethod
    def _parse_response(
        raw_response: str, entity_list: List[str]
    ) -> Dict[str, List[str]]:
        """Fail-safe JSON parser for VLM responses.

        Strategy:
        1. Extract the first ``{...}`` block via regex.
        2. ``json.loads`` — if it fails → return empty safe dict.
        3. Validate schema (three list keys) — if invalid → empty safe dict.
        4. Filter out-of-vocab entities (any entity not in *entity_list*).
        """
        empty = {"visible": [], "absent": [], "partial": []}

        # --- Step 1: extract JSON block ---
        match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if not match:
            logger.warning(
                "TeleportVLM: no JSON block found in response: %s",
                raw_response[:200],
            )
            return empty

        json_str = match.group(0)

        # --- Step 2: parse ---
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(
                "TeleportVLM: JSON decode failed for: %s", json_str[:200]
            )
            return empty

        # --- Step 3: schema validation ---
        if not isinstance(parsed, dict):
            logger.warning(
                "TeleportVLM: parsed JSON is not a dict: %s", type(parsed)
            )
            return empty

        for key in ("visible", "absent", "partial"):
            if key not in parsed or not isinstance(parsed[key], list):
                logger.warning(
                    "TeleportVLM: missing or non-list key '%s' in: %s",
                    key,
                    json_str[:200],
                )
                return empty

        # --- Step 4: out-of-vocab filtering ---
        entity_set = set(entity_list)
        filtered: Dict[str, List[str]] = {}
        for key in ("visible", "absent", "partial"):
            raw_list: list = parsed[key]
            clean = [e for e in raw_list if e in entity_set]
            removed = [e for e in raw_list if e not in entity_set]
            if removed:
                logger.warning(
                    "TeleportVLM: filtered out-of-vocab entities from '%s': %s",
                    key,
                    removed,
                )
            filtered[key] = clean

        return filtered


def build_teleport_vlm(cfg: Optional[dict]) -> Optional[TeleportVLM]:
    """Factory: create a ``TeleportVLM`` instance from a config dict.

    Parameters:
        cfg:
            Dict with keys ``type`` (str), ``model_path`` (str), and optionally
            ``device`` (str) / ``max_new_tokens`` (int).

            Special values that return ``None``:
            - ``cfg is None``
            - ``cfg == {}`` (empty dict)
            - ``cfg["type"] == "disabled"``

    Returns:
        A concrete ``TeleportVLM`` instance, or ``None`` when disabled.

    Raises:
        ValueError: If ``cfg["type"]`` is not recognized.
    """
    if cfg is None or cfg == {}:
        return None

    vlm_type = cfg.get("type", "disabled")
    if vlm_type == "disabled":
        return None

    if vlm_type in ("qwen_vl_3b", "qwen_vl"):
        return QwenTeleportVLM(
            model_path=cfg["model_path"],
            device=cfg.get("device", "cuda"),
            max_new_tokens=cfg.get("max_new_tokens", 256),
        )

    raise ValueError(
        f"Unknown teleport.vlm.type: {vlm_type!r}. "
        "Supported: ['qwen_vl_3b', 'qwen_vl', 'disabled']"
    )

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
from typing import Any, Dict, List, Optional

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
    def analyze(
        self,
        frame_rgb: torch.Tensor,
        prompt_hint_entities: List[str],
    ) -> Dict[str, Any]:
        """Open-vocabulary entity listing + closed-set visibility classification
        in a SINGLE VLM call.

        Parameters:
            frame_rgb: shape [3, H, W] or [1, 3, H, W], values in [0, 1].
            prompt_hint_entities:
                Entities mentioned in the prompt the model should ADDITIONALLY
                classify (even if absent from the frame).  The VLM is told
                to cover the **union** of in_frame + prompt_hint.

        Returns:
            {
                "in_frame": List[str],                # what VLM saw (open-vocab)
                "visibility": {
                    "visible": List[str],
                    "absent":  List[str],
                    "partial": List[str],
                },
            }
            - in_frame ⊆ visibility["visible"] ∪ visibility["partial"]
              (semantic contract; entities filtered out of vocab will not violate)
            - keys of visibility cover the UNION of in_frame and prompt_hint_entities
            - Out-of-vocab handling: in_frame is unconstrained (open-vocab);
              visibility entities NOT in (in_frame ∪ prompt_hint_entities) are
              filtered (same policy as check_visibility).
            - Fail-safe: on any parse failure returns
              ``{"in_frame": [], "visibility": {"visible":[], "absent":[], "partial":[]}}``.
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

    _ANALYZE_SYSTEM_PROMPT: str = (
        "You are an open-vocabulary object analyzer.  Do TWO things in ONE response:\n\n"
        "STEP 1 — List every distinct, concrete entity (object, animal, person, "
        "vehicle, named scene element) that you can SEE in the image.  Use short "
        "common nouns (e.g. 'dog', 'red car'), no articles, lower-case.  Ignore "
        "abstract attributes and background style.\n\n"
        "STEP 2 — The user message includes a list of entities mentioned in the "
        "next caption (prompt_hint).  Considering the UNION of what you saw "
        "(STEP 1) AND prompt_hint, classify each entity as:\n"
        "  - 'visible': clearly visible in the image\n"
        "  - 'absent':  not visible at all\n"
        "  - 'partial': only partially visible (cropped at the edge, heavily occluded)\n\n"
        "Respond with ONLY a JSON object matching this exact schema, with no "
        "additional text, no markdown, no code fences:\n"
        '{"in_frame": [<entity names>], '
        '"visibility": {"visible": [...], "absent": [...], "partial": [...]}}\n\n'
        "Every entity in the UNION (STEP 1 result ∪ prompt_hint) must appear "
        "in exactly one of the three visibility arrays.  Do not invent entities "
        "outside the union."
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

    def analyze(
        self,
        frame_rgb: torch.Tensor,
        prompt_hint_entities: List[str],
    ) -> Dict[str, Any]:
        """See ``TeleportVLM.analyze``."""
        self._load_model()

        from ._debug import tdprint, is_teleport_debug

        if is_teleport_debug():
            tdprint(
                f"analyze() prompt_hint={prompt_hint_entities!r} "
                f"frame_shape={tuple(frame_rgb.shape)} "
                f"frame_minmax=({frame_rgb.min().item():.3f}, "
                f"{frame_rgb.max().item():.3f})"
            )

        raw = self._call_vlm_analyze(frame_rgb, prompt_hint_entities)

        if is_teleport_debug():
            # 截断超长输出，避免刷屏
            truncated = (
                raw if len(raw) <= 800 else raw[:800] + f"... [{len(raw) - 800} more chars]"
            )
            tdprint(f"analyze() raw_response={truncated!r}")

        parsed = self._parse_analyze_response(raw, prompt_hint_entities)

        if is_teleport_debug():
            tdprint(f"analyze() parsed={parsed!r}")

        return parsed

    # ------------------------------------------------------------------
    # VLM call (testable mock points)
    # ------------------------------------------------------------------

    def _call_vlm_analyze(
        self,
        frame_rgb: torch.Tensor,
        prompt_hint_entities: List[str],
    ) -> str:
        """Convert frame → PIL, call VLM with analyze system prompt, return raw text.

        This is the **primary monkey-patch point** for unit tests of analyze().
        """
        pil_image = self._tensor_to_pil(frame_rgb)
        user_prompt = (
            f"prompt_hint: {prompt_hint_entities}. "
            "Run STEP 1 (in_frame open-vocab listing) and STEP 2 "
            "(visibility classification over in_frame ∪ prompt_hint)."
        )
        result = self._expander.extend_with_img(
            prompt=user_prompt,
            system_prompt=self._ANALYZE_SYSTEM_PROMPT,
            image=pil_image,
        )
        return result.prompt

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

    @staticmethod
    def _parse_analyze_response(
        raw_response: str,
        prompt_hint_entities: List[str],
    ) -> Dict[str, Any]:
        """Fail-safe parser for analyze() responses.

        Strategy mirrors _parse_response:
          1. Extract first {...} block
          2. json.loads with try/except → fail-safe empty
          3. Schema validation: top dict has 'in_frame' (list) and 'visibility'
             (dict with 'visible'/'absent'/'partial' lists)
          4. in_frame is open-vocab → no filtering
          5. visibility entities filtered to union(in_frame ∪ prompt_hint_entities)
          6. logging.warning for any out-of-vocab filtering
        """
        empty: Dict[str, Any] = {
            "in_frame": [],
            "visibility": {"visible": [], "absent": [], "partial": []},
        }
        # Step 1-2: extract + parse
        match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if not match:
            logger.warning(
                "TeleportVLM.analyze: no JSON block in response: %s",
                raw_response[:200],
            )
            return empty
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning(
                "TeleportVLM.analyze: JSON decode failed for: %s",
                match.group(0)[:200],
            )
            return empty

        # Step 3: schema validation
        if not isinstance(parsed, dict):
            return empty
        if not isinstance(parsed.get("in_frame"), list):
            logger.warning("TeleportVLM.analyze: missing/invalid 'in_frame'")
            return empty
        vis = parsed.get("visibility")
        if not isinstance(vis, dict):
            logger.warning("TeleportVLM.analyze: missing/invalid 'visibility'")
            return empty
        for key in ("visible", "absent", "partial"):
            if not isinstance(vis.get(key), list):
                logger.warning(
                    "TeleportVLM.analyze: missing/invalid visibility key '%s'", key
                )
                return empty

        # Step 4: in_frame open-vocab, but enforce string + non-empty
        in_frame = [
            str(e) for e in parsed["in_frame"] if isinstance(e, str) and e.strip()
        ]

        # Step 5: visibility filtering — union of in_frame ∪ prompt_hint
        universe = set(in_frame) | set(prompt_hint_entities)
        filtered_vis: Dict[str, List[str]] = {}
        for key in ("visible", "absent", "partial"):
            raw_list = vis[key]
            clean = [e for e in raw_list if isinstance(e, str) and e in universe]
            removed = [
                e for e in raw_list if not isinstance(e, str) or e not in universe
            ]
            if removed:
                logger.warning(
                    "TeleportVLM.analyze: filtered out-of-union entities "
                    "from '%s': %s",
                    key,
                    removed,
                )
            filtered_vis[key] = clean

        return {"in_frame": in_frame, "visibility": filtered_vis}


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

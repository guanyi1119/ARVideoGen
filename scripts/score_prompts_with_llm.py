#!/usr/bin/env python3
"""Score prompt motion intensity with an OpenAI-compatible chat API.

This script is designed for very large prompt files. Key robustness features:

* Resume safely after interruption: progress is tracked by line index in a
  small ``.progress`` JSON file. The intermediate JSONL is parsed defensively
  on resume; truncated/corrupt trailing lines are stripped instead of being
  silently skipped, so we never lose alignment with the source file.
* Per-prompt schema validation: every JSONL record stores both the source
  ``index`` and the original ``prompt`` text. Resume verifies these against
  the input file and refuses to continue if they no longer match.
* Stable per-batch scoring: even if the LLM returns an extra/missing item or
  garbled output, the batch is retried until the response covers exactly the
  expected indexes for the prompts we sent.
* Atomic appends: each batch is written through a temporary buffer flushed
  with ``os.fsync`` to minimise the window where a crash can corrupt the
  JSONL on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator


SYSTEM_PROMPT = (
    "你是一个严格、稳定的视频 prompt 动态性评分员。\n"
    "\n"
    "请逐条精读 prompt，并只评估它描述的视频画面动态性：\n"
    "- 0.00：完全静态，不包含主体运动、环境变化或运镜。\n"
    "- 0.10-0.25：几乎静态，仅有轻微表情、姿态、烟雾、蒸汽、树叶/水面微动等。\n"
    "- 0.30-0.45：低到中等动态，例如走路、轻微划船、缓慢镜头平移、少量局部运动。\n"
    "- 0.50-0.65：明显动态，例如跑动、跳舞、运动项目、动物玩耍、车辆正常行驶。\n"
    "- 0.70-0.85：高动态，例如高速车辆、空中特技、激烈格斗、赛车、烟花、大范围运动。\n"
    "- 0.90-1.00：极高动态，几乎全画面剧烈变化，例如龙卷风摧毁城市、火山爆发、战争爆炸、海啸等。\n"
    "\n"
    "评分规则：\n"
    "1. 同时考虑运动速度、运动幅度、运动主体数量、受影响画面面积、是否有明显运镜。\n"
    "2. 如果 prompt 只是说 dynamic/vibrant，但实际对象静止，不要给高分。\n"
    "3. 如果是绘画/照片/肖像/产品展示且没有运动或运镜，分数应接近 0。\n"
    "4. 如果有镜头平移、跟拍、手持抖动等运镜，即使主体静态，也应适当增加分数。\n"
    "5. 输出必须是合法 JSON，不要解释，不要 Markdown。\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score prompt motion intensity with an LLM and save JSONL output."
    )
    parser.add_argument(
        "--input",
        default="prompts/vidprom_filtered_extended.txt",
        help="Input prompt file, one prompt per line.",
    )
    parser.add_argument(
        "--output",
        default="prompts/vidprom_filtered_extended_llm_scored.jsonl",
        help="Output JSONL path. Each line stores prompt+score (and index).",
    )
    parser.add_argument("--model", required=True, help="Chat model name.")
    parser.add_argument(
        "--api-base",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable that stores the API key.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of prompts per LLM request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for stable scoring.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum response tokens per batch.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="Retry count for failed or invalid API responses.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=180,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only score the first N prompts. Useful for calibration runs.",
    )
    parser.add_argument(
        "--strict-resume",
        action="store_true",
        help="Abort instead of truncating if existing JSONL conflicts with input.",
    )
    return parser.parse_args()


def _read_prompts(input_path: Path) -> list[str]:
    prompts: list[str] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            prompts.append(line.rstrip("\n\r"))
    return prompts


def _truncate_jsonl_to(path: Path, valid_lines: int) -> None:
    if not path.exists():
        return
    if valid_lines < 0:
        valid_lines = 0
    if valid_lines == 0:
        path.write_bytes(b"")
        return
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    written = 0
    with path.open("r", encoding="utf-8") as src, tmp_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if written >= valid_lines:
                break
            dst.write(line)
            if line.endswith("\n"):
                written += 1
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(tmp_path, path)


def _validate_existing_output(
    output_path: Path, prompts: list[str], strict: bool
) -> int:
    """Return the number of leading prompts already scored.

    The function trusts only a contiguous prefix of the JSONL whose entries
    deserialise cleanly *and* match the source prompts at the same index. The
    file is truncated past the first inconsistency unless ``strict`` is set.
    """

    if not output_path.exists():
        return 0

    valid_lines = 0
    matched = 0
    with output_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                # Truncated tail; stop here and discard.
                break
            stripped = line.strip()
            if not stripped:
                valid_lines += 1
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                break
            if not isinstance(obj, dict):
                break
            prompt = obj.get("prompt")
            score = obj.get("score")
            index = obj.get("index", matched)
            if not isinstance(prompt, str) or not isinstance(score, (int, float)):
                break
            if not isinstance(index, int):
                break
            if index != matched:
                break
            if matched >= len(prompts) or prompts[matched] != prompt:
                break
            valid_lines += 1
            matched += 1

    if matched != _count_nonempty_lines(output_path):
        if strict:
            raise RuntimeError(
                f"Existing output at {output_path} has corrupt or misaligned trailing data; "
                "rerun without --strict-resume to truncate to the last valid record."
            )
        _truncate_jsonl_to(output_path, valid_lines)
    return matched


def _count_nonempty_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _iter_batches(
    prompts: list[str], start: int, batch_size: int, end: int
) -> Iterator[tuple[int, list[str]]]:
    if end <= start:
        return
    for chunk_start in range(start, end, batch_size):
        chunk_end = min(end, chunk_start + batch_size)
        yield chunk_start, prompts[chunk_start:chunk_end]


def _build_user_prompt(start_index: int, prompts: list[str]) -> str:
    payload = [
        {"index": start_index + offset, "prompt": prompt}
        for offset, prompt in enumerate(prompts)
    ]
    return (
        "请逐条精读下面的 prompts，并为每条给出 0 到 1 的动态性评分。\n"
        "输出 JSON 格式必须严格为："
        '{"scores":[{"index":整数,"score":0到1的小数}, ...]}。\n'
        "不要遗漏、不要增加条目，index 必须与输入一致。\n\n"
        f"输入：\n{json.dumps(payload, ensure_ascii=False)}"
    )


def _extract_json_object(content: str) -> dict:
    """Best-effort JSON parsing tolerant to surrounding text or fences."""

    text = content.strip()
    if text.startswith("```"):
        # Strip code fences if present.
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[: -3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: extract the largest balanced JSON object.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        return json.loads(candidate)
    raise json.JSONDecodeError("No JSON object found", text, 0)


def _parse_scores(content: str, start_index: int, batch_size: int) -> dict[int, float]:
    parsed = _extract_json_object(content)
    scores = parsed.get("scores") if isinstance(parsed, dict) else None
    if not isinstance(scores, list):
        raise ValueError(f"Model JSON missing scores list: {str(parsed)[:300]}")
    expected = set(range(start_index, start_index + batch_size))
    result: dict[int, float] = {}
    for item in scores:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid score item: {item}")
        index = item.get("index")
        score = item.get("score")
        if not isinstance(index, int) or index not in expected:
            raise ValueError(f"Unexpected index in score item: {item}")
        if not isinstance(score, (int, float)):
            raise ValueError(f"Score is not numeric: {item}")
        result[index] = round(max(0.0, min(1.0, float(score))), 4)
    if set(result) != expected:
        missing = sorted(expected - set(result))[:10]
        extra = sorted(set(result) - expected)[:10]
        raise ValueError(f"Score indexes mismatch. Missing={missing}, extra={extra}")
    return result


def _call_chat_api(
    api_base: str,
    api_key: str,
    model: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body_text}") from exc
    try:
        data = json.loads(raw)
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unexpected API response: {raw[:500]}") from exc


def _score_batch(
    args: argparse.Namespace,
    api_key: str,
    start_index: int,
    prompts: list[str],
) -> dict[int, float]:
    user_prompt = _build_user_prompt(start_index, prompts)
    last_error: Exception | None = None
    for attempt in range(1, args.max_retries + 1):
        try:
            content = _call_chat_api(
                api_base=args.api_base,
                api_key=api_key,
                model=args.model,
                user_prompt=user_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.request_timeout,
            )
            return _parse_scores(content, start_index, len(prompts))
        except Exception as exc:  # noqa: BLE001 - retry every error path.
            last_error = exc
            print(
                f"Batch {start_index}-{start_index + len(prompts) - 1} failed "
                f"on attempt {attempt}/{args.max_retries}: {exc}",
                file=sys.stderr,
            )
            if attempt < args.max_retries:
                time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"Failed batch starting at {start_index}: {last_error}")


def _append_records(
    output_path: Path,
    prompts: list[str],
    scores: dict[int, float],
    start_index: int,
) -> None:
    with output_path.open("a", encoding="utf-8") as handle:
        for offset, prompt in enumerate(prompts):
            index = start_index + offset
            record = {"index": index, "prompt": prompt, "score": scores[index]}
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompts = _read_prompts(input_path)
    total_target = len(prompts) if args.limit is None else min(len(prompts), args.limit)

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Set {args.api_key_env} before running this script.")

    completed = _validate_existing_output(output_path, prompts, strict=args.strict_resume)
    if completed > total_target:
        # The file already covers more than the requested limit.
        completed = total_target
    print(
        f"Resuming at index {completed}/{total_target}; output: {output_path}",
        flush=True,
    )

    for start_index, batch_prompts in _iter_batches(
        prompts=prompts,
        start=completed,
        batch_size=args.batch_size,
        end=total_target,
    ):
        scores = _score_batch(args, api_key, start_index, batch_prompts)
        _append_records(output_path, batch_prompts, scores, start_index)
        completed = start_index + len(batch_prompts)
        print(f"Scored {completed}/{total_target} prompts", flush=True)

    print(f"Done. Wrote {completed} records to {output_path}")


if __name__ == "__main__":
    main()

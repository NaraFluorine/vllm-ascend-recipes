#!/usr/bin/env python3
"""Translate pending YAML recipe fields with DeepSeek.

Reads the manifest produced by detect_yaml_changes.py, groups the pending
(path → English text) entries by file, and asks DeepSeek to translate each
entry into Chinese. Produces a per-file {path → zh} map that
apply_translations.py then writes into models/zh/**/*.yaml.

This is the YAML analogue of vllm-ascend's po_translate.py, but the translation
unit is a JSON path→text pair instead of a gettext msgid/msgstr pair.

Usage:
    python yaml_translate.py [--state-dir .translate] [--max-concurrent 5]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

import translate_common as tc

SYSTEM_PROMPT = (
    "You are a professional technical documentation translator specializing in "
    "translating vLLM-Ascend deployment recipe YAML fields from English to Chinese. "
    "You produce accurate, consistent translations without skipping any entry. You "
    "never add explanations, markdown fences, or extra text outside the JSON output."
)

TRANSLATION_PROMPT = """Translate these YAML recipe fields from English to Chinese.

You are given a JSON object mapping each field's path to its English text.
Return a JSON object with the SAME keys, each mapped to its Chinese translation.

This content comes from vLLM-Ascend deployment recipe YAML files (the
vllm-ascend-recipes repo). Recipes describe how to deploy ML models on Huawei
Ascend NPUs. Field values are technical documentation: short descriptions, step
titles, and multi-paragraph markdown blocks that may contain fenced code blocks,
tables, links, and Jinja-like placeholders.

CRITICAL RULES — violations will cause the output to be rejected:

--- OUTPUT FORMAT ---
1. Return ONLY a JSON object: {"<path>": "translation", ...}. No markdown code
   fences, no explanations, no prose outside the JSON.
2. The set of keys MUST EQUAL the input keys exactly — do not add, drop, rename,
   merge, or reorder keys under any circumstance.

--- WHAT TO PRESERVE VERBATIM (do NOT translate or alter) ---
3. Fenced code blocks (```...```) and inline `code`.
4. Placeholders: {max_num_seqs}, {max_model_len}, etc.
5. Config markers: %%CONFIG:key%% ... %%/CONFIG:key%%.
6. URLs, email addresses, file paths.
7. Model ids (org/name), command-line flags, shell commands, env vars, docker
   image names and version strings.
8. Brand & proper nouns: vLLM, Ascend, CANN, Atlas, ModelScope, HuggingFace,
   GitCode, Modelers, Eagle3, MTP, etc. Keep them as-is.

--- IDENTIFIER-STYLE FIELDS (weight_version, scenario_selector_labels, labels) ---
9. These values are short display names, not prose. Keep the name part verbatim —
   model/weight family names and version/quantization tags such as "GLM-4.5",
   "Qwen3.5-27B", "BF16", "W8A8", "W4A8", "Eagle3 Draft Model".
10. Translate ONLY a parenthetical qualifier that ALREADY EXISTS in the English
    source, e.g. "W8A8 (Pre-quantized)" → "W8A8（预量化）", "W4A8 (Pre-quantized)"
    → "W4A8（预量化）", "(with MTP)" → "（含 MTP）".
11. If the English value has NO parenthetical, return it EXACTLY unchanged.
12. NEVER add a parenthetical annotation or explanation that is not present in
    the English text — e.g. do NOT turn "Eagle3 Draft Model" into
    "Eagle3 Draft Model（投机解码用）". Adding any content is a violation.

--- MARKDOWN RULES ---
13. In links [text](url), translate only the [text], keep (url) exactly.
14. Keep table separator rows and structural cells unchanged; translate only
    prose cells.
15. If a value is purely structural (symbols, code, paths, identifiers only),
    return it unchanged.

--- QUALITY ---
16. Use fluent, natural Chinese technical documentation style; avoid word-for-word
    literalism.
17. Use consistent Chinese technical terminology. If unsure about a term, keep it
    in English.
18. Never invent content; never add or remove information.

Input JSON:
<CONTENT>"""


class YAMLTranslator:
    def __init__(self, api_key: str, max_concurrent: int = 5):
        self.client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.max_concurrent = max_concurrent

    async def _call_api(self, content: str, chunk_info: str = "") -> str | None:
        system = SYSTEM_PROMPT
        if chunk_info:
            system = f"{SYSTEM_PROMPT} ({chunk_info})"
        response = await self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": TRANSLATION_PROMPT.replace("<CONTENT>", content)},
            ],
            max_tokens=8000,
            temperature=0.3,
        )
        text = response.choices[0].message.content
        if not text:
            return None
        return self._clean_response(text)

    @staticmethod
    def _clean_response(text: str) -> str | None:
        text = text.strip()
        # Strip a wrapping ```json ... ``` fence if present.
        if text.startswith("```"):
            lines = text.split("\n")
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            while lines and lines[-1].strip().startswith("```"):
                lines.pop()
            text = "\n".join(lines).strip()
        if not text.startswith("{") or not text.rstrip().endswith("}"):
            return None
        return text

    @staticmethod
    def _chunk_entries(entries: dict[str, str], max_chars: int = 6000) -> list[dict[str, str]]:
        """Split a {path: en} dict into chunks under max_chars of serialized JSON."""
        chunks: list[dict[str, str]] = []
        current: dict[str, str] = {}
        current_chars = 0
        for path, en in entries.items():
            piece = json.dumps({path: en}, ensure_ascii=False)
            if current_chars + len(piece) > max_chars and current:
                chunks.append(current)
                current = {}
                current_chars = 0
            current[path] = en
            current_chars += len(piece)
        if current:
            chunks.append(current)
        return chunks

    async def _translate_chunk(self, chunk: dict[str, str], idx: int, total: int) -> dict[str, str]:
        info = f"chunk {idx + 1}/{total}"
        payload = json.dumps(chunk, ensure_ascii=False, indent=2)
        last_err = "unknown"
        for attempt in range(3):
            try:
                raw = await self._call_api(payload, chunk_info=info)
                if raw is None:
                    last_err = "empty or non-JSON response"
                    continue
                data = json.loads(raw)
                if not isinstance(data, dict):
                    last_err = "response is not a JSON object"
                    continue
                # Only accept keys that match the input exactly.
                out = {k: str(data[k]) for k in chunk if k in data and data[k]}
                return out
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
            if attempt < 2:
                await asyncio.sleep(2)
        print(f"\n    Chunk {idx + 1}/{total} failed after 3 attempts — {last_err}", flush=True)
        return {}

    async def translate_entries(self, entries: dict[str, str]) -> dict[str, str]:
        chunks = self._chunk_entries(entries)
        if not chunks:
            return {}
        print(f"  {len(entries)} entries in {len(chunks)} chunk(s)", flush=True)
        sem = asyncio.Semaphore(self.max_concurrent)

        async def do(idx: int, chunk: dict[str, str]) -> dict[str, str]:
            async with sem:
                return await self._translate_chunk(chunk, idx, len(chunks))

        results = await asyncio.gather(*[do(i, c) for i, c in enumerate(chunks)])
        merged: dict[str, str] = {}
        for r in results:
            merged.update(r)
        return merged


async def async_main() -> int:
    ap = argparse.ArgumentParser(description="Translate pending YAML fields (DeepSeek).")
    ap.add_argument("--state-dir", default=".translate", help="Working dir for inter-stage JSON")
    ap.add_argument("--max-concurrent", type=int, default=5)
    ap.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY"))
    ap.add_argument("--output-json", default=None, help="Override result path (default: <state-dir>/translate.json)")
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    out_path = Path(args.output_json) if args.output_json else state_dir / "translate.json"
    detect_path = state_dir / "detect.json"

    if not detect_path.exists():
        print(f"Error: {detect_path} not found — run detect_yaml_changes.py first", file=sys.stderr)
        return 1

    manifest = tc.load_json(detect_path)
    pending = manifest.get("pending", [])
    if not pending:
        tc.write_json(out_path, {"translations": {}, "success_files": [], "success_count": 0, "failed_files": []})
        print("No pending entries — nothing to translate.")
        return 0

    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY not set", file=sys.stderr)
        return 1

    # Group pending entries by file.
    by_file: dict[str, dict[str, str]] = {}
    for e in pending:
        by_file.setdefault(e["en_file"], {})[e["path"]] = e["en"]

    translator = YAMLTranslator(api_key=api_key, max_concurrent=args.max_concurrent)
    translations: dict[str, dict[str, str]] = {}
    success_files: list[str] = []
    failed_files: list[str] = []

    for en_file, entries in by_file.items():
        print(f"Translating {en_file}", flush=True)
        translated = await translator.translate_entries(entries)
        if translated:
            translations[en_file] = translated
            success_files.append(en_file)
            print(f"  -> {len(translated)}/{len(entries)} translated")
        else:
            failed_files.append(en_file)
            print(f"  -> FAILED (0/{len(entries)})")

    result = {
        "translations": translations,
        "success_files": success_files,
        "failed_files": failed_files,
        "success_count": len(success_files),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    tc.write_json(out_path, result)
    print(f"\nResult: {len(success_files)} file(s) translated -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(async_main()))

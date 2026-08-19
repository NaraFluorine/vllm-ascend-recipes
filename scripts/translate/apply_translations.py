#!/usr/bin/env python3
"""Rebuild models/zh/**/*.yaml from the en mirror + translations, and update memory.

Reads the per-file {path → zh} map produced by yaml_translate.py and, for each
file, loads the English recipe with a ruamel round-trip loader (so key order,
comments and scalar styles survive), overwrites the translatable leaves with the
Chinese values, and writes the result to models/zh/**/*.yaml. It also refreshes
the translation memory (models/translations/**/*.json) so the next detect run can
diff English changes.

Two modes:
- normal (default): process only the files present in translate.json.
- --seed-memory: process ALL en files and write memory only (zh = existing zh
  mirror where available, else en), without touching zh files. Used to bootstrap
  memory from pre-existing hand-authored translations.

Usage:
    python apply_translations.py [--state-dir .translate] [--seed-memory]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import translate_common as tc


def process_file(en_path: Path, translated: dict[str, str], patterns, write_zh: bool) -> dict:
    """Apply translations to one recipe. Returns the new memory dict for it.

    The zh mirror is rebuilt by *surgical text replacement*: we load the existing
    zh file (or the en file when no zh exists yet), locate each translatable
    scalar by its character span, and replace only the strings that changed.
    Non-whitelisted fields (``source``, ``url``, model ids, …) are never touched,
    and list indentation / comments stay byte-identical.
    """
    en_data = tc.load_yaml_safe(en_path)
    entries = tc.extract_translatable(en_data, patterns)

    zh_path = tc.zh_path_for(en_path)
    zh_exists = zh_path.exists()
    base_raw = zh_path.read_text(encoding="utf-8") if zh_exists else en_path.read_text(encoding="utf-8")
    zh_data = tc.load_yaml_safe(zh_path) if zh_exists else en_data
    zh_leaves = {}
    for p, v in tc.iter_leaves(zh_data):
        zh_leaves[tc.path_to_str(p)] = v

    memory: dict[str, dict[str, str]] = {}
    replacements: dict[str, str] = {}
    integrity_failures = 0

    for path, path_str, en in entries:
        # Source priority: new LLM translation > existing zh mirror > en fallback.
        if path_str in translated and translated[path_str]:
            zh = translated[path_str]
            source = "translated"
        elif path_str in zh_leaves and zh_leaves[path_str] and zh_leaves[path_str] != en:
            zh = zh_leaves[path_str]
            source = "adopted"
        else:
            zh = en
            source = "en"

        # Structural integrity guards only LLM output — human-authored zh is
        # trusted as-is. On failure, fall back to English so we never ship a
        # recipe with broken placeholders / dropped code blocks.
        if source == "translated" and zh != en:
            ok, missing = tc.check_integrity(en, zh, path_str)
            if not ok:
                print(f"  WARN integrity: {en_path.name} {path_str} — {missing[:3]}")
                zh = en
                integrity_failures += 1

        memory[path_str] = {"en": en, "zh": zh}

        base_val = zh_leaves.get(path_str) if zh_exists else en
        if write_zh and zh != base_val:
            replacements[path_str] = zh

    if write_zh and replacements:
        new_raw, missing = tc.replace_scalars(base_raw, replacements)
        if missing:
            print(f"  WARN: {en_path.name} — {len(missing)} path(s) not in zh base, skipped: {missing[:3]}")
        zh_path.parent.mkdir(parents=True, exist_ok=True)
        zh_path.write_text(new_raw, encoding="utf-8")
        print(f"  zh: {zh_path.relative_to(tc.REPO_ROOT)} ({len(replacements)} replaced, {integrity_failures} integrity fallbacks)")

    return memory


def main() -> int:
    ap = argparse.ArgumentParser(description="Rebuild zh YAML and update translation memory.")
    ap.add_argument("--state-dir", default=".translate", help="Working dir for inter-stage JSON")
    ap.add_argument("--seed-memory", action="store_true", help="Seed memory for ALL files from existing zh, no zh writes")
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    patterns = tc.load_patterns()

    if args.seed_memory:
        en_files = sorted(tc.EN_DIR.rglob("*.yaml"))
        seeded = 0
        for en_path in en_files:
            memory = process_file(en_path, {}, patterns, write_zh=False)
            tc.save_memory(tc.memory_path_for(en_path), memory)
            seeded += 1
        print(f"Seeded memory for {seeded} recipe(s) (no zh changes).")
        return 0

    translate_path = state_dir / "translate.json"
    if not translate_path.exists():
        print(f"Error: {translate_path} not found — run yaml_translate.py first", file=sys.stderr)
        return 1

    result = tc.load_json(translate_path)
    translations = result.get("translations", {})

    if not translations:
        print("No translations to apply.")
        return 0

    for en_file, translated in translations.items():
        en_path = Path(en_file)
        if not en_path.is_absolute():
            en_path = tc.REPO_ROOT / en_path
        if not en_path.exists():
            print(f"  WARN: {en_file} not found, skipping")
            continue
        memory = process_file(en_path, translated, patterns, write_zh=True)
        tc.save_memory(tc.memory_path_for(en_path), memory)

    print(f"\nApplied translations for {len(translations)} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

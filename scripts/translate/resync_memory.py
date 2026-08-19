#!/usr/bin/env python3
"""Resync translation-memory JSONs from the en + zh mirrors.

``models/translations/**/*.json`` is the translation memory — a
``{path: {"en", "zh"}}`` snapshot that ``detect_yaml_changes.py`` diffs against
the live English recipes. When a developer manually re-translates a recipe
(edits ``models/en/**/*.yaml`` and its ``models/zh/**`` 1:1 mirror together —
e.g. trimming an overview or adding a variant), the memory goes stale. A stale
memory makes the daily translate pipeline mistake already-translated fields for
pending work and re-translate them.

This script rebuilds the memory for the affected recipes so it faithfully
reflects the current en + zh. It is also run automatically on merge by
``.github/workflows/sync_translation_memory.yml``.

Per translatable field the new ``zh`` is chosen as:

- developer edited the zh mirror   → adopt the new zh
- en unchanged                     → keep the existing translation
- en changed but zh not edited     → record ``zh == en`` so the daily pipeline
  re-translates it (never mask a stale translation as up-to-date)
- brand-new field                  → adopt zh if already translated, else ``en``

Usage:
    python scripts/translate/resync_memory.py --all
    python scripts/translate/resync_memory.py --files models/en/Qwen/Qwen3-235B-A22B.yaml
    python scripts/translate/resync_memory.py --all --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import translate_common as tc


def resync_file(en_path: Path, patterns) -> dict:
    """Return a fresh memory dict for *en_path*, derived from en + current zh.

    Existing memory is consulted only to tell "developer edited zh" (current
    zh differs from the memory's zh) apart from "zh left untouched", so a
    changed-English/unchanged-Chinese field is recorded as untranslated rather
    than silently adopted.
    """
    en_data = tc.load_yaml_safe(en_path)
    entries = tc.extract_translatable(en_data, patterns)

    zh_path = tc.zh_path_for(en_path)
    zh_data = tc.load_yaml_safe(zh_path) if zh_path.exists() else None
    zh_leaves: dict[str, str] = {}
    if zh_data is not None:
        for p, v in tc.iter_leaves(zh_data):
            zh_leaves[tc.path_to_str(p)] = v

    old = tc.load_memory(tc.memory_path_for(en_path))

    new: dict[str, dict[str, str]] = {}
    for _path, path_str, en in entries:
        zh_value = zh_leaves.get(path_str)
        old_entry = old.get(path_str)
        old_zh = old_entry.get("zh") if isinstance(old_entry, dict) else None
        old_en = old_entry.get("en") if isinstance(old_entry, dict) else None

        zh_edited = zh_value is not None and zh_value != old_zh

        if zh_edited:
            # Developer manually changed the zh mirror (re-translated / fixed).
            zh = zh_value if zh_value != en else en
        elif old_en is not None and old_en == en:
            # en unchanged: keep whatever translation we already have.
            zh = old_zh if old_zh is not None else en
        elif old_en is None and zh_value and zh_value != en:
            # Brand-new field that already carries a translation in zh.
            zh = zh_value
        else:
            # New field, or en changed without a zh update: leave untranslated
            # so the daily translate pipeline picks it up.
            zh = en

        new[path_str] = {"en": en, "zh": zh}

    return new


def resolve_en_paths(files: list[str], all_files: bool) -> list[Path]:
    if all_files or not files:
        return sorted(tc.EN_DIR.rglob("*.yaml"))

    out: list[Path] = []
    for f in files:
        p = Path(f)
        if not p.is_absolute():
            p = tc.REPO_ROOT / p
        if not p.exists():
            print(f"WARN: {p} not found, skipping", file=sys.stderr)
            continue
        out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Resync translation-memory JSONs from en + zh.")
    ap.add_argument("--all", action="store_true", help="Resync every recipe (default)")
    ap.add_argument("--files", nargs="*", default=[], help="Specific en YAML file(s) to resync")
    ap.add_argument("--check", action="store_true", help="Dry-run: report what would change, write nothing")
    args = ap.parse_args()

    patterns = tc.load_patterns()
    en_paths = resolve_en_paths(args.files, args.all)

    changed = 0
    for en_path in en_paths:
        mem_path = tc.memory_path_for(en_path)
        before = tc.load_memory(mem_path)
        memory = resync_file(en_path, patterns)
        if memory == before:
            continue
        changed += 1
        rel = en_path.relative_to(tc.REPO_ROOT)
        if args.check:
            print(f"  [check] would resync {rel}")
        else:
            tc.save_memory(mem_path, memory)
            print(f"  resynced {rel} -> {mem_path.relative_to(tc.REPO_ROOT)}")

    verb = "Would resync" if args.check else "Resynced"
    print(f"\n{verb} memory for {changed} recipe(s) ({len(en_paths)} scanned).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

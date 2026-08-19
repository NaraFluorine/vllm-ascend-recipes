#!/usr/bin/env python3
"""Detect YAML recipe fields that need English→Chinese translation.

Scans models/en/**/*.yaml, extracts translatable fields (per the allowlist in
yaml_translate_fields.json), and compares them against the translation memory
(models/translations/**/*.json) and the existing Chinese mirror
(models/zh/**/*.yaml). Produces a JSON manifest listing the fields pending
translation, which yaml_translate.py then consumes.

Memory bookkeeping: entries that already carry a valid translation (from memory
or an existing zh mirror) are written back into memory so a later run can detect
"English text changed" diffs. Pending entries are left for apply_translations.py
to fill in. This is the YAML analogue of vllm-ascend's detect_po_changes.py.

Usage:
    python detect_yaml_changes.py [--state-dir .translate] [--dry-run] [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import translate_common as tc


def main() -> int:
    ap = argparse.ArgumentParser(description="Detect YAML recipe fields needing translation.")
    ap.add_argument("--state-dir", default=".translate", help="Working dir for inter-stage JSON")
    ap.add_argument("--dry-run", action="store_true", help="Do not write memory files")
    ap.add_argument("--force", action="store_true", help="Re-queue ALL translatable fields")
    ap.add_argument("--output-json", default=None, help="Override manifest path (default: <state-dir>/detect.json)")
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output_json) if args.output_json else state_dir / "detect.json"

    patterns = tc.load_patterns()

    pending = []
    files_with_pending = []
    mem_missing_files = []
    stats = {"total_entries": 0, "ok": 0, "pending": 0, "new": 0, "changed": 0, "untranslated": 0, "force": 0}

    for en_path in sorted(tc.EN_DIR.rglob("*.yaml")):
        rel = en_path.relative_to(tc.EN_DIR)
        if "translations" in en_path.parts:  # safety
            continue
        data = tc.load_yaml_safe(en_path)
        entries = tc.extract_translatable(data, patterns)
        if not entries:
            continue

        mem_path = tc.memory_path_for(en_path)
        zh_path = tc.zh_path_for(en_path)
        memory = tc.load_memory(mem_path) if mem_path.exists() else {}
        if not mem_path.exists():
            mem_missing_files.append(str(en_path))

        zh_data = tc.load_yaml_safe(zh_path) if zh_path.exists() else None
        zh_leaves = {}
        if zh_data is not None:
            for p, v in tc.iter_leaves(zh_data):
                zh_leaves[tc.path_to_str(p)] = v

        # memory entries we need to (re)write for adopted/ok leaves
        adopted_updates = {}
        file_pending = False

        for path, path_str, en in entries:
            stats["total_entries"] += 1
            mem_entry = memory.get(path_str)
            mem_en = mem_entry.get("en") if isinstance(mem_entry, dict) else None
            mem_zh = mem_entry.get("zh") if isinstance(mem_entry, dict) else None
            file_zh = zh_leaves.get(path_str)

            # Resolve current zh: memory is authoritative when present.
            zh_current = mem_zh if (mem_entry is not None) else file_zh

            reason = None
            if args.force:
                reason = "force"
            elif mem_entry is not None:
                if mem_en != en:
                    reason = "changed"
                elif not mem_zh or not mem_zh.strip() or mem_zh == en:
                    reason = "untranslated"
            else:
                if file_zh is None:
                    reason = "new"
                elif not file_zh.strip() or file_zh == en:
                    reason = "untranslated"

            if reason is None:
                # Up to date (memory or existing zh already valid).
                stats["ok"] += 1
                # Adopt into memory so future "changed" diffs are detectable.
                if mem_entry is None or mem_en != en or (mem_zh or "") != (zh_current or ""):
                    adopted_updates[path_str] = {"en": en, "zh": zh_current or en}
            else:
                stats["pending"] += 1
                stats[reason] = stats.get(reason, 0) + 1
                pending.append({"en_file": str(en_path), "path": path_str, "en": en, "reason": reason})
                file_pending = True

        if file_pending:
            files_with_pending.append(str(en_path))

        # Persist adopted/ok entries back into memory (unless dry-run).
        if adopted_updates and not args.dry_run:
            merged = dict(memory)
            merged.update(adopted_updates)
            tc.save_memory(mem_path, merged)
            print(f"  memory: {mem_path.relative_to(tc.REPO_ROOT)} (+{len(adopted_updates)} adopted)")

    manifest = {
        "has_changes": len(pending) > 0,
        "files": sorted(files_with_pending),
        "pending": pending,
        "stats": stats,
        "memory_missing_files": mem_missing_files,
    }
    tc.write_json(out_path, manifest)

    print(f"\nScanned recipes, {stats['total_entries']} translatable fields:")
    print(f"  pending: {stats['pending']} (new={stats.get('new', 0)}, changed={stats.get('changed', 0)}, "
          f"untranslated={stats.get('untranslated', 0)}, force={stats.get('force', 0)})")
    print(f"  ok: {stats['ok']}")
    print(f"  memory missing: {len(mem_missing_files)} file(s)")
    print(f"Manifest -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Shared helpers for the vllm-ascend-recipes EN→ZH translation pipeline.

This module is imported by detect_yaml_changes.py, yaml_translate.py and
apply_translations.py (all three live in the same directory, so Python's
script-dir-on-sys.path makes ``import translate_common`` work).

It centralises the pieces that must stay consistent across the three stages:

- path representation: every translatable leaf is addressed by a *path tuple*,
  a sequence of ``('k', name)`` / ``('i', index)`` segments, plus a canonical
  string form (``scenarios[0].steps[0].content``) used as JSON keys.
- the translatable-field allowlist (deny-by-default), parsed from
  yaml_translate_fields.json.
- YAML loading (round-trip for en, safe for read-only zh).
- the translation memory layout under models/translations/**.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml as _pyyaml  # PyYAML — used for scalar character offsets (surgical replace)
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString

# --- Paths -----------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
EN_DIR = REPO_ROOT / "models" / "en"
ZH_DIR = REPO_ROOT / "models" / "zh"
MEM_DIR = REPO_ROOT / "models" / "translations"
FIELDS_FILE = SCRIPT_DIR / "yaml_translate_fields.json"


# --- YAML instances --------------------------------------------------------

# Round-trip: preserves comments, key order and scalar styles. Used to rebuild
# the zh mirror from the en file with minimal diff.
RT = YAML(typ="rt")
RT.preserve_quotes = True
RT.width = 4096

# Safe: plain dict/list/str, used for read-only inspection of the zh mirror.
SAFE = YAML(typ="safe")


def load_yaml_safe(path: Path):
    with open(path, encoding="utf-8") as fh:
        return SAFE.load(fh)


def load_yaml_rt(path: Path):
    with open(path, encoding="utf-8") as fh:
        return RT.load(fh)


def dump_yaml_rt(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        RT.dump(data, fh)


# --- JSON helpers ----------------------------------------------------------

def load_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


# --- Field allowlist -------------------------------------------------------

def load_patterns(fields_file: Path = FIELDS_FILE) -> list[list[tuple[str, str | None]]]:
    """Load and parse the translatable-field patterns.

    Each pattern string is parsed into a list of segments: ``('k', name)`` for a
    literal map key, ``('k', None)`` for any map key (``*``), ``('i', None)``
    for any list index (``[]``).
    """
    raw = load_json(fields_file)
    patterns = []
    for pat in raw.get("patterns", []):
        segs: list[tuple[str, str | None]] = []
        for tok in pat.split("."):
            if tok == "*":
                segs.append(("k", None))
            elif tok.endswith("[]"):
                segs.append(("k", tok[:-2]))
                segs.append(("i", None))
            else:
                segs.append(("k", tok))
        patterns.append(segs)
    return patterns


# --- Path tuples -----------------------------------------------------------

def iter_leaves(node, prefix: tuple = ()):
    """Yield (path_tuple, value) for every string leaf in *node*."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from iter_leaves(v, prefix + (("k", k),))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from iter_leaves(v, prefix + (("i", i),))
    elif isinstance(node, str):
        yield prefix, node


def path_to_str(segments) -> str:
    """Render a path tuple as a canonical JSON-style string key."""
    s = ""
    for seg in segments:
        if seg[0] == "k":
            if s:
                s += "."
            s += seg[1]
        else:
            s += f"[{seg[1]}]"
    return s


def match_path(path_tuple: tuple, pattern_segs: list[tuple[str, str | None]]) -> bool:
    """Return True if *path_tuple* matches *pattern_segs*."""
    if len(path_tuple) != len(pattern_segs):
        return False
    for (ct, cv), (pt, pv) in zip(path_tuple, pattern_segs):
        if pt == "k":
            if pv is None:  # any key
                if ct != "k":
                    return False
            else:
                if ct != "k" or cv != pv:
                    return False
        else:  # 'i' — any index
            if ct != "i":
                return False
    return True


def extract_translatable(data, patterns):
    """Return a list of (path_tuple, path_str, en_text) for translatable leaves."""
    entries = []
    for path, val in iter_leaves(data):
        if not val or not val.strip():
            continue
        for psegs in patterns:
            if match_path(path, psegs):
                entries.append((path, path_to_str(path), val))
                break
    return entries


# --- Translation memory ----------------------------------------------------

def memory_path_for(en_path: Path) -> Path:
    rel = en_path.relative_to(EN_DIR)
    return MEM_DIR / rel.with_suffix(".json")


def zh_path_for(en_path: Path) -> Path:
    rel = en_path.relative_to(EN_DIR)
    return ZH_DIR / rel


def load_memory(mem_path: Path) -> dict:
    if not mem_path.exists():
        return {}
    return load_json(mem_path)


def save_memory(mem_path: Path, data: dict) -> None:
    write_json(mem_path, data)


# --- ruamel in-place update ------------------------------------------------

def set_path(root, path_tuple: tuple, value: str) -> None:
    """Set the leaf addressed by *path_tuple* in a ruamel round-trip tree."""
    node = root
    for seg in path_tuple[:-1]:
        node = node[seg[1]]
    last = path_tuple[-1]
    if "\n" in value:
        value = LiteralScalarString(value)
    node[last[1]] = value


# --- Structural integrity --------------------------------------------------

# The integrity check guards only the *machine-consumed* tokens that must never
# be localised: Jinja-style placeholders ({{max_num_seqs}}), config markers
# (%%CONFIG:key%% ... %%/CONFIG:key%%), and fenced code block count parity.
#
# Deliberately NOT checked (humans legitimately localise these, and the LLM is
# told to too): URLs (en → zh-cn doc links), and code-block *contents* (shell
# comments are prose and may be translated).
_PLACEHOLDER_RE = re.compile(r"\{\{[^{}]*\}\}")
_CONFIG_RE = re.compile(r"%%[A-Za-z0-9_:\-/]+%%")

# Identifier-style fields hold short display names (weight versions, selector
# labels) where a parenthetical gloss is never invented during translation.
_IDENTIFIER_FIELD_RE = re.compile(
    r"^(?:weight_download\[\d+\]\.weight_version|scenario_selector_labels\.[^.]+)$"
)


def _has_gloss(text: str) -> bool:
    """True if *text* contains a matched (...) or （...） pair."""
    return ("(" in text and ")" in text) or ("（" in text and "）" in text)


# --- Untranslatable (identifier / brand / numeric) detection -----------------
#
# Some translatable leaves are short display names — weight versions, sequence
# labels, reference titles — that a human or the LLM legitimately keeps in
# English (zh == en). detect_yaml_changes.py must not flag these as pending
# "untranslated" work, otherwise the daily translate pipeline re-runs forever on
# brand names like "vllm-ascend GitHub" or tags like "BF16" / "135K".

# Brand / proper-noun / model-family tokens that are never localised.
_BRAND_TOKENS = frozenset({
    "vllm", "vllm-ascend", "ascend", "github", "modelscope", "huggingface",
    "modelers", "gitcode", "atlas", "cann", "deepseek", "qwen", "glm", "thudm",
    "zhipu", "kimi", "moonshot", "moonshotai", "eagle3", "mtp", "dflash",
    "mooncake", "mindspore", "torch",
})


def _is_brand_only(title: str) -> bool:
    """True if *title* (a reference title) contains no translatable prose word.

    A word counts as prose when it is a plain lowercase word, or a Capitalised
    common noun (Download / Model / Draft / Guide / Documentation / ...). Brand
    names and model/version tokens are skipped, so "vllm-ascend GitHub" and
    "Qwen3.5-27B ModelScope" are brand-only while "ModelScope Model Download"
    and "Eagle3 Draft Model" are not.
    """
    title = re.sub(r"[（(][^）)]*[）)]", "", title).strip()
    for token in re.split(r"[\s/]+", title):
        token = token.strip(".-")
        if not token:
            continue
        low = token.lower()
        if low in _BRAND_TOKENS:
            continue
        # Model id / version token (contains a digit), e.g. "Qwen3.5-27B".
        if re.fullmatch(r"\S*\d\S*", token):
            continue
        # Capitalised common noun: "Download", "Model", "Draft", "Guide", ...
        if len(low) >= 4 and token[:1].isupper() and token[1:].islower():
            return False
        # Plain lowercase word: "download", "model", ...
        if low.islower() and low.isalpha() and len(low) >= 3:
            return False
    return True


def is_untranslatable(path_str: str, en: str) -> bool:
    """True if *en* may legitimately stay verbatim in the zh mirror (zh == en).

    Used by detect_yaml_changes.py to avoid counting identifier / brand /
    numeric labels as pending "untranslated" work.
    """
    text = en.strip()
    if not text:
        return False
    # Purely numeric sequence-length labels: "20K", "135K", "250K", "1M".
    if re.fullmatch(r"\d+(?:\.\d+)?[KMG]?", text):
        return True
    # Identifier-style fields (weight versions, selector labels): the name part
    # stays English; only a parenthetical qualifier is translated.
    if _IDENTIFIER_FIELD_RE.match(path_str) and not _has_gloss(text):
        return True
    # Brand-only reference titles.
    if path_str.startswith("references[") and path_str.endswith("].title"):
        return _is_brand_only(text)
    return False


def protected_tokens(text: str) -> list[str]:
    """Return machine-consumed tokens that must survive translation verbatim."""
    tokens: list[str] = []
    tokens.extend(_PLACEHOLDER_RE.findall(text))
    tokens.extend(_CONFIG_RE.findall(text))
    return tokens


def check_integrity(en: str, zh: str, path_str: str = "") -> tuple[bool, list[str]]:
    """Return (ok, missing): placeholders/markers absent from zh, or fence imbalance."""
    missing = [t for t in protected_tokens(en) if t not in zh]
    if en.count("```") != zh.count("```"):
        missing.append("<fence-count-mismatch>")
    # Identifier-style fields: reject an invented parenthetical gloss that was
    # not in the English source (e.g. "Eagle3 Draft Model" → "…（投机解码用）").
    if _IDENTIFIER_FIELD_RE.match(path_str) and not _has_gloss(en) and _has_gloss(zh):
        missing.append("<added-parenthetical-gloss>")
    return (not missing, missing)


# --- Surgical scalar replacement -------------------------------------------
# Instead of re-serialising the whole YAML tree (which would re-indent lists and
# clobber non-whitelisted localisations like `source: 魔乐社区` or zh-cn URLs),
# we locate each scalar by its character span and replace only its value. This
# keeps the file byte-identical everywhere except the translated strings.

def scalar_spans(text: str) -> dict[str, tuple[int, int, str | None, str]]:
    """Map path_str → (start, end, style, value) for every string scalar leaf."""
    try:
        node = _pyyaml.compose(text)
    except Exception:
        return {}
    spans: dict[str, tuple[int, int, str | None, str]] = {}

    def walk(n, path=()):
        if isinstance(n, _pyyaml.MappingNode):
            for k, v in n.value:
                walk(v, path + (("k", k.value),))
        elif isinstance(n, _pyyaml.SequenceNode):
            for i, v in enumerate(n.value):
                walk(v, path + (("i", i),))
        elif isinstance(n, _pyyaml.ScalarNode):
            if isinstance(n.value, str):
                spans[path_to_str(path)] = (n.start_mark.index, n.end_mark.index, n.style, n.value)

    walk(node)
    return spans


def _needs_quoting(value: str) -> bool:
    if value != value.strip():
        return True
    if value.startswith((" ", "- ", "? ", ": ", "#", "{", "[", "&", "*", "!", "|", ">", "%", "@", "`")):
        return True
    if ": " in value or " #" in value or value.endswith(":"):
        return True
    return False


def _serialize_scalar(value: str, style: str | None, raw: str, start: int, end: int) -> str:
    """Re-serialise a scalar value in-place, matching the original style/indent."""
    if style in ("|", ">"):
        indicator_len = 2 if (start + 1 < len(raw) and raw[start + 1] in "-+") else 1
        indicator = raw[start : start + indicator_len]
        # Base indentation = indent of the first non-blank content line.
        p = start + indicator_len
        while p < len(raw) and raw[p] != "\n":
            p += 1
        p += 1
        indent = 0
        while p < end:
            ls = p
            while p < end and raw[p] != "\n":
                p += 1
            if raw[ls:p].strip():
                indent = len(raw[ls:p]) - len(raw[ls:p].lstrip(" "))
                break
            p += 1
        # Preserve the exact number of trailing newlines the block scalar had in
        # the source (a terminating newline plus any explicit blank lines).
        trailing = 0
        i = end - 1
        while i >= start and raw[i] == "\n":
            trailing += 1
            i -= 1

        lines = value.rstrip("\n").split("\n")
        body = "\n".join((" " * indent + ln) if ln else "" for ln in lines)
        # The scalar's char span includes the newline that terminates the block;
        # re-emit it so the following key stays on its own line.
        return indicator + "\n" + body + ("\n" * trailing)
    if style == '"':
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if style == "'":
        return "'" + value.replace("'", "''") + "'"
    if _needs_quoting(value):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return value


def replace_scalars(raw_text: str, replacements: dict[str, str]) -> tuple[str, list[str]]:
    """In-place replace scalar values. Returns (new_text, missing_paths)."""
    spans = scalar_spans(raw_text)
    missing: list[str] = []
    edits: list[tuple[int, int, str]] = []
    for path_str, new_val in replacements.items():
        if path_str not in spans:
            missing.append(path_str)
            continue
        start, end, style, old_val = spans[path_str]
        if new_val == old_val:
            continue
        edits.append((start, end, _serialize_scalar(new_val, style, raw_text, start, end)))
    edits.sort(key=lambda e: e[0], reverse=True)
    out = raw_text
    for start, end, repl in edits:
        out = out[:start] + repl + out[end:]
    return out, missing

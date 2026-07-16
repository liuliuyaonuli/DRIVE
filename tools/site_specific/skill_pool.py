"""Atomic batch updates from DRIVE's temporary ``Knew`` skill pools."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from AgentOccam.drive import (
    merge_reasoning_group,
    normalize_reasoning_skill,
    reasoning_similarity,
)


def _function_sources(source: str, names: Iterable[str]) -> Dict[str, str]:
    tree = ast.parse(source)
    wanted = set(names)
    return {
        node.name: ast.get_source_segment(source, node) or ""
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name in wanted
    }


def _import_sources(source: str) -> List[str]:
    tree = ast.parse(source)
    return [
        ast.get_source_segment(source, node) or ast.unparse(node)
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]


def _replace_function(source: str, name: str, replacement: str) -> str:
    tree = ast.parse(source)
    node = next(
        (
            item
            for item in tree.body
            if isinstance(item, ast.AsyncFunctionDef) and item.name == name
        ),
        None,
    )
    if node is None:
        return source.rstrip() + f"\n\n# DRIVE Knew batch addition: {name}\n{replacement.strip()}\n"
    lines = source.splitlines(keepends=True)
    start = node.lineno - 1
    end = node.end_lineno or node.lineno
    replacement_text = replacement.rstrip() + "\n"
    return "".join(lines[:start]) + replacement_text + "".join(lines[end:])


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def merge_interaction_pool(pool_path: Path, library_path: Path) -> Dict[str, int]:
    """Apply ``K_i <- K_i union Knew_i`` without dropping the old library."""

    pool_metadata_path = pool_path.with_suffix(".json")
    pool_source = pool_path.read_text(encoding="utf-8")
    pool_metadata = json.loads(pool_metadata_path.read_text(encoding="utf-8"))
    pool_functions = pool_metadata.get("functions", {})
    pool_sources = _function_sources(pool_source, pool_functions)
    if set(pool_functions) - set(pool_sources):
        missing = sorted(set(pool_functions) - set(pool_sources))
        raise ValueError(f"Knew metadata has no matching functions: {missing}")

    if library_path.exists():
        library_source = library_path.read_text(encoding="utf-8")
        metadata_path = library_path.with_suffix(".json")
        library_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        library_source = pool_source
        library_metadata = {"functions": {}, "global_version": 0}

    existing_names = set(library_metadata.get("functions", {}))
    added = replaced = 0
    if library_path.exists():
        imports = set(_import_sources(library_source))
        for import_source in _import_sources(pool_source):
            if import_source not in imports:
                library_source = library_source.rstrip() + "\n" + import_source + "\n"
                imports.add(import_source)
        for name, function_source in pool_sources.items():
            if name in existing_names:
                replaced += 1
            else:
                added += 1
            library_source = _replace_function(library_source, name, function_source)
    else:
        added = len(pool_sources)

    functions = library_metadata.setdefault("functions", {})
    for name, metadata in pool_functions.items():
        functions[name] = metadata
    library_metadata["global_version"] = int(library_metadata.get("global_version", 0)) + 1
    library_metadata["last_batch_update"] = {
        "new_pool": pool_path.name,
        "added": added,
        "replaced": replaced,
    }

    # Parse before committing either side of the pair.
    ast.parse(library_source)
    _atomic_write(library_path, library_source.rstrip() + "\n")
    _atomic_write(
        library_path.with_suffix(".json"),
        json.dumps(library_metadata, indent=2, ensure_ascii=False) + "\n",
    )
    return {"added": added, "replaced": replaced, "remaining": len(functions)}


def merge_reasoning_pool(
    pool_path: Path,
    library_path: Path,
    *,
    site: str,
    similarity_threshold: float = 0.82,
) -> Dict[str, int]:
    """Apply ``BatchMerge_r(K_r union Knew_r)`` atomically."""

    new_items = json.loads(pool_path.read_text(encoding="utf-8"))
    old_items = (
        json.loads(library_path.read_text(encoding="utf-8"))
        if library_path.exists()
        else []
    )
    by_id: Dict[str, Dict[str, Any]] = {}
    for raw in old_items + new_items:
        item = normalize_reasoning_skill(raw, site_hint=site)
        by_id[str(item["skill_id"])] = item

    groups: List[List[Dict[str, Any]]] = []
    for item in by_id.values():
        group = next(
            (
                candidates
                for candidates in groups
                if reasoning_similarity(candidates[0], item) >= similarity_threshold
            ),
            None,
        )
        if group is None:
            groups.append([item])
        else:
            group.append(item)
    merged = [merge_reasoning_group(group, site_hint=site) for group in groups]
    _atomic_write(
        library_path,
        json.dumps(merged, indent=2, ensure_ascii=False) + "\n",
    )
    return {
        "added": len(new_items),
        "merged": sum(max(0, len(group) - 1) for group in groups),
        "remaining": len(merged),
    }

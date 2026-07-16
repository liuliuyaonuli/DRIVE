#!/usr/bin/env python3
"""Stage or commit local DRIVE interaction repairs from feedback.

This tool updates ``Theta(k_i)`` metadata only.  A failure-derived candidate
is *not* an active patch: it remains staged until a source-context replay
reports a successful local postcondition.  The runtime consumes only committed
ordered selector sets and never regenerates the complete operation skill.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, Iterable

try:
    from drive_artifacts import add_interaction_runtime_schema
except ImportError:
    from tools.site_specific.drive_artifacts import add_interaction_runtime_schema


def _unique(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


def _replay_validation(candidate: Dict[str, Any], feedback: Dict[str, Any]) -> Dict[str, Any] | None:
    """Return accepted replay evidence, never treating a failure log as proof."""

    evidence = candidate.get("replay_validation") or candidate.get("validation")
    if not isinstance(evidence, dict):
        evidence = feedback.get("replay_validation")
    if not isinstance(evidence, dict):
        return None
    status = str(evidence.get("status", "")).strip().lower()
    if (
        status in {"pass", "passed", "validated"}
        and evidence.get("source_context_replayed") is True
        and evidence.get("postcondition_success") is True
    ):
        return dict(evidence)
    return None


def _proposal_key(candidate: Dict[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    source = str(candidate.get("source_selector") or candidate.get("failed_selector") or "").strip()
    failed = str(candidate.get("failed_selector") or source).strip()
    alternatives = tuple(
        _unique(candidate.get("working_selectors", []) + candidate.get("fallback_selectors", []))
    )
    return source, failed, alternatives


def _stage_repair_proposal(
    metadata: Dict[str, Any], candidate: Dict[str, Any], feedback: Dict[str, Any]
) -> None:
    """Keep an unverified local change out of the active selector policy."""

    source, failed, alternatives = _proposal_key(candidate)
    if not source or not failed:
        return
    proposals = metadata.setdefault("pending_repair_proposals", [])
    key = (source, failed, alternatives)
    if any(
        _proposal_key(existing) == key
        for existing in proposals
        if isinstance(existing, dict)
    ):
        return
    proposals.append(
        {
            "source_selector": source,
            "failed_selectors": [failed],
            "working_selectors": _unique(candidate.get("working_selectors", [])),
            "fallback_selectors": _unique(candidate.get("fallback_selectors", [])),
            "failure_count": int(candidate.get("failure_count", feedback.get("failure_count", 1)) or 1),
            "required_validation": "source_context_replay_and_postcondition",
            "validation_status": "pending",
        }
    )
    metadata["pending_repair_proposals"] = proposals[-20:]


def _stage_recovery_branch(metadata: Dict[str, Any], branch: Dict[str, Any]) -> None:
    pending = metadata.setdefault("pending_recovery_branches", [])
    trigger = branch.get("trigger", "unexpected_modal")
    if any(
        isinstance(item, dict) and item.get("trigger", "unexpected_modal") == trigger
        for item in pending
    ):
        return
    staged = dict(branch)
    staged["pending_validation"] = True
    staged["required_validation"] = "source_context_replay_and_postcondition"
    pending.append(staged)
    metadata["pending_recovery_branches"] = pending[-20:]


def patch_metadata(
    source: str,
    metadata: Dict[str, Any],
    feedback_kb: Dict[str, Any],
    *,
    remove_after: int = 3,
) -> Dict[str, Any]:
    functions = metadata.setdefault("functions", {})
    feedback_by_skill = feedback_kb.get("skills", {})
    try:
        tree = ast.parse(source)
        function_sources = {
            node.name: ast.get_source_segment(source, node) or source
            for node in tree.body
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
        }
    except SyntaxError:
        function_sources = {}
    for skill_name, skill_metadata in functions.items():
        add_interaction_runtime_schema(
            skill_metadata,
            function_sources.get(skill_name, ""),
        )
        feedback = feedback_by_skill.get(skill_name)
        if not isinstance(feedback, dict):
            continue
        selector_sets = skill_metadata.setdefault("selector_sets", {})
        counts = skill_metadata.setdefault("selector_failure_counts", {})
        committed_patch = False
        for candidate in feedback.get("selector_patch_candidates", []):
            if not isinstance(candidate, dict):
                continue
            selector_source = str(
                candidate.get("source_selector")
                or candidate.get("failed_selector", "")
            ).strip()
            failed = str(candidate.get("failed_selector") or selector_source).strip()
            if not selector_source or not failed:
                continue
            validation = _replay_validation(candidate, feedback)
            if validation is None:
                _stage_repair_proposal(skill_metadata, candidate, feedback)
                continue
            committed_patch = True
            working = _unique(candidate.get("working_selectors", []))
            fallbacks = _unique(candidate.get("fallback_selectors", []))
            count_key = f"{selector_source}::{failed}"
            counts[count_key] = int(counts.get(count_key, 0)) + int(
                candidate.get("failure_count", feedback.get("failure_count", 1)) or 1
            )
            ordered = _unique(working + fallbacks + selector_sets.get(selector_source, []))
            ordered = [value for value in ordered if value != failed]
            if counts[count_key] < remove_after:
                ordered.append(failed)  # failed selector is demoted
            selector_sets[selector_source] = ordered[:8]

            for template in skill_metadata.get("operation_templates", []):
                if (
                    template.get("source_selector") == selector_source
                    or selector_source in template.get("selectors", [])
                ):
                    template["source_selector"] = selector_source
                    template["selectors"] = list(selector_sets[selector_source])
            committed = skill_metadata.setdefault("validated_repairs", [])
            committed.append(
                {
                    "source_selector": selector_source,
                    "failed_selector": failed,
                    "validation": validation,
                }
            )
            skill_metadata["validated_repairs"] = committed[-50:]
        if committed_patch:
            skill_metadata["selector_patch_rounds"] = int(
                skill_metadata.get("selector_patch_rounds", 0)
            ) + 1
        for branch in feedback.get("recovery_branches", []) or []:
            if not isinstance(branch, dict) or not branch.get("selectors"):
                continue
            if _replay_validation(branch, feedback) is None:
                _stage_recovery_branch(skill_metadata, branch)
                continue
            branches = skill_metadata.setdefault("recovery_branches", [])
            trigger = branch.get("trigger", "unexpected_modal")
            if not any(item.get("trigger") == trigger for item in branches if isinstance(item, dict)):
                approved = dict(branch)
                approved.pop("replay_validation", None)
                approved["validated"] = True
                branches.append(approved)
        contract = skill_metadata.get("interaction_contract")
        if isinstance(contract, dict):
            contract["Rec_k"] = list(skill_metadata.get("recovery_branches", []))
    return metadata


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage or commit replay-validated local DRIVE selector repairs"
    )
    parser.add_argument("--file", required=True, help="operation_skills.py")
    parser.add_argument("--metadata", help="metadata JSON; defaults to --file with .json")
    parser.add_argument("--feedback-kb", required=True)
    parser.add_argument("--remove-after", type=int, default=3)
    args = parser.parse_args(argv)

    source_path = Path(args.file)
    metadata_path = Path(args.metadata) if args.metadata else source_path.with_suffix(".json")
    feedback_path = Path(args.feedback_kb)
    source = source_path.read_text(encoding="utf-8")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    patched = patch_metadata(source, metadata, feedback, remove_after=args.remove_after)
    metadata_path.write_text(
        json.dumps(patched, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Processed local selector repair proposals: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

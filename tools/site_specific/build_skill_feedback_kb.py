#!/usr/bin/env python3
"""Build a lightweight skill feedback knowledge base from runtime failures."""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def _read_metadata(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("functions", {})
    except Exception:
        return {}


def _top(counter: Counter, limit: int = 3, text_limit: int = 120) -> List[str]:
    return [_brief(item, text_limit) for item, _ in counter.most_common(limit) if item not in (None, "")]


def _brief(text: Any, limit: int = 240) -> str:
    value = str(text or "").strip()
    return value[:limit]


def _compact_args(skill_args: Any) -> Dict[str, str]:
    if not isinstance(skill_args, dict):
        return {}
    compact = {}
    for key, value in list(skill_args.items())[:4]:
        compact[str(key)[:40]] = _brief(value, 80)
    return compact


def _valid_selector_text(value: Any) -> bool:
    selector = str(value or "").strip()
    return bool(
        selector
        and not selector.endswith(("(", "*=", "[href*="))
        and selector.count("(") == selector.count(")")
        and selector.count("[") == selector.count("]")
    )


def _selectors_from_record(record: Dict[str, Any]) -> List[str]:
    selectors = [
        _brief(item, 240)
        for item in record.get("selectors", [])
        if _valid_selector_text(item)
    ]
    traceback_text = str(record.get("traceback") or "")
    for match in re.finditer(r'locator\((?P<q>["\'])(?P<value>.+?)(?P=q)\)', traceback_text):
        value = match.group("value")
        if _valid_selector_text(value):
            selectors.append(_brief(value, 240))
    return list(dict.fromkeys(selectors))


def _quote_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _fallbacks_for_selector(selector: str) -> List[str]:
    selector = _brief(selector, 300)
    fallbacks = []
    text_match = re.search(r":has-text\([\"'](.+?)[\"']\)", selector)
    if text_match:
        text = _quote_text(text_match.group(1))
        if selector.startswith("button"):
            fallbacks.extend([f'get_by_role("button", name="{text}")', f'text="{text}"'])
        elif selector.startswith("a"):
            fallbacks.extend([f'get_by_role("link", name="{text}")', f'text="{text}"'])
        else:
            fallbacks.append(f'text="{text}"')
    if "input" in selector:
        if "search" in selector.lower() or "q" in selector.lower():
            fallbacks.extend(["input[type='search']", "input[placeholder*='Search']"])
        name_match = re.search(r"name=[\"']([^\"']+)[\"']", selector)
        if name_match:
            fallbacks.append(f"input[name='{name_match.group(1)}']")
    if "textarea" in selector:
        fallbacks.extend(["textarea", "textarea[placeholder]"])
    role_match = re.search(
        r'(?:get_by_role\(["\']([^"\']+)["\']|role=([^,]+))(?:,\s*name=["\']([^"\']+)["\'])?',
        selector,
    )
    if role_match:
        role = role_match.group(1) or role_match.group(2)
        name = role_match.group(3)
        if name:
            fallbacks.extend([f'[role="{role}"][aria-label="{name}"]', f'text={name}'])
        else:
            fallbacks.append(f'[role="{role}"]')
    label_match = re.search(r'get_by_label\(["\']([^"\']+)["\']', selector)
    if label_match:
        label = label_match.group(1)
        fallbacks.extend([f'[aria-label="{label}"]', f'label:has-text("{label}")'])
    text_locator_match = re.search(r'get_by_text\(["\']([^"\']+)["\']', selector)
    if text_locator_match:
        text = text_locator_match.group(1)
        fallbacks.extend([f'text={text}', f':text-is("{text}")'])
    return list(dict.fromkeys(_brief(item, 100) for item in fallbacks if item))[:3]


def _selector_patch_candidates(selectors: List[str]) -> List[Dict[str, Any]]:
    candidates = []
    for selector in selectors[:4]:
        fallbacks = _fallbacks_for_selector(selector)
        if not fallbacks:
            continue
        candidates.append(
            {
                "source_selector": _brief(selector, 300),
                "failed_selector": _brief(selector, 300),
                "fallback_selectors": fallbacks,
                "patch_hint": _brief("Try the original selector, then these fallbacks in order with visibility checks.", 140),
            }
        )
        if len(candidates) >= 3:
            break
    return candidates


def _drive_feedback_as_event(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize the complete online feedback tuple to the KB event shape."""

    if record.get("skill_type") != "interaction" or not record.get("skill_name"):
        return None
    local = record.get("local_outcome") or {}
    execution = record.get("execution_log") or {}
    context = record.get("page_context") or {}
    return {
        "skill_name": record.get("skill_name"),
        "error": local.get("error") or local.get("failure_reason") or "",
        "url": context.get("url") or execution.get("url_before") or "",
        "selectors": execution.get("selectors") or [],
        "selector_trace": execution.get("selector_trace") or [],
        "skill_args": record.get("arguments") or {},
        "page_state_snippet": context.get("page_state_snippet")
        or context.get("observation_snippet")
        or "",
        "objective": record.get("task_instruction") or "",
        "local_success": bool(local.get("success")),
        "final_task_label": record.get("final_task_label"),
        "feedback_id": record.get("feedback_id"),
    }


def _trace_patch_candidates(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build operation-local patches from exact runtime selector traces."""

    by_source: Dict[str, Dict[str, Counter]] = defaultdict(
        lambda: {"failed": Counter(), "working": Counter()}
    )
    for record in records:
        for event in record.get("selector_trace", []) or []:
            if not isinstance(event, dict) or event.get("operation") == "goto":
                continue
            source = _brief(event.get("source_selector"), 300)
            candidate = _brief(event.get("selector"), 300)
            if not source or not candidate:
                continue
            bucket = "working" if event.get("success") is True else "failed"
            by_source[source][bucket].update([candidate])

    candidates = []
    for source, traces in sorted(by_source.items()):
        if not traces["failed"]:
            continue
        failed, failure_count = traces["failed"].most_common(1)[0]
        working = [value for value, _ in traces["working"].most_common(3)]
        fallbacks = list(dict.fromkeys(working + _fallbacks_for_selector(source)))[:5]
        candidates.append(
            {
                "source_selector": source,
                "failed_selector": failed,
                "failure_count": failure_count,
                "working_selectors": working,
                "fallback_selectors": fallbacks,
                "patch_hint": "Demote the failed candidate and promote observed working candidates for this operation only.",
            }
        )
    return candidates[:8]


def _recovery_branches(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    text = " ".join(
        f"{record.get('error', '')} {record.get('page_state_snippet', '')}"
        for record in records
        if not record.get("local_success")
    ).lower()
    if not any(cue in text for cue in ("modal", "dialog", "overlay", "popup", "pop-up")):
        return []
    return [
        {
            "trigger": "unexpected_modal",
            "selectors": [
                'get_by_role("button", name="Close")',
                "button:has-text('Close')",
                "[aria-label='Close']",
            ],
            "max_attempts": 1,
        }
    ]


def _guidance_for_skill(summary: Dict[str, Any]) -> List[str]:
    tips = []
    selectors = summary.get("common_selectors", [])
    errors = " ".join(summary.get("common_errors", [])).lower()
    snippets = " ".join(summary.get("page_state_snippets", [])).lower()

    if selectors:
        tips.append(
            _brief(
                "Add fallback selectors or semantic locator alternatives for failed selectors: "
                + ", ".join(selectors[:3]),
                160,
            )
        )
    if "timeout" in errors or "not found" in errors or "element" in errors:
        tips.append("Wait for dynamic content and verify element visibility before interacting.")
    if "out of stock" in snippets:
        tips.append("Check stock/availability state before cart or purchase mutations.")
    if summary.get("common_urls"):
        tips.append("Revisit known failure URL patterns when validating the workflow.")
    if not tips:
        tips.append("Use prior failure records to avoid repeating the same UI assumptions.")
    return tips


def build_feedback_kb(site_dir: Path, metadata_path: Optional[Path] = None) -> Dict[str, Any]:
    failure_log = site_dir / "skill_failure_log.jsonl"
    feedback_log = site_dir / "skill_feedback_log.jsonl"
    metadata_path = metadata_path or site_dir / "operation_skills.json"
    legacy_failures = _read_jsonl(failure_log)
    complete_feedback = [
        event
        for record in _read_jsonl(feedback_log)
        if (event := _drive_feedback_as_event(record)) is not None
    ]
    complete_ids = {
        str(event.get("feedback_id"))
        for event in complete_feedback
        if event.get("feedback_id")
    }
    legacy_only = [
        record
        for record in legacy_failures
        if not record.get("feedback_id")
        or str(record.get("feedback_id")) not in complete_ids
    ]
    all_records = legacy_only + complete_feedback
    metadata = _read_metadata(metadata_path)

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in all_records:
        grouped[str(record.get("skill_name") or "unknown")].append(record)

    skills = {}
    for skill_name, records in sorted(grouped.items()):
        selector_counter = Counter()
        url_counter = Counter()
        error_counter = Counter()
        objective_counter = Counter()
        snippets = []
        args_examples = []
        local_success_count = 0
        task_success_count = 0
        failure_count = 0

        for record in records:
            local_success = bool(record.get("local_success"))
            if local_success:
                local_success_count += 1
            else:
                failure_count += 1
                selector_counter.update(_selectors_from_record(record))
            if record.get("final_task_label") == 1:
                task_success_count += 1
            url_counter.update([record.get("url", "")])
            if record.get("error"):
                error_counter.update([_brief(record.get("error"), 120)])
            objective_counter.update([_brief(record.get("objective"), 120)])
            snippet = _brief(record.get("page_state_snippet"), 160)
            if snippet and snippet not in snippets:
                snippets.append(snippet)
            skill_args = _compact_args(record.get("skill_args", {}))
            if skill_args and skill_args not in args_examples:
                args_examples.append(skill_args)

        trace_candidates = _trace_patch_candidates(records)
        heuristic_candidates = _selector_patch_candidates(_top(selector_counter, 4, 300))
        summary = {
            "skill_name": skill_name,
            "invocation_count": len(records),
            "failure_count": failure_count,
            "local_success_count": local_success_count,
            "task_success_count": task_success_count,
            "common_errors": _top(error_counter, 3, 120),
            "common_urls": _top(url_counter, 3, 120),
            "common_selectors": _top(selector_counter, 4, 300),
            "objectives": _top(objective_counter, 3, 120),
            "page_state_snippets": snippets[:2],
            "skill_args_examples": args_examples[:1],
            "selector_patch_candidates": trace_candidates or heuristic_candidates,
            "recovery_branches": _recovery_branches(records),
            "scenario_descriptor": metadata.get(skill_name, {}).get("scenario_descriptor", {}),
            "events": metadata.get(skill_name, {}).get("events", [])[-3:],
        }
        summary["generation_guidance"] = _guidance_for_skill(summary)
        skills[skill_name] = summary

    return {
        "schema_version": 2,
        "site_dir": str(site_dir),
        "source_failure_log": str(failure_log),
        "source_feedback_log": str(feedback_log),
        "source_metadata": str(metadata_path),
        "total_feedback": len(all_records),
        "total_failures": len(legacy_only)
        + sum(not event.get("local_success") for event in complete_feedback),
        "total_local_successes": sum(
            bool(event.get("local_success")) for event in complete_feedback
        ),
        "skill_count": len(skills),
        "skills": skills,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="构建运行时技能反馈知识库")
    parser.add_argument("--site-dir", required=True, help="技能站点目录，如 skills/map")
    parser.add_argument("--metadata", help="operation_skills.json 路径；默认使用 site-dir 下的文件")
    parser.add_argument("--out", help="输出 skill_feedback_kb.json 路径；默认写入 site-dir")
    args = parser.parse_args(argv)

    site_dir = Path(args.site_dir)
    metadata_path = Path(args.metadata) if args.metadata else None
    output_path = Path(args.out) if args.out else site_dir / "skill_feedback_kb.json"

    kb = build_feedback_kb(site_dir, metadata_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(kb, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"技能反馈知识库: {output_path}")
    print(f"失败记录: {kb['total_failures']}")
    print(f"涉及技能: {kb['skill_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

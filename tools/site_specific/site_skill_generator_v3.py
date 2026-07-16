#!/usr/bin/env python3
"""
Generic site-specific trajectory skill generator (v3).

This is the non-Reddit counterpart to reddit_skill_generator_v3.py. It keeps the
same output contract:
- operation_skills.py with async Playwright skills
- operation_skills.json with SkillRegistry-compatible metadata

Supported sites are defined in site_skill_profiles.py.
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse


sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from llm_helpers import call_llm_json, call_llm_text
except ImportError as e:
    print(f"错误: 无法导入必需模块: {e}")
    print("请确保 tools/llm_helpers.py 存在")
    sys.exit(1)

try:
    from site_skill_profiles import SiteSkillProfile, get_site_profile, list_supported_sites
except ImportError:
    from tools.site_specific.site_skill_profiles import (
        SiteSkillProfile,
        get_site_profile,
        list_supported_sites,
    )

from site_specific.drive_artifacts import add_interaction_runtime_schema
from site_specific.skill_quality import validate_generated_interaction_skill
from site_specific.failure_attribution import (
    COUNTERFACTUAL_ATTRIBUTION_INSTRUCTION,
    normalize_failure_attribution,
)


SUCCESS_TRAJECTORY_PROMPT = """
You are learning how to use {display_name}. You are building a procedural
knowledge base with Python functions using the Playwright async API.

Site: {site}
Platform: {platform_description}

Write a reusable operation-level skill for the completed task. Make the skill
represent the GENERAL case, not just the concrete trajectory.

Function requirements:
- Output exactly one top-level `async def` function.
- The first argument must be `page`.
- Use Playwright async calls and `await`.
- Include a detailed docstring with what the function does, usage preconditions,
  arguments, return value, and a "Usage Log" section.
- Begin by navigating to a stable initial page with `await page.goto(...)`.
- Prefer semantic parameters such as `project_name`, `product_name`,
  `order_id`, `place_name`, `query`, not raw URLs or numeric element ids.
- Do not define nested functions.
- Do not use broad try/except blocks that only print errors.
- Raise `ElementNotFoundError`, `NavigationError`, or `SubmissionError` when
  an operation cannot be completed.
- Wait for navigation or dynamic content with `await page.wait_for_load_state("networkidle")`
  or `await page.wait_for_selector(...)`.
- Verify the visible UI state after any mutation before returning.
- Represent each grounding operation with an ordered selector list (CSS, text,
  role/label, or nearby-element alternatives). Try selectors in order and use
  the first valid one; do not bind an operation to a single brittle selector.

Verified selectors for {display_name}:
{verified_selectors}

URL patterns:
{url_patterns}

Task families:
{task_families}

Site-specific generation guidance:
{success_generation_guidance}

Site-specific example skill patterns:
{example_skill_patterns}

Relevant prior skill feedback from runtime tests:
{feedback_context}

Successful trajectory:
Task ID: {task_id}
Task Objective: {objective}
Final URL: {final_url}

Action History:
{action_history}

Generate the reusable skill now.
Output ONLY Python code. Do not include markdown code fences, JSON, or explanation.
"""


FAILURE_TRAJECTORY_PROMPT = """
You are analyzing a FAILED {display_name} automation attempt and generating an
operation-level corrective skill.

Site: {site}
Platform: {platform_description}

CRITICAL constraints:
1. Output exactly one top-level `async def` function.
2. No nested functions and no helper functions inside the skill.
3. Generate a COMPLETE operation skill that performs the task, not just a check.
4. Use the verified selectors and URL patterns below.
5. Address the root operation failure described in Failure Analysis.
6. Verify success using visible UI state, URL state, or extracted text before returning.
7. Raise `ElementNotFoundError`, `NavigationError`, or `SubmissionError` on failure.
8. For every grounding operation, try a small ordered selector list and use the
   first valid selector. Keep recovery local (selector retry or modal close).

Verified selectors for {display_name}:
{verified_selectors}

URL patterns:
{url_patterns}

Common operation failures this skill should avoid:
{operation_failures}

Site-specific corrective guidance:
{failure_generation_guidance}

Site-specific example skill patterns:
{example_skill_patterns}

Relevant prior skill feedback from runtime tests:
{feedback_context}

Failed trajectory:
Task ID: {task_id}
Task Objective: {objective}
Final URL: {final_url}

Action History:
{action_history}

Failure Analysis:
{failure_analysis}

Generate the corrective operation skill now.
Output ONLY Python code. Do not include markdown code fences, JSON, or explanation.
"""


def extract_action_history(trajectory: List[Dict[str, Any]]) -> str:
    actions = []
    for i, step in enumerate(trajectory):
        action = step.get("action", "")
        reason = step.get("reason", "")
        url = step.get("url", "")
        obs_desc = step.get("observation_description", "")
        if not obs_desc:
            obs = step.get("observation", "")
            obs_desc = obs[:200] + "..." if isinstance(obs, str) and len(obs) > 200 else obs

        step_info = f"Step {i + 1}:\n"
        step_info += f"  URL: {url}\n"
        if obs_desc:
            obs_text = str(obs_desc)
            step_info += (
                f"  Observation: {obs_text[:150]}...\n"
                if len(obs_text) > 150
                else f"  Observation: {obs_text}\n"
            )
        step_info += f"  Reason: {reason}\n"
        step_info += f"  Action: {action}\n"
        actions.append(step_info)
    return "\n".join(actions)


def _clean_llm_code(response: str) -> str:
    code = response.strip()
    if code.startswith("```python"):
        code = code[len("```python") :]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    lines = []
    forbidden_exception_imports = (
        "from exceptions import ",
        "from openagent.skills.exceptions import ",
        "from skills.exceptions import ",
    )
    for line in code.strip().splitlines():
        if line.strip().startswith(forbidden_exception_imports):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def load_feedback_context(path: Optional[str], limit: int = 3) -> str:
    if not path:
        return "No prior runtime skill feedback is available."
    feedback_path = Path(path)
    if not feedback_path.exists():
        return "No prior runtime skill feedback is available."
    try:
        kb = json.loads(feedback_path.read_text(encoding="utf-8"))
    except Exception:
        return "Prior runtime skill feedback exists but could not be loaded."

    skills = kb.get("skills", {})
    if not isinstance(skills, dict) or not skills:
        return "No prior runtime skill feedback is available."

    ranked = sorted(
        skills.values(),
        key=lambda item: item.get("failure_count", 0) if isinstance(item, dict) else 0,
        reverse=True,
    )
    lines = [
        "Use these historical failures as guidance only. Do not copy old broken selectors blindly.",
        f"Total recorded skill failures: {kb.get('total_failures', 0)}",
    ]
    for item in ranked[:limit]:
        if not isinstance(item, dict):
            continue
        lines.append(f"- Skill `{item.get('skill_name', 'unknown')}` failed {item.get('failure_count', 0)} time(s).")
        errors = item.get("common_errors", [])
        selectors = item.get("common_selectors", [])
        urls = item.get("common_urls", [])
        guidance = item.get("generation_guidance", [])
        patch_candidates = item.get("selector_patch_candidates", [])
        if errors:
            lines.append(f"  Common error: {str(errors[0])[:120]}")
        if selectors:
            lines.append(f"  Selectors needing fallback: {', '.join(str(s)[:80] for s in selectors[:2])}")
        patch_line = _format_selector_patch_candidate(patch_candidates, selectors)
        if patch_line:
            lines.append(patch_line)
        if urls:
            lines.append(f"  Example failure URL: {str(urls[0])[:120]}")
        if guidance:
            lines.append(f"  Feedback guidance: {str(guidance[0])[:160]}")
    return "\n".join(lines)


def _format_selector_patch_candidate(patch_candidates: Any, selectors: List[str]) -> str:
    candidate = None
    if isinstance(patch_candidates, list) and patch_candidates:
        first = patch_candidates[0]
        if isinstance(first, dict):
            candidate = first
    if not candidate and selectors:
        fallback = _fallbacks_for_selector(str(selectors[0]))
        if fallback:
            candidate = {"failed_selector": str(selectors[0]), "fallback_selectors": fallback}
    if not candidate:
        return ""
    failed = str(candidate.get("failed_selector", ""))[:80]
    fallbacks = candidate.get("fallback_selectors", [])
    if not isinstance(fallbacks, list) or not fallbacks:
        return ""
    fallback_text = " | ".join(str(item)[:80] for item in fallbacks[:2])
    return f"  Selector patch candidate: {failed} -> {fallback_text}"


def _fallbacks_for_selector(selector: str) -> List[str]:
    text_match = re.search(r":has-text\([\"'](.+?)[\"']\)", selector)
    if not text_match:
        return []
    text = text_match.group(1).replace('"', '\\"')
    if selector.startswith("button"):
        return [f'get_by_role("button", name="{text}")', f'text="{text}"']
    if selector.startswith("a"):
        return [f'get_by_role("link", name="{text}")', f'text="{text}"']
    return [f'text="{text}"']


def _trajectory_objective(trajectory_data: Dict[str, Any]) -> str:
    objective = trajectory_data.get("intent", "")
    trajectory = trajectory_data.get("trajectory", [])
    if not objective and trajectory:
        objective = trajectory[0].get("objective", "Unknown task")
    return objective or "Unknown task"


def _final_url(trajectory_data: Dict[str, Any]) -> str:
    trajectory = trajectory_data.get("trajectory", [])
    return trajectory[-1].get("url", "") if trajectory else ""


def _unique_nonempty(items: List[str], limit: int = 12) -> List[str]:
    seen = []
    for item in items:
        normalized = str(item).strip()
        if normalized and normalized not in seen:
            seen.append(normalized)
    return seen[:limit]


def _url_pattern(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme:
        return url
    path = parsed.path or "/"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def _infer_task_family(profile: SiteSkillProfile, objective: str, failure_analysis: Optional[Dict[str, Any]]) -> str:
    if failure_analysis and failure_analysis.get("task_family"):
        return str(failure_analysis["task_family"])
    objective_lower = objective.lower()
    best_family = profile.task_families[0]
    best_score = -1
    for family in profile.task_families:
        keywords = profile.family_keywords.get(family, [])
        score = sum(1 for keyword in keywords if keyword.lower() in objective_lower)
        if score > best_score:
            best_family = family
            best_score = score
    return best_family


def _scenario_keywords(profile: SiteSkillProfile, objective: str, task_family: str) -> List[str]:
    keywords = list(profile.family_keywords.get(task_family, []))
    objective_lower = objective.lower()
    for obj, obj_keywords in profile.object_keywords.items():
        if obj in objective_lower or any(keyword.lower() in objective_lower for keyword in obj_keywords):
            keywords.extend([obj] + obj_keywords)
    return _unique_nonempty([kw.lower() for kw in keywords], limit=10)


def build_scenario_descriptor(
    trajectory_data: Dict[str, Any],
    profile: SiteSkillProfile,
    skill_name: str,
    source_type: str,
) -> Dict[str, Any]:
    """Build DRIVE-style d_k=<U_k,W_k> metadata for an operation skill."""
    trajectory = trajectory_data.get("trajectory", [])
    objective = _trajectory_objective(trajectory_data)
    failure_analysis = trajectory_data.get("failure_analysis", {})
    urls = _unique_nonempty([_url_pattern(step.get("url", "")) for step in trajectory])
    final_url = _final_url(trajectory_data)
    if final_url:
        urls = _unique_nonempty(urls + [_url_pattern(final_url)])

    traj_analysis = failure_analysis.get("trajectory_analysis", {}) if isinstance(failure_analysis, dict) else {}
    analyzed_url_patterns = traj_analysis.get("url_patterns", []) if isinstance(traj_analysis, dict) else []
    urls = _unique_nonempty(urls + [str(pattern) for pattern in analyzed_url_patterns])

    task_family = _infer_task_family(profile, objective, failure_analysis if isinstance(failure_analysis, dict) else None)
    keywords = _scenario_keywords(profile, objective, task_family)
    failure_category = failure_analysis.get("failure_category", "") if isinstance(failure_analysis, dict) else ""

    scenario_description = (
        failure_analysis.get("failure_description")
        if isinstance(failure_analysis, dict) and failure_analysis.get("failure_description")
        else f"{profile.display_name} task requiring {task_family.lower().replace('_', ' ')} behavior: {objective}"
    )

    return {
        "U_k": {
            "site": profile.site,
            "display_name": profile.display_name,
            "url_patterns": urls,
            "url_match": "compatible_site",
            "page_context": {
                "start_url": urls[0] if urls else "",
                "final_url": _url_pattern(final_url) if final_url else "",
                "observed_actions": _unique_nonempty([str(step.get("action", "")).split(" ", 1)[0] for step in trajectory], limit=8),
            },
        },
        "W_k": {
            "task_intent": objective,
            "task_family": task_family,
            "semantic_keywords": keywords,
            "scenario_description": scenario_description,
            "failure_category": failure_category,
            "source_type": source_type,
            "skill_name": skill_name,
        },
    }


def generate_skill_from_success_trajectory(
    trajectory_data: Dict[str, Any],
    profile: SiteSkillProfile,
    model: str = "gpt-4.1",
    feedback_context: str = "",
) -> Optional[str]:
    trajectory = trajectory_data.get("trajectory", [])
    if not trajectory:
        return None

    prompt = SUCCESS_TRAJECTORY_PROMPT.format(
        site=profile.site,
        display_name=profile.display_name,
        platform_description=profile.platform_description,
        verified_selectors=profile.verified_selectors.strip(),
        url_patterns=profile.url_patterns.strip(),
        task_families=", ".join(profile.task_families),
        success_generation_guidance=profile.success_generation_guidance.strip(),
        example_skill_patterns=profile.example_skill_patterns.strip(),
        feedback_context=feedback_context or "No prior runtime skill feedback is available.",
        task_id=trajectory_data.get("id", "unknown"),
        objective=_trajectory_objective(trajectory_data),
        final_url=_final_url(trajectory_data),
        action_history=extract_action_history(trajectory),
    )

    try:
        response = call_llm_text(prompt, model=model, max_tokens=2200)
        code = _clean_llm_code(response)
        if "async def" not in code:
            print("  警告: 生成的代码不包含 async def")
            return None
        return code
    except Exception as e:
        print(f"  技能生成失败: {e}")
        return None


def analyze_failure(
    trajectory: List[Dict[str, Any]],
    objective: str,
    profile: SiteSkillProfile,
    model: str = "gpt-4.1",
) -> Dict[str, Any]:
    action_history = extract_action_history(trajectory)
    prompt = f"""
Analyze this FAILED {profile.display_name} automation attempt and identify the root cause.

Task Objective: {objective}

Action History:
{action_history}

Classify whether this needs an operation skill or reasoning guidance. Output JSON:
{{
  "failure_type": "operation|reasoning",
  "root_cause": "Specific root cause of failure",
  "what_went_wrong": "Detailed explanation",
  "missing_steps": ["steps that should have been taken"],
  "suggested_fix": "How an operation skill should avoid this",
  "skill_recommendation": {{
    "should_generate": true,
    "needs_operation_skill": true,
    "needs_reasoning_guidance": false,
    "skill_type": "navigation|interaction|extraction|submission|configuration",
    "skill_name_suggestion": "snake_case_name"
  }}
}}
{COUNTERFACTUAL_ATTRIBUTION_INSTRUCTION}
"""
    try:
        return normalize_failure_attribution(call_llm_json(prompt, model=model))
    except Exception as e:
        print(f"  警告: 失败分析失败: {e}")
        return {
            "failure_type": "unknown",
            "root_cause": "Unable to analyze",
            "what_went_wrong": str(e),
            "missing_steps": [],
            "suggested_fix": "",
            "skill_recommendation": {
                "should_generate": False,
                "needs_operation_skill": False,
                "needs_reasoning_guidance": False,
            },
        }


def generate_skill_from_failure_trajectory(
    trajectory_data: Dict[str, Any],
    profile: SiteSkillProfile,
    model: str = "gpt-4.1",
    feedback_context: str = "",
) -> Optional[str]:
    trajectory = trajectory_data.get("trajectory", [])
    if not trajectory:
        return None

    task_id = trajectory_data.get("id", "unknown")
    objective = _trajectory_objective(trajectory_data)
    failure_analysis = trajectory_data.get("failure_analysis")
    if failure_analysis:
        print("    使用预先的失败分析... ✓")
    else:
        print("    分析失败原因...", end=" ", flush=True)
        failure_analysis = analyze_failure(trajectory, objective, profile, model=model)
        print("✓")

    skill_rec = failure_analysis.get("skill_recommendation", {})
    should_generate = skill_rec.get("should_generate", True)
    needs_operation = skill_rec.get("needs_operation_skill", True)
    if not should_generate or not needs_operation:
        print("    跳过: 不适合生成操作技能")
        return None

    prompt = FAILURE_TRAJECTORY_PROMPT.format(
        site=profile.site,
        display_name=profile.display_name,
        platform_description=profile.platform_description,
        verified_selectors=profile.verified_selectors.strip(),
        url_patterns=profile.url_patterns.strip(),
        operation_failures="\n".join(f"- {item}" for item in profile.operation_failure_examples),
        failure_generation_guidance=profile.failure_generation_guidance.strip(),
        example_skill_patterns=profile.example_skill_patterns.strip(),
        feedback_context=feedback_context or "No prior runtime skill feedback is available.",
        task_id=task_id,
        objective=objective,
        final_url=_final_url(trajectory_data),
        action_history=extract_action_history(trajectory),
        failure_analysis=json.dumps(failure_analysis, indent=2, ensure_ascii=False),
    )

    try:
        print("    生成技能代码...", end=" ", flush=True)
        response = call_llm_text(prompt, model=model, max_tokens=2200)
        code = _clean_llm_code(response)
        if "async def" not in code:
            print("✗ 不包含 async def")
            return None
        print("✓")
        return code
    except Exception as e:
        print(f"✗ 失败: {e}")
        return None


def validate_skill_code(code: str) -> Tuple[bool, str]:
    valid, errors = validate_generated_interaction_skill(code)
    return valid, "; ".join(errors)


def extract_function_name(code: str) -> str:
    match = re.search(r"async\s+def\s+(\w+)\s*\(", code)
    return match.group(1) if match else "unknown_skill"


def load_trajectory(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  无法加载 {path}: {e}")
        return None


def is_successful_trajectory(trajectory_data: Dict[str, Any]) -> bool:
    if "reward" in trajectory_data:
        return trajectory_data.get("reward", 0) == 1.0
    trajectory = trajectory_data.get("trajectory", [])
    if not trajectory:
        return False
    return trajectory[-1].get("reward", 0) == 1.0


def _load_trajectories(args: argparse.Namespace) -> Tuple[List[Tuple[str, Dict[str, Any]]], str]:
    trajectories: List[Tuple[str, Dict[str, Any]]] = []
    if args.jsonl:
        jsonl_path = Path(args.jsonl)
        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL 文件不存在: {jsonl_path}")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    traj_data = json.loads(line)
                    task_id = str(traj_data.get("id", len(trajectories)))
                    trajectories.append((task_id, traj_data))
                except json.JSONDecodeError:
                    continue
        return trajectories, str(jsonl_path)

    if args.trajectory:
        path = Path(args.trajectory)
        if not path.exists():
            raise FileNotFoundError(f"轨迹文件不存在: {path}")
        traj_data = load_trajectory(path)
        if traj_data:
            trajectories.append((path.stem, traj_data))
        return trajectories, str(path)

    dir_path = Path(args.trajectory_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"目录不存在: {dir_path}")
    trajectory_files = sorted(dir_path.glob("*.json"))
    trajectory_files = [f for f in trajectory_files if f.stem.isdigit() or f.stem.startswith("task_")]
    for traj_path in trajectory_files:
        traj_data = load_trajectory(traj_path)
        if traj_data:
            trajectories.append((traj_path.stem, traj_data))
    return trajectories, str(dir_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="多站点单轨迹技能生成器 (v3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 site_skill_generator_v3.py --site gitlab --jsonl all_trajectories.jsonl --filter-success --out skills/gitlab/operation_skills.py
  python3 site_skill_generator_v3.py --site shopping --trajectory 31.json --mode success --out skills/shopping/operation_skills.py
  python3 site_skill_generator_v3.py --site map --trajectory-dir ./trajectories --mode auto --out skills/map/operation_skills.py
        """,
    )
    parser.add_argument("--site", required=True, choices=list_supported_sites(), help="目标站点")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--trajectory", "-t", help="单个轨迹文件路径 (JSON)")
    input_group.add_argument("--trajectory-dir", "-d", help="轨迹文件目录 (批量处理)")
    input_group.add_argument("--jsonl", "-j", help="JSONL 格式的简化轨迹文件")

    parser.add_argument("--out", "-o", required=True, help="输出技能文件路径 (.py)")
    parser.add_argument(
        "--mode",
        "-m",
        choices=["success", "failure", "auto"],
        default="auto",
        help="生成模式: success, failure, auto",
    )
    parser.add_argument("--model", default="gpt-4.1", help="LLM 模型名称")
    parser.add_argument("--filter-success", action="store_true", help="只处理成功轨迹")
    parser.add_argument("--filter-failure", action="store_true", help="只处理失败轨迹")
    parser.add_argument("--limit", type=int, help="只处理前 N 个轨迹")
    parser.add_argument("--resume", action="store_true", help="断点续传/追加到已有技能文件")
    parser.add_argument("--debug", action="store_true", help="显示详细错误信息")
    parser.add_argument("--feedback-kb", help="运行时技能反馈知识库 JSON，用于生成技能时参考历史失败")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    profile = get_site_profile(args.site)

    try:
        trajectories, input_desc = _load_trajectories(args)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return 1

    if args.limit:
        trajectories = trajectories[: args.limit]
    if not trajectories:
        print("错误: 未找到轨迹数据")
        return 1
    feedback_context = load_feedback_context(args.feedback_kb)

    output_path = Path(args.out)
    existing_names = set()
    if args.resume and output_path.exists():
        content = output_path.read_text(encoding="utf-8")
        existing_names = set(re.findall(r"async\s+def\s+(\w+)\s*\(", content))
        print(f"断点续传: 已存在 {len(existing_names)} 个技能")

    print(f"{'=' * 70}")
    print(f"{profile.display_name} 站点单轨迹技能生成器 (v3)")
    print(f"{'=' * 70}")
    print(f"站点: {profile.site}")
    print(f"输入: {input_desc}")
    print(f"输出: {args.out}")
    print(f"轨迹数: {len(trajectories)}")
    print(f"模式: {args.mode}")
    print(f"模型: {args.model}")
    print(f"{'=' * 70}\n")

    skills: List[str] = []
    metadata: List[Dict[str, Any]] = []
    success_count = 0
    failed_count = 0
    skipped_count = 0
    seen_names: Dict[str, Tuple[str, int]] = {}

    for i, (task_id, traj_data) in enumerate(trajectories):
        print(f"[{i + 1}/{len(trajectories)}] 处理 Task {task_id}...", end=" ", flush=True)
        is_success = is_successful_trajectory(traj_data)

        if args.filter_success and not is_success:
            print("跳过 (失败轨迹)")
            skipped_count += 1
            continue
        if args.filter_failure and is_success:
            print("跳过 (成功轨迹)")
            skipped_count += 1
            continue

        mode = "success" if args.mode == "auto" and is_success else "failure" if args.mode == "auto" else args.mode
        print(f"[{mode}] ", end="", flush=True)

        try:
            if mode == "success":
                code = generate_skill_from_success_trajectory(
                    traj_data,
                    profile,
                    model=args.model,
                    feedback_context=feedback_context,
                )
            else:
                code = generate_skill_from_failure_trajectory(
                    traj_data,
                    profile,
                    model=args.model,
                    feedback_context=feedback_context,
                )

            if code is None:
                print("无法生成")
                failed_count += 1
                continue

            is_valid, error_msg = validate_skill_code(code)
            if not is_valid:
                print(f"验证失败: {error_msg}")
                failed_count += 1
                continue

            func_name = extract_function_name(code)
            if func_name in existing_names:
                print("跳过 (已存在)")
                skipped_count += 1
                continue

            meta = {
                "name": func_name,
                "site": profile.site,
                "source_task_id": task_id,
                "source_type": mode,
                "is_success": is_success,
                "source_code": code,
            }
            meta["scenario_descriptor"] = build_scenario_descriptor(traj_data, profile, func_name, mode)
            if func_name in seen_names:
                prev_id, prev_index = seen_names[func_name]
                print(f"替换 {prev_id}", end=" ")
                skills[prev_index] = code
                metadata[prev_index] = meta
                seen_names[func_name] = (task_id, prev_index)
            else:
                seen_names[func_name] = (task_id, len(skills))
                skills.append(code)
                metadata.append(meta)

            success_count += 1
            print(f"✓ {func_name}")

        except Exception as e:
            print(f"异常: {e}")
            failed_count += 1
            if args.debug:
                import traceback

                traceback.print_exc()

    if not skills:
        print("\n错误: 没有成功生成任何技能")
        return 1

    code_lines = [
        '"""',
        f"{profile.display_name} 站点操作级技能库 (v3)",
        "",
        f"从 {len(trajectories)} 个轨迹中生成了 {len(skills)} 个技能",
        "",
        "生成模式:",
        "- 成功轨迹: 从成功执行中提取可复用操作技能",
        "- 操作级失败: 分析失败原因，生成避免失败的操作技能",
        "",
        "技能格式:",
        "- async def 函数",
        "- 第一个参数为 page",
        "- Playwright async API",
        "- 详细 docstring，包含 Usage Log",
        '"""',
        "",
        "import asyncio",
        "import re",
        "from playwright.async_api import Page, TimeoutError as PlaywrightTimeout",
        "",
        "",
        "class ElementNotFoundError(RuntimeError):",
        "    pass",
        "",
        "",
        "class NavigationError(RuntimeError):",
        "    pass",
        "",
        "",
        "class SubmissionError(RuntimeError):",
        "    pass",
        "",
        "",
    ]

    for i, (code, meta) in enumerate(zip(skills, metadata), start=1):
        code_lines.append(f"# {'=' * 68}")
        code_lines.append(f"# Skill {i}/{len(skills)}: {meta['name']}")
        code_lines.append(f"# Source: Task {meta['source_task_id']} ({meta['source_type']})")
        code_lines.append(f"# {'=' * 68}")
        code_lines.append("")
        code_lines.append(code)
        code_lines.append("")
        code_lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_mode = "a" if args.resume and output_path.exists() else "w"
    with open(output_path, write_mode, encoding="utf-8") as f:
        if write_mode == "a":
            f.write("\n\n# ===== 新增技能 (断点续传) =====\n\n")
        f.write("\n".join(code_lines))

    metadata_path = output_path.with_suffix(".json")
    existing_metadata = {"functions": {}, "global_version": 0}
    if args.resume and metadata_path.exists():
        try:
            existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    skillweaver_metadata = existing_metadata
    for meta in metadata:
        skillweaver_metadata["functions"][meta["name"]] = {
            "test_count": 0,
            "success_count": 0,
            "version": 0,
            "references": [],
            "events": [],
            "site": meta["site"],
            "source_task_id": meta["source_task_id"],
            "source_type": meta["source_type"],
            "is_success": meta["is_success"],
            "quality_gate": {"status": "pass", "errors": [], "deterministic": True},
            "scenario_descriptor": meta["scenario_descriptor"],
            "scenario_description": meta["scenario_descriptor"]["W_k"]["scenario_description"],
            "scenario_keywords": meta["scenario_descriptor"]["W_k"]["semantic_keywords"],
            "url_patterns": meta["scenario_descriptor"]["U_k"]["url_patterns"],
        }
        add_interaction_runtime_schema(
            skillweaver_metadata["functions"][meta["name"]],
            meta["source_code"],
        )

    metadata_path.write_text(json.dumps(skillweaver_metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n✓ 技能代码: {output_path}")
    print(f"✓ 元数据: {metadata_path}")
    print(f"成功: {success_count}")
    print(f"跳过: {skipped_count}")
    print(f"失败: {failed_count}")
    print(f"最终技能数: {len(skills)}")
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

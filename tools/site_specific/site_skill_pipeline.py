#!/usr/bin/env python3
"""
Generic trajectory-to-skill pipeline for WebArena sites.

Pipeline:
1. Extract simplified trajectories with extract_trajectory.py
2. Analyze failed trajectories with the selected site's analyzer
3. Split operation-level and reasoning-level failures
4. Generate operation_skills.py/json from success trajectories plus operation failures
5. Generate reasoning_tips.json from reasoning failures
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from site_skill_profiles import SiteSkillProfile, get_site_profile, list_supported_sites
except ImportError:
    from tools.site_specific.site_skill_profiles import (
        SiteSkillProfile,
        get_site_profile,
        list_supported_sites,
    )


SCRIPT_DIR = Path(__file__).parent
TOOLS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
from AgentOccam.drive import generalize_reasoning_text
from tools.site_specific.skill_pool import merge_interaction_pool, merge_reasoning_pool

EXTRACT_TRAJECTORY = PROJECT_ROOT / "extract_trajectory.py"
SKILL_GENERATOR = SCRIPT_DIR / "site_skill_generator_v3.py"
VERIFY_SKILL_SNIPPETS = TOOLS_DIR / "verify_skill_snippets.py"
FEEDBACK_KB_BUILDER = SCRIPT_DIR / "build_skill_feedback_kb.py"
LOCAL_SKILL_PATCHER = SCRIPT_DIR / "patch_interaction_skills.py"


def get_pipeline_components(site: str) -> Dict[str, Path]:
    profile = get_site_profile(site)
    return {
        "extract_trajectory": EXTRACT_TRAJECTORY,
        "analyze_failures": profile.analysis_script,
        "skill_generator": SKILL_GENERATOR,
    }


def run_command(cmd: List[str], description: str) -> bool:
    print(f"\n{'=' * 70}")
    print(f"步骤: {description}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'=' * 70}\n")
    try:
        subprocess.run(cmd, check=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"错误: 命令执行失败 (exit code {e.returncode})")
        return False
    except FileNotFoundError:
        print(f"错误: 找不到命令: {cmd[0]}")
        return False


def step1_extract_trajectories(trajectory_dir: str, merge: bool = True) -> Optional[Path]:
    simplified_dir = Path(trajectory_dir + "_simplified")
    cmd = [
        sys.executable,
        str(EXTRACT_TRAJECTORY),
        trajectory_dir,
        "-o",
        str(simplified_dir),
    ]
    if merge:
        cmd.append("--merge")
    if not run_command(cmd, "提取简化轨迹"):
        return None
    jsonl_path = simplified_dir / "all_trajectories.jsonl"
    if not jsonl_path.exists():
        print(f"警告: 未找到合并的 JSONL 文件: {jsonl_path}")
        return None
    return jsonl_path


def step2_analyze_failures(
    profile: SiteSkillProfile,
    jsonl_path: Path,
    output_dir: Path,
    model: str,
    limit: Optional[int] = None,
    debug: bool = False,
) -> Optional[Path]:
    analyzed_path = output_dir / "failures_analyzed.jsonl"
    has_failures = False
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("reward", 0) != 1.0:
                has_failures = True
                break
    if not has_failures:
        analyzed_path.write_text("", encoding="utf-8")
        print("没有失败轨迹，跳过反事实归因")
        return analyzed_path
    if not profile.analysis_script.exists():
        print(f"错误: 站点失败分析器不存在: {profile.analysis_script}")
        return None

    cmd = [
        sys.executable,
        str(profile.analysis_script),
        "--input",
        str(jsonl_path),
        "--output",
        str(analyzed_path),
        "--model",
        model,
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])
    if debug:
        cmd.append("--debug")
    if not run_command(cmd, f"分析 {profile.display_name} 失败轨迹类型"):
        return None
    return analyzed_path


def step3_split_by_failure_type(
    analyzed_path: Path,
    original_jsonl_path: Path,
    output_dir: Path,
) -> Dict[str, Path]:
    operation_task_ids = set()
    reasoning_task_ids = set()
    analysis_by_id: Dict[Any, Dict[str, Any]] = {}

    with open(analyzed_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_id = record.get("task_id")
            analysis_by_id[task_id] = record
            # DRIVE attribution is binary and exclusive.  Recommendation flags
            # are derived output, never the source of truth for the split.
            failure_level = record.get("failure_level_primary") or record.get("failure_level")
            if failure_level == "operation":
                operation_task_ids.add(task_id)
            elif failure_level == "reasoning":
                reasoning_task_ids.add(task_id)

    operation_failures = []
    reasoning_failures = []
    with open(original_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                traj_data = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_id = traj_data.get("id")
            if task_id in operation_task_ids:
                merged = {**traj_data, "failure_analysis": analysis_by_id.get(task_id, {})}
                operation_failures.append(merged)
            elif task_id in reasoning_task_ids:
                merged = {**traj_data, "failure_analysis": analysis_by_id.get(task_id, {})}
                reasoning_failures.append(merged)

    op_path = output_dir / "operation_failures.jsonl"
    reason_path = output_dir / "reasoning_failures.jsonl"
    with open(op_path, "w", encoding="utf-8") as f:
        for record in operation_failures:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(reason_path, "w", encoding="utf-8") as f:
        for record in reasoning_failures:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n拆分结果:")
    print(f"  操作级失败: {len(operation_failures)} 条 -> {op_path}")
    print(f"  推理级失败: {len(reasoning_failures)} 条 -> {reason_path}")
    return {"operation": op_path, "reasoning": reason_path}


def step4_generate_operation_skills(
    profile: SiteSkillProfile,
    success_jsonl: Path,
    operation_failures_jsonl: Optional[Path],
    output_path: Path,
    model: str,
    limit: Optional[int] = None,
    debug: bool = False,
    feedback_kb_path: Optional[Path] = None,
) -> bool:
    pool_path = output_path.with_name(".drive_knew_operation_skills.py")
    pool_metadata_path = pool_path.with_suffix(".json")
    pool_path.unlink(missing_ok=True)
    pool_metadata_path.unlink(missing_ok=True)
    def cleanup_pool() -> None:
        pool_path.unlink(missing_ok=True)
        pool_metadata_path.unlink(missing_ok=True)
    has_operation_failures = False
    has_successes = False
    with open(success_jsonl, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("reward", 0) == 1.0:
                has_successes = True
                break
    if operation_failures_jsonl and operation_failures_jsonl.exists():
        with open(operation_failures_jsonl, "r", encoding="utf-8") as f:
            has_operation_failures = any(line.strip() for line in f)

    cmd = [
        sys.executable,
        str(SKILL_GENERATOR),
        "--site",
        profile.site,
        "--jsonl",
        str(success_jsonl),
        "--out",
        str(pool_path),
        "--mode",
        "success",
        "--filter-success",
        "--model",
        model,
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])
    if debug:
        cmd.append("--debug")
    if feedback_kb_path and feedback_kb_path.exists():
        cmd.extend(["--feedback-kb", str(feedback_kb_path)])

    if not has_successes and not has_operation_failures:
        print("本轮没有成功轨迹或交互级失败，Knew_i 为空")
        cleanup_pool()
        return True
    success_generation_ok = (
        run_command(cmd, "从成功轨迹生成操作技能") if has_successes else False
    )
    if not success_generation_ok and not has_operation_failures:
        cleanup_pool()
        return False
    if has_successes and not success_generation_ok and has_operation_failures:
        print("警告: 成功轨迹未生成技能，但存在操作级失败；继续从失败轨迹生成操作技能。")

    if has_operation_failures:
        cmd = [
            sys.executable,
            str(SKILL_GENERATOR),
            "--site",
            profile.site,
            "--jsonl",
            str(operation_failures_jsonl),
            "--out",
            str(pool_path),
            "--mode",
            "failure",
            "--model",
            model,
            "--resume",
        ]
        if limit:
            cmd.extend(["--limit", str(limit)])
        if debug:
            cmd.append("--debug")
        if feedback_kb_path and feedback_kb_path.exists():
            cmd.extend(["--feedback-kb", str(feedback_kb_path)])
        if not run_command(cmd, "从操作级失败生成操作技能 (追加)"):
            cleanup_pool()
            return False
    if not pool_path.exists() or not pool_metadata_path.exists():
        print("错误: Knew 交互技能池未生成完整代码/元数据")
        cleanup_pool()
        return False
    if not run_command(
        [sys.executable, str(VERIFY_SKILL_SNIPPETS), "--file", str(pool_path)],
        "验证 Knew 交互技能池",
    ):
        cleanup_pool()
        return False
    try:
        result = merge_interaction_pool(pool_path, output_path)
        print(f"Knew 交互技能批量更新: {result}")
        return True
    except Exception as exc:
        print(f"错误: Knew 交互技能批量更新失败: {exc}")
        return False
    finally:
        cleanup_pool()


def step4_build_skill_feedback_kb(site_dir: Path) -> Optional[Path]:
    failure_log = site_dir / "skill_failure_log.jsonl"
    feedback_log = site_dir / "skill_feedback_log.jsonl"
    if not failure_log.exists() and not feedback_log.exists():
        print("没有运行时技能反馈日志，跳过反馈知识库构建")
        return None
    feedback_kb_path = site_dir / "skill_feedback_kb.json"
    cmd = [
        sys.executable,
        str(FEEDBACK_KB_BUILDER),
        "--site-dir",
        str(site_dir),
        "--out",
        str(feedback_kb_path),
    ]
    if not run_command(cmd, "构建技能反馈知识库"):
        print("警告: 技能反馈知识库构建失败；继续不带反馈生成技能")
        return None
    if not feedback_kb_path.exists():
        print("警告: 技能反馈知识库未生成；继续不带反馈生成技能")
        return None
    return feedback_kb_path


def step5_repair_operation_skills(
    output_path: Path,
    debug: bool = False,
    feedback_kb_path: Optional[Path] = None,
    llm_validate: bool = False,
) -> bool:
    if not output_path.exists():
        print("没有操作级技能文件，跳过明显问题修复")
        return True

    if feedback_kb_path and feedback_kb_path.exists():
        patch_cmd = [
            sys.executable,
            str(LOCAL_SKILL_PATCHER),
            "--file",
            str(output_path),
            "--feedback-kb",
            str(feedback_kb_path),
        ]
        if not run_command(patch_cmd, "按反馈局部修补操作技能选择器集合"):
            return False
        return run_command(
            [sys.executable, str(VERIFY_SKILL_SNIPPETS), "--file", str(output_path)],
            "验证局部修补后的操作技能",
        )

    cmd = [
        sys.executable,
        str(VERIFY_SKILL_SNIPPETS),
        "--file",
        str(output_path),
    ]
    if llm_validate:
        cmd.extend(["--llm-check", "--llm-repair"])
    if debug:
        cmd.append("--verbose")
    return run_command(cmd, "检查并修复操作级技能明显问题")


def step5_generate_reasoning_tips(
    profile: SiteSkillProfile,
    reasoning_failures_jsonl: Path,
    output_path: Path,
) -> bool:
    if not reasoning_failures_jsonl.exists():
        print("没有推理级失败，跳过此步骤")
        return True
    with open(reasoning_failures_jsonl, "r", encoding="utf-8") as f:
        has_content = any(line.strip() for line in f)
    if not has_content:
        print("推理级失败文件为空，跳过此步骤")
        return True

    task_lessons_list = []
    lesson_id_start = 101
    with open(reasoning_failures_jsonl, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            lesson = build_reasoning_lesson(profile, record, lesson_id_start + idx)
            task_lessons_list.append(lesson)

    if not task_lessons_list:
        print("没有有效的推理级失败记录")
        return True

    pool_path = output_path.with_name(".drive_knew_reasoning_tips.json")
    pool_path.write_text(
        json.dumps(task_lessons_list, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    try:
        result = merge_reasoning_pool(pool_path, output_path, site=profile.site)
    except Exception as exc:
        print(f"错误: Knew 推理技能批量更新失败: {exc}")
        return False
    finally:
        pool_path.unlink(missing_ok=True)
    print("\n生成推理级任务经验:")
    print(f"  共 {len(task_lessons_list)} 条任务级经验")
    print(f"  格式: task_lessons (兼容 SkillRegistry)")
    print(f"  已保存: {output_path}")
    print(f"  BatchMerge: {result}")
    return True


def build_reasoning_lesson(
    profile: SiteSkillProfile,
    record: Dict[str, Any],
    lesson_id: int,
) -> Dict[str, Any]:
    failure_analysis = record.get("failure_analysis", {})
    skill_rec = failure_analysis.get("skill_recommendation", {})
    eval_gap = failure_analysis.get("eval_gap", {})
    traj_analysis = failure_analysis.get("trajectory_analysis", {})

    task_id = record.get("id")
    intent = record.get("intent", "")
    task_family = failure_analysis.get("task_family", profile.task_families[0])
    failure_category = failure_analysis.get("failure_category", "")
    failure_description = failure_analysis.get("failure_description", "")
    reasoning_issues = failure_analysis.get("reasoning_issues", [])
    task_level_tips = skill_rec.get("task_level_tips", [])
    if isinstance(reasoning_issues, str):
        reasoning_issues = [reasoning_issues]
    if isinstance(task_level_tips, str):
        task_level_tips = [task_level_tips]

    skill_name = _generate_skill_name(profile, task_family, failure_category, intent)
    semantic_keywords = _extract_keywords(profile, intent, task_family)
    url_patterns = traj_analysis.get("url_patterns", [])
    better_strategy = _convert_tips_to_strategy(task_level_tips)
    verification_strategy = _generate_verification_strategy(profile, task_family, eval_gap)
    mistake = generalize_reasoning_text(failure_description or " ".join(reasoning_issues))
    behavior = generalize_reasoning_text(" ".join(better_strategy))
    verification = generalize_reasoning_text(" ".join(verification_strategy))
    return {
        # Keep cluster_id/cluster_size as compatibility keys for SkillRegistry.
        "cluster_id": lesson_id,
        "skill_name": skill_name,
        "task_family": task_family,
        "cluster_size": 1,
        "skill_id": (
            f"reasoning-{profile.site}-{task_id}"
            if task_id is not None
            else f"reasoning-{profile.site}-{lesson_id}"
        ),
        "reasoning_skill": {
            "M": mistake or "The prior trajectory used an incorrect task-level decision.",
            "B": behavior or "Correct the task interpretation and choose a strategy that satisfies every constraint.",
            "V": verification or "Verify the final state and evaluator requirements before stopping.",
        },
        "scenario_descriptor": {
            "U_k": {
                "site": profile.site,
                "url_patterns": url_patterns,
                "url_match": "compatible_site",
            },
            "W_k": {
                "task_intent": intent,
                "task_family": task_family,
                "semantic_keywords": semantic_keywords,
                "scenario_description": failure_description
                or f"Tasks related to {task_family.lower().replace('_', ' ')}.",
            },
        },
        "statistics": {"N": 0, "S": 0, "lambda": 1.0, "rho": 0.5},
        "task_lessons": {
            "scenario_keywords": semantic_keywords,
            "scenario_description": failure_description
            or f"Tasks related to {task_family.lower().replace('_', ' ')}.",
            "url_patterns": url_patterns,
            "task_pattern": f"Agent attempts to: {intent[:200]}..." if len(intent) > 200 else f"Agent attempts to: {intent}",
            "task_family": task_family,
            "why_failed": failure_description,
            "decision_mistakes": reasoning_issues,
            "better_strategy": better_strategy,
            "failure_signals": _extract_failure_signals(eval_gap, traj_analysis),
            "task_level_tips": [tip if tip.startswith("Tip:") else f"Tip: {tip}" for tip in task_level_tips],
            "expected_final_state": _infer_expected_state(profile, intent, task_family),
            "verification_strategy": verification_strategy,
            "source_task_id": task_id,
        },
    }


def _generate_skill_name(profile: SiteSkillProfile, task_family: str, failure_category: str, intent: str) -> str:
    intent_lower = intent.lower()
    if "count" in intent_lower or "how many" in intent_lower:
        action = "count"
    elif any(word in intent_lower for word in ["find", "search", "show", "extract", "get"]):
        action = "extract"
    elif any(word in intent_lower for word in ["create", "add", "submit", "post", "fork", "save"]):
        action = "create"
    elif any(word in intent_lower for word in ["edit", "update", "change", "set"]):
        action = "update"
    elif any(word in intent_lower for word in ["delete", "remove", "cancel"]):
        action = "remove"
    elif failure_category:
        action = failure_category.split("_")[0]
    else:
        action = task_family.lower().split("_")[0]

    selected_object = profile.default_object
    for obj, keywords in profile.object_keywords.items():
        if any(keyword in intent_lower for keyword in keywords):
            selected_object = obj
            break
    return f"{action}_{selected_object}"


def _extract_keywords(profile: SiteSkillProfile, intent: str, task_family: str) -> List[str]:
    keywords = []
    keywords.extend(profile.family_keywords.get(task_family, []))
    intent_lower = intent.lower()
    for obj, obj_keywords in profile.object_keywords.items():
        if obj in intent_lower or any(keyword in intent_lower for keyword in obj_keywords):
            keywords.extend([obj] + obj_keywords)
    seen = []
    for keyword in keywords:
        normalized = keyword.lower().strip()
        if normalized and normalized not in seen:
            seen.append(normalized)
    return seen[:8]


def _convert_tips_to_strategy(tips: List[str]) -> List[str]:
    strategies = []
    for tip in tips:
        tip_text = tip.replace("Tip:", "").replace("tip:", "").strip()
        if not tip_text:
            continue
        if not tip_text[0].isupper():
            tip_text = tip_text[0].upper() + tip_text[1:]
        strategies.append(tip_text)
    return strategies or ["Verify the task requirements before proceeding."]


def _extract_failure_signals(eval_gap: Dict[str, Any], traj_analysis: Dict[str, Any]) -> List[str]:
    signals = []
    gap_explanation = eval_gap.get("gap_explanation", "") if eval_gap else ""
    if gap_explanation:
        signals.append(gap_explanation[:150])
    must_include = eval_gap.get("must_include_check", {}) if eval_gap else {}
    missing = must_include.get("missing", []) if isinstance(must_include, dict) else []
    if missing:
        signals.append(f"Missing required items: {', '.join(str(item) for item in missing[:3])}")
    missed_actions = traj_analysis.get("missed_actions", []) if traj_analysis else []
    if missed_actions:
        signals.append(f"Missed actions: {missed_actions[0]}")
    return signals or ["Task completion criteria not met."]


def _infer_expected_state(profile: SiteSkillProfile, intent: str, task_family: str) -> str:
    intent_lower = intent.lower()
    if "count" in intent_lower or "how many" in intent_lower:
        return "Agent outputs a numeric count in the required format."
    if task_family in profile.expected_states:
        return profile.expected_states[task_family]
    return f"Task objective '{intent[:100]}...' is successfully completed."


def _generate_verification_strategy(
    profile: SiteSkillProfile,
    task_family: str,
    eval_gap: Dict[str, Any],
) -> List[str]:
    strategies = []
    failed_evaluator = eval_gap.get("failed_evaluator", "") if eval_gap else ""
    if failed_evaluator == "string_match":
        strategies.extend([
            "Verify output matches the exact required format before stopping.",
            "Check every task-requested item is present in the final answer.",
        ])
    elif failed_evaluator == "program_html":
        strategies.extend([
            "Verify the UI state reflects the completed action.",
            "Check required page elements are visible after the operation.",
        ])
    elif failed_evaluator == "url_match":
        strategies.append("Verify the final URL matches the expected page pattern.")
    elif failed_evaluator == "multiple":
        strategies.append("Verify URL, UI state, and final answer before stopping.")
    strategies.extend(profile.verification_strategies.get(task_family, []))
    return strategies or ["Verify task completion before stopping."]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="多站点技能生成完整流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
输出文件:
  {out-dir}/operation_skills.py
  {out-dir}/operation_skills.json
  {out-dir}/reasoning_tips.json
  {out-dir}/failures_analyzed.jsonl

示例:
  python3 site_skill_pipeline.py --site gitlab --trajectories /path/to/AgentOccam_run --out-dir skills/gitlab/
  python3 site_skill_pipeline.py --site shopping --simplified-jsonl all_trajectories.jsonl --out-dir skills/shopping/
        """,
    )
    parser.add_argument("--site", required=True, choices=list_supported_sites(), help="目标站点")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--trajectories", "-t", help="原始轨迹文件夹路径")
    input_group.add_argument("--simplified-jsonl", "-j", help="已简化的轨迹 JSONL 文件路径")
    parser.add_argument("--out-dir", "-o", required=True, help="输出目录路径")
    parser.add_argument("--model", "-m", default="gpt-4.1", help="LLM 模型名称")
    parser.add_argument("--limit", type=int, help="只处理前 N 条记录")
    parser.add_argument("--skip-extract", action="store_true", help="跳过轨迹提取步骤")
    parser.add_argument("--skip-analysis", action="store_true", help="跳过失败分析步骤")
    parser.add_argument("--debug", action="store_true", help="显示详细调试信息")
    parser.add_argument(
        "--llm-validate",
        action="store_true",
        help="额外使用外部 LLM 检查器；默认仅运行确定性质量门禁",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    profile = get_site_profile(args.site)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 70}")
    print(f"{profile.display_name} 技能生成流水线")
    print(f"{'=' * 70}")
    print(f"站点: {profile.site}")
    print(f"输出目录: {out_dir}")
    print(f"模型: {args.model}")
    if args.limit:
        print(f"限制: {args.limit} 条")
    print(f"{'=' * 70}\n")

    if args.simplified_jsonl:
        jsonl_path = Path(args.simplified_jsonl)
        if not jsonl_path.exists():
            print(f"错误: JSONL 文件不存在: {jsonl_path}")
            return 1
        print(f"使用已有的简化轨迹: {jsonl_path}")
    elif args.skip_extract:
        candidate = Path(args.trajectories + "_simplified") / "all_trajectories.jsonl"
        if not candidate.exists():
            print(f"错误: --skip-extract 但未找到: {candidate}")
            return 1
        jsonl_path = candidate
    else:
        jsonl_path = step1_extract_trajectories(args.trajectories)
        if not jsonl_path:
            return 1

    analyzed_path = out_dir / "failures_analyzed.jsonl"
    if args.skip_analysis:
        if not analyzed_path.exists():
            print(f"错误: --skip-analysis 但未找到: {analyzed_path}")
            return 1
        print(f"使用已有分析结果: {analyzed_path}")
    else:
        analyzed_result = step2_analyze_failures(profile, jsonl_path, out_dir, args.model, args.limit, args.debug)
        if not analyzed_result:
            return 1
        analyzed_path = analyzed_result

    split_paths = step3_split_by_failure_type(analyzed_path, jsonl_path, out_dir)
    operation_skills_path = out_dir / "operation_skills.py"
    feedback_kb_path = step4_build_skill_feedback_kb(out_dir)
    if not step4_generate_operation_skills(
        profile,
        jsonl_path,
        split_paths["operation"],
        operation_skills_path,
        args.model,
        args.limit,
        args.debug,
        feedback_kb_path,
    ):
        return 1
    repair_ok = (
        step5_repair_operation_skills(
            operation_skills_path,
            args.debug,
            feedback_kb_path=feedback_kb_path,
            llm_validate=args.llm_validate,
        )
        if feedback_kb_path
        else step5_repair_operation_skills(
            operation_skills_path,
            args.debug,
            llm_validate=args.llm_validate,
        )
    )
    if not repair_ok:
        return 1

    reasoning_tips_path = out_dir / "reasoning_tips.json"
    if not step5_generate_reasoning_tips(profile, split_paths["reasoning"], reasoning_tips_path):
        return 1

    print(f"\n{'=' * 70}")
    print("流水线完成")
    print(f"{'=' * 70}")
    print(f"操作级技能: {operation_skills_path}")
    print(f"技能元数据: {operation_skills_path.with_suffix('.json')}")
    print(f"推理级提示: {reasoning_tips_path}")
    print(f"失败分析: {analyzed_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

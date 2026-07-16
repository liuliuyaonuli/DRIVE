#!/usr/bin/env python3
"""
GitLab 简化轨迹失败分析器 (v2)

分析失败轨迹（含 reference_answer），对比参考答案和任务意图，
判断失败原因是操作级错误还是推理级错误，为后续技能生成提供分类依据。

输入：失败轨迹 JSONL 文件（由 extract_trajectory.py --failures-only --merge 生成）
输出：带失败分类的标注 JSONL 文件

用法:
    # 分析失败轨迹
    python3 tools/site_specific/analyze_gitlab_failures_v2.py \\
        --input /path/to/failed_trajectories.jsonl \\
        --output /path/to/gitlab_failures_analyzed.jsonl

    # 测试前5条
    python3 tools/site_specific/analyze_gitlab_failures_v2.py \\
        --input /path/to/failed_trajectories.jsonl \\
        --output /path/to/gitlab_failures_analyzed.jsonl \\
        --limit 5 --debug
"""

import json
import argparse
import traceback
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

# 添加 tools 目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from llm_helpers import call_llm_json
except ImportError:
    print("错误: 无法导入 llm_helpers")
    print("请确保 tools/llm_helpers.py 文件存在")
    sys.exit(1)

from site_specific.failure_attribution import (
    COUNTERFACTUAL_ATTRIBUTION_INSTRUCTION,
    normalize_failure_attribution,
)


# GitLab 简化轨迹失败分析 Prompt
GITLAB_FAILURE_ANALYSIS_PROMPT = r"""
You are a failure analyst for web automation agents working on GitLab platform.

CRITICAL: ALL input trajectories are CONFIRMED FAILURES from WebArena evaluation.
Your job is NOT to determine success/failure, but to CLASSIFY the failure type.
The "success" field in output should ALWAYS be false.

Analyze the given simplified trajectory, compare with reference_answer, and determine:
1. Whether it's an OPERATION-level or REASONING-level failure
2. Detailed failure analysis with actionable insights for skill generation

CRITICAL: Output ONLY valid JSON. No markdown, no explanations, no code fences.

=====================================================================================
WEBARENA EVALUATION SYSTEM (Why these trajectories failed)
=====================================================================================

WebArena uses 3 evaluation methods. A task fails if ANY evaluator fails:
Final Score = string_match × url_match × program_html (all must be 1.0 to pass)

**1. StringEvaluator (string_match)**
Checks if agent's answer matches expected answer:
- exact_match: Answer must exactly match reference (after lowercase/cleanup)
- must_include: Answer must contain all required phrases
- fuzzy_match: GPT-4 judges semantic equivalence
- ua_match: For impossible tasks, checks if agent correctly identified impossibility

**2. URLEvaluator (url_match)**
Checks if agent's final URL matches expected URL pattern:
- Reference URL path must be contained in agent's final URL path
- All query parameters in reference must exist in agent's URL
- Example failure: Task requires staying on issues LIST page (/issues?label=bug)
  but agent clicked into issue DETAIL page (/issues/1478) → URL mismatch!

**3. HTMLContentEvaluator (program_html)**
Checks if page HTML contains expected elements:
- Navigates to target URL (may be dynamic)
- Uses JS locator (document.querySelector) to find elements
- Checks element content matches required_contents (exact or must_include)
- Example: Verify order status shows "Completed"

COMMON FAILURE PATTERNS:
- Agent gave correct verbal answer but ended on wrong URL → url_match fails
- Agent navigated to right page but gave wrong/incomplete answer → string_match fails
- Agent completed action but page state doesn't reflect it → program_html fails
- Agent clicked into detail view instead of staying on list view → url_match fails

=====================================================================================
INPUT FORMAT (Simplified Trajectory)
=====================================================================================

You will receive a JSON with:
- id: Task ID
- intent: Task objective/goal
- sites: ["gitlab"]
- trajectory: List of steps with {objective, url, plan, reason, action}
- reference_answer: Ground truth containing:
  - answer: Expected answer string
  - must_include: Required substrings in answer
  - must_exclude: Forbidden substrings
  - eval_types: Validation methods (string_match, url_match, program_html)

=====================================================================================
FAILURE TYPE CLASSIFICATION
=====================================================================================

**OPERATION-LEVEL FAILURE** (操作级错误):
Problems with HOW actions were executed. The agent understood the task correctly but failed to execute.

GitLab-specific operation failures:
- Filter not applied: Author/date filter clicked but URL params or UI didn't change
- Pagination incomplete: Didn't scroll/click "Load more" to see all commits/issues
- Async not ready: Didn't wait for Vue.js components to render
- Selector miss: Element not found with given locator
- Form submit failed: Button clicked but form didn't submit
- Navigation failed: Couldn't reach target page
- Verification skipped: Assumed action succeeded without checking URL/UI change
- Over-navigation: Clicked into detail page when should stay on list page
- Multi-step incomplete: Task requires multiple operations (e.g., "fork ALL repos") but agent stopped after partial completion
- Loop not completed: Agent started iterating but didn't finish all iterations

**REASONING-LEVEL FAILURE** (推理级错误):
Problems with WHAT to do. The agent misunderstood the task or made wrong decisions.

GitLab-specific reasoning failures:
- Wrong date interpretation: e.g., "3/5/2023" interpreted as March 5 vs May 3
- Wrong entity: Searched for wrong author/project/branch
- Wrong metric: Counted wrong thing (all commits vs filtered commits)
- Premature conclusion: Said "0" without exhaustive verification
- Wrong page section: Looked in wrong part of GitLab (e.g., issues instead of commits)
- Ignored constraints: Didn't check must_include/must_exclude requirements
- Miscount: Counted visible items incorrectly
- Wrong final state: Understood the answer but navigated to wrong page type
- Unnecessary navigation: Task required staying on list but clicked into detail

=====================================================================================
OUTPUT SCHEMA
=====================================================================================

{
  "task_id": <int>,
  "intent": "<task objective>",
  "task_family": "Repo_Lifecycle|Issue_Management|Commit_Analysis|Merge_Request|Collaboration_Access|Content_Editing|Profile_Settings",

  "success": false,
  "agent_answer": "<what agent concluded in stop action>",
  "expected_answer": "<reference answer>",
  "answer_match": <bool>,

  "failure_level": "operation|reasoning",
  "failure_level_primary": "operation|reasoning",
  "failure_category": "<specific category>",
  "failure_description": "<1-2 sentence description>",

  "operation_issues": [
    "<specific operation-level problem if any>"
  ],
  "reasoning_issues": [
    "<specific reasoning-level problem if any>"
  ],

  "trajectory_analysis": {
    "total_steps": <int>,
    "final_url": "<last URL>",
    "filters_applied": ["<filters agent tried to apply>"],
    "verification_done": <bool>,
    "key_actions": ["<important actions>"],
    "missed_actions": ["<what should have been done>"],
    "multi_step_task": {
      "is_multi_step": <bool>,
      "total_items_required": <int or null>,
      "items_completed": <int or null>,
      "items_remaining": ["<list of unprocessed items if applicable>"]
    }
  },

  "eval_gap": {
    "failed_evaluator": "string_match|url_match|program_html|multiple",
    "would_pass_string_match": <bool>,
    "would_pass_url_match": <bool>,
    "must_include_check": {"required": [...], "found": [...], "missing": [...]},
    "gap_explanation": "<why eval failed - be specific about which evaluator>"
  },

  "skill_recommendation": {
    "needs_operation_skill": <bool>,
    "needs_reasoning_guidance": <bool>,
    "proposed_skill_name": "<snake_case name if operation skill needed>",
    "proposed_skill_description": "<what the skill should do>",
    "task_level_tips": ["<tips for reasoning guidance if needed>"]
  }
}

=====================================================================================
GITLAB TASK FAMILIES
=====================================================================================

1. Repo_Lifecycle: Create/fork/star repos, project templates, license changes
2. Issue_Management: Create/filter/assign issues, milestones, todos
3. Commit_Analysis: Count commits by author/date, find contributors
4. Merge_Request: Create/review MRs, assign reviewers, comment
5. Collaboration_Access: Manage members, roles, groups, follow users
6. Content_Editing: Create files/folders, edit README
7. Profile_Settings: Set status, update profile, manage tokens

=====================================================================================
FAILURE CATEGORIES
=====================================================================================

OPERATION categories:
- "filter_not_applied": Filter action didn't change URL params or UI state
- "pagination_incomplete": Didn't load all pages/items before counting
- "scroll_insufficient": Didn't scroll to load lazy content
- "async_not_ready": Didn't wait for Vue.js components
- "selector_miss": Element locator failed
- "form_submit_failed": Form didn't submit after button click
- "navigation_failed": Couldn't reach target page
- "verification_skipped": Didn't confirm action took effect
- "wrong_element_clicked": Clicked wrong button/link
- "over_navigation": Clicked into detail page when should stay on list (url_match fail)
- "multi_step_incomplete": Task requires N operations but agent only completed M < N (e.g., "fork ALL repos" but only forked some)
- "loop_not_completed": Agent started iterating over items but stopped before processing all
- "action_timeout": Action started but didn't complete within expected time

REASONING categories:
- "wrong_date_format": Misinterpreted date (MM/DD vs DD/MM)
- "wrong_author": Searched for wrong person
- "wrong_project": Navigated to wrong repository
- "wrong_metric": Counted/measured wrong thing
- "premature_stop": Concluded without exhaustive check
- "miscount": Counted visible items incorrectly
- "wrong_page_section": Looked in wrong GitLab section
- "ignored_constraints": Didn't check must_include/exclude
- "false_negative": Said "0" or "not found" when exists
- "false_positive": Said "found" when doesn't exist
- "unnecessary_detail_view": Decided to click into detail when task only needed list view
- "wrong_final_url": Understood task but ended on wrong URL pattern

=====================================================================================
ANALYSIS PROCESS
=====================================================================================

1. Extract agent's answer from last "stop [...]" action

2. Identify which WebArena evaluator(s) failed:
   - Check eval_types in reference_answer to know which evaluators apply
   - For string_match: Compare agent's answer with must_include/answer
   - For url_match: Compare agent's final URL with expected URL pattern
   - For program_html: Check if required page state was achieved

3. Trace through trajectory:
   - Did agent navigate to correct page (commits, issues, etc.)?
   - Did agent apply correct filters (author, date)?
   - Did agent verify filter was applied (URL change, UI change)?
   - Did agent scroll/paginate to see all content?
   - Did agent stay on the right page type (list vs detail)?
   - Did agent provide correct answer?
   - For multi-step tasks (e.g., "fork ALL", "delete ALL", "create N items"):
     * Did agent identify all items that need processing?
     * Did agent complete the operation for EACH item?
     * Did agent verify each individual operation succeeded?
     * Did agent stop only after ALL items were processed?

4. Classify failure:
   - If strategy was correct but execution failed → OPERATION
   - If agent went wrong direction or misunderstood → REASONING
   - If both → MIXED (specify primary)
   - IMPORTANT: "Over-navigation" (clicking into detail when should stay on list)
     is typically REASONING if agent chose to do it, OPERATION if it was accidental

5. Recommend skill or guidance:
   - OPERATION failure → propose skill name and what it should do
   - REASONING failure → propose task-level tips

=====================================================================================
EXAMPLE ANALYSES
=====================================================================================

**Example 1: string_match failure (Miscount)**

Task: "How many commits did Kilian make on 3/5/2023?"

Agent trajectory:
1. Navigated to project → commits page ✓
2. Clicked Author filter ✓
3. Typed "Kilian" and selected ✓
4. Saw commits dated "05 Mar, 2023" and "06 Mar, 2023"
5. Concluded "0 commits on 3/5/2023"

Reference answer: "1"
eval_types: ["string_match"]

Analysis:
- Agent navigated correctly (commits page) ✓
- Agent applied author filter ✓
- BUT: Agent saw "05 Mar, 2023" which IS 3/5/2023 (March 5)
- Agent reported "0" but there was at least 1 commit
- Failed evaluator: string_match (answer "0" doesn't include "1")

Failure classification:
- This is REASONING level: Agent misread/misinterpreted the date display
- The filter was applied (operation worked)
- But agent concluded wrongly (reasoning failed)

failure_level: "reasoning"
failure_category: "miscount" or "wrong_date_format"

**Example 2: url_match failure (Over-navigation into detail page)**

Task: "List all opened issues that report bugs"

Agent trajectory:
1. Navigated to project → Issues page ✓
2. Searched/filtered by "bug" label ✓
3. Found 1 open issue with bug label: "#1478 - Bug 404s..."
4. Clicked into issue #1478 to read details
5. Concluded with detailed description of the bug issue

Reference answer: (doesn't matter for url_match)
eval_types: ["url_match"]
Expected final URL: /{project}/-/issues?label_name=bug (list page)

Analysis:
- Agent navigated to issues page ✓
- Agent applied bug filter correctly ✓
- Agent found the correct issue ✓
- Agent gave detailed correct answer about the issue ✓
- BUT: Agent's final URL is /{project}/-/issues/1478 (detail page)
- Expected URL pattern requires staying on list page: /-/issues?...
- Failed evaluator: url_match (detail page URL doesn't match list page pattern)

Failure classification:
- This is REASONING level: Agent decided to click into detail unnecessarily
- The task asked to "list" issues, not to view details
- Agent should have stopped on the list page after filtering

failure_level: "reasoning"
failure_category: "unnecessary_detail_view" or "wrong_final_url"

**Example 3: program_html failure (Form not submitted)**

Task: "Create an issue titled 'Feature Request'"

Agent trajectory:
1. Navigated to project → New Issue page ✓
2. Filled title field with "Feature Request" ✓
3. Clicked Submit button
4. Concluded "Issue created successfully"

Reference answer: N/A
eval_types: ["program_html"]
program_html checks: page at /issues/{id} should contain "Feature Request"

Analysis:
- Agent navigated to new issue form ✓
- Agent filled title correctly ✓
- Agent clicked submit button ✓
- BUT: Form didn't actually submit (async issue, validation error, etc.)
- Agent assumed success without verifying URL changed to /issues/{id}
- Failed evaluator: program_html (issue doesn't exist, can't find title)

Failure classification:
- This is OPERATION level: Agent's submit action didn't work
- Strategy was correct, but execution failed

failure_level: "operation"
failure_category: "form_submit_failed" or "verification_skipped"

**Example 4: program_html failure (Multi-step task incomplete)**

Task: "Fork all source repos from Akilesh Kannan"

Agent trajectory:
1. Navigated to Akilesh Kannan's profile ✓
2. Found 6 repos: empathy-prompts, CacheEval, nvidia-patch, SimCache, viewgrades-scraper, dots
3. Clicked into empathy-prompts → Fork → Selected namespace → Submitted ✓
4. Clicked into CacheEval → Fork → Selected namespace → Prepared form
5. Stopped and reported "Successfully forked empathy-prompts, CacheEval ready to submit, 4 remaining"

Reference answer: N/A
eval_types: ["program_html"]
program_html checks: All 6 repos should exist in target namespace

Analysis:
- Agent understood the task correctly (fork ALL repos) ✓
- Agent found all 6 repos ✓
- Agent successfully forked 1 repo (empathy-prompts) ✓
- Agent prepared but didn't submit fork for CacheEval ✗
- Agent didn't process remaining 4 repos ✗
- Agent stopped prematurely, reporting partial progress as if task was done
- Failed evaluator: program_html (only 1 of 6 repos forked)

Failure classification:
- This is OPERATION level: Agent understood "fork ALL" but failed to complete all operations
- The strategy was correct, but execution stopped too early
- Root cause: Agent didn't persist through the full loop of operations

failure_level: "operation"
failure_category: "multi_step_incomplete" or "loop_not_completed"
operation_issues: ["multi_step_incomplete", "form_submit_failed", "verification_skipped"]

=====================================================================================
NOW ANALYZE
=====================================================================================

Analyze the GitLab trajectory below and output the JSON analysis:
"""


def extract_agent_answer(trajectory: List[dict]) -> str:
    """从轨迹中提取 agent 的最终答案"""
    if not trajectory:
        return ""

    last_step = trajectory[-1]
    last_action = last_step.get("action", "")

    if last_action.startswith("stop"):
        content = last_action.replace("stop", "").strip()
        if content.startswith("[") and content.endswith("]"):
            content = content[1:-1]
        return content

    return last_action


def analyze_gitlab_trajectory(
    traj: dict,
    model: str = "gpt-4.1",
    debug: bool = False
) -> Optional[dict]:
    """
    分析单个 GitLab 轨迹

    Args:
        traj: 简化的轨迹数据
        model: LLM 模型名称
        debug: 是否打印调试信息

    Returns:
        分析结果字典
    """
    agent_answer = extract_agent_answer(traj.get("trajectory", []))

    input_data = {
        "id": traj.get("id"),
        "intent": traj.get("intent", ""),
        "sites": traj.get("sites", []),
        "trajectory": traj.get("trajectory", []),
        "reference_answer": traj.get("reference_answer"),
        "agent_answer": agent_answer
    }

    prompt = (
        f"{GITLAB_FAILURE_ANALYSIS_PROMPT}\n{COUNTERFACTUAL_ATTRIBUTION_INSTRUCTION}"
        f"\nTRAJECTORY DATA:\n{json.dumps(input_data, indent=2, ensure_ascii=False)}"
    )

    try:
        result = call_llm_json(prompt, model=model)

        if "task_id" not in result:
            result["task_id"] = traj.get("id")

        return normalize_failure_attribution(result)

    except Exception as e:
        if debug:
            print(f"\n    LLM 调用失败: {e}")
            traceback.print_exc()

        return {
            "task_id": traj.get("id"),
            "intent": traj.get("intent", ""),
            "error": str(e),
            "failure_level": "error"
        }


def main():
    parser = argparse.ArgumentParser(
        description="GitLab 简化轨迹失败分析器 (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 分析简化轨迹
  python3 tools/site_specific/analyze_gitlab_failures_v2.py \\
      --input /path/to/all_trajectories.jsonl \\
      --output /path/to/gitlab_failures_analyzed.jsonl

  # 只分析失败轨迹
  python3 tools/site_specific/analyze_gitlab_failures_v2.py \\
      --input /path/to/all_trajectories.jsonl \\
      --output /path/to/gitlab_failures_analyzed.jsonl \\
      --failures-only

  # 测试前5条
  python3 tools/site_specific/analyze_gitlab_failures_v2.py \\
      --input /path/to/all_trajectories.jsonl \\
      --output /path/to/gitlab_failures_analyzed.jsonl \\
      --limit 5 --debug
        """
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="输入的简化轨迹 JSONL 文件路径"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="输出的分析结果 JSONL 文件路径"
    )
    parser.add_argument(
        "--model", "-m",
        default="gpt-4.1",
        help="LLM 模型名称 (默认: gpt-4.1)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="只处理前 N 个轨迹（用于测试）"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续传（跳过已分析的任务）"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="显示详细调试信息"
    )

    args = parser.parse_args()

    # 检查输入文件
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {args.input}")
        return 1

    # 读取轨迹
    trajectories = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                traj = json.loads(line)
                # 只处理 GitLab 轨迹
                if "gitlab" in traj.get("sites", []):
                    trajectories.append(traj)
            except json.JSONDecodeError:
                continue

    if not trajectories:
        print(f"错误: 未找到 GitLab 轨迹")
        return 1

    print(f"找到 {len(trajectories)} 条 GitLab 失败轨迹")

    # 应用 limit
    if args.limit:
        trajectories = trajectories[:args.limit]

    if not trajectories:
        print(f"错误: 没有轨迹可处理")
        return 1

    # 断点续传
    processed_ids = set()
    output_path = Path(args.output)
    if args.resume and output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if "task_id" in rec:
                        processed_ids.add(rec["task_id"])
                except json.JSONDecodeError:
                    continue
        print(f"断点续传: 跳过已分析的 {len(processed_ids)} 条记录")
        trajectories = [t for t in trajectories if t.get("id") not in processed_ids]

    # 打印配置
    print(f"{'='*70}")
    print(f"GitLab 简化轨迹失败分析器 (v2)")
    print(f"{'='*70}")
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print(f"待分析数: {len(trajectories)}")
    print(f"LLM 模型: {args.model}")
    print(f"{'='*70}\n")

    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 统计
    stats = {
        "total": 0,
        "operation": 0,
        "reasoning": 0,
        "mixed": 0,
        "error": 0
    }

    category_counts: Dict[str, int] = {}
    task_family_counts: Dict[str, int] = {}

    # 打开输出文件
    mode = 'a' if args.resume else 'w'
    with open(output_path, mode, encoding='utf-8') as out_f:
        for i, traj in enumerate(trajectories):
            task_id = traj.get("id", i)
            intent = traj.get("intent", "")[:50]

            print(f"[{i+1}/{len(trajectories)}] 任务 {task_id}: {intent}...", end=" ", flush=True)

            result = analyze_gitlab_trajectory(traj, model=args.model, debug=args.debug)

            if result is None:
                print(f"✗ 分析失败")
                stats["error"] += 1
                continue

            # 写入结果
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

            # 更新统计
            stats["total"] += 1
            failure_level = result.get("failure_level", "error")
            task_family = result.get("task_family", "Unknown")
            task_family_counts[task_family] = task_family_counts.get(task_family, 0) + 1

            if failure_level == "operation":
                stats["operation"] += 1
                category = result.get("failure_category", "unknown")
                category_counts[category] = category_counts.get(category, 0) + 1
                print(f"⚙ 操作级 [{task_family}]: {category}")
            elif failure_level == "reasoning":
                stats["reasoning"] += 1
                category = result.get("failure_category", "unknown")
                category_counts[category] = category_counts.get(category, 0) + 1
                print(f"🧠 推理级 [{task_family}]: {category}")
            elif failure_level == "mixed":
                stats["mixed"] += 1
                primary = result.get("failure_level_primary", "unknown")
                category = result.get("failure_category", "unknown")
                category_counts[category] = category_counts.get(category, 0) + 1
                print(f"🔀 混合({primary}) [{task_family}]: {category}")
            else:
                # 包括 "none" 或其他未知类型，视为分类错误
                stats["error"] += 1
                print(f"⚠ 分类异常: {failure_level} [{task_family}]")

    # 打印统计
    print(f"\n{'='*70}")
    print(f"GitLab 失败分析完成")
    print(f"{'='*70}")
    print(f"总计: {stats['total']}")
    print(f"  ⚙ 操作级失败: {stats['operation']}")
    print(f"  🧠 推理级失败: {stats['reasoning']}")
    print(f"  🔀 混合失败: {stats['mixed']}")
    print(f"  ⚠ 分类异常/错误: {stats['error']}")

    if stats['total'] > 0:
        classified_count = stats['operation'] + stats['reasoning'] + stats['mixed']
        if classified_count > 0:
            print(f"\n失败类型占比:")
            print(f"  操作级: {stats['operation']/classified_count*100:.1f}%")
            print(f"  推理级: {stats['reasoning']/classified_count*100:.1f}%")
            print(f"  混合: {stats['mixed']/classified_count*100:.1f}%")

    # 按任务类型统计
    if task_family_counts:
        print(f"\n按任务类型统计:")
        for tf, count in sorted(task_family_counts.items(), key=lambda x: -x[1]):
            print(f"  {tf}: {count}")

    # 按失败类别统计
    if category_counts:
        print(f"\n按失败类别统计:")
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count}")

    print(f"{'='*70}")

    # 下一步提示
    print(f"\n下一步: 使用当前统一流水线生成操作级和推理级技能")
    print(f"  python3 tools/site_specific/site_skill_pipeline.py \\")
    print(f"    --site gitlab \\")
    print(f"    --simplified-jsonl {args.input} \\")
    print(f"    --out-dir skills/gitlab \\")
    print(f"    --model {args.model}")

    return 1 if stats["error"] else 0


if __name__ == "__main__":
    sys.exit(main())

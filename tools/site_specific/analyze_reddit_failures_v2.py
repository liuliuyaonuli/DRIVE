#!/usr/bin/env python3
"""
Reddit 简化轨迹失败分析器 (v2)

分析失败轨迹（含 reference_answer），对比参考答案和任务意图，
判断失败原因是操作级错误还是推理级错误，为后续技能生成提供分类依据。

输入：失败轨迹 JSONL 文件（由 extract_trajectory.py --failures-only --merge 生成）
输出：带失败分类的标注 JSONL 文件

工作流程:
1. 成功轨迹 → reddit_skill_generator_v3.py --mode success → 操作技能
2. 失败轨迹 → 本分析器 → 分类为操作级/推理级
   - 操作级失败 → reddit_skill_generator_v3.py --mode failure → 操作技能
   - 推理级失败 → reddit_skill_pipeline.py → reasoning_tips.json

用法:
    # 分析失败轨迹
    python3 tools/site_specific/analyze_reddit_failures_v2.py \\
        --input /path/to/failed_trajectories.jsonl \\
        --output /path/to/reddit_failures_analyzed.jsonl

    # 测试前5条
    python3 tools/site_specific/analyze_reddit_failures_v2.py \\
        --input /path/to/failed_trajectories.jsonl \\
        --output /path/to/reddit_failures_analyzed.jsonl \\
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


# Reddit 简化轨迹失败分析 Prompt
REDDIT_FAILURE_ANALYSIS_PROMPT = r"""
You are a failure analyst for web automation agents working on Reddit/Postmill platform.

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
- Example failure: Task requires staying on forum LIST page but agent clicked into POST page

**3. HTMLContentEvaluator (program_html)**
Checks if page HTML contains expected elements:
- Navigates to target URL (may be dynamic)
- Uses JS locator (document.querySelector) to find elements
- Checks element content matches required_contents (exact or must_include)
- Example: Verify post was created, comment was added, vote was recorded

COMMON FAILURE PATTERNS:
- Agent gave correct verbal answer but ended on wrong URL → url_match fails
- Agent navigated to right page but gave wrong/incomplete answer → string_match fails
- Agent completed action but page state doesn't reflect it → program_html fails
- Agent clicked into post detail instead of staying on forum list → url_match fails

=====================================================================================
INPUT FORMAT (Simplified Trajectory)
=====================================================================================

You will receive a JSON with:
- id: Task ID
- intent: Task objective/goal (may be in "task" field)
- sites: ["reddit"]
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

Reddit/Postmill-specific operation failures:
- Vote not registered: Clicked upvote/downvote but vote state didn't change (arrow color)
- Post not created: Filled form but post didn't appear in forum
- Comment not added: Submitted comment but it didn't appear under post
- Navigation failed: Couldn't reach target forum/post/user page
- Selector miss: Element not found (vote button, submit button, text input)
- Form submit failed: Post/comment form didn't submit properly
- Scroll incomplete: Didn't scroll to load more posts/comments
- Wait insufficient: Didn't wait for dynamic content (lazy loading posts)
- Wrong element clicked: Clicked wrong button/link
- Subscription failed: Subscribe/unsubscribe action didn't take effect
- Profile update failed: User settings change didn't save
- Flair not set: Post flair selection didn't work
- Sort not applied: Sort option (new/top/hot) didn't change post order

**REASONING-LEVEL FAILURE** (推理级错误):
Problems with WHAT to do. The agent misunderstood the task or made wrong decisions.

Reddit/Postmill-specific reasoning failures:
- Wrong forum: Navigated to wrong subreddit/forum
- Wrong post: Selected wrong post (wrong title, wrong author, wrong time)
- Wrong user: Looked at wrong user's profile/posts
- Wrong count: Miscounted posts/comments/upvotes
- Wrong time period: Didn't filter by correct time (newest/oldest/this week)
- Premature conclusion: Said "not found" without checking all pages
- Wrong answer format: Gave wrong format (full sentence vs just number/name)
- Misread content: Extracted wrong information from post/comment
- Wrong sort order: Used wrong sort when looking for newest/oldest/top
- Ignored constraints: Didn't check must_include requirements
- Wrong calculation: Calculated karma/votes/statistics incorrectly

=====================================================================================
OUTPUT SCHEMA
=====================================================================================

{
  "task_id": <int>,
  "intent": "<task objective>",
  "task_family": "Information_Retrieval|Account_Configuration|Interaction_Voting|Post_Creation|Content_Repost|Comment_Reply|Post_Editing",

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
    "forums_visited": ["<forums agent navigated to>"],
    "actions_performed": ["<key actions like vote, post, comment>"],
    "verification_done": <bool>,
    "key_actions": ["<important actions>"],
    "missed_actions": ["<what should have been done>"]
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
REDDIT TASK FAMILIES (7 Categories)
=====================================================================================

1. Information_Retrieval (~12%): Find and extract info
   - Find newest/oldest post in forum
   - Count posts/comments/subscribers
   - Get user karma/statistics
   - Find post by title/content
   Core operations: navigate to forum/user → sort/filter → read info

2. Account_Configuration (~6%): Profile and settings
   - Update bio/avatar
   - Change notification settings
   - Subscribe/unsubscribe to forums
   Core operations: go to settings → modify form → save

3. Interaction_Voting (~20%): Votes and reactions
   - Upvote/downvote posts or comments
   - Check vote status
   Core operations: find target → click vote arrow → verify state change

4. Post_Creation (~38%): Create new content
   - Submit text/link post to forum
   - Set post flair
   - Cross-post to another forum
   Core operations: go to submit → fill title/body → select flair → submit

5. Content_Repost (~10%): Share existing content
   - Repost content to another forum
   - Share post
   Core operations: find source → use share/crosspost → select target → submit

6. Comment_Reply (~4%): Comments and replies
   - Add comment to post
   - Reply to existing comment
   Core operations: go to post → scroll to comment box → type → submit

7. Post_Editing (~10%): Modify existing content
   - Edit post title/body
   - Delete post/comment
   - Change post flair
   Core operations: find own post → click edit → modify → save

=====================================================================================
FAILURE CATEGORIES
=====================================================================================

OPERATION categories:
- "vote_not_registered": Vote arrow clicked but color didn't change
- "post_not_created": Form submitted but post not visible in forum
- "comment_not_added": Comment submitted but not appearing
- "navigation_failed": Couldn't reach target page (404, wrong redirect)
- "selector_miss": Element not found (button, input, link)
- "form_submit_failed": Submit button clicked but form didn't post
- "scroll_incomplete": Didn't load all lazy content
- "wait_insufficient": Didn't wait for dynamic content
- "subscription_failed": Subscribe action didn't change subscription status
- "profile_update_failed": Settings change didn't save
- "flair_not_set": Flair selection didn't apply to post
- "sort_not_applied": Sort option didn't reorder content
- "wrong_element_clicked": Clicked wrong interactive element

REASONING categories:
- "wrong_forum": Navigated to incorrect subreddit
- "wrong_post": Selected post with wrong attributes
- "wrong_user": Looked at wrong user profile
- "wrong_count": Miscounted items
- "wrong_time_filter": Used incorrect time period
- "premature_stop": Gave up too early without checking all options
- "wrong_answer_format": Correct info but wrong format
- "misread_content": Extracted wrong information
- "wrong_sort": Used incorrect sort order
- "ignored_constraints": Didn't satisfy must_include requirements
- "wrong_calculation": Math error in karma/vote calculation

=====================================================================================
ANALYSIS TASK
=====================================================================================

Analyze the trajectory below and output ONLY valid JSON matching the schema above.
Focus on:
1. Identifying PRIMARY failure type (operation vs reasoning)
2. Pinpointing exact failure category
3. Providing actionable skill recommendation

TRAJECTORY:
{trajectory_json}

OUTPUT (JSON only, no markdown):
"""


def analyze_single_trajectory(
    trajectory_data: Dict[str, Any],
    model: str = "gpt-4.1"
) -> Dict[str, Any]:
    """
    分析单个失败轨迹

    Args:
        trajectory_data: 简化后的轨迹数据
        model: LLM 模型名称

    Returns:
        失败分析结果
    """
    # 构建 prompt
    trajectory_json = json.dumps(trajectory_data, ensure_ascii=False, indent=2)
    # 使用字符串拼接代替 .format()，避免 JSON 示例中的花括号被误解析
    prompt = (
        REDDIT_FAILURE_ANALYSIS_PROMPT.replace("{trajectory_json}", trajectory_json)
        + "\n"
        + COUNTERFACTUAL_ATTRIBUTION_INSTRUCTION
    )

    try:
        result = call_llm_json(prompt, model=model)

        # 确保必要字段存在
        result.setdefault("task_id", trajectory_data.get("id"))
        result.setdefault("intent", trajectory_data.get("intent", trajectory_data.get("task", "")))
        result.setdefault("success", False)
        result.setdefault("failure_level", "unknown")
        result.setdefault("failure_level_primary", "unknown")
        result.setdefault("skill_recommendation", {
            "needs_operation_skill": False,
            "needs_reasoning_guidance": False
        })

        return normalize_failure_attribution(result)

    except Exception as e:
        # 返回错误信息
        return {
            "task_id": trajectory_data.get("id"),
            "intent": trajectory_data.get("intent", trajectory_data.get("task", "")),
            "success": False,
            "failure_level": "unknown",
            "failure_level_primary": "unknown",
            "failure_category": "analysis_error",
            "failure_description": f"Analysis failed: {str(e)}",
            "operation_issues": [],
            "reasoning_issues": [],
            "skill_recommendation": {
                "needs_operation_skill": False,
                "needs_reasoning_guidance": False,
                "error": str(e)
            }
        }


def main():
    parser = argparse.ArgumentParser(
        description="Reddit 简化轨迹失败分析器 (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析失败轨迹
  python3 analyze_reddit_failures_v2.py \\
      --input failed_trajectories.jsonl \\
      --output reddit_failures_analyzed.jsonl

  # 测试前5条
  python3 analyze_reddit_failures_v2.py \\
      --input failed_trajectories.jsonl \\
      --output reddit_failures_analyzed.jsonl \\
      --limit 5 --debug

输出文件格式:
  每行一个 JSON 对象，包含:
  - failure_level: "operation" | "reasoning" (binary; never mixed)
  - failure_level_primary: 主要失败类型
  - skill_recommendation.needs_operation_skill: 是否需要生成操作技能
  - skill_recommendation.needs_reasoning_guidance: 是否需要推理级提示
        """
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="输入失败轨迹 JSONL 文件路径"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="输出分析结果 JSONL 文件路径"
    )
    parser.add_argument(
        "--model", "-m",
        default="gpt-4.1",
        help="LLM 模型名称 (默认: gpt-4.1)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只处理前 N 条记录 (测试用)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续传 (跳过已处理的 task_id)"
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

    # 读取输入数据
    trajectories = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    traj = json.loads(line)
                    # 只处理失败轨迹 (reward != 1.0)
                    if traj.get("reward", 0) != 1.0:
                        trajectories.append(traj)
                except json.JSONDecodeError as e:
                    print(f"警告: JSON 解析错误: {e}")
                    continue

    if not trajectories:
        print("错误: 未找到失败轨迹")
        return 1

    # 应用 limit
    if args.limit:
        trajectories = trajectories[:args.limit]

    # 断点续传: 读取已处理的 task_id
    processed_ids = set()
    output_path = Path(args.output)
    if args.resume and output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    processed_ids.add(record.get("task_id"))
                except:
                    continue
        print(f"断点续传: 已处理 {len(processed_ids)} 条记录")

    print(f"{'='*70}")
    print(f"Reddit 失败轨迹分析器 (v2)")
    print(f"{'='*70}")
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print(f"待分析数: {len(trajectories)}")
    print(f"模型: {args.model}")
    print(f"{'='*70}\n")

    # 统计
    success_count = 0
    error_count = 0
    skipped_count = 0
    operation_count = 0
    reasoning_count = 0
    mixed_count = 0

    # 打开输出文件
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_mode = 'a' if args.resume else 'w'

    with open(output_path, write_mode, encoding='utf-8') as out_f:
        for i, traj in enumerate(trajectories):
            task_id = traj.get("id", i)
            intent = traj.get("intent", traj.get("task", ""))[:50]

            # 跳过已处理
            if task_id in processed_ids:
                print(f"[{i+1}/{len(trajectories)}] 跳过 Task {task_id} (已处理)")
                skipped_count += 1
                continue

            print(f"[{i+1}/{len(trajectories)}] 分析 Task {task_id}: {intent}...", end=" ", flush=True)

            try:
                # 分析
                result = analyze_single_trajectory(traj, model=args.model)

                # 统计失败类型
                level = result.get("failure_level_primary", "unknown")
                if level == "operation":
                    operation_count += 1
                elif level == "reasoning":
                    reasoning_count += 1
                else:
                    error_count += 1
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_f.flush()
                    print(f"✗ 分类失败 [{level}]: {result.get('error', 'invalid attribution')}")
                    continue

                # 写入结果
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()

                success_count += 1
                needs_op = result.get("skill_recommendation", {}).get("needs_operation_skill", False)
                needs_reason = result.get("skill_recommendation", {}).get("needs_reasoning_guidance", False)
                print(f"✓ [{level}] op_skill={needs_op}, reason_guide={needs_reason}")

                if args.debug:
                    print(f"    Category: {result.get('failure_category', 'unknown')}")
                    print(f"    Description: {result.get('failure_description', '')[:80]}")

            except Exception as e:
                error_count += 1
                print(f"✗ 错误: {e}")
                if args.debug:
                    traceback.print_exc()

    # 打印统计
    print(f"\n{'='*70}")
    print(f"分析完成")
    print(f"{'='*70}")
    print(f"成功: {success_count}")
    print(f"跳过: {skipped_count}")
    print(f"错误: {error_count}")
    print(f"\n失败类型分布:")
    print(f"  操作级 (operation): {operation_count}")
    print(f"  推理级 (reasoning): {reasoning_count}")
    print(f"  混合 (mixed): {mixed_count}")
    print(f"\n输出文件: {args.output}")
    print(f"{'='*70}")

    return 1 if error_count else 0


if __name__ == "__main__":
    sys.exit(main())

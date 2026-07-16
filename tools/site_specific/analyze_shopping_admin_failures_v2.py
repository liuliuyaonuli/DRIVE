#!/usr/bin/env python3
"""
Shopping Admin simplified trajectory failure analyzer (v2).

Classifies confirmed failed WebArena trajectories into operation-level and
reasoning-level failures, then emits the same schema used by the current Reddit
pipeline for downstream operation-skill and reasoning-tip generation.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional


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


SHOPPING_ADMIN_FAILURE_ANALYSIS_PROMPT = r"""
You are a failure analyst for web automation agents working on a Magento-style
Shopping Admin console.

CRITICAL: ALL input trajectories are CONFIRMED FAILURES from WebArena evaluation.
Your job is NOT to determine success/failure, but to CLASSIFY the failure type.
The "success" field in output should ALWAYS be false.

Analyze the simplified trajectory, compare with reference_answer, and determine:
1. Whether it is an OPERATION-level or REASONING-level failure
2. Detailed failure analysis with actionable insights for skill generation

CRITICAL: Output ONLY valid JSON. No markdown, no code fences.

=====================================================================================
WEBARENA EVALUATION SYSTEM
=====================================================================================

WebArena task success can require:
- string_match: final answer exactly or semantically matches reference
- url_match: final URL matches the expected URL pattern
- program_html: page HTML contains required state after the action

Admin failures often happen because the agent saved the wrong record, failed to
apply a grid filter, did not wait for a Magento admin page to finish rendering,
or stopped before verifying that a save/bulk action took effect.

=====================================================================================
INPUT FORMAT
=====================================================================================

You will receive JSON with:
- id: Task ID
- intent: Task objective
- sites: ["shopping_admin"]
- trajectory: list of steps with url, reason, action
- reference_answer: WebArena ground truth/evaluation information
- agent_answer: extracted final stop action

=====================================================================================
FAILURE TYPE CLASSIFICATION
=====================================================================================

OPERATION-LEVEL FAILURE:
The agent understood what to do but failed to execute the admin workflow.

Shopping Admin operation failures:
- grid_filter_not_applied: product/order/customer grid filter did not change results
- save_not_completed: Save button clicked but record did not persist
- bulk_action_incomplete: bulk action applied to only some selected rows
- pagination_incomplete: did not inspect all pages of an admin grid
- async_not_ready: acted before Magento admin JS/grid finished loading
- selector_miss: target form field, grid row, or button not found
- wrong_row_selected: clicked or edited a row different from the target record
- form_submit_failed: validation or submit failure after filling fields
- verification_skipped: assumed save/action worked without reloading or checking row state
- navigation_failed: could not reach the correct admin section

REASONING-LEVEL FAILURE:
The agent chose the wrong admin object, section, value, or stopping condition.

Shopping Admin reasoning failures:
- wrong_product_or_order: selected wrong SKU, order, or customer
- wrong_admin_section: used Products instead of Orders, Customers, CMS, etc.
- wrong_status_value: set or reported the wrong status/attribute
- wrong_date_range: filtered by wrong time period
- wrong_metric: counted or aggregated the wrong report value
- miscount: counted visible rows incorrectly
- ignored_constraints: missed must_include/must_exclude or objective constraints
- premature_stop: concluded before all pages/records were checked
- wrong_answer_format: answer content was right but format did not match evaluator

=====================================================================================
OUTPUT SCHEMA
=====================================================================================

{
  "task_id": <int>,
  "intent": "<task objective>",
  "task_family": "Catalog_Product_Admin|Order_Management|Customer_Management|Promotion_Rules|Content_CMS|Store_Configuration|Reports_Analytics",

  "success": false,
  "agent_answer": "<what agent concluded in stop action>",
  "expected_answer": "<reference answer>",
  "answer_match": <bool>,

  "failure_level": "operation|reasoning",
  "failure_level_primary": "operation|reasoning",
  "failure_category": "<specific category>",
  "failure_description": "<1-2 sentence description>",

  "operation_issues": ["<operation-level issue if any>"],
  "reasoning_issues": ["<reasoning-level issue if any>"],

  "trajectory_analysis": {
    "total_steps": <int>,
    "final_url": "<last URL>",
    "filters_applied": ["<filters attempted>"],
    "records_touched": ["<products/orders/customers/etc.>"],
    "verification_done": <bool>,
    "key_actions": ["<important actions>"],
    "missed_actions": ["<what should have been done>"],
    "multi_step_task": {
      "is_multi_step": <bool>,
      "total_items_required": <int or null>,
      "items_completed": <int or null>,
      "items_remaining": ["<unprocessed items if applicable>"]
    }
  },

  "eval_gap": {
    "failed_evaluator": "string_match|url_match|program_html|multiple",
    "would_pass_string_match": <bool>,
    "would_pass_url_match": <bool>,
    "must_include_check": {"required": [...], "found": [...], "missing": [...]},
    "gap_explanation": "<specific evaluator gap>"
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
TASK FAMILIES
=====================================================================================

1. Catalog_Product_Admin: update products, SKU, stock, categories, prices
2. Order_Management: order status, invoices, shipments, comments, refunds
3. Customer_Management: customers, groups, addresses, account settings
4. Promotion_Rules: coupons, catalog rules, cart rules, discounts
5. Content_CMS: CMS pages, blocks, banners, content edits
6. Store_Configuration: tax, shipping, store settings, payment configuration
7. Reports_Analytics: sales/order/product reports and counts

=====================================================================================
ANALYSIS PROCESS
=====================================================================================

1. Extract the final answer from stop action.
2. Identify failed evaluator(s) from reference_answer/eval_types.
3. Trace whether the agent reached the correct admin section, selected the right
   record, applied required filters, completed saves/bulk actions, and verified state.
4. Classify:
   - Correct strategy but failed UI execution -> operation
   - Wrong record/section/value/interpretation -> reasoning
   - Both -> mixed, with a primary level
5. Recommend:
   - operation failure -> needs_operation_skill true
   - reasoning failure -> needs_reasoning_guidance true with task_level_tips

Analyze the Shopping Admin trajectory below and output the JSON analysis:
"""


def extract_agent_answer(trajectory: List[dict]) -> str:
    if not trajectory:
        return ""
    last_action = trajectory[-1].get("action", "")
    if last_action.startswith("stop"):
        content = last_action.replace("stop", "").strip()
        if content.startswith("[") and content.endswith("]"):
            content = content[1:-1]
        return content
    return last_action


def analyze_shopping_admin_trajectory(
    traj: dict,
    model: str = "gpt-4.1",
    debug: bool = False,
) -> Optional[dict]:
    agent_answer = extract_agent_answer(traj.get("trajectory", []))
    input_data = {
        "id": traj.get("id"),
        "intent": traj.get("intent", ""),
        "sites": traj.get("sites", []),
        "trajectory": traj.get("trajectory", []),
        "reference_answer": traj.get("reference_answer"),
        "agent_answer": agent_answer,
    }
    prompt = (
        f"{SHOPPING_ADMIN_FAILURE_ANALYSIS_PROMPT}\n"
        f"{COUNTERFACTUAL_ATTRIBUTION_INSTRUCTION}\n"
        f"TRAJECTORY DATA:\n{json.dumps(input_data, indent=2, ensure_ascii=False)}"
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
            "failure_level": "error",
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Shopping Admin 简化轨迹失败分析器 (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 tools/site_specific/analyze_shopping_admin_failures_v2.py \\
      --input /path/to/all_trajectories.jsonl \\
      --output /path/to/shopping_admin_failures_analyzed.jsonl

  python3 tools/site_specific/analyze_shopping_admin_failures_v2.py \\
      --input /path/to/all_trajectories.jsonl \\
      --output /path/to/shopping_admin_failures_analyzed.jsonl \\
      --limit 5 --debug
        """,
    )
    parser.add_argument("--input", "-i", required=True, help="输入的简化轨迹 JSONL 文件路径")
    parser.add_argument("--output", "-o", required=True, help="输出的分析结果 JSONL 文件路径")
    parser.add_argument("--model", "-m", default="gpt-4.1", help="LLM 模型名称")
    parser.add_argument("--limit", type=int, help="只处理前 N 个轨迹")
    parser.add_argument("--resume", action="store_true", help="断点续传")
    parser.add_argument("--debug", action="store_true", help="显示详细调试信息")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {args.input}")
        return 1

    trajectories = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                traj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "shopping_admin" in traj.get("sites", []):
                trajectories.append(traj)

    if args.limit:
        trajectories = trajectories[: args.limit]
    if not trajectories:
        print("错误: 未找到 Shopping Admin 轨迹")
        return 1

    output_path = Path(args.output)
    processed_ids = set()
    if args.resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "task_id" in rec:
                    processed_ids.add(rec["task_id"])
        trajectories = [t for t in trajectories if t.get("id") not in processed_ids]

    print(f"{'=' * 70}")
    print("Shopping Admin 简化轨迹失败分析器 (v2)")
    print(f"{'=' * 70}")
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print(f"待分析数: {len(trajectories)}")
    print(f"LLM 模型: {args.model}")
    print(f"{'=' * 70}\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats: Dict[str, int] = {"total": 0, "operation": 0, "reasoning": 0, "mixed": 0, "error": 0}
    category_counts: Dict[str, int] = {}
    task_family_counts: Dict[str, int] = {}
    mode = "a" if args.resume else "w"

    with open(output_path, mode, encoding="utf-8") as out_f:
        for i, traj in enumerate(trajectories):
            task_id = traj.get("id", i)
            intent = traj.get("intent", "")[:50]
            print(f"[{i + 1}/{len(trajectories)}] 任务 {task_id}: {intent}...", end=" ", flush=True)

            result = analyze_shopping_admin_trajectory(traj, model=args.model, debug=args.debug)
            if result is None:
                print("失败")
                stats["error"] += 1
                continue

            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

            stats["total"] += 1
            level = result.get("failure_level_primary") or result.get("failure_level", "error")
            if level in stats:
                stats[level] += 1
            else:
                stats["error"] += 1

            category = result.get("failure_category", "unknown")
            category_counts[category] = category_counts.get(category, 0) + 1
            family = result.get("task_family", "unknown")
            task_family_counts[family] = task_family_counts.get(family, 0) + 1
            print(f"{level} / {category}")

    print(f"\n{'=' * 70}")
    print("分析完成")
    print(f"{'=' * 70}")
    print(f"总计: {stats['total']}")
    print(f"操作级: {stats['operation']}")
    print(f"推理级: {stats['reasoning']}")
    print(f"混合: {stats['mixed']}")
    print(f"错误: {stats['error']}")
    print(f"输出: {output_path}")
    print("\n任务族分布:")
    for family, count in sorted(task_family_counts.items(), key=lambda item: item[1], reverse=True):
        print(f"  {family}: {count}")
    print("\n失败类别分布:")
    for category, count in sorted(category_counts.items(), key=lambda item: item[1], reverse=True):
        print(f"  {category}: {count}")
    return 1 if stats["error"] else 0


if __name__ == "__main__":
    sys.exit(main())

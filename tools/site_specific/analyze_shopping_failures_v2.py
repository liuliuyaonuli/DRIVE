#!/usr/bin/env python3
"""
Shopping 简化轨迹失败分析器 (v2)

分析失败轨迹（含 reference_answer），对比参考答案和任务意图，
判断失败原因是操作级错误还是推理级错误，为后续技能生成提供分类依据。

输入：失败轨迹 JSONL 文件（由 extract_trajectory.py --failures-only --merge 生成）
输出：带失败分类的标注 JSONL 文件

用法:
    # 分析失败轨迹
    python3 tools/site_specific/analyze_shopping_failures_v2.py \\
        --input /path/to/failed_trajectories.jsonl \\
        --output /path/to/shopping_failures_analyzed.jsonl

    # 测试前5条
    python3 tools/site_specific/analyze_shopping_failures_v2.py \\
        --input /path/to/failed_trajectories.jsonl \\
        --output /path/to/shopping_failures_analyzed.jsonl \\
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


# Shopping 简化轨迹失败分析 Prompt
SHOPPING_FAILURE_ANALYSIS_PROMPT = r"""
You are a failure analyst for web automation agents working on e-commerce Shopping platform.

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
- Example failure: Task requires staying on product LIST page but agent clicked into product DETAIL page

**3. HTMLContentEvaluator (program_html)**
Checks if page HTML contains expected elements:
- Navigates to target URL (may be dynamic)
- Uses JS locator (document.querySelector) to find elements
- Checks element content matches required_contents (exact or must_include)
- Example: Verify cart contains specific product, order status shows "Shipped"

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
- sites: ["shopping"]
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

Shopping-specific operation failures:
- Filter not applied: Price/brand/category filter clicked but URL params or UI didn't change
- Search not executed: Search input filled but search not triggered
- Pagination incomplete: Didn't click "Load more" or next page to see all products/orders
- Async not ready: Didn't wait for product cards, prices, or reviews to render
- Selector miss: Element not found (Add to Cart button, filter checkbox, etc.)
- Form submit failed: Address form, contact form, or review form didn't submit
- Navigation failed: Couldn't reach target page (product page, order page, etc.)
- Verification skipped: Assumed action succeeded without checking cart count/order status
- Over-navigation: Clicked into product detail when should stay on list page
- Multi-step incomplete: Task requires multiple operations (add multiple items, process multiple orders) but agent stopped early
- Cart operation failed: Add to cart, update quantity, or remove item didn't work
- Login/auth issue: Operation requires login but session expired or not logged in

**REASONING-LEVEL FAILURE** (推理级错误):
Problems with WHAT to do. The agent misunderstood the task or made wrong decisions.

Shopping-specific reasoning failures:
- Wrong product selected: Found wrong product (wrong brand, wrong specs, wrong price range)
- Wrong calculation: Calculated total/refund/discount incorrectly
- Wrong date interpretation: Misinterpreted date format or time period
- Wrong order selected: Found wrong order (wrong status, wrong date range)
- Premature conclusion: Said "not found" or "0" without exhaustive search
- Wrong page section: Looked in wrong section (orders vs wishlist, reviews vs Q&A)
- Ignored constraints: Didn't check must_include requirements (specific product attributes)
- Miscount: Counted products/orders/reviews incorrectly
- Wrong aggregation: Aggregated wrong data for statistics (wrong time period, wrong category)
- Unnecessary navigation: Task required staying on list but clicked into detail
- Wrong answer format: Gave answer in wrong format (full sentence vs just number)

=====================================================================================
OUTPUT SCHEMA
=====================================================================================

{
  "task_id": <int>,
  "intent": "<task objective>",
  "task_family": "Product_Search_Info|Reviews_Ratings|Cart_Wishlist_Account|Order_Status|Order_Statistics|Aftermarket_Service",

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
    "searches_performed": ["<search queries executed>"],
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
SHOPPING TASK FAMILIES (6 Categories)
=====================================================================================

1. Product_Search_Info (~33%, 64 tasks): Product retrieval and information
   围绕"找到什么商品 / 看清它的基本信息"
   Subtypes:
   - Browse/filter by category/brand/price
   - Sorting by price/rating
   - Keyword search
   - Find cheapest/most expensive product
   - Get price range
   - Product recommendations
   Core operations: search + list page operations + read product title/price/specs

2. Reviews_Ratings (~12%, 24 tasks): Reviews and reputation analysis
   围绕"看别人的评价怎么说"
   Subtypes:
   - Find reviewers by content/rating
   - Filter reviews by star rating
   - Summarize reviews
   - Find criticisms
   Core operations: open reviews section → filter by stars/keywords → extract info

3. Cart_Wishlist_Account (~21%, 40 tasks): Cart/wishlist/account operations
   偏"操作型"任务：点按钮、修改资料、提交表单
   Subtypes:
   - Add to cart/wishlist
   - Update shipping address
   - Reorder previous purchase
   - Rate purchased product
   Core operations: add to cart, add to wishlist, modify address, reorder, write review

4. Order_Status (~11%, 21 tasks): Order status and details queries
   围绕"查某一单 / 某几单的具体信息"
   Subtypes:
   - Single order details (status, total, delivery date)
   - Find order by number/status
   - Recent order lookup
   Core operations: filter/open orders → read status, amount, delivery time

5. Order_Statistics (~11%, 21 tasks): Order and spending statistics
   偏"跨多单的统计 / 聚合信息"
   Subtypes:
   - Spending by time period/category
   - Order count by criteria
   - First/last purchase date
   Core operations: iterate order list → aggregate/find earliest/latest records

6. Aftermarket_Service (~11%, 22 tasks): After-sales and customer service
   围绕"退款、优惠、联系客服"
   Subtypes:
   - Calculate refund amount (with/without shipping)
   - Fill contact form (don't submit)
   - Find customer service info
   - Subscribe to newsletter
   Core operations: find after-sales entry → fill form/text → simple arithmetic

=====================================================================================
FAILURE CATEGORIES
=====================================================================================

OPERATION categories:
- "filter_not_applied": Price/rating/brand filter didn't change URL or results
- "search_not_executed": Search input filled but not triggered
- "sort_not_applied": Sort option selected but products not reordered
- "pagination_incomplete": Didn't load all pages before counting/finding
- "scroll_insufficient": Didn't scroll to load lazy content
- "async_not_ready": Didn't wait for product cards/prices/reviews to render
- "selector_miss": Element locator failed (button, input, link not found)
- "form_submit_failed": Form didn't submit after button click
- "navigation_failed": Couldn't reach target page
- "verification_skipped": Didn't confirm action took effect (cart count, order status)
- "wrong_element_clicked": Clicked wrong button/link
- "over_navigation": Clicked into detail page when should stay on list
- "multi_step_incomplete": Task requires N operations but only completed M < N
- "cart_add_failed": Add to cart button didn't work or cart not updated
- "cart_update_failed": Quantity update or remove didn't work
- "wishlist_add_failed": Add to wishlist didn't work
- "login_required": Operation needs login but not authenticated
- "stock_check_missed": Didn't check out-of-stock before adding to cart

REASONING categories:
- "wrong_product": Selected product with wrong attributes (brand, specs, price)
- "wrong_calculation": Calculated total/refund/spending incorrectly
- "wrong_date_format": Misinterpreted date (MM/DD vs DD/MM)
- "wrong_time_period": Used wrong date range for statistics
- "wrong_order": Found order with wrong status/date
- "wrong_metric": Counted/measured wrong thing
- "premature_stop": Concluded without exhaustive search
- "miscount": Counted visible items incorrectly
- "wrong_page_section": Looked in wrong section (orders vs wishlist)
- "ignored_constraints": Didn't check must_include requirements
- "false_negative": Said "not found" when exists
- "false_positive": Said "found" when doesn't exist
- "unnecessary_detail_view": Clicked into detail when task only needed list
- "wrong_final_url": Understood task but ended on wrong URL pattern
- "wrong_answer_format": Correct info but wrong format (sentence vs number)
- "wrong_aggregation": Aggregated wrong data for statistics

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
   - Did agent navigate to correct page (product list, order history, etc.)?
   - Did agent apply correct filters (price, rating, category)?
   - Did agent verify filter was applied (URL change, UI change)?
   - Did agent search with correct keywords?
   - Did agent scroll/paginate to see all content?
   - Did agent stay on the right page type (list vs detail)?
   - Did agent calculate correctly (prices, refunds, totals)?
   - Did agent provide correct answer in correct format?
   - For multi-step tasks:
     * Did agent identify all items that need processing?
     * Did agent complete the operation for EACH item?
     * Did agent verify each individual operation succeeded?

4. Classify failure:
   - If strategy was correct but execution failed → OPERATION
   - If agent went wrong direction or misunderstood → REASONING
   - If both → MIXED (specify primary)
   - "Over-navigation" is typically REASONING if agent chose to do it

5. Recommend skill or guidance:
   - OPERATION failure → propose skill name and what it should do
   - REASONING failure → propose task-level tips

=====================================================================================
EXAMPLE ANALYSES
=====================================================================================

**Example 1: string_match failure (Wrong calculation)**

Task: "How much did I spend on shopping last month?"

Agent trajectory:
1. Navigated to My Account → Order History ✓
2. Found 3 orders in last month ✓
3. Saw order totals: $45.99, $123.50, $67.00
4. Concluded "Total spending: $236.49"

Reference answer: "236.49" (must_include)
eval_types: ["string_match"]

Analysis:
- Agent navigated correctly ✓
- Agent found correct orders ✓
- Agent calculated: 45.99 + 123.50 + 67.00 = 236.49 ✓
- BUT: Agent said "Total spending: $236.49" instead of just "236.49"
- Failed evaluator: string_match (format mismatch)

Failure classification:
- This is REASONING level: Agent gave correct number but wrong format
- The calculation was correct, but answer format was wrong

failure_level: "reasoning"
failure_category: "wrong_answer_format"

**Example 2: url_match failure (Over-navigation)**

Task: "Show me products under $50 in Electronics category"

Agent trajectory:
1. Navigated to Electronics category ✓
2. Applied price filter: max $50 ✓
3. Found 5 products matching criteria ✓
4. Clicked on first product to see details
5. Reported "Found 5 products under $50..."

Reference answer: N/A
eval_types: ["url_match"]
Expected final URL: /electronics?price=0-50 (list page)

Analysis:
- Agent navigated to Electronics ✓
- Agent applied price filter ✓
- Agent found correct products ✓
- BUT: Agent's final URL is /product/12345 (detail page)
- Expected URL requires staying on list page with filters
- Failed evaluator: url_match

Failure classification:
- This is REASONING level: Agent decided to click into detail unnecessarily
- Task asked to "show products" (list), not view details

failure_level: "reasoning"
failure_category: "unnecessary_detail_view"

**Example 3: program_html failure (Cart operation failed)**

Task: "Add the cheapest laptop to my cart"

Agent trajectory:
1. Navigated to Laptops category ✓
2. Sorted by price ascending ✓
3. Found cheapest: "Basic Laptop $299" ✓
4. Clicked "Add to Cart" button
5. Concluded "Added Basic Laptop to cart"

Reference answer: N/A
eval_types: ["program_html"]
program_html checks: Cart should contain "Basic Laptop"

Analysis:
- Agent navigated to Laptops ✓
- Agent sorted by price ✓
- Agent found cheapest product ✓
- Agent clicked Add to Cart ✓
- BUT: Cart still shows empty (program_html check fails)
- Agent didn't verify cart count changed
- Failed evaluator: program_html (product not in cart)

Failure classification:
- This is OPERATION level: Agent's Add to Cart action didn't work
- Strategy was correct, but execution failed (async issue? out of stock?)

failure_level: "operation"
failure_category: "cart_add_failed" or "verification_skipped"

**Example 4: string_match failure (Wrong product selected)**

Task: "What is the price of iPhone 15 Pro Max 256GB?"

Agent trajectory:
1. Searched for "iPhone 15 Pro Max" ✓
2. Found product "iPhone 15 Pro Max 128GB - $999"
3. Concluded "The price is $999"

Reference answer: "$1099" (must_include)
eval_types: ["string_match"]

Analysis:
- Agent searched correctly ✓
- BUT: Agent found 128GB version instead of 256GB
- The 256GB version costs $1099, not $999
- Agent didn't verify the storage capacity matched
- Failed evaluator: string_match (wrong price)

Failure classification:
- This is REASONING level: Agent selected wrong product variant
- Should have verified all specs match (256GB requirement)

failure_level: "reasoning"
failure_category: "wrong_product"

**Example 5: program_html failure (Multi-step incomplete)**

Task: "Add all items from my wishlist to cart"

Agent trajectory:
1. Navigated to Wishlist ✓
2. Found 4 items in wishlist ✓
3. Clicked "Add to Cart" on first item ✓
4. Clicked "Add to Cart" on second item ✓
5. Concluded "Added items to cart"

Reference answer: N/A
eval_types: ["program_html"]
program_html checks: All 4 wishlist items should be in cart

Analysis:
- Agent found wishlist with 4 items ✓
- Agent added 2 items to cart ✓
- BUT: Only 2 of 4 items were added
- Agent stopped after partial completion
- Failed evaluator: program_html (only 2 items in cart)

Failure classification:
- This is OPERATION level: Agent understood "add ALL" but didn't complete
- Strategy was correct, execution stopped too early

failure_level: "operation"
failure_category: "multi_step_incomplete"

=====================================================================================
NOW ANALYZE
=====================================================================================

Analyze the Shopping trajectory below and output the JSON analysis:
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


def analyze_shopping_trajectory(
    traj: dict,
    model: str = "gpt-4.1",
    debug: bool = False
) -> Optional[dict]:
    """
    分析单个 Shopping 轨迹

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
        f"{SHOPPING_FAILURE_ANALYSIS_PROMPT}\n{COUNTERFACTUAL_ATTRIBUTION_INSTRUCTION}"
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
        description="Shopping 简化轨迹失败分析器 (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 分析简化轨迹
  python3 tools/site_specific/analyze_shopping_failures_v2.py \\
      --input /path/to/all_trajectories.jsonl \\
      --output /path/to/shopping_failures_analyzed.jsonl

  # 只分析失败轨迹
  python3 tools/site_specific/analyze_shopping_failures_v2.py \\
      --input /path/to/all_trajectories.jsonl \\
      --output /path/to/shopping_failures_analyzed.jsonl \\
      --failures-only

  # 测试前5条
  python3 tools/site_specific/analyze_shopping_failures_v2.py \\
      --input /path/to/all_trajectories.jsonl \\
      --output /path/to/shopping_failures_analyzed.jsonl \\
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
                # 只处理 Shopping 轨迹
                if "shopping" in traj.get("sites", []):
                    trajectories.append(traj)
            except json.JSONDecodeError:
                continue

    if not trajectories:
        print(f"错误: 未找到 Shopping 轨迹")
        return 1

    print(f"找到 {len(trajectories)} 条 Shopping 失败轨迹")

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
    print(f"Shopping 简化轨迹失败分析器 (v2)")
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

            result = analyze_shopping_trajectory(traj, model=args.model, debug=args.debug)

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
                stats["error"] += 1
                print(f"⚠ 分类异常: {failure_level} [{task_family}]")

    # 打印统计
    print(f"\n{'='*70}")
    print(f"Shopping 失败分析完成")
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
    print(f"    --site shopping \\")
    print(f"    --simplified-jsonl {args.input} \\")
    print(f"    --out-dir skills/shopping \\")
    print(f"    --model {args.model}")

    return 1 if stats["error"] else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Map 简化轨迹失败分析器 (v2)

分析失败轨迹（含 reference_answer），对比参考答案和任务意图，
判断失败原因是操作级错误还是推理级错误，为后续技能生成提供分类依据。

输入：失败轨迹 JSONL 文件（由 extract_trajectory.py --failures-only --merge 生成）
输出：带失败分类的标注 JSONL 文件

用法:
    # 分析失败轨迹
    python3 tools/site_specific/analyze_map_failures_v2.py \\
        --input /path/to/failed_trajectories.jsonl \\
        --output /path/to/map_failures_analyzed.jsonl

    # 测试前5条
    python3 tools/site_specific/analyze_map_failures_v2.py \\
        --input /path/to/failed_trajectories.jsonl \\
        --output /path/to/map_failures_analyzed.jsonl \\
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


# Map 简化轨迹失败分析 Prompt
MAP_FAILURE_ANALYSIS_PROMPT = r"""
You are a failure analyst for web automation agents working on Map/Navigation platforms (Google Maps, OpenStreetMap, etc.).

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
- Example failure: Task requires showing route but agent stayed on search page

**3. HTMLContentEvaluator (program_html)**
Checks if page HTML contains expected elements:
- Navigates to target URL (may be dynamic)
- Uses JS locator (document.querySelector) to find elements
- Checks element content matches required_contents (exact or must_include)
- Example: Verify POI detail page shows correct information

COMMON FAILURE PATTERNS:
- Agent gave correct verbal answer but ended on wrong URL → url_match fails
- Agent navigated to right page but gave wrong/incomplete answer → string_match fails
- Agent completed action but page state doesn't reflect it → program_html fails
- Agent found wrong POI due to ambiguous search → string_match fails

=====================================================================================
INPUT FORMAT (Simplified Trajectory)
=====================================================================================

You will receive a JSON with:
- id: Task ID
- intent: Task objective/goal
- sites: ["map"]
- trajectory: List of steps with {objective, url, plan, reason, action}
- reference_answer: Ground truth containing:
  - answer: Expected answer string
  - must_include: Required substrings in answer
  - must_exclude: Forbidden substrings
  - eval_types: Validation methods (string_match, url_match, program_html)

=====================================================================================
MAP TASK FAMILIES (6 Major Categories)
=====================================================================================

**1. Travel_Time_Distance (出行时间/距离计算 & 路线规划)**
- Single-leg: Calculate time/distance from A to B with specific transport mode
- Multi-leg comparison: Compare walking vs driving time
- Show route/directions: Display route on map with step-by-step directions
- Examples:
  - "How long does it take to walk from CMU to UPitt?"
  - "What is the minimum travel time by car from CMU to University of Pittsburgh?"
  - "Show the route from SCS CMU to where the Declaration of Independence was signed"

**2. Multi_Stop_Itinerary (多点/多段行程规划)**
- Multi-destination trips: Visit multiple points in order or optimal sequence
- Combined transport modes: "Walk to A, then drive to B"
- Total time/distance calculation across multiple legs
- Examples:
  - "Given these 5 universities, start from the first and visit all, what's total distance?"
  - "Walk from school to Starbucks, then drive from Starbucks to airport, total time?"

**3. Nearby_POI_Search (最近/附近 POI 检索)**
- Find nearest X from a location
- Filter by brand/chain/local store
- Filter by time/distance threshold ("within 15 min drive")
- Examples:
  - "Find the nearest Trader Joe's from CMU"
  - "What supermarkets are within 15 min drive from the hotel?"
  - "Nearest national park and driving time from here"

**4. Reachability_Check (可达性判断/距离阈值判断)**
- Yes/No questions about reaching destinations within constraints
- Distance/time threshold verification
- Examples:
  - "Can I drive from CMU to this government office within 1 hour?"
  - "Is there a mall within 50km of here?"

**5. POI_Attribute_Query (地点属性信息查询)**
- Address/ZIP code lookup
- Coordinates (latitude/longitude) query
- Contact info (phone, website, hours, operator)
- Examples:
  - "What are the complete addresses of international airports within 100 miles?"
  - "What is the ZIP code of Carnegie Mellon University?"
  - "What are the coordinates of this coffee shop in decimal degrees?"
  - "What are the opening hours of this Japanese restaurant?"

**6. Knowledge_Location_Resolution (地图页面定位 + 地理知识结合)**
- Open specific POI detail page on map
- Resolve location from indirect description (events, movies, famous people)
- Combine world knowledge with map navigation
- Examples:
  - "Pull up the description page of Carnegie Music Hall on Map"
  - "Navigate to where Mr. Rogers was filmed"
  - "What's the driving time from CMU to where Nash equilibrium author studied undergrad?"
  - "Get directions from where the 1980 Super Bowl champions play to 1991 champions' stadium"

=====================================================================================
FAILURE TYPE CLASSIFICATION
=====================================================================================

**OPERATION-LEVEL FAILURE** (操作级错误):
Problems with HOW actions were executed. The agent understood the task correctly but failed to execute.

Map-specific operation failures:
- Search not executed: Search query typed but not submitted
- Route not calculated: Origin/destination set but route calculation not triggered
- Transport mode not selected: Default mode used instead of specified (walk/drive/bike/transit)
- POI not selected: Search results appeared but correct POI not clicked
- Directions not displayed: Route exists but step-by-step view not opened
- Async not ready: Map tiles or results didn't finish loading
- Wrong POI clicked: Multiple search results, clicked wrong one
- Zoom/pan insufficient: Didn't zoom to see required details
- Multi-stop not completed: Added some waypoints but not all
- Form interaction failed: Input field not properly filled
- Coordinates not extracted: Found location but didn't read coordinates

**REASONING-LEVEL FAILURE** (推理级错误):
Problems with WHAT to do. The agent misunderstood the task or made wrong decisions.

Map-specific reasoning failures:
- Wrong destination: Searched for wrong place entirely
- Knowledge gap: Couldn't resolve indirect location reference (e.g., didn't know which team won 1980 Super Bowl)
- Wrong transport mode interpretation: Task said "walk" but agent searched for driving
- Wrong metric: Gave distance when time was asked, or vice versa
- Unit confusion: Gave km when miles expected, or wrong time format
- Premature conclusion: Said "no route" without trying alternatives
- Wrong comparison: Compared wrong pair of options
- Misread map data: Read wrong value from displayed route info
- Wrong POI selected due to name confusion: Found similar-named but wrong location
- Incomplete answer: Answered only part of multi-part question
- Wrong attribute extracted: Asked for phone but gave address
- Ignored constraints: Didn't apply brand/time/distance filters correctly

=====================================================================================
OUTPUT SCHEMA
=====================================================================================

{
  "task_id": <int>,
  "intent": "<task objective>",
  "task_family": "Travel_Time_Distance|Multi_Stop_Itinerary|Nearby_POI_Search|Reachability_Check|POI_Attribute_Query|Knowledge_Location_Resolution",

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
    "search_queries": ["<searches agent performed>"],
    "transport_mode_used": "<walk|drive|bike|transit|none>",
    "route_calculated": <bool>,
    "poi_found": <bool>,
    "verification_done": <bool>,
    "key_actions": ["<important actions>"],
    "missed_actions": ["<what should have been done>"],
    "multi_stop_task": {
      "is_multi_stop": <bool>,
      "total_stops_required": <int or null>,
      "stops_added": <int or null>,
      "stops_remaining": ["<list of unvisited stops if applicable>"]
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
FAILURE CATEGORIES
=====================================================================================

OPERATION categories:
- "search_not_submitted": Search query typed but Enter/Search not clicked
- "route_not_calculated": Origin/destination set but route not triggered
- "transport_mode_not_set": Didn't select correct transport mode (walk/drive/bike/transit)
- "poi_not_clicked": Correct POI in results but not selected
- "directions_not_opened": Route calculated but detailed directions not viewed
- "async_not_ready": Map/results still loading when action taken
- "wrong_poi_selected": Clicked incorrect POI from multiple results
- "zoom_insufficient": Didn't zoom to required detail level
- "multi_stop_incomplete": Some waypoints added but not all
- "input_field_failed": Couldn't type in search/origin/destination field
- "coordinates_not_read": Found location but failed to extract coordinates
- "attribute_not_extracted": Found POI page but didn't read required field
- "selector_miss": Element not found with given locator
- "navigation_failed": Couldn't reach target page/view

REASONING categories:
- "wrong_destination": Searched for completely wrong location
- "knowledge_resolution_failed": Couldn't map description to real place (e.g., sports team → stadium)
- "wrong_transport_mode": Used driving when walking was specified, etc.
- "wrong_metric_type": Gave distance when time asked, or vice versa
- "unit_error": Wrong units (km vs miles, hours vs minutes)
- "premature_conclusion": Said impossible without exhaustive check
- "wrong_comparison": Compared wrong options in comparison task
- "misread_value": Read incorrect value from map display
- "poi_name_confusion": Selected similarly-named but wrong POI
- "incomplete_answer": Answered only part of multi-part question
- "wrong_attribute": Extracted wrong field from POI info
- "filter_not_applied": Didn't use brand/time/distance constraints
- "wrong_origin": Started route from wrong location
- "multi_stop_order_wrong": Visited stops in wrong sequence

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
   - Did agent search for the correct location(s)?
   - Did agent select the correct transport mode?
   - Did agent successfully calculate the route?
   - Did agent read the correct values (time, distance, address, etc.)?
   - Did agent visit POI detail page when needed?
   - Did agent resolve knowledge-based location references correctly?
   - For multi-stop tasks:
     * Did agent add all required waypoints?
     * Did agent calculate total time/distance correctly?
     * Did agent visit stops in correct order (if required)?

4. Classify failure:
   - If strategy was correct but execution failed → OPERATION
   - If agent went wrong direction or misunderstood → REASONING
   - If both → MIXED (specify primary)

5. Recommend skill or guidance:
   - OPERATION failure → propose skill name and what it should do
   - REASONING failure → propose task-level tips

=====================================================================================
EXAMPLE ANALYSES
=====================================================================================

**Example 1: string_match failure (Wrong transport mode)**

Task: "How long does it take to walk from CMU to UPitt?"

Agent trajectory:
1. Navigated to Google Maps ✓
2. Entered "CMU" as origin ✓
3. Entered "UPitt" as destination ✓
4. Clicked "Directions" ✓
5. (Default driving mode selected)
6. Read "12 minutes" from driving route
7. Concluded "12 minutes"

Reference answer: "45 minutes" (walking time)
eval_types: ["string_match"]

Analysis:
- Agent navigated correctly to maps ✓
- Agent set correct origin and destination ✓
- BUT: Agent used driving mode instead of walking
- "12 minutes" is driving time, not walking time
- Failed evaluator: string_match (answer doesn't match walking time)

Failure classification:
- This is OPERATION level if agent intended to select walking but failed to click it
- This is REASONING level if agent didn't realize walking mode was required

failure_level: "operation" (if walking icon was clicked but didn't register)
failure_category: "transport_mode_not_set"

**Example 2: string_match failure (Knowledge resolution failed)**

Task: "What's the driving time from CMU to where Nash equilibrium author studied undergrad?"

Agent trajectory:
1. Navigated to Google Maps ✓
2. Searched "Nash equilibrium author university"
3. Got confused results, clicked on Princeton University
4. Set route from CMU to Princeton
5. Concluded "5 hours 30 minutes"

Reference answer: "4 hours 20 minutes" (to Carnegie Tech/CMU - Nash studied at Carnegie)
eval_types: ["string_match"]

Analysis:
- Agent tried to resolve the knowledge reference
- BUT: John Nash studied undergrad at Carnegie Institute of Technology (now CMU)
- Agent incorrectly mapped to Princeton (where Nash did PhD, not undergrad)
- This is a knowledge/reasoning error

failure_level: "reasoning"
failure_category: "knowledge_resolution_failed"
reasoning_issues: ["Incorrectly identified Nash's undergrad school as Princeton instead of Carnegie Tech"]

**Example 3: url_match failure (POI page not opened)**

Task: "Pull up the description page of Carnegie Music Hall on Map"

Agent trajectory:
1. Navigated to Google Maps ✓
2. Searched "Carnegie Music Hall Pittsburgh" ✓
3. Saw search results with Carnegie Music Hall listed
4. Concluded "Found Carnegie Music Hall at 4400 Forbes Ave"

Reference answer: N/A
eval_types: ["url_match"]
Expected final URL: URL containing place ID for Carnegie Music Hall detail page

Analysis:
- Agent searched correctly ✓
- Agent found correct POI in results ✓
- BUT: Agent didn't click to open the POI detail page
- Agent's URL is still on search results, not detail page
- Failed evaluator: url_match

Failure classification:
- This is OPERATION level: Agent found it but didn't click to open detail

failure_level: "operation"
failure_category: "poi_not_clicked"

**Example 4: Multi-stop task incomplete**

Task: "Walk from CMU to Starbucks on Craig St, then drive from there to Pittsburgh Airport. What's the total travel time?"

Agent trajectory:
1. Navigated to Google Maps ✓
2. Searched for route from CMU to Starbucks Craig St ✓
3. Selected walking mode ✓
4. Found walking time: 15 minutes ✓
5. Concluded "15 minutes"

Reference answer: "1 hour 10 minutes" (15 min walk + 55 min drive)
eval_types: ["string_match"]

Analysis:
- Agent correctly calculated first leg ✓
- BUT: Agent stopped after first leg
- Didn't calculate second leg (Starbucks to Airport by car)
- Didn't add up total time

failure_level: "operation"
failure_category: "multi_stop_incomplete"
operation_issues: ["Only calculated first leg, didn't add second leg to airport"]

**Example 5: Nearby POI with constraints**

Task: "Find supermarkets within 15 minutes drive from Hilton Hotel Downtown Pittsburgh"

Agent trajectory:
1. Navigated to Google Maps ✓
2. Searched "Hilton Hotel Downtown Pittsburgh" ✓
3. Searched "supermarket near me" ✓
4. Listed all visible supermarkets
5. Concluded with list of 10 supermarkets

Reference answer: "Giant Eagle, Trader Joe's, Target" (only 3 are within 15 min)
eval_types: ["string_match"]

Analysis:
- Agent found the hotel ✓
- Agent searched for supermarkets ✓
- BUT: Agent didn't filter by 15-minute drive constraint
- Listed all supermarkets, including ones >15 min away

failure_level: "reasoning"
failure_category: "filter_not_applied"
reasoning_issues: ["Didn't verify each supermarket is within 15 min drive time"]

=====================================================================================
NOW ANALYZE
=====================================================================================

Analyze the Map trajectory below and output the JSON analysis:
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


def analyze_map_trajectory(
    traj: dict,
    model: str = "gpt-4.1",
    debug: bool = False
) -> Optional[dict]:
    """
    分析单个 Map 轨迹

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
        f"{MAP_FAILURE_ANALYSIS_PROMPT}\n{COUNTERFACTUAL_ATTRIBUTION_INSTRUCTION}"
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
        description="Map 简化轨迹失败分析器 (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 分析简化轨迹
  python3 tools/site_specific/analyze_map_failures_v2.py \\
      --input /path/to/all_trajectories.jsonl \\
      --output /path/to/map_failures_analyzed.jsonl

  # 只分析失败轨迹
  python3 tools/site_specific/analyze_map_failures_v2.py \\
      --input /path/to/all_trajectories.jsonl \\
      --output /path/to/map_failures_analyzed.jsonl \\
      --failures-only

  # 测试前5条
  python3 tools/site_specific/analyze_map_failures_v2.py \\
      --input /path/to/all_trajectories.jsonl \\
      --output /path/to/map_failures_analyzed.jsonl \\
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
                # 只处理 Map 轨迹
                if "map" in traj.get("sites", []):
                    trajectories.append(traj)
            except json.JSONDecodeError:
                continue

    if not trajectories:
        print(f"错误: 未找到 Map 轨迹")
        return 1

    print(f"找到 {len(trajectories)} 条 Map 失败轨迹")

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
    print(f"Map 简化轨迹失败分析器 (v2)")
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

            result = analyze_map_trajectory(traj, model=args.model, debug=args.debug)

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
    print(f"Map 失败分析完成")
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
    print(f"    --site map \\")
    print(f"    --simplified-jsonl {args.input} \\")
    print(f"    --out-dir skills/map \\")
    print(f"    --model {args.model}")

    return 1 if stats["error"] else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Reddit 站点单轨迹技能生成器 (v3)

针对单个轨迹生成技能，支持:
1. 成功轨迹: 从成功的任务执行轨迹生成可复用技能 (参考 SkillWeaver)
2. 失败轨迹 (操作级): 从失败轨迹分析生成避免失败的技能

工作流程:
1. 成功轨迹 (reward=1.0) → 直接生成操作技能
2. 失败轨迹 → 先用 analyze_reddit_failures_v2.py 分析
   - 操作级失败 → 本工具生成操作技能
   - 推理级失败 → reddit_skill_pipeline.py 生成 reasoning_tips.json

输入格式:
- 单个 JSON 文件 (--trajectory)
- JSON 文件目录 (--trajectory-dir)
- JSONL 文件 (--jsonl)，支持简化轨迹格式

生成的技能格式严格符合 SkillWeaver 标准:
- async def 函数，使用 Playwright API
- 详细的 docstring，包含 Usage Log
- page.goto() 设置初始状态
- 返回有意义的结果

用法:
    # 从 JSONL 文件生成技能（推荐）
    python3 tools/site_specific/reddit_skill_generator_v3.py \\
        --jsonl /path/to/all_trajectories.jsonl \\
        --out skills/reddit/skills.py \\
        --filter-success  # 只处理成功轨迹

    # 从单个成功轨迹生成技能
    python3 tools/site_specific/reddit_skill_generator_v3.py \\
        --trajectory /path/to/trajectory.json \\
        --mode success \\
        --out skills/reddit/success_skills.py

    # 从目录批量处理
    python3 tools/site_specific/reddit_skill_generator_v3.py \\
        --trajectory-dir /path/to/trajectories/ \\
        --out skills/reddit/batch_skills.py
"""

import json
import argparse
import re
import sys
import ast
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Sequence
from urllib.parse import urlparse

# 添加tools目录到path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from llm_helpers import call_llm_json, call_llm_text
except ImportError as e:
    print(f"错误: 无法导入必需模块: {e}")
    print("请确保 tools/llm_helpers.py 存在")
    sys.exit(1)

from site_specific.drive_artifacts import add_interaction_runtime_schema
from site_specific.skill_quality import validate_generated_interaction_skill
from site_specific.failure_attribution import (
    COUNTERFACTUAL_ATTRIBUTION_INSTRUCTION,
    normalize_failure_attribution,
)


# ============================================================================
# SkillWeaver 风格的技能生成 Prompt (用于成功轨迹)
# ============================================================================

SKILLWEAVER_BASE_PROMPT = """You are learning how to use a Reddit-like website (Postmill). You are building a procedural knowledge base with Python functions using the Playwright API.

Write 'skills' - Python code snippets representing logical procedures to perform the task you just completed. Make this represent the GENERAL case, not a specific case.

**Function Declaration Requirements**
- Write a detailed docstring (with triple quotes) describing:
  - What the function does
  - How to use it
  - Any unexpected behavior observed
  - A "Usage Log" section describing when you've used it and what happened
- Begin with a `page.goto("/...")` call using relative URL to set initial state
- Use `page` as the first argument
- Do NOT use `dict` as parameter type
- Avoid `*_id` or `*_url` parameters (prefer `post_title`, `username`, etc.)
- Function must be async with `await`
- No nested functions
- No global try-catch that only prints errors
- For every grounding operation, use a small ordered selector list (CSS,
  text/role/label, or nearby-element alternatives), try it in order, and use
  the first valid selector.

**Selectors - IMPORTANT: Use Postmill-compatible selectors**

Postmill uses different HTML structure than Reddit. Use these VERIFIED selectors:

1. Navigation & Search:
   - Search box: `input[name="q"], #site-nav-search, input[type="search"]`
   - Forums link: `get_by_role("link", name="Forums", exact=True)`
   - Submit link: `a:has-text('Submit')` or `get_by_role("link", name="Submit")`

2. Posts & Articles:
   - Post containers: `article` (NOT `.post`, `.thing`, or Reddit-specific classes)
   - Post links: `a[href^="/f/"]` with pattern `/f/{forum}/{id}/{slug}`
   - Post title: Look inside `article` for `a` or `h2`, `h3` elements

3. Forms (Submit/Edit):
   - Title input: `input[name='title']` or `get_by_label("Title", exact=True)`
   - Body textarea: `textarea[name='body']` or `get_by_label("Body", exact=True)`
   - Forum selector: `get_by_role("combobox")` then `get_by_role("option", name=forum_name)`
   - Submit button: `button:has-text('Create submission')` or `get_by_role("button", name="Create submission")`
   - Edit button: `a:has-text("Edit")`
   - Save button: `button:has-text("Edit submission")` or `button:has-text("Save")`

4. Voting:
   - Upvote: `button:has-text('Upvote')`
   - Downvote: `button:has-text('Downvote')`
   - Retract upvote: `button:has-text('Retract upvote')`
   - Retract downvote: `button:has-text('Retract downvote')`
   - Vote scores: `button:has-text('Upvote') span` or `[class*='score']`

5. Comments:
   - Comment textarea: `textarea` or `textarea[placeholder*="comment"]`
   - Post comment button: `button:has-text('Post')`
   - Comment containers: Look for comment-related classes

6. User & Settings:
   - User menu: `get_by_role("button")` (first button in nav)
   - User settings: `get_by_role("link", name="User settings", exact=True)`
   - Edit biography: `get_by_role("link", name="Edit biography", exact=True)`
   - User profile: `/user/{username}` or `/user/{username}/submissions`

7. Subscription:
   - Subscribe: `button:has-text('Subscribe')`
   - Unsubscribe: `button:has-text('Unsubscribe')`

**URL Patterns**
- Forums: `/f/{forum_name}` (e.g., `/f/books`, `/f/nyc`)
- Forum sorted: `/f/{forum_name}/top?t=all` (top all time)
- Posts: `/f/{forum_name}/{post_id}/{slug}` (e.g., `/f/books/123/my-post-title`)
- User submissions: `/user/{username}/submissions`
- User comments: `/user/{username}/comments`
- Submit page: `/submit` (global) or navigate to forum then click Submit
- Forum creation: `/create_forum`
- Forums list: `/forums`

**Error Handling**
- Either recover from exceptions or reraise them using custom exceptions:
  - `ElementNotFoundError`: When elements are not found
  - `NavigationError`: When navigation fails
  - `SubmissionError`: When form submission fails
- Don't just print errors
- Use `await page.wait_for_load_state("networkidle")` after navigation
- Check for 404 pages: `if "not found" in page_content.lower() or "404" in page_content`

**Best Practices from Verified Skills**
- Always use `await page.wait_for_selector(...)` before interacting with dynamic content
- Use `query_selector_all` to find multiple elements, then iterate
- For forum matching, normalize names: `forum_name.lower()` and handle spaces
- After voting, verify state change by waiting for "Retract" button
- Use `page.url` to get current URL after navigation
- Use `re.compile(r"/f/[^/]+/\\d+/.*")` for post URL patterns
"""

SUCCESS_TRAJECTORY_PROMPT = """
{base_prompt}

=====================================================================================
EXAMPLE SKILLS (Reference these patterns)
=====================================================================================

Example 1 - Upvoting a post:
```python
async def upvote_post(page: Page, forum_name: str, *, newest: bool = True) -> bool:
    \"\"\"
    Upvote a post in a specified forum.

    Usage preconditions:
    - You must be logged in to an account with permission to upvote posts.

    Args:
        page: A Playwright Page instance.
        forum_name: The name of the forum (subreddit) to target.
        newest: If True, upvote the newest (topmost) post.

    Returns:
        True if the upvote was successful.

    Usage Log:
    - Successfully upvoted the newest post in the "deeplearning" forum.
    \"\"\"
    await page.goto(f"/f/{{forum_name}}")
    await page.wait_for_selector("article", timeout=5000)

    posts = await page.query_selector_all("article")
    if not posts:
        raise ElementNotFoundError(f"No posts found in forum '{{forum_name}}'")

    post_article = posts[0]
    upvote_button = await post_article.query_selector("button:has-text('Upvote')")
    if upvote_button:
        await upvote_button.click()
        await post_article.wait_for_selector("button:has-text('Retract upvote')", timeout=5000)
        return True
    return False
```

Example 2 - Creating a post:
```python
async def create_post(page: Page, forum_name: str, post_title: str, post_body: str) -> str:
    \"\"\"
    Create a new post in a specified forum.

    Usage preconditions:
    - You must be logged in before calling this function.

    Args:
        page: A Playwright Page instance.
        forum_name: The name of the forum to post in.
        post_title: The title of the post.
        post_body: The body/content of the post.

    Returns:
        The URL of the newly created post.

    Usage Log:
    - Successfully submitted posts to "relationship_advice", "movies", "books".
    \"\"\"
    await page.goto(f"/f/{{forum_name}}")
    await page.wait_for_selector("a:has-text('Submit')", timeout=5000)
    await page.click("a:has-text('Submit')")

    await page.wait_for_selector("form", timeout=5000)
    await page.fill("input[name='title']", post_title)
    await page.fill("textarea[name='body']", post_body)
    await page.click("button:has-text('Create submission')")

    await page.wait_for_url(re.compile(rf"/f/{{forum_name}}/\\d+/.*"), timeout=10000)
    return page.url
```

Example 3 - Finding and opening a forum:
```python
async def find_and_open_forum_by_search(page: Page, forum_query: str) -> str:
    \"\"\"
    Navigates to the forum list, searches for a forum, and opens it.

    Args:
        page: A Playwright Page instance.
        forum_query: The search string to locate the desired forum.

    Returns:
        The forum name as it appears in the URL.

    Usage Log:
    - Successfully searched for and opened the "nyc" forum by searching "NYC".
    \"\"\"
    await page.goto("/forums")

    search_selector = 'input[name="q"], #site-nav-search, input[type="search"]'
    await page.wait_for_selector(search_selector, timeout=10000)
    search_box = await page.query_selector(search_selector)
    await search_box.fill(forum_query)
    await search_box.press("Enter")

    await page.wait_for_load_state("networkidle")

    forum_links = await page.query_selector_all('a[href^="/f/"]')
    # Find best matching forum...

    await page.goto(f"/f/{{forum_name}}")
    return forum_name
```

=====================================================================================
SUCCESSFUL TASK EXECUTION TRAJECTORY
=====================================================================================

Task Objective: {objective}

Action History:
{action_history}

Final URL: {final_url}
Task was: SUCCESSFUL

=====================================================================================
RELEVANT PRIOR SKILL FEEDBACK FROM RUNTIME TESTS
=====================================================================================

{feedback_context}

=====================================================================================
GENERATE SKILL
=====================================================================================

Based on this SUCCESSFUL trajectory, synthesize a reusable skill that:
1. Generalizes the specific actions into a parameterized function
2. Handles the common case, not just this specific instance
3. Uses the VERIFIED Postmill selectors from the examples above
4. Includes proper waits and error recovery
5. Documents the expected behavior in docstring

Output a Python async function that follows SkillWeaver format.
Use the custom exception classes: ElementNotFoundError, NavigationError, SubmissionError.

Output ONLY the Python code, no JSON wrapper, no markdown code fences.
"""


# ============================================================================
# 失败轨迹技能生成 Prompt
# ============================================================================

FAILURE_TRAJECTORY_PROMPT = """
You are analyzing a FAILED Reddit/Postmill automation attempt to generate a skill that avoids this failure.

**CRITICAL CONSTRAINTS - MUST FOLLOW:**
1. NO nested functions - write flat, linear code only
2. NO helper functions inside the main function
3. Generate a COMPLETE skill that performs the ENTIRE task, not just verification
4. Use ONLY the verified Postmill selectors listed below
5. Verify state changes using BUTTON TEXT (e.g., "Retract downvote"), NOT CSS classes
6. Encode each grounding operation as an ordered selector list, try selectors
   in order, and use the first valid one. Limit recovery to local selector or
   modal fallbacks rather than regenerating unrelated behavior.

**VERIFIED Postmill Selectors - USE THESE EXACTLY:**

Navigation:
- Search box: `input[name="q"]` or `input[type="search"]`
- Forums link: Click text "Forums"
- Submit link: `a:has-text('Submit')`

Posts:
- Post containers: `article` (NOT `.post`, `.thing`)
- Post links: `a[href^="/f/"]`

Forms:
- Title: `input[name='title']`
- Body: `textarea[name='body']`
- URL input: `input[name='url']`
- Forum selector: Use `role=combobox` then `role=option`
- Submit: `button:has-text('Create submission')`
- Edit: `a:has-text("Edit")`
- Save: `button:has-text("Edit submission")`

Voting:
- Upvote: `button:has-text('Upvote')`
- Downvote: `button:has-text('Downvote')`
- Verify upvote success: Wait for `button:has-text('Retract upvote')`
- Verify downvote success: Wait for `button:has-text('Retract downvote')`

URL Patterns:
- Forums: `/f/{{forum_name}}`
- Sorted by top: `/f/{{forum_name}}/top?t=all`
- Posts: `/f/{{forum_name}}/{{post_id}}/{{slug}}`
- User submissions: `/user/{{username}}/submissions`

=====================================================================================
FAILED TASK EXECUTION TRAJECTORY
=====================================================================================

Task Objective: {objective}

Action History:
{action_history}

Final URL: {final_url}
Task was: FAILED

Failure Analysis:
{failure_analysis}

=====================================================================================
RELEVANT PRIOR SKILL FEEDBACK FROM RUNTIME TESTS
=====================================================================================

{feedback_context}

=====================================================================================
GENERATE CORRECTIVE SKILL
=====================================================================================

Based on this FAILED trajectory, generate a skill that:
1. Performs the COMPLETE task (not just verification)
2. Addresses the root cause of failure
3. Uses ONLY the verified Postmill selectors above
4. NO nested/helper functions - flat code only
5. Verifies success using button text changes (e.g., "Retract downvote" appears)
6. Includes proper waits: `await page.wait_for_load_state("networkidle")`

**Example of CORRECT voting verification:**
```python
# CORRECT - verify by button text
await downvote_btn.click()
await article.wait_for_selector("button:has-text('Retract downvote')", timeout=3000)

# WRONG - do NOT use CSS classes
# class_attr = await article.get_attribute("class")
# if "vote--user-downvoted" in class_attr:  # DON'T DO THIS
```

**Example of CORRECT forum selection:**
```python
# CORRECT - use role selectors
combobox = await page.wait_for_selector('role=combobox', timeout=5000)
await combobox.click()
options = await page.query_selector_all('role=option')
for option in options:
    text = (await option.inner_text()).strip().lower()
    if text == forum_name.lower():
        await option.click()
        break

# WRONG - do NOT use these
# forum_combobox = await page.wait_for_selector('input[role="combobox"]')  # DON'T DO THIS
```

Output a single async function with NO nested functions:

```python
async def skill_name(page, param1: str, param2: str) -> return_type:
    \"\"\"
    Brief description of what this skill does.

    This skill [description], with handling to avoid [failure pattern].

    Usage preconditions:
    - User must be authenticated

    Args:
        page: A Playwright Page instance.
        param1: Description.
        param2: Description.

    Returns:
        Description of return value.

    Raises:
        ElementNotFoundError: When elements are not found.
        SubmissionError: When operation fails.

    Usage Log:
    - Generated from failed task {task_id}: {failure_summary}
    \"\"\"
    # Step 1: Navigate
    await page.goto("/path")
    await page.wait_for_load_state("networkidle")

    # Step 2: Find elements (use verified selectors)
    await page.wait_for_selector("article", timeout=5000)
    articles = await page.query_selector_all("article")

    # Step 3: Perform action
    # ... flat code, no nested functions ...

    # Step 4: Verify success using button text
    # await element.wait_for_selector("button:has-text('Expected text')", timeout=3000)

    return result
```

Output ONLY the Python code, no JSON wrapper, no markdown code fences.
"""



def extract_action_history(trajectory: List[Dict]) -> str:
    """
    从轨迹中提取动作历史，格式化为可读字符串
    """
    actions = []
    for i, step in enumerate(trajectory):
        action = step.get("action", "")
        reason = step.get("reason", "")
        url = step.get("url", "")

        # 简化 observation，只保留关键信息
        obs_desc = step.get("observation_description", "")
        if not obs_desc:
            obs = step.get("observation", "")
            if obs and len(obs) > 200:
                obs_desc = obs[:200] + "..."
            else:
                obs_desc = obs

        step_info = f"Step {i+1}:\n"
        step_info += f"  URL: {url}\n"
        if obs_desc:
            step_info += f"  Observation: {obs_desc[:150]}...\n" if len(obs_desc) > 150 else f"  Observation: {obs_desc}\n"
        step_info += f"  Reason: {reason}\n"
        step_info += f"  Action: {action}\n"

        actions.append(step_info)

    return "\n".join(actions)


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


def analyze_failure(trajectory: List[Dict], objective: str, model: str = "gpt-4.1") -> Dict:
    """
    分析失败轨迹，提取失败原因
    """
    action_history = extract_action_history(trajectory)

    prompt = f"""
Analyze this FAILED Reddit automation attempt and identify the root cause.

Task Objective: {objective}

Action History:
{action_history}

Analyze and output JSON:
{{
    "failure_type": "operation|reasoning|external",
    "root_cause": "Specific root cause of failure",
    "what_went_wrong": "Detailed explanation of what went wrong",
    "missing_steps": ["Step that should have been taken but wasn't"],
    "suggested_fix": "How to fix this in a skill",
    "skill_recommendation": {{
        "should_generate": true/false,
        "skill_type": "navigation|interaction|extraction|submission",
        "skill_name_suggestion": "suggested_function_name"
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
            "skill_recommendation": {"should_generate": False}
        }


def generate_skill_from_success_trajectory(
    trajectory_data: Dict,
    model: str = "gpt-4.1",
    feedback_context: str = "",
) -> Optional[str]:
    """
    从成功轨迹生成技能 (SkillWeaver 风格)

    支持两种格式:
    1. 原始轨迹: objective 在 trajectory[0] 中
    2. 简化轨迹: intent 在顶层，objective 在 trajectory[0] 中

    Args:
        trajectory_data: 完整的轨迹 JSON 数据
        model: LLM 模型名称

    Returns:
        生成的 Python 函数代码字符串，如果失败返回 None
    """
    trajectory = trajectory_data.get("trajectory", [])
    if not trajectory:
        return None

    # 提取目标 (优先顶层 intent，其次 trajectory[0].objective)
    objective = trajectory_data.get("intent", "")
    if not objective and trajectory:
        objective = trajectory[0].get("objective", "Unknown task")

    # 提取动作历史
    action_history = extract_action_history(trajectory)

    # 获取最终 URL
    final_url = trajectory[-1].get("url", "") if trajectory else ""

    # 构建 prompt
    prompt = SUCCESS_TRAJECTORY_PROMPT.format(
        base_prompt=SKILLWEAVER_BASE_PROMPT,
        objective=objective,
        action_history=action_history,
        final_url=final_url,
        feedback_context=feedback_context or "No prior runtime skill feedback is available.",
    )

    try:
        # 调用 LLM 生成代码
        response = call_llm_text(prompt, model=model, max_tokens=2000)

        # 清理响应，提取 Python 代码
        code = response.strip()

        # 移除可能的 markdown 代码块标记
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]

        code = code.strip()

        # 验证是 async def
        if "async def" not in code:
            print("  警告: 生成的代码不包含 async def")
            return None

        return code

    except Exception as e:
        print(f"  技能生成失败: {e}")
        return None


def generate_skill_from_failure_trajectory(
    trajectory_data: Dict,
    model: str = "gpt-4.1",
    feedback_context: str = "",
) -> Optional[str]:
    """
    从失败轨迹生成避免失败的技能

    支持两种格式:
    1. 原始轨迹: objective 在 trajectory[0] 中
    2. 简化轨迹: intent 在顶层

    如果 trajectory_data 中已经包含 failure_analysis 字段，则使用预先的分析结果，
    否则会调用 analyze_failure 进行分析。

    Args:
        trajectory_data: 完整的轨迹 JSON 数据
        model: LLM 模型名称

    Returns:
        生成的 Python 函数代码字符串，如果失败或不适合生成技能返回 None
    """
    trajectory = trajectory_data.get("trajectory", [])
    if not trajectory:
        return None

    task_id = trajectory_data.get("id", "unknown")

    # 提取目标 (优先顶层 intent)
    objective = trajectory_data.get("intent", "")
    if not objective and trajectory:
        objective = trajectory[0].get("objective", "Unknown task")

    action_history = extract_action_history(trajectory)
    final_url = trajectory[-1].get("url", "") if trajectory else ""

    # 检查是否已有预先的失败分析结果
    failure_analysis = trajectory_data.get("failure_analysis")
    if failure_analysis:
        print("    使用预先的失败分析...", end=" ", flush=True)
        print("✓")
    else:
        # 没有预先分析，则进行分析
        print("    分析失败原因...", end=" ", flush=True)
        failure_analysis = analyze_failure(trajectory, objective, model=model)
        print("✓")

    # 检查是否应该生成技能
    skill_rec = failure_analysis.get("skill_recommendation", {})
    if not skill_rec.get("should_generate", True) or not skill_rec.get("needs_operation_skill", True):
        print("    跳过: 不适合生成技能")
        return None

    # 构建 prompt
    failure_analysis_str = json.dumps(failure_analysis, indent=2, ensure_ascii=False)

    prompt = FAILURE_TRAJECTORY_PROMPT.format(
        objective=objective,
        action_history=action_history,
        final_url=final_url,
        failure_analysis=failure_analysis_str,
        task_id=task_id,
        failure_summary=failure_analysis.get("root_cause", "unknown failure"),
        feedback_context=feedback_context or "No prior runtime skill feedback is available.",
    )

    try:
        print("    生成技能代码...", end=" ", flush=True)
        response = call_llm_text(prompt, model=model, max_tokens=2000)

        # 清理响应
        code = response.strip()
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        code = code.strip()

        if "async def" not in code:
            print("✗ 不包含 async def")
            return None

        print("✓")
        return code

    except Exception as e:
        print(f"✗ 失败: {e}")
        return None


def validate_skill_code(code: str) -> Tuple[bool, str]:
    """Validate DRIVE executability, closed-loop checks, and selector fallbacks."""
    valid, errors = validate_generated_interaction_skill(code)
    return valid, "; ".join(errors)


def extract_function_name(code: str) -> str:
    """从代码中提取函数名"""
    match = re.search(r'async\s+def\s+(\w+)\s*\(', code)
    if match:
        return match.group(1)
    return "unknown_skill"


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


def _trajectory_objective(trajectory_data: Dict[str, Any]) -> str:
    objective = trajectory_data.get("intent", "")
    trajectory = trajectory_data.get("trajectory", [])
    if not objective and trajectory:
        objective = trajectory[0].get("objective", "Unknown task")
    return objective or "Unknown task"


def _infer_reddit_task_family(objective: str, failure_analysis: Optional[Dict[str, Any]]) -> str:
    if failure_analysis and failure_analysis.get("task_family"):
        return str(failure_analysis["task_family"])
    objective_lower = objective.lower()
    if any(word in objective_lower for word in ["upvote", "downvote", "vote", "like", "dislike"]):
        return "Interaction_Voting"
    if any(word in objective_lower for word in ["post", "submit", "create", "publish"]):
        return "Post_Creation"
    if any(word in objective_lower for word in ["subscribe", "settings", "profile", "biography"]):
        return "Account_Configuration"
    if any(word in objective_lower for word in ["find", "search", "count", "extract", "show"]):
        return "Information_Retrieval"
    return "Reddit_Task"


def _reddit_keywords(objective: str, task_family: str) -> List[str]:
    family_keywords = {
        "Information_Retrieval": ["search", "find", "count", "extract", "forum", "post", "comment"],
        "Post_Creation": ["post", "submit", "create", "forum", "title", "body"],
        "Interaction_Voting": ["upvote", "downvote", "vote", "post", "comment"],
        "Account_Configuration": ["subscribe", "settings", "profile", "biography", "forum"],
        "Content_Repost": ["repost", "share", "promote", "post"],
        "Reddit_Task": ["reddit", "postmill", "forum", "post"],
    }
    keywords = list(family_keywords.get(task_family, []))
    objective_lower = objective.lower()
    for keyword in [
        "forum",
        "subreddit",
        "post",
        "comment",
        "user",
        "thread",
        "book",
        "recommendation",
        "subscription",
    ]:
        if keyword in objective_lower:
            keywords.append(keyword)
    return _unique_nonempty([kw.lower() for kw in keywords], limit=10)


def build_scenario_descriptor(
    trajectory_data: Dict[str, Any],
    skill_name: str,
    source_type: str,
) -> Dict[str, Any]:
    """Build DRIVE-style d_k=<U_k,W_k> metadata for a Reddit operation skill."""
    trajectory = trajectory_data.get("trajectory", [])
    objective = _trajectory_objective(trajectory_data)
    failure_analysis = trajectory_data.get("failure_analysis", {})
    urls = _unique_nonempty([_url_pattern(step.get("url", "")) for step in trajectory])
    final_url = _url_pattern(trajectory[-1].get("url", "")) if trajectory else ""
    if final_url:
        urls = _unique_nonempty(urls + [final_url])

    traj_analysis = failure_analysis.get("trajectory_analysis", {}) if isinstance(failure_analysis, dict) else {}
    analyzed_url_patterns = traj_analysis.get("url_patterns", []) if isinstance(traj_analysis, dict) else []
    urls = _unique_nonempty(urls + [str(pattern) for pattern in analyzed_url_patterns])

    task_family = _infer_reddit_task_family(objective, failure_analysis if isinstance(failure_analysis, dict) else None)
    failure_category = failure_analysis.get("failure_category", "") if isinstance(failure_analysis, dict) else ""
    scenario_description = (
        failure_analysis.get("failure_description")
        if isinstance(failure_analysis, dict) and failure_analysis.get("failure_description")
        else f"Reddit/Postmill task requiring {task_family.lower().replace('_', ' ')} behavior: {objective}"
    )

    return {
        "U_k": {
            "site": "reddit",
            "display_name": "Reddit/Postmill",
            "url_patterns": urls,
            "url_match": "compatible_site",
            "page_context": {
                "start_url": urls[0] if urls else "",
                "final_url": final_url,
                "observed_actions": _unique_nonempty([str(step.get("action", "")).split(" ", 1)[0] for step in trajectory], limit=8),
            },
        },
        "W_k": {
            "task_intent": objective,
            "task_family": task_family,
            "semantic_keywords": _reddit_keywords(objective, task_family),
            "scenario_description": scenario_description,
            "failure_category": failure_category,
            "source_type": source_type,
            "skill_name": skill_name,
        },
    }


def load_trajectory(path: Path) -> Optional[Dict]:
    """加载轨迹文件"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"  无法加载 {path}: {e}")
        return None


def is_successful_trajectory(trajectory_data: Dict) -> bool:
    """
    判断轨迹是否成功 (reward == 1.0 才算成功)

    支持两种格式:
    1. 原始轨迹: reward 在 trajectory[-1] 中
    2. 简化轨迹: reward 在顶层
    """
    # 简化轨迹格式: reward 在顶层
    if "reward" in trajectory_data:
        return trajectory_data.get("reward", 0) == 1.0

    # 原始轨迹格式: reward 在最后一步
    trajectory = trajectory_data.get("trajectory", [])
    if not trajectory:
        return False

    last_step = trajectory[-1]
    reward = last_step.get("reward", 0)

    return reward == 1.0


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Reddit 站点单轨迹技能生成器 (v3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从 JSONL 文件生成技能（推荐方式）
  python3 reddit_skill_generator_v3.py --jsonl all_trajectories.jsonl --filter-success --out skills.py

  # 从单个成功轨迹生成
  python3 reddit_skill_generator_v3.py --trajectory 31.json --mode success --out skill.py

  # 从目录批量处理成功轨迹
  python3 reddit_skill_generator_v3.py --trajectory-dir ./trajectories/ --filter-success --out skills.py

  # 自动检测模式（根据轨迹结果）
  python3 reddit_skill_generator_v3.py --trajectory 31.json --mode auto --out skill.py
        """
    )

    # 输入选项（三选一）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--trajectory", "-t",
        help="单个轨迹文件路径 (JSON)"
    )
    input_group.add_argument(
        "--trajectory-dir", "-d",
        help="轨迹文件目录 (批量处理)"
    )
    input_group.add_argument(
        "--jsonl", "-j",
        help="JSONL 格式的轨迹文件 (简化轨迹格式)"
    )

    parser.add_argument("--out", "-o", required=True, help="输出技能文件路径 (.py)")
    parser.add_argument(
        "--mode", "-m",
        choices=["success", "failure", "auto"],
        default="auto",
        help="生成模式: success (成功轨迹), failure (失败轨迹), auto (自动检测)"
    )
    parser.add_argument("--model", default="gpt-4.1", help="LLM 模型名称")
    parser.add_argument("--filter-success", action="store_true", help="只处理成功的轨迹")
    parser.add_argument("--filter-failure", action="store_true", help="只处理失败的轨迹")
    parser.add_argument("--limit", type=int, help="只处理前N个轨迹")
    parser.add_argument("--resume", action="store_true", help="断点续传")
    parser.add_argument("--debug", action="store_true", help="显示详细错误信息")
    parser.add_argument("--feedback-kb", help="运行时技能反馈知识库 JSON，用于生成技能时参考历史失败")

    args = parser.parse_args(argv)

    # 收集轨迹数据
    trajectories = []  # List of (task_id, traj_data)

    if args.jsonl:
        # 从 JSONL 文件加载
        jsonl_path = Path(args.jsonl)
        if not jsonl_path.exists():
            print(f"错误: JSONL 文件不存在: {jsonl_path}")
            return 1

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        traj_data = json.loads(line)
                        task_id = str(traj_data.get("id", len(trajectories)))
                        trajectories.append((task_id, traj_data))
                    except json.JSONDecodeError:
                        continue

        input_desc = str(jsonl_path)

    elif args.trajectory:
        path = Path(args.trajectory)
        if not path.exists():
            print(f"错误: 轨迹文件不存在: {path}")
            return 1
        traj_data = load_trajectory(path)
        if traj_data:
            trajectories.append((path.stem, traj_data))
        input_desc = str(path)

    else:  # args.trajectory_dir
        dir_path = Path(args.trajectory_dir)
        if not dir_path.exists():
            print(f"错误: 目录不存在: {dir_path}")
            return 1
        trajectory_files = sorted(dir_path.glob("*.json"))
        # 排除 summary 等非轨迹文件
        trajectory_files = [f for f in trajectory_files if f.stem.isdigit() or f.stem.startswith("task_")]

        for traj_path in trajectory_files:
            traj_data = load_trajectory(traj_path)
            if traj_data:
                trajectories.append((traj_path.stem, traj_data))

        input_desc = str(dir_path)

    if not trajectories:
        print("错误: 未找到轨迹数据")
        return 1
    feedback_context = load_feedback_context(args.feedback_kb)

    # 应用 limit
    if args.limit:
        trajectories = trajectories[:args.limit]

    # 断点续传：读取已生成的函数名
    existing_names = set()
    output_path = Path(args.out)
    if args.resume and output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            existing_names = set(re.findall(r'async\s+def\s+(\w+)\s*\(', content))
        print(f"断点续传: 已存在 {len(existing_names)} 个技能")

    print(f"{'='*70}")
    print(f"Reddit 站点单轨迹技能生成器 (v3)")
    print(f"{'='*70}")
    print(f"输入: {input_desc}")
    print(f"输出: {args.out}")
    print(f"轨迹数: {len(trajectories)}")
    print(f"模式: {args.mode}")
    print(f"模型: {args.model}")
    print(f"{'='*70}\n")

    # 处理轨迹
    skills = []
    metadata = []
    success_count = 0
    failed_count = 0
    skipped_count = 0
    seen_names = {}

    for i, (task_id, traj_data) in enumerate(trajectories):
        print(f"[{i+1}/{len(trajectories)}] 处理 Task {task_id}...", end=" ", flush=True)

        # 判断成功/失败
        is_success = is_successful_trajectory(traj_data)

        # 根据过滤条件跳过
        if args.filter_success and not is_success:
            print("⚠ 跳过 (失败轨迹)")
            skipped_count += 1
            continue
        if args.filter_failure and is_success:
            print("⚠ 跳过 (成功轨迹)")
            skipped_count += 1
            continue

        # 确定处理模式
        if args.mode == "auto":
            mode = "success" if is_success else "failure"
        else:
            mode = args.mode

        print(f"[{mode}] ", end="", flush=True)

        try:
            # 生成技能
            if mode == "success":
                code = generate_skill_from_success_trajectory(
                    traj_data,
                    model=args.model,
                    feedback_context=feedback_context,
                )
            else:
                code = generate_skill_from_failure_trajectory(
                    traj_data,
                    model=args.model,
                    feedback_context=feedback_context,
                )

            if code is None:
                print("✗ 无法生成")
                failed_count += 1
                continue

            # 验证代码
            is_valid, error_msg = validate_skill_code(code)
            if not is_valid:
                print(f"✗ 验证失败: {error_msg}")
                failed_count += 1
                continue

            # 提取函数名
            func_name = extract_function_name(code)

            # 检查是否已存在
            if func_name in existing_names:
                print(f"⚠ 跳过 (已存在)")
                skipped_count += 1
                continue

            # 检查重名
            if func_name in seen_names:
                prev_id, prev_index = seen_names[func_name]
                print(f"↻ 替换 {prev_id}")
                skills[prev_index] = code
                metadata[prev_index] = {
                    "name": func_name,
                    "source_task_id": task_id,
                    "source_type": mode,
                    "is_success": is_success,
                    "source_code": code,
                    "scenario_descriptor": build_scenario_descriptor(traj_data, func_name, mode),
                }
                seen_names[func_name] = (task_id, prev_index)
            else:
                current_index = len(skills)
                seen_names[func_name] = (task_id, current_index)
                skills.append(code)
                metadata.append({
                    "name": func_name,
                    "source_task_id": task_id,
                    "source_type": mode,
                    "is_success": is_success,
                    "source_code": code,
                    "scenario_descriptor": build_scenario_descriptor(traj_data, func_name, mode),
                })

            success_count += 1
            print(f"✓ {func_name}")

        except Exception as e:
            print(f"✗ 异常: {e}")
            failed_count += 1
            if args.debug:
                import traceback
                traceback.print_exc()

    if not skills:
        print("\n错误: 没有成功生成任何技能")
        return 1

    # 生成输出文件
    print(f"\n{'='*70}")
    print(f"生成技能代码文件...")
    print(f"{'='*70}\n")

    # 文件头部
    code_lines = [
        '"""',
        'Reddit 站点技能库 (v3)',
        '',
        f'从 {len(trajectories)} 个轨迹中生成了 {len(skills)} 个技能',
        '',
        '生成模式:',
        '- 成功轨迹: SkillWeaver 风格，从成功执行中提取可复用技能',
        '- 失败轨迹: 分析失败原因，生成避免失败的技能',
        '',
        '技能格式符合 SkillWeaver 标准:',
        '- async def 函数',
        '- 详细 docstring (含 Usage Log)',
        '- page.goto() 设置初始状态',
        '- Playwright API',
        '"""',
        '',
        'import asyncio',
        'import re',
        'from playwright.async_api import Page, TimeoutError as PlaywrightTimeout',
        '',
        '',
        'class SkillExecutionError(RuntimeError):',
        '    """Base exception for generated Reddit skills."""',
        '    pass',
        '',
        '',
        'class ElementNotFoundError(SkillExecutionError):',
        '    """Raised when a required page element cannot be found."""',
        '    pass',
        '',
        '',
        'class NavigationError(SkillExecutionError):',
        '    """Raised when navigation does not reach the expected page."""',
        '    pass',
        '',
        '',
        'class SubmissionError(SkillExecutionError):',
        '    """Raised when a form/action submission does not complete."""',
        '    pass',
        '',
        ''
    ]

    # 添加技能代码
    for i, (code, meta) in enumerate(zip(skills, metadata)):
        code_lines.append(f"# {'='*68}")
        code_lines.append(f"# Skill {i+1}/{len(skills)}: {meta['name']}")
        code_lines.append(f"# Source: Task {meta['source_task_id']} ({meta['source_type']})")
        code_lines.append(f"# {'='*68}")
        code_lines.append("")
        code_lines.append(code)
        code_lines.append("")
        code_lines.append("")

    # 写入代码文件
    code_content = "\n".join(code_lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_mode = 'a' if args.resume and output_path.exists() else 'w'
    with open(output_path, write_mode, encoding='utf-8') as f:
        if write_mode == 'a':
            f.write("\n\n# ===== 新增技能 (断点续传) =====\n\n")
        f.write(code_content)

    # 写入元数据文件 (SkillWeaver 格式)
    metadata_path = output_path.with_suffix('.json')

    # 读取已有元数据
    existing_metadata = {"functions": {}, "global_version": 0}
    if args.resume and metadata_path.exists():
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                existing_metadata = json.load(f)
        except:
            pass

    # 更新元数据
    skillweaver_metadata = existing_metadata
    for meta in metadata:
        func_name = meta["name"]
        skillweaver_metadata["functions"][func_name] = {
            # SkillWeaver 必需字段
            "test_count": 0,
            "success_count": 0,
            "version": 0,
            "references": [],
            "events": [],
            # 来源信息
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
            skillweaver_metadata["functions"][func_name],
            meta["source_code"],
        )

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(skillweaver_metadata, f, indent=2, ensure_ascii=False)

    # 打印统计
    print(f"✓ 技能代码: {output_path}")
    print(f"✓ 元数据: {metadata_path}")
    print(f"\n{'='*70}")
    print(f"统计信息")
    print(f"{'='*70}")
    print(f"成功: {success_count}")
    print(f"跳过: {skipped_count}")
    print(f"失败: {failed_count}")
    print(f"总计: {len(trajectories)}")
    print(f"最终技能数: {len(skills)}")

    # 按来源类型统计
    success_source = sum(1 for m in metadata if m["source_type"] == "success")
    failure_source = sum(1 for m in metadata if m["source_type"] == "failure")
    print(f"\n按来源统计:")
    print(f"  成功轨迹生成: {success_source}")
    print(f"  失败轨迹生成: {failure_source}")

    print(f"{'='*70}")

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

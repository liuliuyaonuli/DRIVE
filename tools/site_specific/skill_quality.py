"""Deterministic admission checks for newly induced interaction skills."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import List, Tuple


# Site-specific generator wrappers are also executable as standalone scripts.
# Ensure the repository package is importable in that mode before using the
# shared browser-only safety validator.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from drive_artifacts import extract_operation_templates
except ImportError:
    from tools.site_specific.drive_artifacts import extract_operation_templates

from AgentOccam.skill_runtime import validate_skill_module_safety


_BROWSER_OPERATIONS = {
    "goto",
    "click",
    "fill",
    "check",
    "select_option",
    "locator",
    "query_selector",
    "query_selector_all",
    "get_by_role",
    "get_by_label",
    "get_by_text",
    "get_by_placeholder",
}
_VERIFICATION_OPERATIONS = {
    "wait_for_url",
    "wait_for_selector",
    "wait_for_load_state",
    "is_visible",
    "is_checked",
    "count",
    "text_content",
    "inner_text",
    "input_value",
    "get_attribute",
    "content",
    "title",
}


def validate_generated_interaction_skill(code: str) -> Tuple[bool, List[str]]:
    """Reject syntactically valid but non-executable or brittle LLM output."""

    errors: List[str] = []
    if not code.strip():
        return False, ["代码为空"]
    if re.search(r"\b(?:TODO|FIXME|YOUR_[A-Z0-9_]*)\b|<actual_", code, re.IGNORECASE):
        errors.append("代码包含占位符或未完成标记")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, [f"语法错误 (行{exc.lineno}): {exc.msg}"]
    errors.extend(validate_skill_module_safety(code))

    async_defs = [node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)]
    sync_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(async_defs) != 1 or sync_defs:
        errors.append("必须且只能包含一个顶层 async def，不能附带同步辅助函数")
        return False, errors
    function = async_defs[0]
    first_param = function.args.args[0].arg if function.args.args else None
    if first_param != "page":
        errors.append("技能函数第一个参数必须是 page")
    if not ast.get_docstring(function):
        errors.append("技能函数必须包含使用前提、返回值和验证方式的 docstring")
    nested = [
        node
        for node in ast.walk(function)
        if node is not function and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if nested:
        errors.append("技能函数不能包含嵌套函数")
    if any(isinstance(node, ast.ExceptHandler) and node.type is None for node in ast.walk(function)):
        errors.append("技能函数不能使用 bare except")

    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    call_names = {
        node.func.attr
        for node in calls
        if isinstance(node.func, ast.Attribute)
    }
    if not (call_names & _BROWSER_OPERATIONS):
        errors.append("技能没有执行任何浏览器操作")
    has_page_url_check = any(
        isinstance(node, ast.Attribute) and node.attr == "url" for node in ast.walk(function)
    )
    if not (call_names & _VERIFICATION_OPERATIONS) and not has_page_url_check:
        errors.append("技能缺少局部结果或最终页面状态验证")

    templates = extract_operation_templates(code)
    if not templates:
        errors.append("技能没有可追踪的页面 grounding 操作")
    for template in templates:
        if len(template.get("selectors", [])) < 2:
            errors.append(
                f"操作 {template.get('operation_id')} 未提供可用的 selector fallback"
            )
    return not errors, errors

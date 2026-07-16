"""Schema helpers shared by DRIVE skill generators and migration tools."""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, Iterable, List


_SELECTOR_OPERATIONS = {
    "locator",
    "query_selector_all",
    "query_selector",
    "wait_for_selector",
    "click",
    "fill",
    "check",
    "select_option",
}
_SEMANTIC_OPERATIONS = {
    "get_by_role",
    "get_by_label",
    "get_by_text",
    "get_by_title",
    "get_by_placeholder",
}
_SELECTOR_CALL = re.compile(
    r"(?P<operation>locator|query_selector_all|query_selector|wait_for_selector|click|fill|check|select_option)"
    r"\(\s*(?P<quote>[\"'])(?P<selector>.+?)(?P=quote)\s*[,)]"
)


def _unique(values: Iterable[str], limit: int = 8) -> List[str]:
    return list(dict.fromkeys(value for value in values if _looks_valid_selector(value)))[:limit]


def _looks_valid_selector(value: Any) -> bool:
    selector = str(value or "").strip()
    if not selector or selector.endswith(("(", "*=", "[href*=", "[role=")):
        return False
    if selector.count("[") != selector.count("]"):
        return False
    if selector.count("(") != selector.count(")"):
        return False
    return True


def _split_selector_candidates(selector: str) -> List[str]:
    """Split top-level CSS selector unions while preserving commas in syntax."""

    parts: List[str] = []
    current: List[str] = []
    quote = ""
    escaped = False
    depth = 0
    for char in selector:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\":
            current.append(char)
            escaped = True
            continue
        if quote:
            current.append(char)
            if char == quote:
                quote = ""
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
            continue
        if char in "([":
            depth += 1
        elif char in ")]" and depth:
            depth -= 1
        if char == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return _unique(parts)


def _fallback_selectors(selector: str) -> List[str]:
    """Derive small, conservative alternatives in a different locator family."""

    fallbacks: List[str] = []
    text_match = re.search(r":has-text\([\"'](.+?)[\"']\)", selector)
    if text_match:
        text = text_match.group(1).replace('"', '\\"')
        if selector.startswith("button"):
            fallbacks.append(f'get_by_role("button", name="{text}")')
        elif selector.startswith("a"):
            fallbacks.append(f'get_by_role("link", name="{text}")')
        fallbacks.append(f'text="{text}"')

    role_match = re.fullmatch(
        r'get_by_role\(["\']([^"\']+)["\'](?:,\s*name=["\']([^"\']+)["\'])?(?:,\s*exact=(?:true|false|True|False))?\)',
        selector,
    )
    if role_match:
        role, name = role_match.groups()
        if name:
            fallbacks.extend(
                [f'[role="{role}"][aria-label="{name}"]', f'text="{name}"']
            )
        else:
            fallbacks.append(f'[role="{role}"]')

    label_match = re.fullmatch(
        r'get_by_label\(["\']([^"\']+)["\'](?:,\s*exact=(?:true|false|True|False))?\)',
        selector,
    )
    if label_match:
        label = label_match.group(1)
        fallbacks.extend([f'[aria-label="{label}"]', f'label:has-text("{label}")'])

    placeholder_match = re.fullmatch(
        r'get_by_placeholder\(["\']([^"\']+)["\'](?:,\s*exact=(?:true|false|True|False))?\)', selector
    )
    if placeholder_match:
        value = placeholder_match.group(1)
        fallbacks.extend([f'input[placeholder="{value}"]', f'input[placeholder*="{value}"]'])

    text_locator = re.fullmatch(
        r'get_by_text\(["\']([^"\']+)["\'](?:,\s*exact=(?:true|false|True|False))?\)',
        selector,
    )
    if text_locator:
        text = text_locator.group(1)
        fallbacks.extend([f'text="{text}"', f':text-is("{text}")'])
    title_match = re.fullmatch(
        r'get_by_title\(["\']([^"\']+)["\'](?:,\s*exact=(?:true|false|True|False))?\)',
        selector,
    )
    if title_match:
        title = title_match.group(1)
        fallbacks.extend([f'[title="{title}"]', f'text="{title}"'])

    if "input" in selector:
        if "search" in selector.lower() or "query" in selector.lower():
            fallbacks.extend(
                ["input[type='search']", "input[placeholder*='Search']", "input[name='query']"]
            )
        name_match = re.search(r"name=[\"']([^\"']+)[\"']", selector)
        if name_match:
            fallbacks.append(f"input[name='{name_match.group(1)}']")
    if "textarea" in selector:
        fallbacks.extend(["textarea", "textarea[placeholder]"])
    if not selector.startswith("get_by_") and ":visible" not in selector:
        # Playwright's visible pseudo-class is a useful local fallback when a
        # broad selector resolves both hidden and active controls.
        fallbacks.append(f"{selector}:visible")
    return _unique(fallbacks)


def _ordered_candidates(source_selector: str) -> List[str]:
    candidates = _split_selector_candidates(source_selector) or [source_selector]
    expanded = list(candidates)
    for candidate in candidates:
        expanded.extend(_fallback_selectors(candidate))
    return _unique(expanded)


def _render_semantic_selector(node: ast.Call) -> str:
    if not node.args or not isinstance(node.args[0], ast.Constant):
        return ""
    if not isinstance(node.args[0].value, str):
        return ""
    selector = f"{node.func.attr}({json.dumps(node.args[0].value)}"
    for keyword in node.keywords:
        if keyword.arg in {"name", "exact"} and isinstance(keyword.value, ast.Constant):
            rendered = (
                json.dumps(keyword.value.value)
                if isinstance(keyword.value.value, str)
                else repr(keyword.value.value)
            )
            selector += f", {keyword.arg}={rendered}"
    return selector + ")"


def extract_operation_templates(source: str) -> List[Dict[str, Any]]:
    """Extract ``Theta(k_i)=[(r_j, Sigma_j), ...]`` from executable code."""

    templates: List[Dict[str, Any]] = []
    seen = set()

    def add(operation: str, source_selector: str, candidates: Iterable[str] = ()) -> None:
        source_selector = str(source_selector or "").strip()
        if not _looks_valid_selector(source_selector):
            return
        ordered = _unique(list(candidates) or _ordered_candidates(source_selector))
        if not ordered:
            return
        key = (operation, source_selector)
        if key in seen:
            return
        seen.add(key)
        templates.append(
            {
                "operation_id": f"r_{len(templates) + 1}",
                "operation": operation,
                "source_selector": source_selector,
                "selectors": ordered,
            }
        )

    try:
        tree = ast.parse(source)
    except SyntaxError:
        for match in _SELECTOR_CALL.finditer(source):
            add(match.group("operation"), match.group("selector"))
        return templates

    string_bindings: Dict[str, str] = {}
    sequence_bindings: Dict[str, List[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = getattr(node, "value", None)
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [target.id for target in targets if isinstance(target, ast.Name)]
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            for name in names:
                string_bindings[name] = value.value
        elif isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            values = [
                item.value
                for item in value.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            ]
            if len(values) == len(value.elts) and values:
                for name in names:
                    sequence_bindings[name] = values

    loop_bindings: Dict[str, List[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.AsyncFor)) or not isinstance(node.target, ast.Name):
            continue
        if isinstance(node.iter, ast.Name) and node.iter.id in sequence_bindings:
            loop_bindings[node.target.id] = sequence_bindings[node.iter.id]
        elif isinstance(node.iter, (ast.List, ast.Tuple)):
            values = [
                item.value
                for item in node.iter.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            ]
            if len(values) == len(node.iter.elts) and values:
                loop_bindings[node.target.id] = values

    def render_selector(node: ast.AST) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id in string_bindings:
            return string_bindings[node.id]
        if isinstance(node, ast.JoinedStr):
            parts = []
            for value in node.values:
                if isinstance(value, ast.Constant):
                    parts.append(str(value.value))
                elif isinstance(value, ast.FormattedValue):
                    parts.append("{" + ast.unparse(value.value) + "}")
            return "".join(parts)
        return ""

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        operation = node.func.attr
        if operation in _SELECTOR_OPERATIONS and node.args:
            first_arg = node.args[0]
            if isinstance(first_arg, ast.Name) and first_arg.id in loop_bindings:
                candidates = _unique(loop_bindings[first_arg.id])
                if candidates:
                    add(operation, candidates[0], candidates)
            else:
                selector = render_selector(first_arg)
                if selector:
                    add(operation, selector)
        elif operation in _SEMANTIC_OPERATIONS:
            selector = _render_semantic_selector(node)
            if selector:
                add(operation, selector)
    return templates


def selector_sets_from_templates(templates: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    result = {}
    for template in templates:
        source = str(template.get("source_selector") or "")
        selectors = _unique(str(item) for item in template.get("selectors", []) if item)
        if source and selectors:
            result[source] = selectors
    return result


def extract_signature_contract(source: str) -> tuple[Dict[str, Any], str] | None:
    """Derive the explicit ``Sig_k`` portion of an interaction artifact."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
        ),
        None,
    )
    if function is None:
        return None

    def annotation(node: ast.arg) -> str:
        return ast.unparse(node.annotation) if node.annotation is not None else "Any"

    inputs: Dict[str, Dict[str, Any]] = {}
    positional = [*function.args.posonlyargs, *function.args.args]
    required_positional = len(positional) - len(function.args.defaults)
    for index, parameter in enumerate(positional):
        if index == 0 and parameter.arg == "page":
            continue
        inputs[parameter.arg] = {
            "annotation": annotation(parameter),
            "required": index < required_positional,
        }
    for parameter, default in zip(function.args.kwonlyargs, function.args.kw_defaults):
        inputs[parameter.arg] = {
            "annotation": annotation(parameter),
            "required": default is None,
        }
    if function.args.vararg is not None:
        inputs[f"*{function.args.vararg.arg}"] = {
            "annotation": annotation(function.args.vararg),
            "required": False,
        }
    if function.args.kwarg is not None:
        inputs[f"**{function.args.kwarg.arg}"] = {
            "annotation": annotation(function.args.kwarg),
            "required": False,
        }
    return (
        {
            "inputs": inputs,
            "returns": ast.unparse(function.returns) if function.returns is not None else "Any",
        },
        function.name,
    )


def add_interaction_runtime_schema(metadata: Dict[str, Any], source: str) -> Dict[str, Any]:
    """Refresh operation templates and feedback counters for one skill."""

    templates = extract_operation_templates(source)
    previous_sets = metadata.get("selector_sets") or {}
    if templates:
        refreshed_sets: Dict[str, List[str]] = {}
        for template in templates:
            source_selector = template["source_selector"]
            preserved = previous_sets.get(source_selector, [])
            ordered = _unique(list(preserved) + list(template["selectors"]))
            template["selectors"] = ordered
            refreshed_sets[source_selector] = ordered
        metadata["operation_templates"] = templates
        metadata["selector_sets"] = refreshed_sets
    else:
        metadata.setdefault("operation_templates", [])
        metadata["selector_sets"] = {
            str(source_selector): _unique(str(item) for item in ordered)
            for source_selector, ordered in previous_sets.items()
            if _looks_valid_selector(source_selector)
        }
    metadata.setdefault("selector_failure_counts", {})
    metadata.setdefault("selector_candidate_failure_counts", {})
    metadata.setdefault("selector_patch_rounds", 0)
    metadata.setdefault("recovery_branches", [])
    metadata.setdefault(
        "preconditions",
        [
            "The typed descriptor matches the current page context.",
            "All required typed inputs are bound to non-placeholder values.",
        ],
    )
    metadata.setdefault(
        "postconditions",
        [
            "The procedure terminates without exception and produces an observable page effect or meaningful local result."
        ],
    )
    metadata.setdefault(
        "precondition_checks",
        [
            {"type": "descriptor_match"},
            {"type": "required_arguments_bound"},
        ],
    )
    metadata.setdefault(
        "postcondition_checks",
        [{"type": "observable_effect_or_result"}],
    )
    contract = dict(metadata.get("interaction_contract") or {})
    signature = extract_signature_contract(source)
    if signature is not None:
        sig_k, callable_name = signature
        contract["Sig_k"] = sig_k
    else:
        callable_name = str(metadata.get("skill_name") or "")
        contract.setdefault("Sig_k", {"inputs": "derived from callable signature", "returns": "Any"})
    contract.setdefault("Pre_k", list(metadata["preconditions"]))
    body = dict(contract.get("Body_k") or {})
    if callable_name:
        body["callable"] = callable_name
    body.setdefault("action_space", "predefined_browser_actions")
    contract["Body_k"] = body
    contract.setdefault("Post_k", list(metadata["postconditions"]))
    # ``Rec_k`` mirrors only active, replay-validated recovery branches.
    contract["Rec_k"] = list(metadata["recovery_branches"])
    metadata["interaction_contract"] = contract
    metadata.setdefault(
        "repair_validation_policy",
        {
            "required": "source_context_replay_and_postcondition",
            "unvalidated_repair": "stage_only",
        },
    )
    metadata.setdefault(
        "reuse_profile",
        {
            "S": "related_operations_under_compatible_page_types_and_local_states",
            "E": [
                "semantic_compatibility",
                "page_context_match",
                "argument_availability",
                "procedural_preconditions",
            ],
            "I": "direct_parameterized_procedure_invocation",
            "F": [
                "grounding_results",
                "exceptions",
                "state_changes",
                "recovery_events",
                "postcondition_outcome",
            ],
        },
    )
    metadata.setdefault("invocation_count", 0)
    metadata.setdefault("local_success_count", 0)
    metadata.setdefault("task_success_count", 0)
    metadata.setdefault("utility_score", 0.5)
    metadata.setdefault("active", True)
    return metadata

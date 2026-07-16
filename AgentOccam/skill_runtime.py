"""Runtime support for closed-loop DRIVE interaction skills.

The public skills use Playwright's async API while AgentOccam's WebArena
environment is synchronous.  The converter operates on the *whole source
module* so helpers, aliases, constants, and custom exceptions remain available.

The page proxy consumes ordered selector sets written by the local patch
operator.  It tries selectors in order and records a compact execution trace
for skill-level feedback.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence
from urllib.parse import urljoin
import ast
import inspect
import json
from pathlib import Path


# ``from __future__ import annotations`` occurs in a few public skill modules
# and has no runtime capability.  Keep it distinct from executable imports
# such as filesystem or networking clients.
_ALLOWED_IMPORT_ROOTS = {
    "__future__", "asyncio", "re", "typing", "datetime", "math", "json",
    "playwright", "urllib",
}
_FORBIDDEN_CALLS = {"open", "eval", "exec", "compile", "__import__", "input", "breakpoint"}
_FORBIDDEN_IMPORT_ROOTS = {
    "os", "sys", "subprocess", "pathlib", "shutil", "socket", "requests",
    "httpx", "aiohttp", "urllib3", "ftplib", "telnetlib", "importlib",
}


def validate_skill_module_safety(source: str) -> list[str]:
    """Reject capabilities outside the paper's browser-only action space."""

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error: {exc.msg}"]
    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in _FORBIDDEN_IMPORT_ROOTS or root not in _ALLOWED_IMPORT_ROOTS:
                    errors.append(f"forbidden import: {alias.name}")
                if root == "urllib" and not alias.name.startswith("urllib.parse"):
                    errors.append(f"forbidden urllib capability: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".", 1)[0]
            if root in _FORBIDDEN_IMPORT_ROOTS or root not in _ALLOWED_IMPORT_ROOTS:
                errors.append(f"forbidden import: {module}")
            if root == "urllib" and not module.startswith("urllib.parse"):
                errors.append(f"forbidden urllib capability: {module}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
                errors.append(f"forbidden call: {node.func.id}")
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "system", "popen", "Popen", "run", "call", "check_call", "check_output",
                "connect", "request", "urlopen", "urlretrieve",
            }:
                errors.append(f"forbidden external capability: {node.func.attr}")
    return list(dict.fromkeys(errors))


class _AsyncToSync(ast.NodeTransformer):
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        replacement = ast.FunctionDef(
            name=node.name,
            args=node.args,
            body=node.body,
            decorator_list=node.decorator_list,
            returns=node.returns,
            type_comment=node.type_comment,
        )
        return ast.copy_location(replacement, node)

    def visit_Await(self, node: ast.Await) -> ast.expr:
        return self.visit(node.value)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> ast.For:
        self.generic_visit(node)
        replacement = ast.For(
            target=node.target,
            iter=node.iter,
            body=node.body,
            orelse=node.orelse,
            type_comment=node.type_comment,
        )
        return ast.copy_location(replacement, node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> ast.With:
        self.generic_visit(node)
        replacement = ast.With(items=node.items, body=node.body, type_comment=node.type_comment)
        return ast.copy_location(replacement, node)

    def visit_Call(self, node: ast.Call) -> ast.Call:
        self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "asyncio"
            and node.func.attr == "sleep"
        ):
            node.func.value.id = "time"
        return node


def convert_async_module_to_sync(source: str) -> str:
    """Convert a valid async skill module to executable sync Python."""

    source = source.replace("playwright.async_api", "playwright.sync_api")
    tree = ast.parse(source)
    converted = _AsyncToSync().visit(tree)
    ast.fix_missing_locations(converted)
    rendered = ast.unparse(converted)
    # The converted source is embedded after a small runtime prelude, so a
    # future import can no longer legally occupy the module's first statement.
    rendered = "\n".join(
        line for line in rendered.splitlines()
        if not line.startswith("from __future__ import ")
    )
    return rendered + "\n\nimport time\n"


def get_skill_module_source(skill_fn: Callable[..., Any]) -> str:
    """Read the complete defining module, falling back to the function body."""

    module = inspect.getmodule(skill_fn)
    if module is not None:
        source_file = inspect.getsourcefile(module)
        if source_file:
            path = Path(source_file)
            if path.exists():
                return path.read_text(encoding="utf-8")
        try:
            return inspect.getsource(module)
        except (OSError, TypeError):
            pass
    return inspect.getsource(skill_fn)


def _is_missing_result(method_name: str, result: Any) -> bool:
    if method_name in {"query_selector", "wait_for_selector", "text_content", "inner_text"}:
        return result is None
    if method_name in {"query_selector_all", "all"}:
        return not result
    if method_name in {"count"}:
        return result == 0
    if method_name in {"is_visible", "is_enabled", "is_checked"}:
        return result is False
    return False


_SEMANTIC_LOCATORS = {"get_by_role", "get_by_label", "get_by_text", "get_by_title", "get_by_placeholder"}


def _semantic_selector_spec(method: str, first: Any, kwargs: Mapping[str, Any]) -> str:
    value = getattr(first, "value", first)
    rendered = f"{method}({json.dumps(str(value))}"
    if kwargs.get("name") is not None:
        rendered += f", name={json.dumps(str(kwargs['name']))}"
    if kwargs.get("exact") is not None:
        rendered += f", exact={bool(kwargs['exact'])!r}"
    return rendered + ")"


def _locator_from_spec(page: Any, selector: str) -> Any:
    """Resolve CSS/text selectors or a small safe semantic-locator expression."""

    try:
        expression = ast.parse(selector, mode="eval").body
    except SyntaxError:
        return page.locator(selector)
    if (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Name)
        and expression.func.id in _SEMANTIC_LOCATORS
    ):
        args = [ast.literal_eval(item) for item in expression.args]
        kwargs = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in expression.keywords
            if keyword.arg is not None
        }
        return getattr(page, expression.func.id)(*args, **kwargs)
    return page.locator(selector)


class FallbackLocator:
    """Locator facade that retries an ordered selector set on local failure."""

    def __init__(
        self,
        page: Any,
        selectors: Sequence[str],
        trace: list[dict[str, Any]],
        *,
        locators: Optional[Sequence[Any]] = None,
        source_selector: Optional[str] = None,
        recovery: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._page = page
        self._selectors = list(selectors)
        self._trace = trace
        self._source_selector = source_selector or (self._selectors[0] if self._selectors else "")
        self._recovery = recovery
        self._locators = (
            list(locators)
            if locators is not None
            else [_locator_from_spec(page, item) for item in selectors]
        )

    def _derived(self, transform: Callable[[Any], Any]) -> "FallbackLocator":
        return FallbackLocator(
            self._page,
            self._selectors,
            self._trace,
            locators=[transform(locator) for locator in self._locators],
            source_selector=self._source_selector,
            recovery=self._recovery,
        )

    @property
    def first(self) -> "FallbackLocator":
        return self._derived(lambda locator: locator.first)

    @property
    def last(self) -> "FallbackLocator":
        return self._derived(lambda locator: locator.last)

    def nth(self, index: int) -> "FallbackLocator":
        return self._derived(lambda locator: locator.nth(index))

    def locator(self, selector: str, *args: Any, **kwargs: Any) -> "FallbackLocator":
        return self._derived(lambda locator: locator.locator(selector, *args, **kwargs))

    def __getattr__(self, name: str) -> Any:
        sample = getattr(self._locators[0], name)
        if not callable(sample):
            return sample

        def call_with_fallback(*args: Any, **kwargs: Any) -> Any:
            last_error: Optional[BaseException] = None
            for attempt in range(2):
                for selector, locator in zip(self._selectors, self._locators):
                    try:
                        result = getattr(locator, name)(*args, **kwargs)
                        if _is_missing_result(name, result):
                            self._trace.append(
                                {
                                    "operation": name,
                                    "source_selector": self._source_selector,
                                    "selector": selector,
                                    "success": False,
                                    "error": "no match",
                                }
                            )
                            continue
                        self._trace.append(
                            {
                                "operation": name,
                                "source_selector": self._source_selector,
                                "selector": selector,
                                "success": True,
                            }
                        )
                        return result
                    except Exception as exc:  # Playwright exposes several timeout subclasses.
                        last_error = exc
                        self._trace.append(
                            {
                                "operation": name,
                                "source_selector": self._source_selector,
                                "selector": selector,
                                "success": False,
                                "error": str(exc)[:300],
                            }
                        )
                if attempt == 0 and self._recovery and self._recovery():
                    continue
                break
            if last_error is not None:
                raise last_error
            return None

        return call_with_fallback


class SkillPageProxy:
    """Page facade implementing relative navigation and selector fallbacks."""

    _DIRECT_SELECTOR_METHODS = {
        "check",
        "click",
        "fill",
        "hover",
        "press",
        "query_selector",
        "query_selector_all",
        "select_option",
        "set_checked",
        "uncheck",
        "wait_for_selector",
    }

    def __init__(
        self,
        page: Any,
        *,
        base_origin: str = "",
        selector_sets: Optional[Mapping[str, Sequence[str]]] = None,
        recovery_branches: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> None:
        object.__setattr__(self, "_page", page)
        object.__setattr__(self, "_base_origin", base_origin)
        object.__setattr__(
            self,
            "_selector_sets",
            {str(key): list(value) for key, value in (selector_sets or {}).items() if value},
        )
        object.__setattr__(self, "trace", [])
        object.__setattr__(self, "_recovery_branches", list(recovery_branches or []))

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or name == "trace":
            object.__setattr__(self, name, value)
        else:
            setattr(self._page, name, value)

    def _ordered(self, selector: str) -> list[str]:
        values = self._selector_sets.get(selector)
        return list(dict.fromkeys(values or [selector]))

    def _run_recovery_branches(self) -> bool:
        for branch in self._recovery_branches:
            for selector in branch.get("selectors", []) or []:
                try:
                    locator = _locator_from_spec(self._page, str(selector))
                    if hasattr(locator, "is_visible") and not locator.is_visible():
                        continue
                    locator.click()
                    self.trace.append(
                        {
                            "operation": "recovery",
                            "source_selector": branch.get("trigger", "unexpected_modal"),
                            "selector": selector,
                            "success": True,
                        }
                    )
                    return True
                except Exception:
                    continue
        return False

    def goto(self, url: str, *args: Any, **kwargs: Any) -> Any:
        resolved = urljoin(self._base_origin.rstrip("/") + "/", url) if self._base_origin else url
        try:
            result = self._page.goto(resolved, *args, **kwargs)
            self.trace.append({"operation": "goto", "selector": resolved, "success": True})
            return result
        except Exception as exc:
            self.trace.append(
                {"operation": "goto", "selector": resolved, "success": False, "error": str(exc)[:300]}
            )
            raise

    def locator(self, selector: str, *args: Any, **kwargs: Any) -> Any:
        ordered = self._ordered(selector)
        if selector not in self._selector_sets:
            return self._page.locator(selector, *args, **kwargs)
        return FallbackLocator(
            self._page,
            ordered,
            self.trace,
            source_selector=selector,
            recovery=self._run_recovery_branches,
        )

    def _direct_selector_call(self, name: str, selector: str, *args: Any, **kwargs: Any) -> Any:
        last_error: Optional[BaseException] = None
        for attempt in range(2):
            for candidate in self._ordered(selector):
                try:
                    if candidate.split("(", 1)[0] in _SEMANTIC_LOCATORS:
                        locator = _locator_from_spec(self._page, candidate)
                        if name == "query_selector":
                            result = locator.element_handle()
                        elif name == "query_selector_all":
                            result = locator.element_handles()
                        elif name == "wait_for_selector":
                            locator.wait_for(*args, **kwargs)
                            result = locator
                        else:
                            result = getattr(locator, name)(*args, **kwargs)
                    else:
                        result = getattr(self._page, name)(candidate, *args, **kwargs)
                    if _is_missing_result(name, result):
                        self.trace.append(
                            {
                                "operation": name,
                                "source_selector": selector,
                                "selector": candidate,
                                "success": False,
                                "error": "no match",
                            }
                        )
                        continue
                    self.trace.append(
                        {
                            "operation": name,
                            "source_selector": selector,
                            "selector": candidate,
                            "success": True,
                        }
                    )
                    return result
                except Exception as exc:
                    last_error = exc
                    self.trace.append(
                        {
                            "operation": name,
                            "source_selector": selector,
                            "selector": candidate,
                            "success": False,
                            "error": str(exc)[:300],
                        }
                    )
            if attempt == 0 and self._run_recovery_branches():
                continue
            break
        if last_error is not None:
            raise last_error
        return [] if name == "query_selector_all" else None

    def __getattr__(self, name: str) -> Any:
        if name in _SEMANTIC_LOCATORS:
            def semantic_locator(first: Any, *args: Any, **kwargs: Any) -> Any:
                key = _semantic_selector_spec(name, first, kwargs)
                ordered = list(self._selector_sets.get(key) or [key])
                original_locator = getattr(self._page, name)(first, *args, **kwargs)
                locators = [
                    original_locator if candidate == key else _locator_from_spec(self._page, candidate)
                    for candidate in ordered
                ]
                return FallbackLocator(
                    self._page,
                    ordered,
                    self.trace,
                    locators=locators,
                    source_selector=key,
                    recovery=self._run_recovery_branches,
                )

            return semantic_locator
        if name in self._DIRECT_SELECTOR_METHODS:
            return lambda selector, *args, **kwargs: self._direct_selector_call(
                name, selector, *args, **kwargs
            )
        return getattr(self._page, name)


def prepare_skill_page(
    page: Any,
    *,
    base_origin: str = "",
    selector_sets: Optional[Mapping[str, Sequence[str]]] = None,
    recovery_branches: Optional[Sequence[Mapping[str, Any]]] = None,
) -> SkillPageProxy:
    return SkillPageProxy(
        page,
        base_origin=base_origin,
        selector_sets=selector_sets,
        recovery_branches=recovery_branches,
    )

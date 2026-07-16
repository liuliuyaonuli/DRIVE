import json
import re as _re
import asyncio
import inspect
import traceback
from browser_env import (
    create_id_based_action,
    create_id_based_actions,
    StateInfo,
    Trajectory,
    ActionTypes,
    ScriptBrowserEnv
)
from evaluation_harness.evaluators import evaluator_router
from AgentOccam.obs_opt import (
    prune_tree,
    translate_node_to_str,
)
from AgentOccam.skill_runtime import (
    convert_async_module_to_sync,
    get_skill_module_source,
    validate_skill_module_safety,
)


def _extract_selectors_from_source(source_code: str) -> list:
    """Extract CSS selectors and Playwright locator patterns from skill source code.

    Performs static analysis on the skill's Python source to identify all
    selectors the skill attempts to use. Useful for debugging 'element not found' failures.
    """
    selectors = []
    # CSS selectors in query_selector / wait_for_selector / click / fill calls
    for match in _re.finditer(
        r'(?:query_selector_all|query_selector|wait_for_selector|\.click|\.fill)\(\s*["\']([^"\']+)["\']',
        source_code,
    ):
        selectors.append(match.group(1))
    # page.locator(...) patterns
    for match in _re.finditer(r'\.locator\(\s*["\']([^"\']+)["\']', source_code):
        selectors.append(match.group(1))
    # get_by_role patterns
    for match in _re.finditer(
        r'get_by_role\(\s*["\']([^"\']+)["\'](?:.*?name\s*=\s*["\']([^"\']+)["\'])?',
        source_code,
    ):
        role = match.group(1)
        name = match.group(2) or ""
        selectors.append(
            f'get_by_role({json.dumps(role)}'
            + (f', name={json.dumps(name)}' if name else '')
            + ')'
        )
    # get_by_label patterns
    for match in _re.finditer(r'get_by_label\(\s*["\']([^"\']+)["\']', source_code):
        selectors.append(f'get_by_label({json.dumps(match.group(1))})')
    # get_by_text patterns
    for match in _re.finditer(r'get_by_text\(\s*["\']([^"\']+)["\']', source_code):
        selectors.append(f'get_by_text({json.dumps(match.group(1))})')
    # goto URLs (navigation targets)
    for match in _re.finditer(r'\.goto\(\s*(?:f?["\']([^"\']+)["\'])', source_code):
        selectors.append(f'goto:{match.group(1)}')
    # Deduplicate preserving order
    return list(dict.fromkeys(selectors))


def _has_meaningful_skill_result(result) -> bool:
    """Whether a skill returned an explicit, locally checkable outcome."""

    if result is None:
        return False
    if isinstance(result, bool):
        return result
    if isinstance(result, str):
        return bool(result.strip())
    if isinstance(result, (list, tuple, set, dict)):
        return bool(result)
    # Numeric zero is a valid extraction result (for example, a count).
    return isinstance(result, (int, float))


def _derive_local_skill_success(
    execution_succeeded: bool,
    result,
    *,
    url_changed: bool,
    page_state_changed: bool,
    postcondition_checks=None,
) -> tuple[bool, str | None]:
    """Apply DRIVE's local-effect rule after an invoked interaction skill.

    A procedure is locally successful only when it completed without an explicit
    failure *and* produced either an observable page effect or a meaningful
    returned result.  Merely returning without raising is not enough.
    """

    if not execution_succeeded:
        return False, None
    checks = list(postcondition_checks or [{"type": "observable_effect_or_result"}])
    for check in checks:
        kind = str(check.get("type", "")) if isinstance(check, dict) else str(check)
        if kind == "observable_effect_or_result":
            if url_changed or page_state_changed or _has_meaningful_skill_result(result):
                continue
            return False, "Postcondition failed: no observable page effect or meaningful result"
        if kind == "url_changed":
            if url_changed:
                continue
            return False, "Postcondition failed: URL did not change"
        if kind == "page_state_changed":
            if page_state_changed:
                continue
            return False, "Postcondition failed: page state did not change"
        if kind == "result_truthy":
            if _has_meaningful_skill_result(result):
                continue
            return False, "Postcondition failed: result is not meaningful"
        if kind == "result_field_truthy":
            field = str(check.get("field", "")) if isinstance(check, dict) else ""
            if isinstance(result, dict) and result.get(field):
                continue
            return False, f"Postcondition failed: result field '{field}' is not truthy"
        return False, f"Unsupported postcondition check: {kind}"
    return True, None


class WebArenaEnvironmentWrapper():
    def __init__(self, config_file, max_browser_rows=300, max_steps=50, slow_mo=1, observation_type="accessibility_tree", current_viewport_only=False, viewport_size={"width": 1280, "height": 720}, headless=False, global_config=None, skill_registry=None):
        self.webarena_env = ScriptBrowserEnv(
                    headless=headless,
                    slow_mo=slow_mo,
                    observation_type=observation_type,
                    current_viewport_only=current_viewport_only,
                    viewport_size=viewport_size,
                    global_config=global_config
                )
        self.config_file = config_file
        with open(self.config_file, "r") as f:
            self.config = json.load(f)
        self.global_config = global_config
        self.skill_registry = skill_registry  # Store skill registry for USE_SKILL execution

        self.obs, self.info = self.webarena_env.reset(options={"config_file": self.config_file})
        self.terminated = False
        self.objective = self.config["intent"]
        self.url = self.config["start_url"]
        self.max_browser_rows = max_browser_rows
        self.max_steps = max_steps
        self.steps = 0
        self.is_done = False
        self.reward = 0.0
        self.verification_requested = False  # Track if we've already asked for verification

        # ===== 新增：技能执行状态跟踪 =====
        self.last_skill_execution = None  # 记录最近一次技能执行的结果 {'success': bool, 'skill_name': str, 'result': any}

        self.trajectory: Trajectory = []
        self.update_webarena_metrics()
        
    def reset(self):
        self.obs, self.info = self.webarena_env.reset(options={"config_file": self.config_file})

    def close(self):
        self.webarena_env.close()
        
    def get_url(self):
        return self.url
    
    def get_objective(self):
        return self.objective 
    
    def get_sites(self):
        return self.config["sites"]
        
    def observation(self):
        self.url = self.webarena_env.page.url

        # ===== 提取技能执行结果（如果有）=====
        skill_result_prefix = ""

        if isinstance(self.obs, dict) and 'text' in self.obs:
            obs_text = self.obs['text']
            # Handle both list and tuple
            if isinstance(obs_text, (list, tuple)) and len(obs_text) > 0 and isinstance(obs_text[0], str):
                # 检查是否包含技能执行结果
                has_skill = "SKILL EXECUTION" in obs_text[0]
                if has_skill:
                    print(f"[DEBUG observation()] 检测到 SKILL EXECUTION 在 obs_text[0] 中！")
                    print(f"[DEBUG observation()] obs_text[0] 前 300 字符: {obs_text[0][:300]}")
                if has_skill:
                    # 提取技能结果部分 - 匹配从第一个分隔符到最后一个分隔符（贪婪匹配）
                    import re
                    # 使用贪婪匹配来获取整个技能结果块（包含 Result、NEXT STEP 等所有信息）
                    match = re.search(r'(={50,}[\s\S]*?SKILL EXECUTION[\s\S]*?={50,}[\s\S]*?={50,})', obs_text[0])
                    if match:
                        skill_result_prefix = match.group(1) + "\n\n"
                        print(f"[DEBUG observation()] 成功提取 skill_result_prefix, 长度={len(skill_result_prefix)}")

        if self.global_config and self.global_config.env.prune:
            root_node = self.obs["text"][1]
            DOM_root_node = prune_tree(objective=self.objective, root_node=root_node, mode="node")
            DOM_str = translate_node_to_str(node=DOM_root_node, mode="concise")
            # 在 prune 模式下，将技能结果添加到 DOM 字符串前面
            if skill_result_prefix:
                DOM_str = skill_result_prefix + DOM_str
                print(f"[DEBUG observation()] 返回的 DOM_str 包含技能结果，前 300 字符: {DOM_str[:300]}")
            return {"text": DOM_str, "image": self.obs["image"], "node": DOM_root_node}
        else:
            browser_content = self.obs["text"][0]
            browser_content = browser_content.split("\n")[:self.max_browser_rows]
            browser_content = "\n".join(browser_content)
            return browser_content
    
    def done(self):
        if self.is_done:
            return True
        return False
    
    def status(self):
        return {
            'done': self.is_done,
            'reward': self.reward,
            'success': float(self.reward == 1.0),
            'num_actions': self.steps,
        }

    def _capture_skill_failure_context(self, skill_fn, skill_args_dict):
        """Capture page state at the moment of skill failure.

        Returns a dict with url, selectors, skill_args, page_state_snippet.
        All capture attempts are wrapped in try/except to avoid secondary failures.
        """
        context = {}
        # Current URL
        try:
            context['url'] = self.webarena_env.page.url
        except Exception:
            context['url'] = 'unknown'

        # Selectors used in skill source code
        try:
            # Feedback must be local to the invoked procedure.  Whole-module
            # source is used for execution, but unrelated skills' selectors
            # must not be blamed for this invocation.
            source = inspect.getsource(skill_fn)
            context['selectors'] = _extract_selectors_from_source(source)
        except Exception:
            context['selectors'] = []

        # Skill arguments (copy to avoid mutation)
        context['skill_args'] = dict(skill_args_dict) if skill_args_dict else {}

        # Accessibility tree snippet (first 2000 chars)
        try:
            obs = self.webarena_env.observation_handler.get_observation(
                self.webarena_env.page,
                self.webarena_env.get_page_client(self.webarena_env.page),
            )
            if isinstance(obs, dict) and 'text' in obs:
                text = obs['text']
                if isinstance(text, (list, tuple)) and len(text) > 0:
                    context['page_state_snippet'] = str(text[0])[:2000]
                elif isinstance(text, str):
                    context['page_state_snippet'] = text[:2000]
                else:
                    context['page_state_snippet'] = ''
            else:
                context['page_state_snippet'] = ''
        except Exception:
            context['page_state_snippet'] = ''

        return context

    def should_force_verification(self):
        """
        Check if we should force LLM to verify task completion before stopping.
        Only asks once per state to avoid infinite loops.

        Checks:
        1. Don't allow STOP immediately after TYPE without any click
        2. Don't allow STOP with very few steps (< 3) unless it's clearly complete
        3. Encourage exploration before giving up
        """
        force_verification = bool(
            self.global_config
            and hasattr(self.global_config, "env")
            and getattr(self.global_config.env, "force_stop_verification", False)
        )
        if not force_verification:
            return False

        if not self.trajectory:
            return False

        # If we already requested verification, allow STOP
        if self.verification_requested:
            print(f"[Stop Validation] ✓ Verification already requested, allowing STOP through.")
            return False

        # Get last few actions (filter out StateInfo, only get actions)
        recent_actions = [item for item in self.trajectory if isinstance(item, dict) and "action_type" in item]

        if not recent_actions:
            return False

        # Count actual web actions (exclude stop actions in trajectory)
        num_actions = len([a for a in recent_actions if a.get("action_type") != ActionTypes.STOP])

        # Check 1: Enforce verification if last action was TYPE (most risky case)
        last_action = recent_actions[-1]
        if last_action.get("action_type") == ActionTypes.TYPE:
            print(f"[Stop Validation] ⚠️  Last action was TYPE. Requesting verification.")
            self.verification_requested = True
            return True

        # Check 2: If stopping too early (< 3 meaningful actions), require verification
        # Exception: If the answer is clearly a found value (contains numbers/specific info)
        MIN_ACTIONS_BEFORE_STOP = 3
        if num_actions < MIN_ACTIONS_BEFORE_STOP:
            # Check if this looks like a "give up" stop (contains "N/A", "not found", "no", etc.)
            last_stop_in_trajectory = next((a for a in reversed(recent_actions) if a.get("action_type") == ActionTypes.STOP), None)
            if last_stop_in_trajectory:
                answer = last_stop_in_trajectory.get("answer", "").lower()
                give_up_indicators = ["n/a", "not found", "no ", "cannot", "unable", "does not", "don't", "doesn't", "isn't", "aren't"]
                if any(indicator in answer for indicator in give_up_indicators):
                    print(f"[Stop Validation] ⚠️  Attempting to stop too early (only {num_actions} actions) with 'give up' answer.")
                    print(f"[Stop Validation] Answer preview: {answer[:100]}...")
                    self.verification_requested = True
                    return True

        return False
    
    def step(self, action):
        self.steps = self.steps + 1
        print(f"[Step {self.steps}] {action}")
        print("*"*100)
        if self.steps > self.max_steps:
            print(f"Steps {self.steps} exceeded maximum {self.max_steps}")
            self.is_done = True
            action_cmd = create_id_based_action(f"stop [Trajectory failed: Steps {self.steps} exceeded maximum {self.max_steps}.]")
            self.update_webarena_metrics(action_cmd)
            return self.status()

        if action is None or action == "":
            action_cmds = []
        else:
            try:
                action_cmds = create_id_based_actions(action)
                if not action_cmds:
                    return False
            except Exception as e:
                print(f"Invalid action syntax: {e}")
                action_cmds = []

        stop_rejected = False  # Track if STOP was rejected for verification

        for action_cmd in action_cmds:
            try:
                # ===== DEBUG: 打印 action 类型 =====
                action_type = action_cmd.get("action_type")
                print(f"[DEBUG] action_type: {action_type}, USE_SKILL: {ActionTypes.USE_SKILL}")
                print(f"[DEBUG] action_type == USE_SKILL: {action_type == ActionTypes.USE_SKILL}")
                print(f"[DEBUG] skill_registry is None: {self.skill_registry is None}")

                # Validate STOP actions - require verification check
                if action_cmd.get("action_type") == ActionTypes.STOP:
                    if self.should_force_verification():
                        print(f"[Stop Validation] STOP rejected - verification required first.")
                        stop_rejected = True
                        # Don't execute STOP, skip this action
                        # The verification prompt will be added by returning special status
                        continue

                # Handle USE_SKILL actions separately
                if action_cmd.get("action_type") == ActionTypes.USE_SKILL:
                    if self.skill_registry is None:
                        error_msg = "USE_SKILL action requested but no skill_registry available"
                        print(f"Error: {error_msg}")
                        # ===== 记录技能执行失败 =====
                        self.last_skill_execution = {
                            'success': False,
                            'skill_name': 'unknown',
                            'error': error_msg
                        }

                        # Add error to observation so Agent knows
                        base_obs = self.webarena_env.observation_handler.get_observation(
                            self.webarena_env.page, self.webarena_env.get_page_client(self.webarena_env.page)
                        )
                        # Preserve the observation format (dict with 'text' key)
                        skill_error_text = f"\n\n{'='*70}\nSKILL EXECUTION FAILED:\nError: {error_msg}\nPlease use basic actions (click, type, etc.) to complete this task instead.\n{'='*70}\n\n"
                        if isinstance(base_obs, dict) and 'text' in base_obs:
                            if isinstance(base_obs['text'], list) and len(base_obs['text']) > 0:
                                base_obs['text'][0] = skill_error_text + str(base_obs['text'][0])
                            elif isinstance(base_obs['text'], str):
                                base_obs['text'] = skill_error_text + base_obs['text']
                            self.obs = base_obs
                        else:
                            base_obs_str = str(base_obs) if base_obs else ""
                            self.obs = {"text": [skill_error_text + base_obs_str, None], "image": None}
                        self.update_webarena_metrics(action_cmd)
                        continue

                    skill_name = action_cmd.get("skill_name", "")
                    skill_args = action_cmd.get("skill_args", {})

                    # Get the skill from registry
                    skill = self.skill_registry.get_skill(skill_name)
                    if skill is None:
                        error_msg = f"Skill '{skill_name}' not found in registry"
                        print(f"Error: {error_msg}")
                        # ===== 记录技能执行失败 =====
                        self.last_skill_execution = {
                            'success': False,
                            'skill_name': skill_name,
                            'error': error_msg
                        }

                        # Add error to observation so Agent knows
                        base_obs = self.webarena_env.observation_handler.get_observation(
                            self.webarena_env.page, self.webarena_env.get_page_client(self.webarena_env.page)
                        )
                        # Preserve the observation format (dict with 'text' key)
                        skill_error_text = f"\n\n{'='*70}\nSKILL EXECUTION FAILED:\nSkill: {skill_name}\nError: {error_msg}\nAvailable skills: {', '.join(self.skill_registry.list_skills()) if self.skill_registry else 'None'}\nPlease use basic actions (click, type, etc.) to complete this task instead.\n{'='*70}\n\n"
                        if isinstance(base_obs, dict) and 'text' in base_obs:
                            if isinstance(base_obs['text'], list) and len(base_obs['text']) > 0:
                                base_obs['text'][0] = skill_error_text + str(base_obs['text'][0])
                            elif isinstance(base_obs['text'], str):
                                base_obs['text'] = skill_error_text + base_obs['text']
                            self.obs = base_obs
                        else:
                            base_obs_str = str(base_obs) if base_obs else ""
                            self.obs = {"text": [skill_error_text + base_obs_str, None], "image": None}
                        self.update_webarena_metrics(action_cmd)
                        continue

                    # DRIVE applicability gate.  Rejection falls back to
                    # primitive actions and is not logged as a skill failure.
                    try:
                        self.skill_registry.validate_skill_invocation(
                            skill_name,
                            skill_args,
                            url=self.webarena_env.page.url,
                            observation=self.obs,
                        )
                    except ValueError as applicability_error:
                        error_msg = str(applicability_error)
                        self.last_skill_execution = {
                            'success': False,
                            'invoked': False,
                            'failure_type': 'applicability_rejection',
                            'skill_name': skill_name,
                            'error': error_msg,
                        }
                        base_obs = self.webarena_env.observation_handler.get_observation(
                            self.webarena_env.page,
                            self.webarena_env.get_page_client(self.webarena_env.page),
                        )
                        rejection_text = (
                            f"\n\n{'='*70}\nSKILL NOT INVOKED (applicability check failed)\n"
                            f"Skill: {skill_name}\nReason: {error_msg}\n"
                            "Fall back to primitive actions for this step.\n"
                            f"{'='*70}\n\n"
                        )
                        if isinstance(base_obs, dict) and 'text' in base_obs:
                            text_content = base_obs['text']
                            if isinstance(text_content, (list, tuple)) and text_content:
                                values = list(text_content)
                                values[0] = rejection_text + str(values[0])
                                base_obs['text'] = values
                            else:
                                base_obs['text'] = rejection_text + str(text_content)
                            self.obs = base_obs
                        self.update_webarena_metrics(action_cmd)
                        continue

                    # Execute the skill with async handling
                    page = self.webarena_env.page
                    skill_result = None
                    skill_success = False
                    selector_trace = []
                    page_url_before = page.url
                    page_state_before = str(self.obs)[:2000]

                    # Get base URL for resolving relative paths in skills
                    # Config files retain placeholders such as ``__REDDIT__``;
                    # the live page URL is the authoritative resolved origin.
                    base_url = page.url or getattr(self, 'url', None)
                    if base_url:
                        # Extract just the origin (protocol + host + port)
                        from urllib.parse import urlparse
                        parsed = urlparse(base_url)
                        base_origin = f"{parsed.scheme}://{parsed.netloc}"
                    else:
                        base_origin = None

                    try:
                        # === Sync Playwright Skill Execution ===
                        # AgentOccam uses sync Playwright, but skills are written as async.
                        # We need to convert async skill code to sync execution.
                        import tempfile
                        import importlib
                        import sys
                        import os as skill_os
                        import re as skill_re
                        import typing

                        # Convert the complete defining module.  Converting only
                        # ``inspect.getsource(skill.fn)`` loses helper functions,
                        # aliases, constants, and exception classes.
                        skill_code = get_skill_module_source(skill.fn)
                        safety_errors = validate_skill_module_safety(skill_code)
                        if safety_errors:
                            raise SkillExecutionError(
                                "Interaction skill violates browser-only runtime policy: "
                                + "; ".join(safety_errors)
                            )
                        sync_skill_code = convert_async_module_to_sync(skill_code)

                        # === Auto-convert types based on function signature ===
                        # Get the function's type hints to check parameter types
                        try:
                            type_hints = typing.get_type_hints(skill.fn)
                        except Exception:
                            type_hints = {}

                        # Process skill_args: convert types based on function signature
                        processed_skill_args = {}
                        empty_required_params = []  # Track empty string params that might be required

                        for k, v in skill_args.items():
                            param_type = type_hints.get(k)

                            # Check for empty string in potentially required parameters
                            if isinstance(v, str) and v.strip() == '':
                                empty_required_params.append(k)

                            # === Type conversion based on function signature ===

                            # Check if parameter type is bool
                            is_bool_type = param_type is bool

                            # Check if parameter type is int
                            is_int_type = param_type is int

                            # Check if parameter type is list or List[...]
                            is_list_type = False
                            if param_type is not None:
                                origin = typing.get_origin(param_type)
                                if origin is list or param_type is list:
                                    is_list_type = True
                                elif hasattr(param_type, '__origin__') and param_type.__origin__ is list:
                                    is_list_type = True
                                elif str(param_type).startswith('list[') or str(param_type).startswith('typing.List['):
                                    is_list_type = True

                            # Apply conversions
                            if is_bool_type and isinstance(v, list) and len(v) == 1:
                                processed_skill_args[k] = bool(v[0])
                            elif is_int_type and isinstance(v, list) and len(v) == 1:
                                processed_skill_args[k] = int(v[0])
                            elif param_type is str and isinstance(v, str) and v.startswith('[') and v.endswith(']'):
                                processed_skill_args[k] = v[1:-1]
                            elif is_bool_type and isinstance(v, str):
                                # Convert string "True"/"False" to boolean
                                if v.lower() in ('true', '1', 'yes'):
                                    processed_skill_args[k] = True
                                    print(f"[DEBUG] Auto-converted '{k}' from string to bool: '{v}' -> True")
                                elif v.lower() in ('false', '0', 'no'):
                                    processed_skill_args[k] = False
                                    print(f"[DEBUG] Auto-converted '{k}' from string to bool: '{v}' -> False")
                                else:
                                    processed_skill_args[k] = v
                            elif is_int_type and isinstance(v, str):
                                # Convert string to int
                                try:
                                    processed_skill_args[k] = int(v)
                                    print(f"[DEBUG] Auto-converted '{k}' from string to int: '{v}' -> {int(v)}")
                                except ValueError:
                                    processed_skill_args[k] = v
                            elif is_list_type and isinstance(v, str):
                                # Convert single string to list with one element
                                processed_skill_args[k] = [v]
                                print(f"[DEBUG] Auto-converted '{k}' from string to list: '{v}' -> ['{v}']")
                            else:
                                processed_skill_args[k] = v

                        # Warn about empty required parameters
                        if empty_required_params:
                            print(f"[WARNING] Empty string values detected for parameters: {empty_required_params}")
                            print(f"[WARNING] This may cause the skill to fail or produce unexpected results.")

                        # Build args string for the skill call
                        args_str_parts = []
                        for k, v in processed_skill_args.items():
                            args_str_parts.append(f'{k}={repr(v)}')
                        args_str = ", ".join(args_str_parts)
                        selector_sets = skill.metadata.get('selector_sets', {})
                        recovery_branches = skill.metadata.get('recovery_branches', [])

                        # Create temporary module with sync skill code
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                            temp_filename = f.name

                            # Resolve relative URLs helper
                            resolve_url_code = f'''
def _resolve_url(url):
    """Convert relative URL to absolute URL"""
    base = "{base_origin or ''}"
    if base and isinstance(url, str) and url.startswith("/"):
        return base.rstrip("/") + url
    return url
'''
                            # Patch page.goto in the module to handle relative URLs
                            goto_patch_code = '''
# Patch page.goto to handle relative URLs
_original_goto = None

def _patch_page_goto(page):
    global _original_goto
    if _original_goto is None:
        _original_goto = page.goto

    class GotoWrapper:
        def __init__(self, original_fn):
            self._original = original_fn

        def __call__(self, url, *args, **kwargs):
            resolved_url = _resolve_url(url)
            return self._original(resolved_url, *args, **kwargs)

    page.goto = GotoWrapper(_original_goto)
    return page

def _restore_page_goto(page):
    global _original_goto
    if _original_goto is not None:
        page.goto = _original_goto
'''

                            module_code = f'''
import re
import time
from datetime import datetime
from typing import List, Dict, Any
from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError
from AgentOccam.skill_runtime import prepare_skill_page
PlaywrightTimeout = PlaywrightTimeoutError  # Alias used in some skills

# Custom exception classes used in skills
class SkillExecutionError(Exception):
    """技能执行过程中的错误"""
    pass

class ElementNotFoundError(SkillExecutionError):
    """元素未找到错误"""
    pass

class NavigationError(SkillExecutionError):
    """导航错误"""
    pass

class SubmissionError(SkillExecutionError):
    """提交错误"""
    pass

{resolve_url_code}

{goto_patch_code}

{sync_skill_code}

def act(page):
    """Wrapper function to execute the skill (sync version)"""
    skill_page = prepare_skill_page(
        page,
        base_origin={repr(base_origin or '')},
        selector_sets={repr(selector_sets)},
        recovery_branches={repr(recovery_branches)},
    )
    result = {skill.name}(skill_page{', ' if args_str else ''}{args_str})
    return result, skill_page.trace
'''
                            f.write(module_code)

                        # Debug: print the generated code to help diagnose issues
                        print(f"[DEBUG] Generated sync skill code saved to: {temp_filename}")
                        if 'await' in sync_skill_code:
                            print(f"[WARNING] 'await' still found in converted code!")
                            # Find lines with await
                            for i, line in enumerate(sync_skill_code.split('\n'), 1):
                                if 'await' in line:
                                    print(f"  Line {i}: {line.strip()}")

                        # Also check the full module code
                        if 'await' in module_code:
                            print(f"[WARNING] 'await' found in module_code!")

                        # Import and execute the module
                        module_dir = skill_os.path.dirname(temp_filename)
                        module_name = skill_os.path.basename(temp_filename)[:-3]

                        # Clean up any previous import
                        if module_name in sys.modules:
                            del sys.modules[module_name]

                        sys.path.insert(0, module_dir)
                        skill_exec_error = None
                        skill_success = False
                        try:
                            module = importlib.import_module(module_name)

                            # Execute synchronously
                            skill_result, selector_trace = module.act(page)

                            # === Check if skill result indicates actual success ===
                            # Some skills return dict with success indicators
                            skill_success = True
                            failure_reason = None

                            if isinstance(skill_result, dict):
                                # Check common failure patterns in skill results

                                # Pattern 1: subscribed/not_found pattern (e.g., subscribe_to_forums_by_names)
                                if 'subscribed' in skill_result and 'not_found' in skill_result:
                                    subscribed = skill_result.get('subscribed', [])
                                    not_found = skill_result.get('not_found', [])
                                    already_subscribed = skill_result.get('already_subscribed', [])
                                    # Fail if nothing was subscribed AND there are items not found AND nothing was already subscribed
                                    if not subscribed and not_found and not already_subscribed:
                                        skill_success = False
                                        failure_reason = f"No forums were subscribed. Not found: {not_found}"

                                # Pattern 2: explicit success field
                                if 'success' in skill_result:
                                    if skill_result['success'] is False:
                                        skill_success = False
                                        failure_reason = skill_result.get('error', skill_result.get('message', 'Skill returned success=False'))

                                # Pattern 3: error field present
                                if 'error' in skill_result and skill_result['error']:
                                    skill_success = False
                                    failure_reason = skill_result['error']

                                # Pattern 4: upvoted/downvoted count patterns
                                if 'upvoted' in skill_result or 'downvoted' in skill_result:
                                    count = skill_result.get('upvoted', 0) or skill_result.get('downvoted', 0) or skill_result.get('downvoted_count', 0)
                                    attempted = skill_result.get('attempted_count', 0) or skill_result.get('total_found', 0)
                                    if attempted > 0 and count == 0:
                                        skill_success = False
                                        failure_reason = f"Found {attempted} items but none were processed successfully"

                                # Pattern 5: not_found is True
                                if skill_result.get('not_found') is True:
                                    skill_success = False
                                    failure_reason = "Target not found"

                            elif isinstance(skill_result, bool):
                                # If skill returns boolean, use it directly
                                if skill_result is False:
                                    skill_success = False
                                    failure_reason = "Skill returned False"

                            if skill_success:
                                print(f"✓ Successfully executed skill '{skill_name}' with result: {skill_result}")
                            else:
                                print(f"✗ Skill '{skill_name}' executed but returned failure result: {skill_result}")
                                print(f"  Failure reason: {failure_reason}")

                        except Exception as exec_error:
                            # On error, keep the temp file for debugging
                            skill_exec_error = exec_error
                            print(f"[DEBUG] Error occurred. Temp file kept at: {temp_filename}")
                            print(f"[DEBUG] Run 'cat {temp_filename}' to see the generated code")
                            # Print full traceback
                            import traceback as tb
                            print("[DEBUG] Full traceback:")
                            tb.print_exc()
                            raise

                        finally:
                            sys.path.remove(module_dir)
                            if module_name in sys.modules:
                                del sys.modules[module_name]
                            # Only delete temp file on success
                            if skill_success and skill_exec_error is None:
                                try:
                                    skill_os.unlink(temp_filename)
                                except:
                                    pass

                        # ===== 记录技能执行结果（基于返回值判断成功/失败）=====
                        if skill_success:
                            self.last_skill_execution = {
                                'success': True,
                                'skill_name': skill_name,
                                'result': skill_result
                            }
                        else:
                            failure_ctx = self._capture_skill_failure_context(
                                skill.fn, processed_skill_args
                            )
                            self.last_skill_execution = {
                                'success': False,
                                'skill_name': skill_name,
                                'result': skill_result,
                                'failure_reason': failure_reason,
                                'url': failure_ctx.get('url', 'unknown'),
                                'selectors': failure_ctx.get('selectors', []),
                                'skill_args': failure_ctx.get('skill_args', {}),
                                'page_state_snippet': failure_ctx.get('page_state_snippet', ''),
                            }

                    except Exception as e:
                        print(f"✗ Error executing skill '{skill_name}': {e}")
                        skill_result = {"error": str(e), "skill_name": skill_name}
                        # ===== 记录技能执行失败（含丰富上下文）=====
                        failure_ctx = self._capture_skill_failure_context(
                            skill.fn, skill_args
                        )
                        self.last_skill_execution = {
                            'success': False,
                            'skill_name': skill_name,
                            'error': str(e),
                            'traceback': traceback.format_exc(),
                            'url': failure_ctx.get('url', 'unknown'),
                            'selectors': failure_ctx.get('selectors', []),
                            'skill_args': failure_ctx.get('skill_args', {}),
                            'page_state_snippet': failure_ctx.get('page_state_snippet', ''),
                        }

                        if self.global_config and getattr(self.global_config, 'debug', False):
                            traceback.print_exc()

                    # Buffer the complete DRIVE feedback tuple.  The final task
                    # label is attached by ``finalize_episode_feedback`` when
                    # WebArena evaluation completes.
                    try:
                        current_url = self.webarena_env.page.url
                        latest_obs = self.webarena_env.observation_handler.get_observation(
                            self.webarena_env.page,
                            self.webarena_env.get_page_client(self.webarena_env.page),
                        )
                        page_state_after = str(latest_obs)[:2000]
                        args_for_feedback = (
                            processed_skill_args
                            if 'processed_skill_args' in locals()
                            else dict(skill_args or {})
                        )
                        source_selectors = self._capture_skill_failure_context(
                            skill.fn, args_for_feedback
                        ).get('selectors', [])
                        url_changed = current_url != page_url_before
                        page_state_changed = page_state_after != page_state_before
                        local_success, missing_effect_reason = _derive_local_skill_success(
                            bool(skill_success),
                            skill_result,
                            url_changed=url_changed,
                            page_state_changed=page_state_changed,
                            postcondition_checks=skill.metadata.get("postcondition_checks"),
                        )
                        if not local_success and missing_effect_reason:
                            if self.last_skill_execution is None:
                                self.last_skill_execution = {}
                            self.last_skill_execution.update(
                                {
                                    'success': False,
                                    'failure_reason': missing_effect_reason,
                                }
                            )
                        local_outcome = {
                            'success': local_success,
                            'result': skill_result,
                            'observable_effect': url_changed or page_state_changed,
                            'error': self.last_skill_execution.get('error')
                            if self.last_skill_execution else None,
                            'failure_reason': self.last_skill_execution.get('failure_reason')
                            if self.last_skill_execution else None,
                        }
                        execution_log = {
                            'url_before': page_url_before,
                            'url_after': current_url,
                            'url_changed': url_changed,
                            'page_state_changed': page_state_changed,
                            'selectors': source_selectors,
                            'selector_trace': selector_trace,
                            'exception': self.last_skill_execution.get('traceback', '')
                            if self.last_skill_execution else '',
                        }
                        feedback = self.skill_registry.record_interaction_feedback(
                            skill_name,
                            task_instruction=self.objective,
                            page_context={
                                'url': page_url_before,
                                'site': self.skill_registry.site_name,
                                'page_state_snippet': page_state_before,
                            },
                            arguments=args_for_feedback,
                            local_outcome=local_outcome,
                            execution_log=execution_log,
                        )
                        if self.last_skill_execution is not None:
                            self.last_skill_execution['feedback_id'] = feedback['feedback_id']
                            self.last_skill_execution['feedback_recorded'] = True
                        if not local_success:
                            legacy_failure = dict(self.last_skill_execution or {})
                            legacy_failure.update(
                                {
                                    'task_id': self.config.get('task_id'),
                                    'objective': self.objective,
                                    'selectors': source_selectors,
                                }
                            )
                            self.skill_registry.record_skill_failure(legacy_failure)
                    except Exception as feedback_error:
                        print(f"[Warning] Failed to buffer DRIVE skill feedback: {feedback_error}")

                    # Update observation after skill execution
                    # IMPORTANT: Inject skill result into the observation so Agent can see it
                    base_obs = self.webarena_env.observation_handler.get_observation(
                        self.webarena_env.page, self.webarena_env.get_page_client(self.webarena_env.page)
                    )

                    # Preserve the observation format (dict with 'text' key)
                    # base_obs should be a dict like {"text": [...], "image": ...}
                    if skill_success and skill_result is not None:
                        skill_result_text = f"""

{'='*70}
SKILL EXECUTION: SUCCESS
{'='*70}
Skill: {skill_name}
Status: SUCCESS
Result: {skill_result}
Use the local result and current page to assess the remaining task. Follow the
active reasoning skill's V instruction before stopping. DRIVE retrieval will
run again for the updated page scene.
{'='*70}

"""
                    else:
                        # Skill failed - add detailed error message to observation
                        # Use failure_reason if available (from result analysis), otherwise fall back to error field
                        if 'failure_reason' in dir() and failure_reason:
                            error_msg = failure_reason
                        elif isinstance(skill_result, dict) and skill_result.get("error"):
                            error_msg = skill_result["error"]
                        elif isinstance(skill_result, dict):
                            error_msg = f"Skill returned failure result: {skill_result}"
                        else:
                            error_msg = "Skill execution failed"
                        skill_result_text = f"""

{'='*70}
SKILL EXECUTION: FAILED
{'='*70}
Skill: {skill_name}
Status: FAILED
Result: {skill_result}
Failure Reason: {error_msg}

NEXT STEP: Fall back to primitive actions for this scene. DRIVE retrieval will
run again at the next environment step.
- click [id] - Click an element
- type [id] [text] [enter] - Type text
- scroll [up/down] - Scroll the page
- go_back - Go back to previous page
{'='*70}

"""

                    # Prepend skill result to observation while preserving format
                    if isinstance(base_obs, dict) and 'text' in base_obs:
                        # Preserve the dict format - prepend skill result to text content
                        # Handle both list and tuple (tuple is immutable, need to convert)
                        text_content = base_obs['text']
                        if isinstance(text_content, (list, tuple)) and len(text_content) > 0:
                            # base_obs['text'] is typically (accessibility_tree_str, root_node) or [...]
                            # Convert to list if tuple to allow modification
                            text_list = list(text_content)
                            text_list[0] = skill_result_text + str(text_list[0])
                            base_obs['text'] = text_list
                        elif isinstance(text_content, str):
                            base_obs['text'] = skill_result_text + text_content
                        self.obs = base_obs
                    else:
                        # Fallback: create proper dict format
                        base_obs_str = str(base_obs) if base_obs else ""
                        self.obs = {"text": [skill_result_text + base_obs_str, None], "image": None}

                    self.update_webarena_metrics(action_cmd)

                    # Reset verification flag after successful action (non-TYPE)
                    if action_cmd.get("action_type") != ActionTypes.TYPE:
                        self.verification_requested = False
                else:
                    # Regular webarena action execution
                    self.obs, _, self.terminated, _, self.info = self.webarena_env.step(action_cmd)
                    self.update_webarena_metrics(action_cmd)

                    # Reset verification flag after successful action (non-TYPE)
                    if action_cmd.get("action_type") != ActionTypes.TYPE:
                        self.verification_requested = False
            except Exception as e:
                print(f"Error occurred while taking step: {e}")
                if self.global_config and getattr(self.global_config, 'debug', False):
                    traceback.print_exc()
                # Print the action_cmd for debugging
                print(f"[DEBUG] Failed action_cmd: {action_cmd}")

        # If STOP was rejected for verification, return special status
        if stop_rejected:
            status = self.status()
            status['verification_required'] = True

            # Get number of actions taken
            recent_actions = [item for item in self.trajectory if isinstance(item, dict) and "action_type" in item]
            num_actions = len([a for a in recent_actions if a.get("action_type") != ActionTypes.STOP])

            # Customize prompt based on why STOP was rejected
            if num_actions < 3:
                status['verification_prompt'] = (
                    f"⚠️  You are attempting to STOP after only {num_actions} action(s). This is too early!\n\n"
                    "Before giving up on this task, you MUST:\n"
                    "1. Try at least 3-5 different approaches (different pages, search, navigation)\n"
                    "2. Use the search function if available (look for search boxes)\n"
                    "3. Check related sections: My Account, Help, About, FAQ, footer links\n"
                    "4. Try searching for keywords related to your objective\n"
                    "5. Scroll down the current page - information might be below the fold\n\n"
                    "What other pages or methods can you try? Continue exploring instead of stopping."
                )
            else:
                status['verification_prompt'] = (
                    "Before stopping, please verify that the task is truly complete:\n"
                    "1. Describe what you see on the current page\n"
                    "2. Confirm whether the task objective has been fully achieved\n"
                    "3. If not achieved, what is still missing?\n"
                    "Only STOP if you are certain the task is complete."
                )
            return status

        return self.status()
    
    def update_webarena_metrics(self, action_cmd=None):
        # Append action (if any) and resulting sate
        if action_cmd:
            self.trajectory.append(action_cmd)
            if action_cmd["action_type"]== ActionTypes.STOP:
                self.is_done = True

        if not self.is_done: # If we are done, no need to append state
            state_info: StateInfo = {"observation": self.obs, "info": self.info}
            self.trajectory.append(state_info)
            
        if self.is_done:
            try:
                evaluator = evaluator_router(self.config_file)
                self.reward = evaluator(trajectory=self.trajectory, config_file=self.config_file, page=self.webarena_env.page, client=self.webarena_env.get_page_client(self.webarena_env.page))
                print(f"[Evaluation] Task completed successfully. Reward: {self.reward}")
            except Exception as e:
                print("="*70)
                print(f"[Evaluation ERROR] Exception occurred during evaluation!")
                print(f"Exception type: {type(e).__name__}")
                print(f"Exception message: {e}")
                print(f"Config file: {self.config_file}")
                print(f"Page URL: {self.webarena_env.page.url if hasattr(self.webarena_env, 'page') else 'N/A'}")
                print(f"Trajectory length: {len(self.trajectory)}")

                # Print trajectory structure info
                if self.trajectory:
                    last_elem = self.trajectory[-1]
                    print(f"Last trajectory element type: {type(last_elem)}")
                    if isinstance(last_elem, dict):
                        print(f"Last element keys: {list(last_elem.keys())[:10]}")
                        if 'action_type' in last_elem:
                            print(f"Last element action_type: {last_elem.get('action_type')}")
                        if 'answer' in last_elem:
                            answer = last_elem.get('answer', '')
                            print(f"Last element answer: {answer[:100]}..." if len(str(answer)) > 100 else f"Last element answer: {answer}")

                print("\nFull traceback:")
                traceback.print_exc()
                print("="*70)
                self.reward = 0

            if self.skill_registry is not None:
                self.skill_registry.finalize_episode_feedback(self.reward)

from AgentOccam.obs_opt import parse_node_descendants, parse_node_ancestors, parse_node_siblings, action_set_invisible, action_set_visible, action_set_visible_if_with_name, translate_node_to_str, construct_new_DOM_with_visible_nodes
from AgentOccam.llms.claude import call_claude, call_claude_with_messages, arrange_message_for_claude
from AgentOccam.llms.mistral import call_mistral, call_mistral_with_messages, arrange_message_for_mistral
from AgentOccam.llms.cohere import call_cohere, call_cohere_with_messages, arrange_message_for_cohere
from AgentOccam.llms.llama import call_llama, call_llama_with_messages, arrange_message_for_llama
from AgentOccam.llms.titan import call_titan, call_titan_with_messages, arrange_message_for_titan
from AgentOccam.llms.gpt import call_gpt, call_gpt_with_messages, arrange_message_for_gpt, get_token_stats, reset_token_stats
from AgentOccam.llms.gemini import call_gemini, call_gemini_with_messages, arrange_message_for_gemini
from AgentOccam.utils import CURRENT_DIR, HOMEPAGE_URL
from AgentOccam.skill_registry import SkillRegistry, load_skills_from_site
from AgentOccam.operation_level_matcher import OperationLevelMatcher
from AgentOccam.skill_prompt_injector import SkillPromptInjector
from browser_env import create_id_based_actions, ActionTypes

from typing import Dict
import re
import copy
import os
from functools import partial
import random
import json
from pathlib import Path
import logging

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


DEFAULT_DOCUMENTED_INTERACTION_ELEMENTS = ["observation", "action"]
DEFAULT_ONLINE_INTERACTION_ELEMENTS = ["url", "observation"]
MODEL_FAMILIES = ["claude", "mistral", "cohere", "llama", "titan", "gpt", "gemini", "qwen"]
CALL_MODEL_MAP = {
    "claude": call_claude,
    "mistral": call_mistral,
    "cohere": call_cohere,
    "llama": call_llama,
    "titan": call_titan,
    "gpt": call_gpt,
    "gemini": call_gemini,
    # Qwen is evaluated through the configured OpenAI-compatible endpoint.
    "qwen": call_gpt,
}
CALL_MODEL_WITH_MESSAGES_FUNCTION_MAP = {
    "claude": call_claude_with_messages,
    "mistral": call_mistral_with_messages,
    "cohere": call_cohere_with_messages,
    "llama": call_llama_with_messages,
    "titan": call_titan_with_messages,
    "gpt": call_gpt_with_messages,
    "gemini": call_gemini_with_messages,
    "qwen": call_gpt_with_messages,
}
ARRANGE_MESSAGE_FOR_MODEL_MAP = {
    "claude": arrange_message_for_claude,
    "mistral": arrange_message_for_mistral,
    "cohere": arrange_message_for_cohere,
    "llama": arrange_message_for_llama,
    "titan": arrange_message_for_titan,
    "gpt": arrange_message_for_gpt,
    "gemini": arrange_message_for_gemini,
    "qwen": arrange_message_for_gpt,
}


def resolve_model_family(model_id: str) -> str:
    """Resolve native and OpenAI-compatible model identifiers."""

    normalized = str(model_id).lower()
    for family in MODEL_FAMILIES:
        if family in normalized:
            return family
    # OpenAI-compatible providers often use vendor/model names unknown to the
    # baseline.  They share the GPT message transport configured by API base.
    if any(name in normalized for name in ("deepseek", "openai", "moonshot")):
        return "gpt"
    raise ValueError(f"Unsupported model family for model: {model_id}")


def _parse_type_action_text(action_str: str):
    """Parse AgentOccam type actions with tolerant content/flag syntax."""

    action_str = action_str.strip()
    bracketed = re.fullmatch(
        r"type ?\[(\d+)\] ?\[(.*?)\](?: ?\[(?:press_enter_after=)?(0|1)\])?",
        action_str,
        re.DOTALL,
    )
    if bracketed:
        element_id, text, enter_flag = bracketed.groups()
        return element_id, text, enter_flag or "1"

    unbracketed = re.fullmatch(
        r"type ?\[(\d+)\] ?(.+?) ?\[(?:press_enter_after=)?(0|1)\]",
        action_str,
        re.DOTALL,
    )
    if unbracketed:
        element_id, text, enter_flag = unbracketed.groups()
        return element_id, text.strip(), enter_flag

    return None

class Agent:
    def __init__(self, config, objective, prompt_template):
        self.config = config
        self.objective = objective
        self.prompt_template = prompt_template

        if hasattr(self.config, "documented_interaction_elements"):
            self.previous_interactions = {k: [] for k in set(DEFAULT_DOCUMENTED_INTERACTION_ELEMENTS+self.config.documented_interaction_elements)}
        else:
            self.previous_interactions = {k: [] for k in DEFAULT_DOCUMENTED_INTERACTION_ELEMENTS}
        if hasattr(self.config, "online_interaction_elements"):
            self.online_interaction = {k: None for k in set(DEFAULT_ONLINE_INTERACTION_ELEMENTS+self.config.online_interaction_elements)}
        else:
            self.online_interaction = {k: None for k in DEFAULT_ONLINE_INTERACTION_ELEMENTS}

        self.model_family = resolve_model_family(self.config.model)
        self.call_model = partial(CALL_MODEL_MAP[self.model_family], model_id=self.config.model)
        self.call_model_with_message = partial(CALL_MODEL_WITH_MESSAGES_FUNCTION_MAP[self.model_family], model_id=self.config.model)
        self.arrange_message_for_model = ARRANGE_MESSAGE_FOR_MODEL_MAP[self.model_family]

        # 加载技能库（如果配置中指定了）
        self.skill_registry = None
        self._load_skills()

    def shift_model(self, model_id):
        self.model_family = resolve_model_family(model_id)
        self.call_model = partial(CALL_MODEL_MAP[self.model_family], model_id=model_id)
        self.call_model_with_message = partial(CALL_MODEL_WITH_MESSAGES_FUNCTION_MAP[self.model_family], model_id=model_id)
        self.arrange_message_for_model = ARRANGE_MESSAGE_FOR_MODEL_MAP[self.model_family]

    def _load_skills(self):
        """
        加载技能库（根据配置）

        配置项（从 config.skills 读取）：
        - use_skills: 是否启用技能库
        - skill_site: 技能站点名称（如 'reddit', 'shopping'）
        - skill_dir: 自定义技能目录路径（可选，默认为 'skills'）
        - top_k: 提示词中展示的技能数量（可选，默认为 8）
        """
        from pathlib import Path
        import logging

        preloaded_registry = getattr(self.config, "preloaded_skill_registry", None)
        if preloaded_registry is not None:
            self.skill_registry = preloaded_registry
            preloaded_cfg = getattr(self.config, "skills", None)
            if preloaded_cfg and hasattr(self.skill_registry, "coordinator"):
                getter = (
                    (lambda key, default: preloaded_cfg.get(key, default))
                    if isinstance(preloaded_cfg, dict)
                    else (lambda key, default: getattr(preloaded_cfg, key, default))
                )
                self.skill_registry.coordinator.use_interaction_skills = getter(
                    "use_interaction_skills", True
                )
                self.skill_registry.coordinator.use_reasoning_skills = getter(
                    "use_reasoning_skills", True
                )
                self.skill_registry.coordinator.interaction_threshold = float(
                    getter("interaction_threshold", 0.06)
                )
                self.skill_registry.coordinator.reasoning_threshold = float(
                    getter("reasoning_threshold", 0.06)
                )
                self.skill_registry.coordinator.contract_mode = getter(
                    "contract_mode", "drive"
                )
            self.operation_level_matcher = OperationLevelMatcher(self.skill_registry)
            self.skill_prompt_injector = SkillPromptInjector(self.operation_level_matcher)
            self.current_filtered_skills = []
            self.current_related_lessons = []
            self.skill_injection_prompt = ""
            return

        # 从 config.skills 读取技能配置
        cfg_sk = getattr(self.config, "skills", None)
        if not cfg_sk:
            self.skill_registry = None
            return

        # 兼容字典和对象两种配置格式
        def get_value(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            else:
                return getattr(obj, key, default)

        def configure_drive_levels():
            if self.skill_registry is None or not hasattr(self.skill_registry, "coordinator"):
                return
            self.skill_registry.coordinator.use_interaction_skills = get_value(
                cfg_sk, "use_interaction_skills", True
            )
            self.skill_registry.coordinator.use_reasoning_skills = get_value(
                cfg_sk, "use_reasoning_skills", True
            )
            self.skill_registry.coordinator.interaction_threshold = float(
                get_value(cfg_sk, "interaction_threshold", 0.06)
            )
            self.skill_registry.coordinator.reasoning_threshold = float(
                get_value(cfg_sk, "reasoning_threshold", 0.06)
            )
            self.skill_registry.coordinator.contract_mode = get_value(
                cfg_sk, "contract_mode", "drive"
            )

        if not get_value(cfg_sk, "use_skills", False):
            self.skill_registry = None
            return

        # 获取技能站点名称
        skill_site = get_value(cfg_sk, "skill_site")
        if not skill_site:
            self.skill_registry = None
            if self.config.verbose >= 1:
                print("⚠️  技能库配置错误: skill_site 未指定")
            return

        try:
            # 获取技能目录（默认为项目根目录的 skills/）
            skill_dir = get_value(cfg_sk, "skill_dir")
            if skill_dir:
                skills_base_dir = Path(skill_dir)
            else:
                # 默认路径：AgentOccam/../skills/
                skills_base_dir = Path(CURRENT_DIR).parent / "skills"

            # 获取技能文件名配置
            skill_file = get_value(cfg_sk, "skill_file")
            skill_metadata = get_value(cfg_sk, "skill_metadata")

            # 加载技能库
            self.skill_registry = load_skills_from_site(
                site_name=skill_site,
                skills_base_dir=skills_base_dir,
                skill_file=skill_file,
                skill_metadata=skill_metadata
            )

            # 加载任务级经验（如果配置了）
            task_lessons_path = get_value(cfg_sk, "task_lessons_path")
            if task_lessons_path:
                # CURRENT_DIR 已经是项目根目录，不需要 .parent
                task_lessons_file = Path(CURRENT_DIR) / task_lessons_path
                self.skill_registry.load_external_task_lessons(task_lessons_file)

            if self.config.verbose >= 1:
                print(f"✓ 成功加载技能库: {self.skill_registry}")
                verified_count = len(self.skill_registry.get_verified_skills())
                print(f"  - 总技能数: {len(self.skill_registry)}")
                print(f"  - 已验证技能: {verified_count}")
                print(f"  - 任务级经验: {len(self.skill_registry.external_task_lessons)}")

            # 初始化操作级匹配器和提示词注入器
            self.operation_level_matcher = OperationLevelMatcher(self.skill_registry)
            self.skill_prompt_injector = SkillPromptInjector(self.operation_level_matcher)
            configure_drive_levels()

            # 跟踪当前任务的过滤技能和经验
            self.current_filtered_skills = []
            self.current_related_lessons = []
            self.skill_injection_prompt = ""

        except FileNotFoundError as e:
            if self.config.verbose >= 1:
                print(f"⚠️  技能库未找到: {e}")
                print(
                    "   提示: 使用当前站点流水线生成技能库，例如 "
                    f"'python tools/site_specific/site_skill_pipeline.py --site {skill_site} "
                    f"--trajectories /path/to/run --out-dir skills/{skill_site}'"
                )
            self.skill_registry = None
            self.operation_level_matcher = None
            self.skill_prompt_injector = None
        except Exception as e:
            if self.config.verbose >= 1:
                print(f"⚠️  加载技能库失败: {e}")
            if self.config.debug:
                import traceback
                traceback.print_exc()
            self.skill_registry = None
            self.operation_level_matcher = None
            self.skill_prompt_injector = None

        # 如果没有初始化过，设为None
        if not hasattr(self, 'operation_level_matcher'):
            self.operation_level_matcher = None
        if not hasattr(self, 'skill_prompt_injector'):
            self.skill_prompt_injector = None
        if not hasattr(self, 'current_filtered_skills'):
            self.current_filtered_skills = []
        if not hasattr(self, 'current_related_lessons'):
            self.current_related_lessons = []
        if not hasattr(self, 'skill_injection_prompt'):
            self.skill_injection_prompt = ""

    def get_skills_summary(self, top_k=10, only_verified=True):
        """
        获取技能摘要（用于添加到提示词）

        Args:
            top_k: 返回前 N 个技能
            only_verified: 只返回经过验证的技能

        Returns:
            str: 格式化的技能列表
        """
        if not self.skill_registry:
            return ""

        return self.skill_registry.summarize_skills(
            top_k=top_k,
            only_verified=only_verified
        )

    def search_relevant_skills(self, query, top_k=5):
        """
        搜索与任务相关的技能

        Args:
            query: 搜索查询（如任务目标）
            top_k: 返回前 N 个相关技能

        Returns:
            list[Skill]: 相关技能列表
        """
        if not self.skill_registry:
            return []

        return self.skill_registry.search_skills(query, top_k=top_k)

    def has_skills(self):
        """检查是否加载了技能库"""
        return self.skill_registry is not None and len(self.skill_registry) > 0

    def _prepare_skill_context(
        self,
        objective: str,
        force_refresh: bool = False,
        *,
        url: str = "",
        observation=None,
        step: int = None,
    ):
        """Run DRIVE retrieval for the *current* task-page scene.

        ``force_refresh`` is kept for compatibility with older callers.  DRIVE
        intentionally does not cache a task-level selection: both libraries are
        structurally filtered and semantically ranked again at every step.
        """

        if not self.skill_registry:
            self.skill_injection_prompt = ""
            return
        try:
            current_url = url or self.online_interaction.get("url", "") or ""
            current_observation = (
                observation
                if observation is not None
                else self.online_interaction.get("observation", "")
            )
            # DRIVE is asymmetric: retrieve reasoning guidance, form g_t with
            # that guidance, and only then retrieve an executable procedure.
            reasoning_stage = self.skill_registry.retrieve_reasoning_stage(
                objective, current_url, current_observation
            )
            _, reasoning_candidate, _ = reasoning_stage
            current_intent = self._form_drive_intent(
                objective,
                current_observation,
                reasoning_candidate.skill if reasoning_candidate else None,
            )
            selection = self.skill_registry.retrieve_dual_skills(
                objective=objective,
                url=current_url,
                observation=current_observation,
                current_intent=current_intent,
                step=step,
                reasoning_stage=reasoning_stage,
            )
            self.current_filtered_skills = (
                [selection.interaction.skill] if selection.interaction else []
            )
            self.current_related_lessons = (
                [selection.reasoning.skill] if selection.reasoning else []
            )
            self.skill_injection_prompt = self.skill_prompt_injector.generate_skill_assistance_prompt(
                filtered_skills=self.current_filtered_skills,
                related_lessons=self.current_related_lessons,
                # The executable skill is selected against g_t, not merely
                # the original task text.  Show the same intent to the actor
                # so retrieval and subsequent binding share one state.
                task_intent=selection.current_intent,
                skill_registry=self.skill_registry,
                skill_already_used=getattr(self, "skill_attempted", False),
                skill_succeeded=getattr(self, "skill_succeeded", False),
            )
            token_budget = int(getattr(getattr(self.config, "skills", None), "skill_context_token_budget", 0) or 0)
            if token_budget:
                import tiktoken
                encoding = tiktoken.get_encoding("cl100k_base")
                prompt_tokens = len(encoding.encode(self.skill_injection_prompt))
                if prompt_tokens > token_budget:
                    raise ValueError(
                        f"Selected skill bundle requires {prompt_tokens} tokens, exceeding fixed budget {token_budget}"
                    )
                while len(encoding.encode(self.skill_injection_prompt)) < token_budget:
                    self.skill_injection_prompt += " x"
            if self.config.verbose >= 1:
                interaction_name = (
                    selection.interaction.skill.name if selection.interaction else "primitive fallback"
                )
                reasoning_name = (
                    selection.reasoning.skill.get("skill_id")
                    if selection.reasoning
                    else "none"
                )
                print(
                    f"[DRIVE retrieval] mode={selection.contract_mode}, step={step}, "
                    f"intent={selection.current_intent[:120]!r}, "
                    f"interaction={interaction_name}, reasoning={reasoning_name}, "
                    f"slot={selection.shared_slot_id or 'typed'}"
                )
        except Exception as e:
            self.current_filtered_skills = []
            self.current_related_lessons = []
            self.skill_injection_prompt = ""
            logger.warning(f"Failed to prepare DRIVE skill context: {e}")
            if getattr(self.config, "debug", False):
                import traceback
                traceback.print_exc()

    def _form_drive_intent(self, objective, observation, reasoning_skill=None) -> str:
        """Form the current high-level intent between the two retrieval stages."""

        guidance = ""
        if reasoning_skill:
            fields = reasoning_skill.get("reasoning_skill", {})
            guidance = "\n".join(
                f"{field}: {fields.get(field, '')}" for field in ("M", "B", "V")
            )
        active_plan = ""
        if hasattr(self, "get_active_plan"):
            try:
                active_plan = self.get_active_plan()
            except Exception:
                active_plan = ""
        observation_text = str(observation)
        prompt = f"""Form the single current web-task intent for the next decision.
Use the task, current plan, page evidence, and corrective reasoning guidance.
Do not choose browser selectors or emit an action. Return exactly one line:
CURRENT_INTENT: <concise intent>

TASK: {objective}
CURRENT PLAN: {active_plan}
REASONING GUIDANCE:
{guidance or 'None'}
PAGE EVIDENCE:
{observation_text[:4000]}
"""
        fallback = active_plan or objective
        try:
            response = self.call_model(
                prompt,
                system_prompt=(
                    "You form high-level web-agent intents after reasoning guidance "
                    "and before interaction-procedure retrieval."
                ),
            )
            if not response:
                return fallback
            match = re.search(r"CURRENT_INTENT:\s*(.+)", str(response), re.IGNORECASE)
            intent = (match.group(1) if match else str(response).splitlines()[0]).strip()
            return intent[:500] or fallback
        except Exception as exc:
            logger.warning("DRIVE intent formation failed; using active plan: %s", exc)
            return fallback

    def _get_dynamic_skill_injection(self, step: int) -> str:
        """Refresh the dual-level singleton selection for this environment step."""
        self._prepare_skill_context(
            self.objective,
            force_refresh=True,
            url=self.online_interaction.get("url", "") or "",
            observation=self.online_interaction.get("observation", ""),
            step=step,
        )
        return self.skill_injection_prompt

    def prune_message_list(self, message_list):
        return self.merge_adjacent_text([m for m in message_list if not (m[0]=="text" and len(m[1])==0)])

    def merge_adjacent_text(self, message_list):
        merged_list = []
        current_tuple = None

        for tup in message_list:
            if tup[0] == "text":
                if current_tuple:
                    current_tuple = (current_tuple[0], current_tuple[1] + tup[1])
                else:
                    current_tuple = tup
            else:
                if current_tuple:
                    merged_list.append(current_tuple)
                    current_tuple = None
                merged_list.append(tup)

        if current_tuple:
            merged_list.append(current_tuple)

        return merged_list

    def get_step(self):
        return len(self.previous_interactions["action"])

    def update_objective(self, objective):
        self.objective = objective

    def update_online_state(self, **online_states):
        for k in online_states.keys():
            if k in self.online_interaction.keys():
                self.online_interaction[k] = online_states[k]

    def update_history(self, **interaction_dict):
        for k in interaction_dict.keys():
            if k in self.previous_interactions.keys():
                self.previous_interactions[k].append(interaction_dict[k])

    def equal_history_length(self):
        lengths = [len(self.previous_interactions[k]) for k in self.previous_interactions.keys()]
        return (len(set(lengths)) == 1)

    def parse_elements(self, text, key_list):
        element_dict = {}
        for k in key_list:
            # _match = re.search(rf'{k.upper()}:\s*(.*?)\s*(?=\n[A-Z\d\s\W]*: *\n|$)', text, re.DOTALL)
            _match = re.search(rf'{k.upper()}:\s*(.*?)\s*(?=\n[A-Z\s]*:|$)', text, re.DOTALL)
            element_dict[k] = _match.group(1).strip() if _match else ""
        return element_dict

    def get_output_specifications(self):
        output_specifications = "\n".join([f"{o.upper()}:\n" + "".join(open(os.path.join(CURRENT_DIR, "AgentOccam", "prompts", "output_specifications", "{}.txt".format(o.replace(" ", "_"))), "r").readlines()) for o in self.config.output])
        return output_specifications

    def parse_stipulated_action_list(self, text: str, action: str, actions: list) -> str:
        pattern = rf'({re.escape(action)}\s*(.*?))(?=\n(?:{"|".join(map(re.escape, actions))})|$)'
        return [match[0].strip() for match in re.findall(pattern, text, re.DOTALL)]

    def parse_str_to_action_list(self, text:str, actions: list):
        remain_text = copy.deepcopy(text)
        action_list = []
        while remain_text:
            find_action = False
            for action in actions:
                if remain_text.startswith(action):
                    match = re.search(rf'({re.escape(action)}\s*(.*?))(?=\n(?:{"|".join(map(re.escape, actions))})|$)', remain_text, re.DOTALL)
                    action_list.append(match[0])
                    remain_text = remain_text[len(match[0]):].strip()
                    find_action = True
            if not find_action:
                break
        return action_list
    
    def get_observation_text(self, idx=None):
        if isinstance(self.online_interaction["observation"], dict):
            if idx:
                return self.previous_interactions["observation"][idx]["text"]
            return self.online_interaction["observation"]["text"]
        elif isinstance(self.online_interaction["observation"], str):
            if idx:
                return self.previous_interactions["observation"][idx]
            return self.online_interaction["observation"]
        
    def get_observation_image(self, idx=None):
        if isinstance(self.online_interaction["observation"], dict):
            if idx:
                return self.previous_interactions["observation"][idx]["image"]
            return self.online_interaction["observation"]["image"]
        elif isinstance(self.online_interaction["observation"], str):
            return None
        
    def get_observation_node(self, idx=None):
        if isinstance(self.online_interaction["observation"], dict):
            if idx != None:
                return self.previous_interactions["observation"][idx]["node"]
            return self.online_interaction["observation"]["node"]
        elif isinstance(self.online_interaction["observation"], str):
            return None
        
    def get_observation_node_str(self, idx=None):
        if isinstance(self.online_interaction["observation"], dict):
            if idx != None:
                return self.previous_interactions["observation"][idx]["node_str"]
            return translate_node_to_str(self.online_interaction["observation"]["node"], mode="name_only")
        elif isinstance(self.online_interaction["observation"], str):
            return None
        
    def del_observation_node(self):
        if isinstance(self.online_interaction["observation"], str):
            return
        if isinstance(self.online_interaction["observation"], dict):
            for idx in range(len(self.previous_interactions["observation"])):
                if "node" in self.previous_interactions["observation"][idx].keys() and self.previous_interactions["observation"][idx]["node"]:
                    node_str = translate_node_to_str(self.previous_interactions["observation"][idx]["node"], mode="name_only")
                    self.previous_interactions["observation"][idx]["node_str"] = node_str
                    self.previous_interactions["observation"][idx]["node"].delete_tree()
                    self.previous_interactions["observation"][idx]["node"] = None

class PlanTreeNode:
    def __init__(self, id, type, text, level, url, step):
        self.visible = True
        self.id = id
        self.type = type
        self.text = text
        self.level = level
        self.url = url
        self.step = step
        self.children = []
        self.parent = None
        self.note = []
        self.hint = []
        self.resume_reason = []
        self.steps_taken = []

    def reset(self):
        self.visible = True
        self.note = []
        self.hint = []
        self.steps_taken = []

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def search_node_by_id(self, target_id):
        if self.visible and self.id == target_id:
            return self
        for child in self.children:
            result = child.search_node_by_id(target_id)
            if result:
                return result
        return None
    
    def traverse(self, action=None, tree_buffer=[]):
        res_action = action(self)
        if res_action:
            if isinstance(res_action, list):
                tree_buffer.extend(res_action)
            else:
                tree_buffer.append(res_action)
        for child in self.children:
            child.traverse(action, tree_buffer=tree_buffer)

class QAActor(Agent):
    def __init__(self, config, objective, prompt_template):
        super().__init__(config, objective, prompt_template)
    def get_instruction(self):
        return self.prompt_template["instruction_template"]
    def get_online_input(self):
        return [("text", self.prompt_template["input_template"].replace("{current_observation}", self.get_observation_text()).replace("{objective}", self.objective))]
    def get_action(self, instruction, online_input):
        model_response = self.call_model_with_message(system_prompt=instruction, messages=self.arrange_message_for_model(online_input))
        action_elements = self.parse_elements(text=model_response, key_list=self.config.output)
        action = action_elements["response"]
        action_elements["action"] = f"note [{action}]"
        action_elements["instruction"] = instruction
        action_elements["input"] = online_input
        return model_response, action_elements
    
class PlanningActor(Agent):
    def __init__(self, config, objective, prompt_template):
        super().__init__(config, objective, prompt_template)
        self.instruction = None

    def get_planning_specifications(self):
        return "\n".join(["- " + "".join(open(os.path.join(CURRENT_DIR, "AgentOccam", "prompts", "planning_specifications", f"{p}.txt"), "r").readlines()) for p in self.config.planning_command])
    
    def get_instruction(self):
        if self.instruction:
            return self.instruction
        output_specifications = self.get_output_specifications()
        self.instruction = self.prompt_template["instruction_template"].replace("{output_specifications}", output_specifications).replace("{planning_specifications}", self.get_planning_specifications())
        return self.instruction
    
    def get_online_input(self):
        return None
    
    def get_action(self, instruction, online_input):
        model_response = self.call_model_with_message(system_prompt=instruction, messages=self.arrange_message_for_model(online_input))
        action_elements = self.parse_elements(text=model_response, key_list=self.config.output)
        action_elements["action"] = copy.deepcopy(action_elements["plan"])
        del action_elements["plan"]
        action_elements["reason"] = "N/A"
        action_elements["instruction"] = instruction
        action_elements["input"] = online_input
        return model_response, action_elements

class ReflectionActor(Agent):
    def __init__(self, config, objective, prompt_template):
        super().__init__(config, objective, prompt_template)
        self.instruction = None

    def get_planning_specifications(self):
        return "\n".join(["- " + "".join(open(os.path.join(CURRENT_DIR, "AgentOccam", "prompts", "planning_specifications", f"{p}.txt"), "r").readlines()) for p in self.config.planning_command])
    
    def get_navigation_specifications(self):
        return "\n".join(["- " + "".join(open(os.path.join(CURRENT_DIR, "AgentOccam", "prompts", "navigation_specifications", f"{n}.txt"), "r").readlines()) for n in self.config.navigation_command])
    
    def get_instruction(self):
        if self.instruction:
            return self.instruction
        output_specifications = self.get_output_specifications()
        planning_specifications = self.get_planning_specifications()
        navigation_specifications = self.get_navigation_specifications()

        # 注入技能库（如果已加载且有可用技能）
        skills_block = ""
        cfg_sk = getattr(self.config, "skills", None)

        # Debug: 打印技能库状态
        if self.config.verbose >= 1:
            has_cfg = cfg_sk is not None
            has_registry = self.skill_registry is not None
            registry_size = len(self.skill_registry) if has_registry else 0
            print(f"[DEBUG] 技能库状态: cfg_sk={has_cfg}, skill_registry={has_registry}, size={registry_size}")


        # DRIVE exposes only the singleton selections for this task-page scene.

        # ===== Dynamic DRIVE Skill Injection =====
        # Use dynamic skill injection that adapts based on skill execution status
        if cfg_sk and self.skill_registry:
            objective = getattr(self, 'objective', '')
            if objective:
                try:
                    # Get current step number
                    current_step = self.get_step() if hasattr(self, 'get_step') else 0

                    # Use dynamic skill injection which checks skill_attempted and skill_succeeded
                    dynamic_prompt = self._get_dynamic_skill_injection(current_step)

                    if dynamic_prompt:
                        # Override skills_block with dynamic prompt
                        skills_block = (
                            "\n\n" + "=" * 70 + "\n"
                            + dynamic_prompt
                            + "=" * 70 + "\n"
                        )

                        # Log skill injection status
                        skill_attempted = getattr(self, 'skill_attempted', False)
                        skill_succeeded = getattr(self, 'skill_succeeded', False)
                        if skill_attempted:
                            if skill_succeeded:
                                print(f"[Skill Status] ✅ Skill succeeded - prompting to use result")
                            else:
                                print(f"[Skill Status] ❌ Previous skill failed - retrieval refreshed")
                        else:
                            print(f"[Skill Status] ⏳ Skills available - Agent will decide when to use")

                except Exception as e:
                    print(f"⚠️  Dynamic skill injection failed: {e}")
                    if self.config.debug:
                        import traceback
                        traceback.print_exc()

        navigation_specifications_with_skills = navigation_specifications + skills_block

        instruction = self.prompt_template["instruction_template"]
        instruction = instruction.replace("{output_specifications}", output_specifications)
        instruction = instruction.replace("{planning_specifications}", planning_specifications)
        instruction = instruction.replace("{navigation_specifications}", navigation_specifications_with_skills)
        self.instruction = instruction
        return self.instruction
    
    def get_online_input(self):
        return None
    
    def get_action(self, instruction, online_input):
        model_response = self.call_model_with_message(system_prompt=instruction, messages=self.arrange_message_for_model(online_input))
        action_elements = self.parse_elements(text=model_response, key_list=self.config.output)
        action_elements["instruction"] = instruction
        action_elements["input"] = online_input
        return model_response, action_elements

IDENTITY_CLASS_MAP = {
    "QA": QAActor,
    "planning": PlanningActor,
    "reflection": ReflectionActor,
}

class Actor(Agent):
    def __init__(self, config, objective, prompt_template, plan_tree_node):
        super().__init__(config, objective, prompt_template)
        self.plan_tree_root = plan_tree_node
        self.active_node = plan_tree_node
        self.output_specifications = None
        self.planning_specifications = None
        self.navigation_specifications = None
        self.criticism_element_list = None

        self.output_play_path = os.path.join(CURRENT_DIR, f"play-{self.config.others.logname}.txt") if getattr(self.config.others, "logname", "") != "" else os.path.join(CURRENT_DIR, f"play.txt")
        self.output_trash_path = os.path.join(CURRENT_DIR, f"trash-{self.config.others.logname}.txt") if getattr(self.config.others, "logname", "") != "" else os.path.join(CURRENT_DIR, f"trash.txt")

        self.identities = []
        if hasattr(self.config, "identities"):
            i = 0
            while hasattr(self.config.identities, f"identity_{i}"):
                identity_config = getattr(self.config.identities, f"identity_{i}")
                self.identities.append(IDENTITY_CLASS_MAP[identity_config.name](identity_config, objective=objective, prompt_template=prompt_template[identity_config.name]))
                i += 1

        # Skill 失败跟踪
        self.invalid_skill_count = 0  # 无效 skill 调用计数
        self.max_invalid_skills = 3  # 最大允许失败次数

        # ===== 新增：技能使用状态跟踪 =====
        self.skill_attempted = False  # 是否已尝试过使用技能
        self.skill_succeeded = False  # 技能是否执行成功
        self.skill_execution_result = None  # 技能执行结果信息

        # Invalid action tracking - prevent infinite loops
        self.invalid_action_counter = {}  # Track repeat count of each invalid action
        self.max_same_invalid_action = 5  # Max repeats for same invalid action
        self.total_invalid_actions = 0  # Total invalid action attempts
        self.max_total_invalid_actions = 20  # Max total invalid action attempts

    def record_skill_execution(self, success: bool, result_info: str = None):
        """
        Record skill execution result

        Args:
            success: Whether the skill executed successfully
            result_info: Additional info about the skill execution result
        """
        self.skill_attempted = True
        self.skill_succeeded = success
        self.skill_execution_result = result_info

        if success:
            print("✅ Skill executed successfully; DRIVE will retrieve again for the new scene.")
            if result_info:
                print(f"   Result: {result_info[:200]}...")
        else:
            print("❌ Skill execution failed; DRIVE will demote it and retrieve again.")
            if result_info:
                print(f"   Reason: {result_info[:200]}...")

    def update_online_state(self, **online_states):
        super().update_online_state(**online_states)
        for identity in self.identities:
            identity.update_online_state(**online_states)

    def is_planning(self, action):
        for c in self.config.planning_command:
            if action.startswith(c):
                return c
        return False

    def is_navigation(self, action):
        action_without_note = re.sub(rf'(note\s*(.*?))(?=\n(?:{"|".join(map(re.escape, self.config.navigation_command))})|$)', "", action).strip()
        for c in self.config.navigation_command:
            if action_without_note.startswith(c):
                return c
        return False
    
    def is_valid_action(self, action_str):
        action = action_str.split()[0].strip() if action_str.split() else ""
        match action:
            case "click":
                match = re.search(r"click ?\[(\d+)\]", action_str)
                if not match:
                    return False
                element_id = match.group(1)
                if element_id in self.get_observation_text():
                    return True
                return False
            case "type":
                parsed = _parse_type_action_text(action_str)
                if not parsed:
                    return False
                element_id, text, enter_flag = parsed
                enter_flag = True if enter_flag == "1" else False
                if enter_flag:
                    text += "\n"
                if element_id in self.get_observation_text():
                    return True
            case "go_back":
                return True
            case "go_home":
                return True
            case "note":
                return True
            case "stop":
                return True
            case "branch":
                return True
            case "prune":
                return True
            case "goto":
                return True
            case "scroll":
                return True
            case "press":
                return True
            case "use_skill":
                # 验证 skill 调用格式
                # 格式: use_skill skill_name arg1=value1 arg2=value2 ...
                parts = action_str.split()
                if len(parts) < 2:
                    return False  # 至少需要 "use_skill skill_name"

                skill_name = parts[1]

                # 检查 skill 是否存在
                if not hasattr(self, 'skill_registry') or not self.skill_registry:
                    return False

                skill = self.skill_registry.get_skill(skill_name)
                if not skill:
                    return False
                try:
                    parsed_commands = create_id_based_actions(action_str)
                    if len(parsed_commands) != 1 or parsed_commands[0].get("action_type") != ActionTypes.USE_SKILL:
                        return False
                    if hasattr(self.skill_registry, "validate_skill_invocation"):
                        self.skill_registry.validate_skill_invocation(
                            skill_name,
                            parsed_commands[0].get("skill_args", {}),
                            url=self.online_interaction.get("url", "") or "",
                            observation=self.online_interaction.get("observation", ""),
                        )
                except Exception:
                    return False
                return True
            case _:
                # 未知操作类型
                return False

    def are_valid_actions(self, actions):
        action_list = self.parse_str_to_action_list(actions, self.config.planning_command+self.config.navigation_command+["goto"])
        if not action_list:
            return False
        for action in action_list:
            if not self.is_valid_action(action):
                return False
        return True

    def get_previous_plans(self, verbose=False):
        def action_return_visible_node(node, verbose=False):
            if node.id == self.active_node.id:
                basic = "\t" * node.level + f"[{node.id}] (Active Plan) {node.text}" if node.visible else None
            else:
                basic = "\t" * node.level + f"[{node.id}] {node.text}" if node.visible else None
            if basic and len(node.resume_reason) > 0:
                basic += f" # Was resumed to this step {len(node.resume_reason)} time(s) for:"
                for i, reason in enumerate(node.resume_reason):
                    basic += f" {i}. {reason}"
            if verbose and basic and len(node.note) > 0:
                for i, note in enumerate(node.note):
                    basic += "\n" + "\t" * node.level + f"Note {i}. {note}"
            return basic
        plan_tree_buffer = []
        parse_node_descendants(self.plan_tree_root, partial(action_return_visible_node, verbose=verbose), tree_buffer=plan_tree_buffer)
        plan_text = "\n".join(plan_tree_buffer)

        # Debug: print if there are any notes
        if verbose and any(len(node.note) > 0 for node in [self.active_node]):
            print(f"[DEBUG get_previous_plans] Active node [{self.active_node.id}] has {len(self.active_node.note)} notes")
            print(f"[DEBUG get_previous_plans] Plan text preview (first 500 chars):\n{plan_text[:500]}")

        return plan_text
    
    def get_active_plan(self):
        return f"[{self.active_node.id}] {self.active_node.text}"
    
    def get_interaction_history(self, interaction_history_config=False, mode="highlight"):
        interaction_history_config = interaction_history_config if interaction_history_config else self.config.interaction_history

        previous_observation = []
        for i in self.active_node.steps_taken:
            if self.get_observation_node_str() and self.get_observation_node_str(i) and not self.get_observation_node_str() == self.get_observation_node_str(i):
                if self.previous_interactions["observation highlight"][i] and mode == "highlight" and len(translate_node_to_str(self.previous_interactions["observation highlight"][i], mode="name_only", retained_ids=self.previous_interactions["retained element ids"][i]).split()) < 200:
                    try:
                        previous_observation.append({"text": translate_node_to_str(self.previous_interactions["observation highlight"][i], mode="name_only", retained_ids=self.previous_interactions["retained element ids"][i]), "image": self.get_observation_image(i)})
                    except:
                        print(i, self.previous_interactions["observation"][i]["text"])
                        raise ValueError("Cannot translate highlight node to text.")
                else:
                    previous_observation.append({"text": self.previous_interactions["observation summary"][i], "image": self.get_observation_image(i)})
            elif not self.get_observation_node() or mode == "full":
                if len(self.get_observation_text(i).split()) < 200:
                    previous_observation.append({"text": self.get_observation_text(i), "image": self.get_observation_image(i)})
                else:
                    previous_observation.append({"text": self.previous_interactions["observation summary"][i], "image": self.get_observation_image(i)})
            else:
                previous_observation.append({"text": "The same as the CURRENT OBSERVATION (see below CURRENT OBSERVATION section).", "image": self.get_observation_image(i)})

        previous_observation_summary = [self.previous_interactions["observation summary"][i] for i in self.active_node.steps_taken]

        def get_text(obs):
            if isinstance(obs, dict):
                return obs["text"]
            elif isinstance(obs, str):
                return obs

        def get_image(obs):
            if isinstance(obs, dict):
                return obs["image"]
            elif isinstance(obs, str):
                return obs

        if interaction_history_config.step_num == "all":
            textual_observations = [get_text(obs) for obs in previous_observation] if interaction_history_config.verbose else previous_observation_summary
            visual_observations = [get_image(obs) for obs in previous_observation]
        else:
            textual_observations = previous_observation_summary[:-interaction_history_config.step_num]
            visual_observations = [None] * len(previous_observation_summary[:-interaction_history_config.step_num])
            textual_observations += [get_text(obs) for obs in previous_observation][-interaction_history_config.step_num:] if interaction_history_config.verbose else previous_observation_summary[-interaction_history_config.step_num:]
            visual_observations += [get_image(obs) for obs in previous_observation][-interaction_history_config.step_num:]

        plans = [self.previous_interactions["plan"][i] for i in self.active_node.steps_taken]
        reasons = [self.previous_interactions["reason"][i] for i in self.active_node.steps_taken]
        actions = [self.previous_interactions["action"][i] for i in self.active_node.steps_taken]
            
        if "image" in interaction_history_config.type:
            message_list = []
            for step, (obs, vi_obs, plan, reason, action) in enumerate(zip(textual_observations, visual_observations, plans, reasons, actions)):
                message_list.append(("text", f"<step_{step}_interaction>\n"))
                if vi_obs:
                    message_list.append(("text", "VISUAL OBSERVATION:\n"))
                    message_list.append(("image", vi_obs))
                if self.active_node.id != 0:
                    message_list.append(("text", f"TEXTUAL OBSERVATION:\n{obs}\nACTIVE PLAN:\n{plan}\nREASON FOR ACTION:\n{reason}\nACTION:\n{action}\n</step_{step}_interaction>\n"))
                else:
                    message_list.append(("text", f"TEXTUAL OBSERVATION:\n{obs}\nREASON FOR ACTION:\n{reason}\nACTION:\n{action}\n</step_{step}_interaction>\n"))
            return self.prune_message_list(message_list=message_list)
        else:
            message = ""
            for step, (obs, plan, reason, action) in enumerate(zip(textual_observations, plans, reasons, actions)):
                if self.active_node.id != 0:
                    message += f"<step_{step}_interaction>\nOBSERVATION:\n{obs}\nACTIVE PLAN:\n{plan}\nREASON FOR ACTION:\n{reason}\nACTION:\n{action}\n</step_{step}_interaction>\n" # f"<step_{step}_interaction>\nOBSERVATION:\n{obs}\nACTIVE PLAN:\n{plan}\nREASON FOR ACTION:\n{reason}\nACTION:\n{action}\n</step_{step}_interaction>\n"
                else:
                    message += f"<step_{step}_interaction>\nOBSERVATION:\n{obs}\nREASON FOR ACTION:\n{reason}\nACTION:\n{action}\n</step_{step}_interaction>\n" # f"<step_{step}_interaction>\nOBSERVATION:\n{obs}\nREASON FOR ACTION:\n{reason}\nACTION:\n{action}\n</step_{step}_interaction>\n"
            return self.prune_message_list(message_list=[("text", message)])
        
    def pre_process_atomic_actions(self, atomic_action_list=["combobox"]):
        if self.get_observation_node() and "combobox" in atomic_action_list:
            # 保留技能执行结果前缀（如果有）
            import re
            skill_result_prefix = ""
            current_text = self.online_interaction["observation"]["text"]
            if isinstance(current_text, str) and "SKILL EXECUTION" in current_text:
                # 提取技能结果部分
                match = re.search(r'(={50,}[\s\S]*?SKILL EXECUTION[\s\S]*?={50,}[\s\S]*?={50,})', current_text)
                if match:
                    skill_result_prefix = match.group(1) + "\n\n"

            new_text = translate_node_to_str(self.get_observation_node(), mode="concise", hidden_roles=["menu", "combobox", "listbox"])
            # 将技能结果前缀添加到新生成的文本前面
            self.online_interaction["observation"]["text"] = skill_result_prefix + new_text

    def get_online_input(self, criticism_elements):
        input_template = self.prompt_template["input_template"]
        input_prefix, input_suffix = input_template.split("{input}")
        INPUT_TYPE_TO_CONTENT_MAP = {
            "step": self.get_step(),
            "objective": self.objective,
            "previous plans": self.get_previous_plans(verbose=True),
            "interaction history": self.get_interaction_history(),
            "current observation": self.get_observation_text(),
            "current visual observation": self.get_observation_image()
        }
        input_list = []
        for input_type in self.config.input:
            input_content = None
            if input_type == "current visual observation":
                continue
            elif input_type in INPUT_TYPE_TO_CONTENT_MAP.keys():
                input_content = INPUT_TYPE_TO_CONTENT_MAP[input_type]
            elif input_type.startswith("critic: ") and criticism_elements and input_type[len("critic: "):] in criticism_elements.keys() and criticism_elements[input_type[len("critic: "):]]:
                input_type = input_type[len("critic: "):]
                input_content = criticism_elements[input_type]
                input_type = "FROM USER: " + input_type
            if input_content and isinstance(input_content, str):
                input_list.append(("text", f"{input_type.upper()}:\n{input_content}\n"))
            elif input_content and isinstance(input_content, list):
                input_list.append(("text", f"{input_type.upper()}:\n"))
                input_list += input_content if len(input_content) > 0 else ["N/A"]

        if "image" in self.config.current_observation.type:
            input_type = "current visual observation"
            input_list.append(("text", f"{input_type.upper()}:\n"))
            input_list.append(("image", INPUT_TYPE_TO_CONTENT_MAP["current visual observation"]))

        return self.prune_message_list(message_list=[("text", input_prefix)] + input_list + [("text", input_suffix)])
    
    def get_planning_specifications(self):
        if self.planning_specifications:
            return self.planning_specifications
        self.planning_specifications = "\n".join(["- " + "".join(open(os.path.join(CURRENT_DIR, "AgentOccam", "prompts", "planning_specifications", f"{p}.txt"), "r").readlines()) for p in self.config.planning_command])
        return self.planning_specifications
    
    def get_navigation_specifications(self):
        if self.navigation_specifications:
            return self.navigation_specifications
        self.navigation_specifications = "\n".join(["- " + "".join(open(os.path.join(CURRENT_DIR, "AgentOccam", "prompts", "navigation_specifications", f"{n}.txt"), "r").readlines()) for n in self.config.navigation_command])
        return self.navigation_specifications
    
    def get_actor_instruction(self, examples=None):
        if self.config.planning_command:
            instruction = self.prompt_template["instruction_template"]["with_planning"]
        else:
            instruction = self.prompt_template["instruction_template"]["without_planning"]
        output_specifications = self.get_output_specifications()
        planning_specifications = self.get_planning_specifications()
        navigation_specifications = self.get_navigation_specifications()

        # 注入技能库（如果已加载且有可用技能）
        skills_block = ""
        cfg_sk = getattr(self.config, "skills", None)

        # Debug: 打印技能库状态
        if self.config.verbose >= 1:
            has_cfg = cfg_sk is not None
            has_registry = self.skill_registry is not None
            registry_size = len(self.skill_registry) if has_registry else 0
            print(f"[DEBUG] 技能库状态: cfg_sk={has_cfg}, skill_registry={has_registry}, size={registry_size}")


        # DRIVE exposes only the singleton selections for this task-page scene.

        # ===== Dynamic DRIVE Skill Injection =====
        # Use dynamic skill injection that adapts based on skill execution status
        if cfg_sk and self.skill_registry:
            objective = getattr(self, 'objective', '')
            if objective:
                try:
                    # Get current step number
                    current_step = self.get_step() if hasattr(self, 'get_step') else 0

                    # Use dynamic skill injection which checks skill_attempted and skill_succeeded
                    dynamic_prompt = self._get_dynamic_skill_injection(current_step)

                    if dynamic_prompt:
                        # Override skills_block with dynamic prompt
                        skills_block = (
                            "\n\n" + "=" * 70 + "\n"
                            + dynamic_prompt
                            + "=" * 70 + "\n"
                        )

                        # Log skill injection status
                        skill_attempted = getattr(self, 'skill_attempted', False)
                        skill_succeeded = getattr(self, 'skill_succeeded', False)
                        if skill_attempted:
                            if skill_succeeded:
                                print(f"[Skill Status] ✅ Skill succeeded - prompting to use result")
                            else:
                                print(f"[Skill Status] ❌ Previous skill failed - retrieval refreshed")
                        else:
                            print(f"[Skill Status] ⏳ Skills available - Agent will decide when to use")

                except Exception as e:
                    print(f"⚠️  Dynamic skill injection failed: {e}")
                    if self.config.debug:
                        import traceback
                        traceback.print_exc()

        navigation_specifications_with_skills = navigation_specifications + skills_block

        instruction = instruction.replace("{output_specifications}", output_specifications)
        instruction = instruction.replace("{planning_specifications}", planning_specifications)
        instruction = instruction.replace("{navigation_specifications}", navigation_specifications_with_skills)

        example_source = examples if examples is not None else self.prompt_template.get("examples", [])
        if len(example_source) > 0:
            instruction += f"\n\n## Here are a few examples:"
            for i, example in enumerate(example_source):
                example_input = example["input"]
                example_output = example["output"]
                if "example_template" in self.prompt_template.keys():
                    instruction += "\n\n"
                    instruction += self.prompt_template.get("example_template", "| Example {i}\n### Input:\n{example_input}\n### Response: Let's think step by step.\n{example_response}").replace("{i}", i).replace("{example_input}", example_input).replace("{example_output}", example_output)
                else:
                    instruction += f"\n\n| Example {i}\n\n### Input:\n{example_input}\n\n### Response: Let's think step by step.\n{example_output}"
        
        if self.get_step() == self.config.others.max_steps - 1:
            instruction += f"\n\nWARNING: You have a {self.config.others.max_steps}-step budget, and this would be your FINAL STEP. Wrap up your observations and return your answer with `stop [answer]` to maximize the reward."
        # else:
        #     instruction += f"\n\nWARNING: You have a {self.config.others.max_steps}-step budget, and there are {self.config.others.max_steps-self.get_step()} remaining attempts."

        return instruction
    
    def verbose(self, instruction, online_input, model_response_list, action_element_list):
        action_element_keys = [k for k in self.config.play if k in action_element_list[0].keys()]
        other_play_keys = [k for k in self.config.play if k not in action_element_list[0].keys()]

        VERBOSE_TO_CONTENT_MAP = {
            "step": self.get_step(),
            "objective": self.objective,
            "previous plans": self.get_previous_plans(verbose=True),
            "url": self.online_interaction["url"],
            "observation": self.get_observation_text(),
            "response": "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n".join([f"|\tAgent {i}:\n{model_response}" for i, model_response in enumerate(model_response_list[:self.config.number])]) if self.config.number > 1 else model_response_list[0],
            "instruction": instruction,
            "online input": "\n".join([i[1] for i in online_input if i[0]=="text"]),
            "alter ego response": "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n".join(["|\tAgent {}:\n{}".format(identity.config.name, response) for identity, response in zip(self.identities, model_response_list[self.config.number:])])
        }

        if self.config.others.verbose > 0 and self.config.verbose > 0:
            with open(self.output_trash_path, "a") as af:
                af.write("-"*32+"ACTOR"+"-"*32+"\n")
            for t in self.config.trash:
                content = VERBOSE_TO_CONTENT_MAP.get(t, "")
                with open(self.output_trash_path, "a") as af:
                    af.write(f"{t.upper()}:\n{content}\n\n")
            with open(self.output_play_path, "w") as _:
                pass
            for p in other_play_keys:
                content = VERBOSE_TO_CONTENT_MAP.get(p, "")
                with open(self.output_play_path, "a") as af:
                    af.write(f"{p.upper()}:\n{content}\n\n")
            for i, action_elements in enumerate(action_element_list):
                if len(action_element_list) > 1:
                    with open(self.output_play_path, "a") as af:
                        af.write("-"*32+f"AGENT {i}"+"-"*32+"\n")
                for action_element_key in action_element_keys:
                    content = action_elements.get(action_element_key, "N/A")
                    with open(self.output_play_path, "a") as af:
                        af.write(f"{action_element_key.upper()}:\n{content}\n\n")
    
    def parse_plan(self, planning):
        planning_type = self.is_planning(action=planning)
        match = re.search(
            rf"{planning_type} ?\[(\d+)\] ?\[(.+)\]", planning, re.DOTALL
        )
        if not match:
            raise ValueError("Invalid planning command.")
        node_id, planning_content = (
            int(match.group(1)),
            match.group(2)
        )
        return planning_type, node_id, planning_content
    
    def prune_planning(self, node:PlanTreeNode, planning_content):
        def set_invisible(node:PlanTreeNode):
            node.visible = False
        def return_steps_taken(node:PlanTreeNode):
            return [node.step] + node.steps_taken
        after_node = False
        if node.id > 0:
            for child in node.parent.children:
                if not after_node and child != node:
                    continue
                elif child == node:
                    after_node = True
                    continue
                child.visible = False
        node.traverse(set_invisible)
        node.reset()
        steps_taken = []
        node.traverse(action=return_steps_taken, tree_buffer=steps_taken)
        node.steps_taken = sorted(list(set(steps_taken)), reverse=False)
        node.resume_reason.append(planning_content)
        navigation = f"goto [{node.url}] [1]"
        self.active_node = node
        return navigation
    
    def branch_planning(self, node, planning_content):
        new_node = PlanTreeNode(id=self.active_node.id+1, type=type, text=planning_content, level=node.level+1, url=self.online_interaction["url"], step=self.get_step())
        self.active_node = new_node
        node.add_child(new_node)
    
    def planning(self, action):
        if action and self.is_planning(action):
            try:
                planning_type, node_id, planning_content = self.parse_plan(planning=action)
                node = self.plan_tree_root.search_node_by_id(node_id)
                if not node:
                    raise ValueError(f"Invalid node id {node_id}: {action}.")
                if planning_type == "prune":
                    navigation_action = self.prune_planning(node=node, planning_content=planning_content)
                    return navigation_action
                elif planning_type == "branch":
                    self.branch_planning(node=node, planning_content=planning_content)
                else:
                    raise ValueError(f"Invalid planning operation {planning_type}: {action}.")
            except Exception as e:
                print("Invalid plan node:", str(e))
                flaw_node = self.active_node
                flaw_node.note.append(f"You previously generate plan \"{action}\", which has INVALID syntax. User planning command like `branch [parent_plan_id] [new_subplan_intent]` or `prune [resume_plan_id] [reason]`.")
        else:
            self.active_node.steps_taken.append(self.get_step())
        return None
    
    def go_home(self, action):
        if "go_home" in action:
            return f"goto [{HOMEPAGE_URL}] [1]"
        return None
    
    def parse_action(self, action_str):
        try:
            DOM_root_node = self.get_observation_node()
            action_str = action_str.strip()
            action = (
                action_str.split("[")[0].strip()
                if "[" in action_str
                else action_str.split()[0].strip()
            )
            match action:
                case "click":
                    match = re.search(r"click ?\[(\d+)\]", action_str)
                    if not match:
                        raise ValueError(f"Invalid click action {action_str}")
                    element_id = match.group(1)
                    node = DOM_root_node.search_node_by_id(element_id)
                    return f"click [{element_id}] ({node.role} {node.name})"
                case "hover":
                    match = re.search(r"hover ?\[(\d+)\]", action_str)
                    if not match:
                        raise ValueError(f"Invalid hover action {action_str}")
                    element_id = match.group(1)
                    node = DOM_root_node.search_node_by_id(element_id)
                    return f"hover [{element_id}] ({node.role} {node.name})"
                case "type":
                    parsed = _parse_type_action_text(action_str)
                    if not parsed:
                        raise ValueError(f"Invalid type action {action_str}")
                    element_id, text, enter_flag = parsed
                    enter_flag = True if enter_flag == "1" else False
                    if enter_flag:
                        text += "\n"
                    node = DOM_root_node.search_node_by_id(element_id)
                    return action + f" ({node.name})"
                case "scroll":
                    return action_str
                case "goto":
                    return action
                case "new_tab":
                    return action
                case "go_back":
                    return action
                case "go_forward":
                    return action
                case "stop":
                    return action

            return False
        except:
            return False
    
    def parse_actions_to_element_ids(self, actions):
        action_str_list = []
        for a in self.config.navigation_command:
            action_str_list += self.parse_stipulated_action_list(text=actions, action=a, actions=self.config.planning_command+self.config.navigation_command+["goto"])
        retained_element_ids = []
        for action_str in action_str_list:
            try:
                action_str = action_str.strip()
                action = (
                    action_str.split("[")[0].strip()
                    if "[" in action_str
                    else action_str.split()[0].strip()
                )
                match action:
                    case "click":
                        match = re.search(r"click ?\[(\d+)\]", action_str)
                        if not match:
                            raise ValueError(f"Invalid click action {action_str}")
                        element_id = match.group(1)
                        element_id = int(element_id)
                        retained_element_ids.append(element_id)
                    case "hover":
                        match = re.search(r"hover ?\[(\d+)\]", action_str)
                        if not match:
                            raise ValueError(f"Invalid hover action {action_str}")
                        element_id = match.group(1)
                        element_id = int(element_id)
                        retained_element_ids.append(element_id)
                    case "type":
                        parsed = _parse_type_action_text(action_str)
                        if not parsed:
                            raise ValueError(f"Invalid type action {action_str}")
                        element_id, text, enter_flag = parsed
                        element_id = int(element_id)
                        retained_element_ids.append(element_id)
                    case "scroll":
                        pass
                    case "goto":
                        pass
                    case "new_tab":
                        pass
                    case "go_back":
                        pass
                    case "go_forward":
                        pass
                    case "stop":
                        pass
                    case "note":
                        pass

                return retained_element_ids
            except:
                continue

        return retained_element_ids
    
    def take_note(self, action, note_as_action=True):
        if action and "note [" in action:
            none_note_action_list = []
            action_list = self.parse_str_to_action_list(action, actions=self.config.planning_command+self.config.navigation_command+["goto"])
            for a in action_list:
                if "note [" in a:
                    note = re.search(r"note ?\[?(.+)", a, re.DOTALL).group(1)
                    if note.endswith("]"):
                        note = note[:-1]
                    self.active_node.note.append(f"STEP {self.get_step()}: {note}")
                    self.note_buffer = note
                else:
                    none_note_action_list.append(a)
            if note_as_action:
                return action
            return "\n".join(none_note_action_list)
        # action_note = self.parse_action(action)
        # if action_note:
        #     self.active_node.note.append(f"STEP {self.get_step()} ACTION: {action_note}")
        return action
        
    def get_observation_highlight(self, action_elements:dict):
        action_elements["observation highlight idxs"] = copy.deepcopy(action_elements.get("observation highlight", ""))
        DOM_root_node = self.get_observation_node()
        if not DOM_root_node:
            action_elements["observation highlight"] = None
            return
        observation_highlight_idxs = [int(idx.strip()) for idx in action_elements.get("observation highlight", "").split(",") if idx.strip().isdigit()]
        if observation_highlight_idxs:
            parse_node_descendants(node=DOM_root_node, action=action_set_invisible)
            for idx in observation_highlight_idxs:
                try:
                    node = DOM_root_node.search_node_by_id(idx)
                    parse_node_descendants(node=node, action=action_set_visible)
                    parse_node_ancestors(node=node, action=action_set_visible)
                    parse_node_siblings(node=node, action=action_set_visible_if_with_name)
                except:
                    pass
        try: 
            assert DOM_root_node.get_visible_node_number() < 30 and construct_new_DOM_with_visible_nodes(DOM_root=DOM_root_node)
            action_elements["observation highlight"] = construct_new_DOM_with_visible_nodes(DOM_root=DOM_root_node)
            parse_node_descendants(node=DOM_root_node, action=action_set_visible)
        except:
            parse_node_descendants(node=DOM_root_node, action=action_set_visible)
            action_elements["observation highlight"] = None

        action_elements["retained element ids"] = self.parse_actions_to_element_ids(action_elements["action"])

    def parse_action_from_action_candidates(self, action_elements):
        if "action" in action_elements.keys():
            return action_elements
        assert any("action candidates" in k for k in action_elements.keys())
        action_candidates_key = [k for k in action_elements.keys() if "action candidates" in k][0]
        def parse_reasons_and_actions(input_string):
            pattern = r'- reason: \[(.*?)\]\s*(?:- action: \[(.*?)\])?\s*(?:\n|\Z)'

            matches = re.findall(pattern, input_string, re.DOTALL)

            parsed_data = []
            for match in matches:
                reason = match[0].strip()
                action = match[1].strip()
                if reason and action:
                    parsed_data.append({'reason': reason, 'action': action})

            return parsed_data
        action_elements[action_candidates_key] = parse_reasons_and_actions(action_elements[action_candidates_key])
        return action_elements

    def handle_invalid_action(self, invalid_action_str):
        """
        处理无效动作，跟踪重复次数并决定是否需要强制终止

        Returns:
            (should_break, warning_message):
                - should_break: 是否应该强制跳出循环
                - warning_message: 要添加到提示中的警告信息
        """
        # 更新总计数
        self.total_invalid_actions += 1

        # 更新特定动作的计数
        if invalid_action_str not in self.invalid_action_counter:
            self.invalid_action_counter[invalid_action_str] = 0
        self.invalid_action_counter[invalid_action_str] += 1

        repeat_count = self.invalid_action_counter[invalid_action_str]

        # 生成警告信息
        warning_message = f"\nGenerating the command `{invalid_action_str}` will be severely punished! Don't generate invalid actions! We don't have that element id in the current observation!"

        # 检查是否超过重复阈值
        if repeat_count >= self.max_same_invalid_action:
            print("="*70)
            print(f"⚠️  CRITICAL: Same invalid action repeated {repeat_count} times!")
            print(f"⚠️  Action: {invalid_action_str}")
            print(f"⚠️  Forcing agent to try a different approach...")
            print("="*70)
            warning_message = (
                f"\n\n❌ CRITICAL ERROR: You have tried the SAME invalid action `{invalid_action_str}` {repeat_count} times!\n"
                f"This element ID does NOT exist in the current observation.\n"
                f"You MUST try a COMPLETELY DIFFERENT approach:\n"
                f"- Look for OTHER element IDs in the observation\n"
                f"- Try a different navigation strategy (go_back, search, click on different elements)\n"
                f"- Re-read the observation carefully for available actions\n"
                f"DO NOT repeat the same action again!"
            )
            return True, warning_message

        # 检查总无效动作次数
        if self.total_invalid_actions >= self.max_total_invalid_actions:
            print("="*70)
            print(f"⚠️  CRITICAL: Total invalid actions ({self.total_invalid_actions}) exceeded limit!")
            print(f"⚠️  Forcing agent to use fallback strategy...")
            print("="*70)
            warning_message = (
                f"\n\n❌ You have made {self.total_invalid_actions} invalid action attempts.\n"
                f"Please be more careful and only use element IDs that exist in the observation!"
            )
            return True, warning_message

        # 根据重复次数增强警告
        if repeat_count >= 3:
            warning_message += (
                f"\n\n⚠️  WARNING: You have tried `{invalid_action_str}` {repeat_count} times already!\n"
                f"This element does NOT exist. Try something DIFFERENT!"
            )

        return False, warning_message

    def predict_action(self, criticism_elements):
        if self.config.debug > 1:
            action_elements = {k: "" for k in self.config.output}
            human_input = input("ACTION: ")
            action_elements["action"] = human_input
            return [action_elements]
        
        self.pre_process_atomic_actions()
        instruction = self.get_actor_instruction()
        online_input = self.get_online_input(criticism_elements=criticism_elements)
        model_response_list = []
        action_element_list = []
        for _ in range(self.config.number):
            get_valid_actions = False
            repetitive_note = False
            invalid_actions = False
            warning_message = ""  # 初始化警告消息
            while not get_valid_actions:
                if repetitive_note:
                    model_response = self.call_model_with_message(system_prompt=instruction+"\nGenerating the command `note [{}]` will be severely punished! Don't generate repetitive notes!".format(getattr(self, "note_buffer", "")), messages=self.arrange_message_for_model(online_input))
                elif invalid_actions:
                    # 使用handle_invalid_action返回的警告消息
                    model_response = self.call_model_with_message(system_prompt=instruction+warning_message, messages=self.arrange_message_for_model(online_input))
                else:
                    model_response = self.call_model_with_message(system_prompt=instruction, messages=self.arrange_message_for_model(online_input))
                action_elements = self.parse_elements(text=model_response, key_list=self.config.output)
                action_elements = self.parse_action_from_action_candidates(action_elements=action_elements)
                assert not ("action" in action_elements.keys() and any("action candidates" in k for k in action_elements.keys()))
                if "action" in action_elements.keys():
                    if self.are_valid_actions(action_elements["action"]):
                        note_buffer = getattr(self, "note_buffer", "")
                        if note_buffer and f"note [{note_buffer}" in action_elements["action"]:
                            print(f"Repetitive note: {note_buffer}")
                            repetitive_note = True
                            continue
                        get_valid_actions = True
                        action_elements["input"] = online_input
                        model_response_list.append(model_response)
                        action_element_list.append(action_elements)
                    else:
                        invalid_action_str = action_elements["action"]
                        print(f"Invalid actions: {invalid_action_str}")

                        # 检测无效的 skill 调用
                        if invalid_action_str.strip().startswith("use_skill"):
                            self.invalid_skill_count += 1
                            print(f"⚠️  无效 skill 调用次数: {self.invalid_skill_count}/{self.max_invalid_skills}")

                            if self.invalid_skill_count >= self.max_invalid_skills:
                                self.invalid_skill_count = 0
                                print("当前技能调用不适用；本步回退到基本操作，下一步重新检索。")

                        # 使用新的无效动作处理方法
                        should_break, warning_msg = self.handle_invalid_action(invalid_action_str)
                        warning_message = warning_msg  # 保存警告消息用于下次迭代
                        if should_break:
                            # 强制跳出循环，使用fallback动作
                            print(f"⚠️  Forcing fallback: using 'go_back' as safe action")
                            action_elements["action"] = "go_back"
                            action_elements["input"] = online_input
                            model_response_list.append(model_response)
                            action_element_list.append(action_elements)
                            break

                        invalid_actions = True
                        invalid_action_str = invalid_action_str  # 保留用于下次提示
                elif any("action candidates" in k for k in action_elements.keys()):
                    action_candidates_key = [k for k in action_elements.keys() if "action candidates" in k][0]
                    if isinstance(action_elements[action_candidates_key], str):
                        continue
                    filtered_action_candidates = []
                    note_buffer = getattr(self, "note_buffer", "")
                    for action_reason_pair in action_elements[action_candidates_key]:
                        action = action_reason_pair["action"]
                        reason = action_reason_pair["reason"]
                        if self.are_valid_actions(action):
                            if note_buffer and f"note [{note_buffer}" in action:
                                print(f"Repetitive note: {note_buffer}")
                                repetitive_note = True
                                continue
                            filtered_action_candidates.append({'reason': reason, 'action': action})
                        else:
                            invalid_action_str = action
                            print(f"Invalid actions: {invalid_action_str}")

                            # 检测无效的 skill 调用
                            if invalid_action_str.strip().startswith("use_skill"):
                                self.invalid_skill_count += 1
                                print(f"⚠️  无效 skill 调用次数: {self.invalid_skill_count}/{self.max_invalid_skills}")

                                if self.invalid_skill_count >= self.max_invalid_skills:
                                    self.invalid_skill_count = 0
                                    print("当前技能调用不适用；本步回退到基本操作，下一步重新检索。")

                            # 使用新的无效动作处理方法跟踪重复
                            should_break, warning_message = self.handle_invalid_action(invalid_action_str)
                            invalid_actions = True

                    if filtered_action_candidates:
                        action_elements[action_candidates_key] = filtered_action_candidates
                        get_valid_actions = True
                        action_elements["input"] = online_input
                        model_response_list.append(model_response)
                        action_element_list.append(action_elements)
                else:
                    raise NotImplementedError("You have to generate either action or action candidates.")
        # if self.config.number != 1:
        if True:
            for identity in self.identities:
                identity_instruction = identity.get_instruction() if identity.get_instruction() else instruction
                identity_online_input = identity.get_online_input() if identity.get_online_input() else online_input
                get_valid_actions = False
                invalid_actions = False
                warning_message = ""  # 初始化警告消息
                while not get_valid_actions:
                    if invalid_actions:
                        # 使用handle_invalid_action返回的警告消息
                        model_response, action_elements = identity.get_action(identity_instruction+warning_message, identity_online_input)
                    else:
                        model_response, action_elements = identity.get_action(identity_instruction, identity_online_input)      
                    if self.are_valid_actions(action_elements["action"]):
                        get_valid_actions = True
                        model_response_list.append(model_response)
                        action_element_list.append(action_elements)
                    else:
                        invalid_action_str = action_elements["action"]
                        print(f"Invalid actions: {invalid_action_str}")

                        # 检测无效的 skill 调用
                        if invalid_action_str.strip().startswith("use_skill"):
                            self.invalid_skill_count += 1
                            print(f"⚠️  无效 skill 调用次数: {self.invalid_skill_count}/{self.max_invalid_skills}")

                            if self.invalid_skill_count >= self.max_invalid_skills:
                                self.invalid_skill_count = 0
                                print("当前技能调用不适用；本步回退到基本操作，下一步重新检索。")

                        # 使用新的无效动作处理方法
                        should_break, warning_msg = self.handle_invalid_action(invalid_action_str)
                        warning_message = warning_msg  # 保存警告消息用于下次迭代
                        if should_break:
                            # 强制跳出循环，使用fallback动作
                            print(f"⚠️  Forcing fallback for identity: using 'go_back' as safe action")
                            action_elements["action"] = "go_back"
                            model_response_list.append(model_response)
                            action_element_list.append(action_elements)
                            break

                        invalid_actions = True

        self.verbose(instruction=instruction, online_input=online_input, model_response_list=model_response_list, action_element_list=action_element_list)

        if self.config.others.debug or self.config.debug:
            for i in range(len(action_element_list)):
                human_input = input(f"ACTION {i}: ")
                if human_input != "":
                    action_element_list[i]["action"] = human_input

        return action_element_list
    
    def finalize_action(self, action_elements):
        self.get_observation_highlight(action_elements=action_elements)
        action = action_elements["action"]
        navigation_action = self.planning(action=action)
        if navigation_action:
            action_elements["navigation action"] = navigation_action
        action = self.take_note(action)
        action_elements["action"] = action
        navigation_action = self.go_home(action=action)
        if navigation_action:
            action_elements["navigation action"] = navigation_action
        return action_elements

class Critic(Agent):
    def __init__(self, config, objective, prompt_template):
        super().__init__(config, objective, prompt_template)
        self.instruction = None
        self.actor_basic_info_dict = None

        self.output_play_path = os.path.join(CURRENT_DIR, f"play-{self.config.others.logname}.txt") if getattr(self.config.others, "logname", "") != "" else os.path.join(CURRENT_DIR, f"play.txt")
        self.output_trash_path = os.path.join(CURRENT_DIR, f"trash-{self.config.others.logname}.txt") if getattr(self.config.others, "logname", "") != "" else os.path.join(CURRENT_DIR, f"trash.txt")

    def verbose(self, instruction, online_input, model_response):
        VERBOSE_TO_CONTENT_MAP = {
            "url": self.online_interaction["url"],
            "objective": self.objective,
            "instruction": instruction,
            "online input": "\n".join([i[1] for i in online_input if i[0]=="text"]),
            "response": model_response
        }
        if self.config.others.verbose > 0 and self.config.verbose > 0:
            with open(self.output_trash_path, "a") as af:
                af.write("-"*32+"CRITIC"+"-"*32+"\n")
            for t in self.config.trash:
                content = VERBOSE_TO_CONTENT_MAP[t]
                with open(self.output_trash_path, "a") as af:
                    af.write(f"{t.upper()}:\n{content}\n\n")

    def update_actor_basic_info(self, **actor_basic_info_dict):
        self.actor_basic_info_dict = actor_basic_info_dict

    def get_output_specifications(self):
        output_specification_filepath_list = []
        for o in self.config.output:
            if os.path.exists(os.path.join(CURRENT_DIR, "AgentOccam", "prompts", "output_specifications", "{}_{}.txt".format(o.replace(" ", "_"), self.config.character))):
                output_specification_filepath_list.append(os.path.join(CURRENT_DIR, "AgentOccam", "prompts", "output_specifications", "{}_{}.txt".format(o.replace(" ", "_"), self.config.character)))
            else:
                output_specification_filepath_list.append(os.path.join(CURRENT_DIR, "AgentOccam", "prompts", "output_specifications", "{}.txt".format(o.replace(" ", "_"))))
        output_specifications = "\n".join([f"{o.upper()}:\n" + "".join(open(filepath, "r").readlines()) for o, filepath in zip(self.config.output, output_specification_filepath_list)])
        return output_specifications

    def get_critic_instruction(self):
        if self.instruction:
            return self.instruction
        instruction = self.prompt_template["instruction_template"]
        output_specifications = self.get_output_specifications()
        instruction = instruction.replace("{output_specifications}", output_specifications)
        instruction = instruction.replace("{planning_specifications}", self.actor_basic_info_dict["planning_specifications"])
        instruction = instruction.replace("{navigation_specifications}", self.actor_basic_info_dict["navigation_specifications"])
        self.instruction = instruction
        return self.instruction
    
    def get_online_input(self):
        input_template = self.prompt_template["input_template"]
        input_prefix, input_suffix = input_template.split("{input}")
        # ["objective", "previous plans", "interaction history", "step", "current observation"]
        INPUT_TYPE_TO_CONTENT_MAP = {
            "step": self.actor_basic_info_dict["step"],
            "objective": self.objective,
            "previous plans": self.actor_basic_info_dict["previous_plans"],
            "interaction history": self.actor_basic_info_dict["interaction_history"],
            "current observation": self.get_observation_text(),
            "current visual observation": self.get_observation_image()
        }
        input_list = []
        for input_type in self.config.input:
            input_content = None
            if input_type == "current visual observation":
                continue
            elif input_type in INPUT_TYPE_TO_CONTENT_MAP.keys():
                input_content = INPUT_TYPE_TO_CONTENT_MAP[input_type]
            if input_content and isinstance(input_content, str):
                input_list.append(("text", f"{input_type.upper()}:\n{input_content}\n"))
            elif input_content and isinstance(input_content, list):
                input_list.append(("text", f"{input_type.upper()}:\n"))
                input_list += input_content if len(input_content) > 0 else ["N/A"]

        if "image" in self.config.current_observation.type:
            input_type = "current visual observation"
            input_list.append(("text", f"{input_type.upper()}:\n"))
            input_list.append(("image", INPUT_TYPE_TO_CONTENT_MAP["current visual observation"]))

        return self.prune_message_list(message_list=[("text", input_prefix)] + input_list + [("text", input_suffix)])

    def get_criticism_elements(self):
        if not self.config.mode:
            return {}
        if self.config.debug > 1:
            criticism_elements = {k: random.choice(["I don't think the task is finished. Don't issue identical actions like taking the same notes. It's annoying. Continue.", "You have make a reasoning mistake. Continue.", "You have missed important details on this page. Continue.", "You don't follow the task requirements. Continue.", "The task assigner might just want to challenge you to answer no and there might be no answer for this brain teaser question. Who knows?", "You should break down the task by using the planning commands.", "You have not gone over all the relevant pages. Continue."]) for k in self.config.output}
            # criticism_elements = {k: input(f"{k.upper()}: ") for k in self.config.output}
            return criticism_elements
        
        instruction = self.get_critic_instruction()
        online_input = self.get_online_input()
        model_response = self.call_model_with_message(system_prompt=instruction, messages=self.arrange_message_for_model(online_input))
        self.verbose(instruction=instruction, online_input=online_input, model_response=model_response)

        criticism_elements = self.parse_elements(text=model_response, key_list=self.config.output) # key_list=self.config.output)
        criticism_elements["input"] = online_input

        if self.config.others.debug or self.config.debug:
            for k in self.config.output:
                human_input = input(f"{k.upper()}: ")
                if not human_input == "":
                    criticism_elements[k] = human_input
        
        return criticism_elements

class Judge(Agent):
    def __init__(self, config, objective, prompt_template):
        super().__init__(config, objective, prompt_template)
        self.instruction = None
        self.actor_basic_info_dict = None

        self.output_play_path = os.path.join(CURRENT_DIR, f"play-{self.config.others.logname}.txt") if getattr(self.config.others, "logname", "") != "" else os.path.join(CURRENT_DIR, f"play.txt")
        self.output_trash_path = os.path.join(CURRENT_DIR, f"trash-{self.config.others.logname}.txt") if getattr(self.config.others, "logname", "") != "" else os.path.join(CURRENT_DIR, f"trash.txt")
    
    def update_actor_basic_info(self, **actor_basic_info_dict):
        self.actor_basic_info_dict = actor_basic_info_dict

    def get_judge_instruction(self):
        if self.instruction:
            return self.instruction
        instruction = self.prompt_template["instruction_template"]
        output_specifications = self.get_output_specifications()
        instruction = instruction.replace("{output_specifications}", output_specifications)
        instruction = instruction.replace("{planning_specifications}", self.actor_basic_info_dict["planning_specifications"])
        instruction = instruction.replace("{navigation_specifications}", self.actor_basic_info_dict["navigation_specifications"])
        self.instruction = instruction
        return self.instruction
    
    def get_online_input(self, action_element_list):
        input_template = self.prompt_template["input_template"]
        input_prefix, input_suffix = input_template.split("{input}")
        INPUT_TYPE_TO_CONTENT_MAP = {
            "step": self.actor_basic_info_dict["step"],
            "objective": self.objective,
            "previous plans": self.actor_basic_info_dict["previous_plans"],
            "interaction history": self.actor_basic_info_dict["interaction_history"],
            "current observation": self.get_observation_text(),
            "current visual observation": self.get_observation_image(),
            "action choices": "\n\n".join(["|\taction [{}]:\n{}\n|\treason for action [{}]:\n{}".format(i, action_element["action"], i, action_element.get("reason", "N/A")) for i, action_element in enumerate(action_element_list)])
        }
        input_list = []
        for input_type in self.config.input:
            input_content = None
            if input_type == "current visual observation":
                continue
            elif input_type in INPUT_TYPE_TO_CONTENT_MAP.keys():
                input_content = INPUT_TYPE_TO_CONTENT_MAP[input_type]
            if input_content and isinstance(input_content, str):
                input_list.append(("text", f"{input_type.upper()}:\n{input_content}\n"))
            elif input_content and isinstance(input_content, list):
                input_list.append(("text", f"{input_type.upper()}:\n"))
                input_list += input_content if len(input_content) > 0 else ["N/A"]

        if "image" in self.config.current_observation.type:
            input_type = "current visual observation"
            input_list.append(("text", f"{input_type.upper()}:\n"))
            input_list.append(("image", INPUT_TYPE_TO_CONTENT_MAP["current visual observation"]))

        return self.prune_message_list(message_list=[("text", input_prefix)] + input_list + [("text", input_suffix)])
    
    def verbose(self, instruction, online_input, model_response):
        VERBOSE_TO_CONTENT_MAP = {
            "url": self.online_interaction["url"],
            "objective": self.objective,
            "instruction": instruction,
            "online input": "\n".join([i[1] for i in online_input if i[0]=="text"]),
            "response": model_response
        }
        if self.config.others.verbose > 0 and self.config.verbose > 0:
            with open(self.output_trash_path, "a") as af:
                af.write("-"*32+"JUDGE"+"-"*32+"\n")
            for t in self.config.trash:
                content = VERBOSE_TO_CONTENT_MAP[t]
                with open(self.output_trash_path, "a") as af:
                    af.write(f"{t.upper()}:\n{content}\n\n")

    def flatten_action_element_list(self, action_element_list):
        new_action_element_list = []
        for action_element in action_element_list:
            if any("action candidates" in k for k in action_element.keys()):
                action_candidates_key = [k for k in action_element.keys() if "action candidates" in k][0]
                new_action_element = copy.deepcopy(action_element)
                for action_reason_pair in action_element[action_candidates_key]:
                    new_action_element["action"] = action_reason_pair["action"]
                    new_action_element["reason"] = action_reason_pair["reason"]
                    new_action_element_list.append(copy.deepcopy(new_action_element))
            else:
                new_action_element_list.append(action_element)
        random.shuffle(new_action_element_list)

        return new_action_element_list
    
    def judge(self, action_element_list):
        action_element_list = self.flatten_action_element_list(action_element_list)
        if not self.config.mode or self.config.debug > 1:
            return action_element_list[0], {}
        if all(action_elements["action"]==action_element_list[0]["action"] for action_elements in action_element_list):
            return action_element_list[0], {}
        
        def deduplicate_action_element_list_strict(lst): # deduplicate, remove action_elements with only note or stop command
            seen = set()
            note_list = []
            stop_list = []
            deduplicated_list = []
    
            for i, item in enumerate(lst):
                item = copy.deepcopy(item)
                action_list = self.parse_str_to_action_list(item["action"], self.actor_basic_info_dict["planning_command"]+self.actor_basic_info_dict["navigation_command"])
                note_list.append([])
                none_note_stop_action_list = []
                for a in action_list:
                    if a.startswith("stop ["):
                        stop_list.append((a, i))
                    elif a.startswith("note ["):
                        note_list[-1].append(a)
                    else:
                        none_note_stop_action_list.append(a)
                item["action"] = "\n".join(none_note_stop_action_list)
                if item["action"] and item["action"] not in seen:
                    seen.add(item["action"])
                    deduplicated_list.append(item)
            note_list = [("\n".join(notes), i) for i, notes in enumerate(note_list)]
            return note_list, stop_list, deduplicated_list
          
        def deduplicate_action_element_list(lst): # deduplicate, remove action_elements with only note or stop command
            seen = set()
            deduplicated_list = []
    
            for item in lst:
                item = copy.deepcopy(item)
                if item["action"] and item["action"] not in seen:
                    seen.add(item["action"])
                    deduplicated_list.append(item)
            return deduplicated_list

        if hasattr(self.config, "strict") and self.config.strict:
            note_list, stop_list, deduplicated_action_element_list = deduplicate_action_element_list_strict(action_element_list)
            if len(stop_list) >= 0.6 * len(action_element_list):
                stop_action_choice = max([s[0] for s in stop_list], key=len)
                stop_action_id = [s[1] for s in stop_list if s[0]==stop_action_choice][0]
                return action_element_list[stop_action_id], {}
            if not deduplicated_action_element_list:
                note_action_choice = max([n[0] for n in note_list], key=len)
                note_action_id = [n[1] for n in note_list if n[0]==note_action_choice][0]
                action_elements = action_element_list[note_action_id]
                action_elements["action"] = note_action_choice
                return action_elements, {}
            elif len(deduplicated_action_element_list) == 1:
                action_elements = deduplicated_action_element_list[0]
                note_action_choice = max([n[0] for n in note_list], key=len)
                action_elements["action"] = note_action_choice + "\n" + action_elements["action"]
                return action_elements, {}
        else:
            deduplicated_action_element_list = deduplicate_action_element_list(action_element_list)
        
        instruction = self.get_judge_instruction()
        online_input = self.get_online_input(deduplicated_action_element_list)
        model_response = self.call_model_with_message(system_prompt=instruction, messages=self.arrange_message_for_model(online_input))
        self.verbose(instruction=instruction, online_input=online_input, model_response=model_response)

        judgement_elements = self.parse_elements(text=model_response, key_list=self.config.output) # key_list=self.config.output)
        judgement_elements["input"] = online_input

        if self.config.others.debug or self.config.debug:
            for k in self.config.output:
                human_input = input(f"{k.upper()}: ")
                if not human_input == "":
                    judgement_elements[k] = human_input

        try:
            action_selection = int(re.search(r'\d+', judgement_elements["action selection"]).group())
            selected_action_elements = deduplicated_action_element_list[action_selection]
            if hasattr(self.config, "strict") and self.config.strict:
                note_action_choice = max([n[0] for n in note_list], key=len)
                if note_action_choice:
                    selected_action_elements["action"] = note_action_choice + "\n" + selected_action_elements["action"]
            return selected_action_elements, judgement_elements
        except:
            return action_element_list[0], judgement_elements

class AgentOccam:
    def __init__(self,
                 config = None,
                 prompt_dict: Dict = None,
                 ):
        self.config = config
        self.prompt_dict = {} if prompt_dict is None else prompt_dict

        self.objective = None
        self.online_observation = None
        self.online_url = None
        self.actor = None
        self.critic = None

        self.trajectory = []

    def get_refined_objective(self):
        model_response = call_claude(self.root_prompt_template["objective_rephrasing_query"].replace("{objective}", self.objective))
        objective_match = re.search(r'REFINED OBJECTIVE:\s*(.*?)\s*(?=\n[A-Z]|$)', model_response, re.DOTALL) 
        self.objective_refined = objective_match.group(1) if objective_match else None
        
    def get_observation_text(self):
        if isinstance(self.online_observation, dict):
            return self.online_observation["text"]
        else:
            return self.online_observation
    
    def init_actor(self):
        self.config.actor.others = self.config.others
        # Copy skills configuration from root to actor config
        if hasattr(self.config, 'skills'):
            self.config.actor.skills = self.config.skills
        if len(self.sites) > 1:
            self.config.actor.navigation_command += ["go_home"]
        self.actor = Actor(
            config=self.config.actor,
            objective=self.objective,
            prompt_template=self.prompt_dict["actor"],
            plan_tree_node=PlanTreeNode(id=0, type="branch", text=f"Find the solution to \"{self.objective}\"", level=0, url=self.online_url, step=0)
        )
        with open(self.actor.output_trash_path, "w") as _:
            pass

    def init_critic(self):
        self.config.critic.others = self.config.others
        self.critic = Critic(
            config=self.config.critic,
            objective=self.objective,
            prompt_template=self.prompt_dict["critic"][self.config.critic.character],
        )
    
    def init_judge(self):
        self.config.judge.others = self.config.others
        self.judge = Judge(
            config=self.config.judge,
            objective=self.objective,
            prompt_template=self.prompt_dict["judge"],
        )
        
    def predict_action(self):
        self.critic.update_actor_basic_info(step=self.get_step(), planning_specifications=self.actor.get_planning_specifications(), navigation_specifications=self.actor.get_navigation_specifications(), interaction_history=self.actor.get_interaction_history(interaction_history_config=self.critic.config.interaction_history), previous_plans=self.actor.get_previous_plans(verbose=True))
        criticism_elements = self.critic.get_criticism_elements() if not self.get_step()==0 else {}
        action_element_list = self.actor.predict_action(criticism_elements=criticism_elements)
        self.judge.update_actor_basic_info(step=self.get_step(), planning_specifications=self.actor.get_planning_specifications(), navigation_specifications=self.actor.get_navigation_specifications(), interaction_history=self.actor.get_interaction_history(interaction_history_config=self.judge.config.interaction_history), previous_plans=self.actor.get_previous_plans(verbose=True), planning_command=self.actor.config.planning_command, navigation_command=self.actor.config.navigation_command)
        selected_action_elements, judgement_elements = self.judge.judge(action_element_list)
        selected_action_elements = self.actor.finalize_action(selected_action_elements)
        return {**selected_action_elements, **{"critic:"+k: criticism_elements[k] for k in criticism_elements.keys()}, **{"judge:"+k: judgement_elements[k] for k in judgement_elements.keys()}}, action_element_list
    
    def update_online_state(self, url, observation):
        self.online_url = url
        self.online_observation = observation

    def get_step(self):
        return self.actor.get_step()
    
    def is_navigation(self, action):
        return self.actor.is_navigation(action=action)
    
    def get_actor_active_plan(self):
        return self.actor.get_active_plan()
    
    def get_trajectory(self):
        return self.trajectory

    def act(self, objective, env):
        # 重置 token 统计
        reset_token_stats()

        self.objective = objective
        self.env = env  # Store env reference for skill filtering
        self.sites = env.get_sites()
        observation = env.observation()
        url = env.get_url()
        self.update_online_state(url=url, observation=observation)
        self.init_actor()
        # The selector, executor, and feedback manager must share one library
        # instance so the environment can enforce the active singleton skill.
        if self.actor.skill_registry is not None:
            env.skill_registry = self.actor.skill_registry
            task_id = env.config.get("task_id") if isinstance(getattr(env, "config", None), dict) else None
            self.actor.skill_registry.start_episode(objective, task_id=task_id)
        self.init_critic()
        self.init_judge()
        while not env.done():
            observation = env.observation()
            url = env.get_url()
            self.update_online_state(url=url, observation=observation)
            self.actor.update_online_state(url=url, observation=observation)
            self.critic.update_online_state(url=url, observation=observation)
            self.judge.update_online_state(url=url, observation=observation)
            action_elements, action_element_list = self.predict_action()
            action = action_elements["action"]
            navigation_action = action_elements["action"] if not action_elements.get("navigation action", "") else action_elements.get("navigation action", "")
            status = env.step(navigation_action)

            # ===== Check skill execution status and update Actor =====
            if hasattr(env, 'last_skill_execution') and env.last_skill_execution is not None:
                skill_exec = env.last_skill_execution
                skill_name = skill_exec.get('skill_name', 'unknown')
                if skill_exec.get('success', False):
                    # Skill executed successfully
                    result_info = str(skill_exec.get('result', ''))
                    self.actor.record_skill_execution(success=True, result_info=result_info)
                    print(f"[Skill Status] ✅ Skill '{skill_name}' succeeded, can use any action to continue")
                else:
                    # Skill execution failed
                    error_info = skill_exec.get('error', skill_exec.get('failure_reason', 'Unknown error'))
                    self.actor.record_skill_execution(success=False, result_info=error_info)
                    print(f"[Skill Status] ❌ Skill '{skill_name}' failed; continuing with refreshed retrieval")
                # Clear skill execution status to avoid duplicate processing
                env.last_skill_execution = None

            # Handle verification requirement for STOP actions
            verification_required = isinstance(status, dict) and status.get('verification_required', False)

            if verification_required:
                flaw_node = self.actor.active_node
                verification_prompt = status.get('verification_prompt', '')
                flaw_node.note.append(f"STEP {self.get_step()}: VERIFICATION REQUIRED - {verification_prompt}")
                print(f"[Verification Required] {verification_prompt}")
                print(f"[DEBUG] Added verification note to plan node [{flaw_node.id}]. Total notes: {len(flaw_node.note)}")
            elif isinstance(status, dict) and status.get('stop_rejected', False):
                # Legacy support for old rejection format
                flaw_node = self.actor.active_node
                rejection_reason = status.get('rejection_reason', 'Unknown reason')
                flaw_node.note.append(f"STEP {self.get_step()}: Your STOP action was REJECTED. Reason: {rejection_reason}. Please continue working to complete the task properly.")
            elif navigation_action and self.is_navigation(action=navigation_action) and status == False: # means invalid action
                flaw_node = self.actor.active_node
                flaw_node.note.append(f"STEP {self.get_step()}: You generate action \"{action}\", which has INVALID syntax. Strictly follow the action specifications.")
            DOCUMENTED_INTERACTION_ELEMENT_KEY_TO_CONTENT_MAP = {
                "observation": observation,
                "action": action,
                "url": url,
                "plan": self.get_actor_active_plan(),
                "reason": action_elements.get("reason", ""),
                "observation highlight": action_elements.get("observation highlight", ""),
                "retained element ids": action_elements.get("retained element ids", []),
                "observation summary": action_elements.get("observation description", "")
            }
            self.actor.update_history(**DOCUMENTED_INTERACTION_ELEMENT_KEY_TO_CONTENT_MAP)
            self.actor.del_observation_node()
            assert self.actor.equal_history_length()

            if len(action_element_list) > 1:
                if self.config.others.logging:
                    self.log_step(
                        status=status if "status" in locals() and isinstance(status, dict) else env.status(),
                        plan=self.get_actor_active_plan(),
                        **action_elements,
                        **{f"actor {i}:{k}": _action_elements[k] for i, _action_elements in enumerate(action_element_list) for k in _action_elements.keys() if k != "input" and k != "instruction"}
                    )
            else:
                if self.config.others.logging:
                    self.log_step(
                        status=status if "status" in locals() and isinstance(status, dict) else env.status(),
                        plan=self.get_actor_active_plan(),
                        **action_elements,
                    )

            # After updating history, if verification was required, continue to next iteration
            if verification_required:
                continue

        # 记录 token 统计
        token_stats = get_token_stats()
        logger.info(f"[Task Token Summary] objective='{objective[:50]}...', total_input_tokens={token_stats['total_input_tokens']}, llm_calls={token_stats['call_count']}")
        print(f"\n{'='*60}")
        print(f"[Token Statistics] Task completed")
        print(f"  Total input tokens: {token_stats['total_input_tokens']:,}")
        print(f"  Total LLM calls: {token_stats['call_count']}")
        if token_stats['call_count'] > 0:
            print(f"  Average tokens per call: {token_stats['total_input_tokens'] // token_stats['call_count']:,}")
        print(f"{'='*60}\n")

        final_status = status if "status" in locals() and isinstance(status, dict) else env.status()
        if env.skill_registry is not None:
            env.skill_registry.finalize_episode_feedback(final_status.get("reward", 0.0))
        return final_status
    
    def log_step(self, status, **kwargs):
        def serialize_message_list(message_list):
            if not isinstance(message_list, list):
                return message_list
            return "".join([m[1] for m in message_list if m[0]=="text"])
        data_to_log = {}
        data_to_log['objective'] = self.objective
        data_to_log['url'] = self.online_url
        data_to_log['observation'] = self.get_observation_text()
        # 添加 token 统计到每步日志
        token_stats = get_token_stats()
        data_to_log['cumulative_input_tokens'] = token_stats['total_input_tokens']
        data_to_log['cumulative_llm_calls'] = token_stats['call_count']
        for (k, v) in status.items():
            data_to_log[k] = v
        for k in kwargs.keys():
            try:
                json.dumps(kwargs[k])
                data_to_log[k.replace(" ", "_")] = kwargs[k] if not "input" in k else serialize_message_list(kwargs[k])
            except:
                pass
        self.trajectory.append(data_to_log)

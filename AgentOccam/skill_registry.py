# AgentOccam/skill_registry.py
"""
技能注册表：加载和管理从 SkillWeaver 生成的技能库。

约定：
- 技能函数必须是 async def
- 第一个参数必须是 page (Playwright Page 对象)
- 技能文件格式：skills/<site>/iter_N/kb_post_code.py
- 元数据文件：skills/<site>/iter_N/kb_post_metadata.json
"""

import ast
import asyncio
import importlib.util
import inspect
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Any, Optional, List, Sequence
import logging

from AgentOccam.drive import (
    DualSkillSelection,
    ScenarioAwareCoordinator,
    extract_context,
    interaction_descriptor,
    merge_reasoning_group,
    normalize_reasoning_skill,
    reasoning_similarity,
    structural_match,
)

logger = logging.getLogger(__name__)


class Skill:
    """单个技能的封装"""

    def __init__(
        self,
        name: str,
        fn: Callable,
        doc: str = "",
        signature: Optional[inspect.Signature] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.fn = fn
        self.doc = doc or (inspect.getdoc(fn) or "")
        self.signature = signature or inspect.signature(fn)
        self.metadata = metadata or {}

        # Canonical DRIVE interaction representation:
        # k^i=<Sig_k, Pre_k, Body_k, Post_k, Rec_k>.  Older artifacts are
        # upgraded at load time, while newly generated artifacts persist the
        # same fields through ``add_interaction_runtime_schema``.
        self._ensure_interaction_contract()

        # 验证技能格式
        self._validate()

    def _ensure_interaction_contract(self) -> None:
        signature = {
            name: {
                "annotation": (
                    str(param.annotation)
                    if param.annotation is not inspect.Parameter.empty
                    else "Any"
                ),
                "required": (
                    param.default is inspect.Parameter.empty
                    and param.kind
                    not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                ),
            }
            for name, param in list(self.signature.parameters.items())[1:]
        }
        required = self.get_required_args()
        contract = dict(self.metadata.get("interaction_contract") or {})
        existing_sig = contract.get("Sig_k")
        if not isinstance(existing_sig, dict) or existing_sig.get("inputs") == "derived from callable signature":
            contract["Sig_k"] = {
                "inputs": signature,
                "returns": (
                    str(self.signature.return_annotation)
                    if self.signature.return_annotation is not inspect.Signature.empty
                    else "Any"
                ),
            }
        contract.setdefault(
            "Pre_k",
            [
                "The typed descriptor matches the current page context.",
                "All required typed inputs are bound to non-placeholder values.",
            ],
        )
        body = dict(contract.get("Body_k") or {})
        body["callable"] = self.name
        body.setdefault("action_space", "predefined_browser_actions")
        contract["Body_k"] = body
        contract.setdefault(
            "Post_k",
            [
                "The procedure terminates without exception and produces an observable page effect or meaningful local result."
            ],
        )
        contract["Rec_k"] = list(self.metadata.get("recovery_branches", []))
        self.metadata["interaction_contract"] = contract
        self.metadata.setdefault("preconditions", list(contract["Pre_k"]))
        self.metadata.setdefault("postconditions", list(contract["Post_k"]))
        self.metadata.setdefault(
            "precondition_checks",
            [
                {"type": "descriptor_match"},
                {"type": "required_arguments_bound", "arguments": required},
            ],
        )
        self.metadata.setdefault(
            "postcondition_checks",
            [{"type": "observable_effect_or_result"}],
        )
        self.metadata.setdefault(
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

    def _validate(self):
        """验证技能函数符合约定"""
        # 检查是否是异步函数
        if not asyncio.iscoroutinefunction(self.fn):
            raise ValueError(f"技能 '{self.name}' 必须是 async 函数")

        # 检查第一个参数是否是 'page'
        params = list(self.signature.parameters.keys())
        if not params or params[0] != "page":
            raise ValueError(
                f"技能 '{self.name}' 的第一个参数必须是 'page'，当前: {params}"
            )

    async def __call__(self, page, **kwargs):
        """异步调用技能"""
        return await self.fn(page, **kwargs)

    def get_args_spec(self) -> Dict[str, str]:
        """获取参数规格（除了 page）"""
        args_spec = {}
        params = list(self.signature.parameters.items())[1:]  # 跳过 page
        for name, param in params:
            # 获取类型注解
            if param.annotation != inspect.Parameter.empty:
                args_spec[name] = str(param.annotation)
            else:
                args_spec[name] = "Any"
        return args_spec

    def get_required_args(self) -> List[str]:
        """Return required task arguments (the implicit ``page`` is excluded)."""

        return [
            name
            for name, param in list(self.signature.parameters.items())[1:]
            if param.default is inspect.Parameter.empty
            and param.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]

    def validate_args(self, arguments: Optional[Dict[str, Any]]) -> None:
        """Validate invocation arguments before touching the browser page."""

        arguments = dict(arguments or {})
        try:
            # ``page`` is supplied by the environment at invocation time.
            self.signature.bind(object(), **arguments)
        except TypeError as exc:
            raise ValueError(f"技能 '{self.name}' 参数不合法: {exc}") from exc

        empty_required = [
            name
            for name in self.get_required_args()
            if isinstance(arguments.get(name), str) and not arguments[name].strip()
        ]
        if empty_required:
            raise ValueError(
                f"技能 '{self.name}' 的必填参数不能为空: {', '.join(empty_required)}"
            )
        placeholder_required = [
            name
            for name in self.get_required_args()
            if _contains_placeholder(arguments.get(name))
        ]
        if placeholder_required:
            raise ValueError(
                f"技能 '{self.name}' 的必填参数仍是占位符: {', '.join(placeholder_required)}"
            )

    def get_function_signature(self) -> str:
        """
        获取完整的函数签名字符串（类似 SkillWeaver 格式）

        Returns:
            格式如: async def subscribe_to_forums_by_names(page, forum_names: list[str]) -> dict
        """
        params_list = []
        for name, param in self.signature.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                type_str = str(param.annotation)
                # 简化类型显示
                type_str = type_str.replace("typing.", "").replace("<class '", "").replace("'>", "")
                params_list.append(f"{name}: {type_str}")
            else:
                params_list.append(name)

        # 获取返回类型
        return_annotation = self.signature.return_annotation
        if return_annotation != inspect.Signature.empty:
            return_str = str(return_annotation)
            return_str = return_str.replace("typing.", "").replace("<class '", "").replace("'>", "")
            return f"async def {self.name}({', '.join(params_list)}) -> {return_str}"
        else:
            return f"async def {self.name}({', '.join(params_list)})"

    def get_skillweaver_format(self) -> str:
        """
        获取 SkillWeaver 风格的完整技能描述

        Returns:
            包含完整签名和文档字符串的格式化字符串
        """
        signature = self.get_function_signature()
        docstring = self.doc.strip() if self.doc else "No description available."

        return f"# Skill: {signature}\n{docstring}\n"

    def get_call_example(self) -> str:
        """
        获取技能调用示例，包含参数类型提示

        Returns:
            格式如: use_skill subscribe_to_forums_by_names forum_names=["forum1", "forum2"]
        """
        args_examples = []
        params = list(self.signature.parameters.items())[1:]  # 跳过 page

        for name, param in params:
            annotation = param.annotation
            if annotation != inspect.Parameter.empty:
                type_str = str(annotation).lower()
                # 根据类型生成示例值
                if 'list[str]' in type_str or 'list' in type_str:
                    args_examples.append(f'{name}=["<actual_{name}_1>", "<actual_{name}_2>"]')
                elif 'bool' in type_str:
                    args_examples.append(f'{name}=True')
                elif 'int' in type_str:
                    args_examples.append(f'{name}=10')
                elif 'str' in type_str:
                    args_examples.append(f'{name}="<actual_{name}>"')
                else:
                    args_examples.append(f'{name}="<{type_str}>"')
            else:
                args_examples.append(f'{name}="<actual_{name}>"')

        if args_examples:
            return f"use_skill {self.name} {' '.join(args_examples)}"
        else:
            return f"use_skill {self.name}"

    def get_summary(self) -> str:
        """获取技能的一句话摘要（用于提示词）"""
        # 取 docstring 的第一行
        if self.doc:
            first_line = self.doc.split("\n")[0].strip()
            return first_line
        return "No description"

    def get_full_description(self) -> str:
        """获取完整描述（用于详细展示）"""
        args_spec = self.get_args_spec()
        args_str = ", ".join([f"{k}: {v}" for k, v in args_spec.items()])

        return f"""
技能名称: {self.name}
参数: ({args_str})
描述: {self.doc}
测试次数: {self.metadata.get('test_count', 0)}
版本: {self.metadata.get('version', 0)}
        """.strip()


class SkillRegistry:
    """技能注册表：管理所有加载的技能"""

    def __init__(self):
        self.skills: Dict[str, Skill] = {}
        self.site_name: Optional[str] = None
        self.site_dir: Optional[Path] = None        # 站点目录路径（用于写入失败日志）
        self.metadata_path: Optional[Path] = None   # 元数据 JSON 文件路径
        self.external_task_lessons: List[Dict[str, Any]] = []  # 外部任务级经验
        self.task_lessons_path: Optional[Path] = None

        # DRIVE online coordination and episode feedback state.
        self.coordinator = ScenarioAwareCoordinator(self)
        self.active_selection: Optional[DualSkillSelection] = None
        self.active_interaction_skill: Optional[str] = None
        self.selection_initialized = False
        self.current_episode: Dict[str, Any] = {}
        self.pending_feedback: List[Dict[str, Any]] = []
        self._reasoning_activation_keys: set[tuple[str, Any]] = set()

        # 技能与任务族的映射索引
        self.skill_to_task_families: Dict[str, List[int]] = {}  # 技能名 → 任务族ID列表
        self.task_family_to_skills: Dict[int, List[str]] = {}   # 任务族ID → 技能名列表

    def load_external_task_lessons(self, task_lessons_path: Path):
        """
        加载外部任务级经验（从JSON文件）

        Args:
            task_lessons_path: task_level_lessons.json 文件路径
        """
        import json

        if not task_lessons_path.exists():
            logger.warning(f"任务级经验文件不存在: {task_lessons_path}")
            return

        try:
            with open(task_lessons_path, 'r', encoding='utf-8') as f:
                lessons_list = json.load(f)

            if not isinstance(lessons_list, list):
                raise ValueError("reasoning skill library must be a JSON list")
            self.task_lessons_path = task_lessons_path
            self.external_task_lessons = [
                normalize_reasoning_skill(item, site_hint=self.site_name or "")
                for item in lessons_list
                if isinstance(item, dict)
            ]
            logger.info(f"✓ 成功加载 {len(lessons_list)} 条任务级经验")

            # 建立技能与任务族的映射索引
            self._build_skill_task_family_index()

        except Exception as e:
            logger.warning(f"加载任务级经验失败: {e}")

    def _build_skill_task_family_index(self):
        """
        从任务级经验中建立技能与任务族的映射索引

        这个索引用于快速查找某个任务族对应的所有技能

        注意: 虚拟任务族自动创建已禁用 (2024-12)
        只使用手动编写的任务级经验，不再自动为缺少经验的技能创建虚拟任务族
        """
        self.skill_to_task_families.clear()
        self.task_family_to_skills.clear()

        # 从 external_task_lessons 建立映射。
        # 字段名 cluster_id 是兼容旧 schema 的 lesson ID，不再表示聚类流程。
        for lesson in self.external_task_lessons:
            cluster_id = lesson.get("cluster_id")
            skill_name = lesson.get("skill_name")

            if skill_name and cluster_id is not None:
                # 建立 skill_name → lesson_id 的映射
                if skill_name not in self.skill_to_task_families:
                    self.skill_to_task_families[skill_name] = []
                if cluster_id not in self.skill_to_task_families[skill_name]:
                    self.skill_to_task_families[skill_name].append(cluster_id)

                # 建立 lesson_id → skill_name 的反向映射
                if cluster_id not in self.task_family_to_skills:
                    self.task_family_to_skills[cluster_id] = []
                if skill_name not in self.task_family_to_skills[cluster_id]:
                    self.task_family_to_skills[cluster_id].append(skill_name)

        logger.info(f"✓ 已建立技能-任务族映射索引: {len(self.skill_to_task_families)} 个技能")

    def _extract_keywords_from_skill(self, skill) -> List[str]:
        """从技能名称和描述中提取关键词"""
        keywords = []

        # 从技能名称中提取 (下划线分隔)
        name_parts = skill.name.split('_')
        keywords.extend([p for p in name_parts if len(p) > 2])

        # 从描述中提取关键词
        summary = skill.get_summary().lower()
        common_words = {'the', 'a', 'an', 'in', 'on', 'to', 'for', 'of', 'and', 'or', 'is', 'are', 'by', 'with', 'from'}
        words = summary.replace(',', ' ').replace('.', ' ').replace('(', ' ').replace(')', ' ').split()
        keywords.extend([w for w in words if len(w) > 3 and w not in common_words])

        # 去重并限制数量
        return list(set(keywords))[:10]

    def get_skills_by_task_families(self, task_family_ids: List[int]) -> List[Skill]:
        """
        根据任务族ID获取相关的操作级技能

        Args:
            task_family_ids: 任务族ID列表

        Returns:
            相关的Skill对象列表
        """
        skill_names = set()
        for tf_id in task_family_ids:
            if tf_id in self.task_family_to_skills:
                skill_names.update(self.task_family_to_skills[tf_id])

        # 返回存在的技能对象
        result = []
        for name in skill_names:
            if name in self.skills:
                result.append(self.skills[name])

        return result

    def load_from_kb_file(
        self,
        kb_code_path: Path,
        kb_metadata_path: Optional[Path] = None,
    ):
        """
        从 SkillWeaver 生成的知识库文件加载技能

        Args:
            kb_code_path: kb_post_code.py 文件路径
            kb_metadata_path: kb_post_metadata.json 文件路径（可选）
        """
        logger.info(f"从 {kb_code_path} 加载技能库...")

        # 加载元数据
        metadata_dict = {}
        if kb_metadata_path and kb_metadata_path.exists():
            self.metadata_path = kb_metadata_path
            with open(kb_metadata_path, "r") as f:
                meta = json.load(f)
                metadata_dict = meta.get("functions", {})
            logger.info(f"加载元数据: {len(metadata_dict)} 个函数")

        # 动态导入模块
        module = self._load_module_from_path(kb_code_path)

        # 提取所有 async def 函数
        loaded_count = 0
        for name, obj in vars(module).items():
            if asyncio.iscoroutinefunction(obj) and not name.startswith("_"):
                try:
                    # When metadata exists it is the authoritative K^i entry
                    # set.  Helper functions and compatibility aliases remain
                    # available through whole-module execution but must not be
                    # retrieved as independent skills.
                    if metadata_dict and name not in metadata_dict:
                        continue
                    function_metadata = metadata_dict.get(name, {})
                    if function_metadata.get("active", True) is False:
                        logger.info("跳过已从 DRIVE 技能库移除的技能 '%s'", name)
                        continue
                    skill = Skill(
                        name=name,
                        fn=obj,
                        metadata=function_metadata,
                    )
                    self.skills[name] = skill
                    loaded_count += 1
                except ValueError as e:
                    logger.warning(f"跳过无效技能 '{name}': {e}")

        logger.info(f"✓ 成功加载 {loaded_count} 个技能")

    def load_from_site_dir(self, site_dir: Path, use_latest: bool = True,
                           skill_file: str = None, skill_metadata: str = None):
        """
        从站点目录加载技能（支持多种目录结构）

        支持的目录结构:
        1. 直接格式: skills/<site>/operation_skills.py (优先)
        2. 迭代格式: skills/<site>/iter_XX/kb_post_code.py

        Args:
            site_dir: skills/<site>/ 目录
            use_latest: 是否使用最新迭代（仅对迭代格式有效）
            skill_file: 技能文件名 (如 'operation_skills.py')
            skill_metadata: 元数据文件名 (如 'operation_skills.json')
        """
        self.site_name = site_dir.name
        self.site_dir = site_dir

        # 方案1: 尝试直接格式 (优先检查配置的文件名或默认名)
        direct_files = [
            (skill_file, skill_metadata) if skill_file else None,
            ("operation_skills.py", "operation_skills.json"),
            ("kb_final_code.py", "kb_final_metadata.json"),
        ]

        for file_pair in direct_files:
            if file_pair is None:
                continue
            code_name, meta_name = file_pair
            kb_code = site_dir / code_name
            kb_meta = site_dir / meta_name if meta_name else None

            if kb_code.exists():
                logger.info(f"使用直接格式: {kb_code.name}")
                self.load_from_kb_file(kb_code, kb_meta if kb_meta and kb_meta.exists() else None)
                return

        # 方案2: 尝试迭代格式 (iter_* 目录)
        def extract_iter_num(dir_path):
            """提取迭代号，支持 iter_0, iter_0160_validated 等格式"""
            import re
            name = dir_path.name
            match = re.match(r'iter_(\d+)', name)
            if match:
                return int(match.group(1))
            return 0

        iter_dirs = sorted(
            [d for d in site_dir.glob("iter_*") if d.is_dir()],
            key=extract_iter_num,
        )

        if not iter_dirs:
            raise FileNotFoundError(f"在 {site_dir} 下没有找到 iter_* 目录")

        # 选择要加载的迭代
        if use_latest:
            selected_iter = iter_dirs[-1]
            logger.info(f"使用最新迭代: {selected_iter.name}")
        else:
            # TODO: 实现基于质量指标选择最佳迭代
            selected_iter = iter_dirs[-1]

        # 加载知识库文件
        kb_code = selected_iter / "kb_post_code.py"
        kb_meta = selected_iter / "kb_post_metadata.json"

        if not kb_code.exists():
            raise FileNotFoundError(f"未找到 {kb_code}")

        self.load_from_kb_file(kb_code, kb_meta if kb_meta.exists() else None)

    def _load_module_from_path(self, pyfile: Path):
        """动态加载 Python 模块"""
        import sys

        spec = importlib.util.spec_from_file_location(pyfile.stem, pyfile)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载模块: {pyfile}")
        mod = importlib.util.module_from_spec(spec)

        # 注入必要的类型定义到模块的全局命名空间
        # 必须在 exec_module 之前注入，因为技能代码会在导入时使用这些类型
        try:
            from playwright.async_api import Page, Browser, TimeoutError
            # 将这些类型添加到 sys.modules，这样技能代码的 import 语句就能找到它们
            mod.Page = Page
            mod.Browser = Browser
            mod.TimeoutError = TimeoutError
            mod.PlaywrightTimeout = TimeoutError  # 别名
        except ImportError:
            # 如果 playwright 未安装，使用 typing.Any 作为占位符
            from typing import Any
            mod.Page = Any
            mod.Browser = Any
            mod.TimeoutError = Any
            mod.PlaywrightTimeout = Any

        spec.loader.exec_module(mod)
        return mod

    def get_skill(self, name: str) -> Optional[Skill]:
        """获取指定技能"""
        return self.skills.get(name)

    def get_all_skills(self) -> List[Skill]:
        """获取所有技能"""
        return list(self.skills.values())

    def list_skills(self) -> List[str]:
        """获取所有技能名称列表"""
        return list(self.skills.keys())

    def get_verified_skills(self, min_test_count: int = 1, require_success: bool = False) -> List[Skill]:
        """获取经过测试验证的技能

        技能被视为"已验证"的条件：
        - require_success=False: test_count >= min_test_count (默认要求 test_count >= 1)
        - require_success=True: success_count > 0 (必须有成功的测试)

        Args:
            min_test_count: 最小测试次数（仅在 require_success=False 时使用）
            require_success: 是否要求技能必须有成功的测试（success_count > 0）

        注意：仅有 references 但 test_count == 0 的技能不被视为可调用
        """
        if require_success:
            # 只返回测试成功的技能 (success_count > 0)
            return [
                sk
                for sk in self.skills.values()
                if sk.metadata.get("success_count", 0) > 0
            ]
        else:
            # 返回已测试的技能 (test_count >= min_test_count)
            return [
                sk
                for sk in self.skills.values()
                if sk.metadata.get("test_count", 0) >= min_test_count
            ]

    def search_skills(self, query: str, top_k: int = 5) -> List[Skill]:
        """
        简单的技能搜索（基于名称和描述的关键词匹配）

        TODO: 实现基于 embedding 的语义搜索（参考 SkillWeaver 的 retrieve 方法）
        """
        query_lower = query.lower()
        matches = []

        for skill in self.skills.values():
            score = 0
            # 名称匹配
            if query_lower in skill.name.lower():
                score += 10
            # 描述匹配
            if query_lower in skill.doc.lower():
                score += 5

            if score > 0:
                matches.append((score, skill))

        # 按分数排序
        matches.sort(key=lambda x: x[0], reverse=True)
        return [skill for _, skill in matches[:top_k]]

    def match_skills_by_docstring(
        self,
        objective: str,
        top_k: int = 5,
        only_verified: bool = True,
        require_success: bool = False,
        model: str = "gpt-4.1",
        api_base: str = None
    ) -> List[Skill]:
        """
        方案3: 直接使用 LLM 比较任务意图和技能的 docstring，跳过任务族映射

        这个方法解决了 task_level_lessons.json 与 kb_post_metadata.json 不同步的问题。
        直接让 LLM 根据技能的 docstring 判断哪些技能与当前任务最相关。

        Args:
            objective: 任务目标描述
            top_k: 返回的最大技能数
            only_verified: 是否只考虑已验证的技能（test_count > 0）
            require_success: 是否只考虑测试成功的技能（success_count > 0）
                            如果为 True，则 only_verified 被忽略
            model: LLM 模型名称
            api_base: API base URL

        Returns:
            匹配的技能列表（按相关性排序）
        """
        import json
        import os

        # 获取候选技能 - 始终使用所有技能，不再区分是否验证
        candidate_skills = self.get_all_skills()
        filter_type = "all"

        if not candidate_skills:
            logger.warning("没有候选技能可供匹配")
            return []

        print(f"📚 候选技能数量: {len(candidate_skills)} ({filter_type})")

        # 构建技能摘要供 LLM 匹配
        skill_summaries = []
        for i, skill in enumerate(candidate_skills):
            # 提取 docstring 的前 500 字符作为描述
            doc_preview = skill.doc[:500] if len(skill.doc) > 500 else skill.doc
            # 移除多余空白行
            doc_preview = "\n".join(line for line in doc_preview.split("\n") if line.strip())

            skill_summaries.append({
                "index": i,
                "name": skill.name,
                "summary": skill.get_summary(),
                "docstring_preview": doc_preview,
                "args": skill.get_args_spec(),
                "test_count": skill.metadata.get("test_count", 0),
                "scenario_descriptor": skill.metadata.get("scenario_descriptor", {}),
                "scenario_description": skill.metadata.get("scenario_description", ""),
                "scenario_keywords": skill.metadata.get("scenario_keywords", []),
                "url_patterns": skill.metadata.get("url_patterns", []),
            })

        # 调用 LLM 进行匹配
        try:
            from openai import OpenAI
        except ImportError:
            logger.warning("OpenAI library not available, falling back to keyword search")
            print(f"⚠️  OpenAI 库不可用，回退到关键词搜索")
            return self.search_skills(objective, top_k)

        prompt = f"""You are a skill matching expert. Select the most relevant skills for the given task.

# Current Task
Objective: {objective}

# Available Skills (with docstrings)
{json.dumps(skill_summaries, indent=2, ensure_ascii=False)}

# Instructions
1. Carefully analyze the task objective
2. For each skill, check if its functionality (based on name, summary, and docstring) matches what the task needs
3. Consider:
   - Prefer the structured scenario_descriptor when present:
     * U_k describes compatible page context such as site and URL patterns
     * W_k describes compatible task semantics such as intent, task family, and semantic keywords
   - Does the skill perform the operation described in the task?
   - Does the skill work on the right type of content (posts, comments, users, forums, etc.)?
   - Are the skill's parameters compatible with the task requirements?
4. Select skills that are DIRECTLY relevant to accomplishing the task
5. Rank them by relevance (most relevant first)

Output a JSON array of skill indices in order of relevance, e.g., [2, 0, 5] or [] if no match.
Output ONLY the JSON array, nothing else.
"""

        try:
            base_url = api_base or os.environ.get("LLM_API_BASE_URL", "http://localhost:4141/v1")
            api_key = os.environ.get("LLM_API_KEY", "dummy")

            client = OpenAI(base_url=base_url, api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )

            result_text = response.choices[0].message.content.strip()
            print(f"🤖 LLM 技能匹配返回: {result_text}")

            # 解析 JSON 数组
            import re
            if result_text.startswith("["):
                matched_indices = json.loads(result_text)
            else:
                # 尝试从文本中提取 JSON
                match = re.search(r'\[[\d,\s]*\]', result_text)
                if match:
                    matched_indices = json.loads(match.group())
                else:
                    matched_indices = []

            # 返回匹配的技能对象
            matched_skills = []
            for idx in matched_indices[:top_k]:
                if 0 <= idx < len(candidate_skills):
                    matched_skills.append(candidate_skills[idx])

            logger.info(f"✓ LLM 匹配到 {len(matched_skills)} 个相关技能")
            if matched_skills:
                print(f"✓ 匹配到的技能: {[s.name for s in matched_skills]}")

            return matched_skills

        except Exception as e:
            logger.warning(f"LLM 技能匹配失败: {e}, 回退到关键词搜索")
            print(f"⚠️  LLM 匹配失败: {e}, 回退到关键词搜索")
            return self.search_skills(objective, top_k)

    def summarize_skills(
        self, top_k: int = 10, only_verified: bool = True
    ) -> str:
        """
        生成技能清单摘要（用于 Agent 提示词）

        包含两部分：
        1. CALLABLE SKILLS (Tested): 可以直接调用的已验证技能
        2. FAILURE LESSONS (Untested): 未验证技能的失败经验，用于指导决策

        Args:
            top_k: 返回前 N 个技能
            only_verified: 只返回经过验证的技能（已弃用，现在总是显示两种类型）

        Returns:
            格式化的技能列表字符串
        """
        # 分类技能：Tested vs Untested (with task_lessons or failure_lessons)
        tested_skills = []
        untested_with_lessons = []

        for sk in self.skills.values():
            test_count = sk.metadata.get("test_count", 0)
            has_lessons = "task_lessons" in sk.metadata or "failure_lessons" in sk.metadata

            # Tested: ONLY test_count >= 1
            if test_count >= 1:
                tested_skills.append(sk)
            # Untested with lessons: test_count == 0 且有 task_lessons 或 failure_lessons
            elif test_count == 0 and has_lessons:
                untested_with_lessons.append(sk)

        lines = []

        # === 第一部分：CALLABLE SKILLS (Tested) ===
        if tested_skills:
            # 按优先级排序：version 高的优先，然后是 references 多的
            tested_skills.sort(
                key=lambda sk: (
                    sk.metadata.get("version", 0),
                    len(sk.metadata.get("references", [])),
                    sk.metadata.get("test_count", 0),
                ),
                reverse=True,
            )

            lines.append("# ✅ CALLABLE SKILLS (已验证，可直接调用)")
            lines.append("")
            lines.append(f"Found {len(tested_skills)} verified skills. These skills are pre-tested and reliable.")
            lines.append("IMPORTANT: Use these skills whenever applicable to complete tasks efficiently!")
            lines.append("")

            for i, sk in enumerate(tested_skills[:top_k], 1):
                summary = sk.get_summary()
                args_spec = sk.get_args_spec()

                # 构建参数说明
                if args_spec:
                    args_str = ", ".join([f"{k}: {v}" for k, v in args_spec.items()])
                else:
                    args_str = "no arguments"

                version = sk.metadata.get("version", 0)
                refs = len(sk.metadata.get("references", []))
                test_count = sk.metadata.get("test_count", 0)

                # 添加技能信息
                lines.append(f"{i}. {sk.name}")
                lines.append(f"   Arguments: {args_str}")
                lines.append(f"   Description: {summary}")
                lines.append(f"   Quality: v{version}, tested {test_count} times, used {refs} times in exploration")
                lines.append("")

        # 如果没有已验证技能，显示所有可用技能
        if not tested_skills:
            all_skills = list(self.skills.values())
            if all_skills:
                # 按名称排序
                all_skills.sort(key=lambda sk: sk.name)

                lines.append("# 🔧 AVAILABLE SKILLS (可用技能)")
                lines.append("")
                lines.append(f"Found {len(all_skills)} skills. These skills are available for use.")
                lines.append("NOTE: Skills are not yet verified but can be called to complete tasks.")
                lines.append("")

                for i, sk in enumerate(all_skills[:top_k], 1):
                    summary = sk.get_summary()
                    args_spec = sk.get_args_spec()

                    # 构建参数说明
                    if args_spec:
                        args_str = ", ".join([f"{k}: {v}" for k, v in args_spec.items()])
                    else:
                        args_str = "no arguments"

                    # 添加技能信息
                    lines.append(f"{i}. {sk.name}")
                    lines.append(f"   Arguments: {args_str}")
                    lines.append(f"   Description: {summary}")
                    lines.append("")
            else:
                lines.append("# NO SKILLS AVAILABLE")
                lines.append("")
                lines.append("No skills found. Use basic actions (click, type, etc.) to complete the task.")

        return "\n".join(lines)

    def get_skill_code(self, name: str) -> str:
        """获取技能的源代码（用于展示或调试）"""
        skill = self.get_skill(name)
        if not skill:
            return ""
        return inspect.getsource(skill.fn)

    # ------------------------------------------------------------------
    # DRIVE scenario-aware coordination
    # ------------------------------------------------------------------

    def retrieve_dual_skills(
        self,
        objective: str,
        url: str,
        observation: Any,
        *,
        current_intent: Optional[str] = None,
        step: Optional[int] = None,
        reasoning_stage: Optional[tuple[Any, Any, Sequence[Any]]] = None,
    ) -> DualSkillSelection:
        """Run typed retrieval in paper order and activate one skill per level."""

        return self.coordinator.select(
            objective=objective,
            url=url,
            observation=observation,
            current_intent=current_intent,
            step=step,
            reasoning_stage=reasoning_stage,
        )

    def retrieve_reasoning_stage(
        self,
        objective: str,
        url: str,
        observation: Any,
    ) -> tuple[Any, Any, Sequence[Any]]:
        """Execute Algorithm 1's first retrieval stage exactly once."""

        context = extract_context(url, observation, site_hint=self.site_name or "")
        selected, ranked = self.coordinator.retrieve_reasoning(objective, context)
        return context, selected, ranked

    def retrieve_reasoning_skill(
        self,
        objective: str,
        url: str,
        observation: Any,
    ) -> tuple[Optional[Any], Any]:
        """Return ``(k_t^r, C_t)`` without selecting an interaction skill."""

        context, selected, _ = self.retrieve_reasoning_stage(objective, url, observation)
        return selected, context

    def set_active_selection(
        self,
        selection: DualSkillSelection,
        *,
        objective: str,
        step: Optional[int],
    ) -> None:
        """Publish the current singleton selection to the execution layer."""

        self.active_selection = selection
        self.active_interaction_skill = (
            selection.interaction.skill.name if selection.interaction else None
        )
        self.selection_initialized = True
        self.current_episode.setdefault("interaction_statuses", []).append(
            {
                "step": step,
                "skill_name": self.active_interaction_skill,
                "status": "NI",
            }
        )

        if selection.reasoning is None:
            return
        lesson = selection.reasoning.skill
        skill_id = str(lesson.get("skill_id", lesson.get("cluster_id", "reasoning_skill")))
        activation_key = (skill_id, step)
        if activation_key in self._reasoning_activation_keys:
            return
        self._reasoning_activation_keys.add(activation_key)
        self.pending_feedback.append(
            self._make_feedback_record(
                skill_type="reasoning",
                skill_name=skill_id,
                descriptor=selection.reasoning.descriptor,
                task_instruction=objective,
                page_context=selection.context.as_feedback_dict(),
                arguments={},
                local_outcome={"activated": True, "semantic_score": selection.reasoning.score},
                execution_log={"environment_step": step},
            )
        )

    def validate_skill_invocation(
        self,
        skill_name: str,
        arguments: Optional[Dict[str, Any]],
        *,
        url: str,
        observation: Any,
    ) -> Skill:
        """Check singleton activation, descriptor applicability, and arguments.

        A rejected applicability check is not an interaction-skill failure: the
        caller should fall back to primitive action generation.
        """

        skill = self.get_skill(skill_name)
        if skill is None:
            raise ValueError(f"Skill '{skill_name}' not found in registry")
        if self.selection_initialized and self.active_interaction_skill != skill_name:
            active = self.active_interaction_skill or "none"
            raise ValueError(
                f"Skill '{skill_name}' is not the active interaction skill for this scene "
                f"(active: {active})"
            )

        descriptor = interaction_descriptor(skill, site_hint=self.site_name or "")
        context = extract_context(url, observation, site_hint=self.site_name or "")
        if not structural_match(context, descriptor["U_k"]):
            raise ValueError(
                f"Skill '{skill_name}' is not applicable to the current page context"
            )
        skill.validate_args(arguments)
        for check in skill.metadata.get("precondition_checks", []):
            kind = str(check.get("type", "")) if isinstance(check, dict) else str(check)
            if kind in {"descriptor_match", "required_arguments_bound"}:
                continue
            if kind == "page_type":
                expected = str(check.get("value", "")).lower()
                if expected and context.page_type.lower() == expected:
                    continue
                raise ValueError(
                    f"Skill '{skill_name}' precondition failed: expected page type '{expected}'"
                )
            if kind == "ui_cue":
                cue = str(check.get("value", "")).lower()
                if cue and cue in context.observation.lower():
                    continue
                raise ValueError(
                    f"Skill '{skill_name}' precondition failed: missing UI cue '{cue}'"
                )
            raise ValueError(
                f"Skill '{skill_name}' has unsupported precondition check '{kind}'"
            )
        return skill

    # ------------------------------------------------------------------
    # DRIVE skill-level feedback and continual refinement
    # ------------------------------------------------------------------

    def start_episode(self, task_instruction: str, task_id: Any = None) -> str:
        """Start a feedback episode shared by reasoning and interaction skills."""

        if self.pending_feedback:
            # A prior episode that crashed before evaluation is a failed task,
            # not feedback that should silently disappear.
            self.finalize_episode_feedback(0.0)
        episode_id = uuid.uuid4().hex
        self.current_episode = {
            "episode_id": episode_id,
            "task_id": task_id,
            "task_instruction": task_instruction,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "interaction_statuses": [],
            "finalized": False,
        }
        self.pending_feedback = []
        self._reasoning_activation_keys.clear()
        self.active_selection = None
        self.active_interaction_skill = None
        self.selection_initialized = False
        return episode_id

    def _make_feedback_record(
        self,
        *,
        skill_type: str,
        skill_name: str,
        descriptor: Dict[str, Any],
        task_instruction: str,
        page_context: Dict[str, Any],
        arguments: Dict[str, Any],
        local_outcome: Dict[str, Any],
        execution_log: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "feedback_id": uuid.uuid4().hex,
            "episode_id": self.current_episode.get("episode_id"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "skill_type": skill_type,
            "skill_name": skill_name,
            "descriptor": _safe_serialize(descriptor),
            "task_id": self.current_episode.get("task_id"),
            "task_instruction": task_instruction,
            "page_context": _safe_serialize(page_context),
            "arguments": _safe_serialize(arguments),
            "local_outcome": _safe_serialize(local_outcome),
            "execution_log": _safe_serialize(execution_log),
            "final_task_label": None,
        }

    def record_interaction_feedback(
        self,
        skill_name: str,
        *,
        task_instruction: str,
        page_context: Dict[str, Any],
        arguments: Dict[str, Any],
        local_outcome: Dict[str, Any],
        execution_log: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Buffer the complete local invocation result until the final label exists."""

        skill = self.get_skill(skill_name)
        if skill is None:
            raise ValueError(f"Cannot record feedback for unknown skill '{skill_name}'")
        record = self._make_feedback_record(
            skill_type="interaction",
            skill_name=skill_name,
            descriptor=interaction_descriptor(skill, site_hint=self.site_name or ""),
            task_instruction=task_instruction,
            page_context=page_context,
            arguments=arguments,
            local_outcome=local_outcome,
            execution_log=execution_log,
        )
        record["invocation_status"] = (
            "OK" if bool(local_outcome.get("success")) else "FAIL"
        )
        for status in reversed(self.current_episode.get("interaction_statuses", [])):
            if status.get("skill_name") == skill_name and status.get("status") == "NI":
                status["status"] = record["invocation_status"]
                break
        self.pending_feedback.append(record)
        return record

    @staticmethod
    def _success_label(final_task_label: Any) -> int:
        if isinstance(final_task_label, bool):
            return int(final_task_label)
        try:
            return int(float(final_task_label) == 1.0)
        except (TypeError, ValueError):
            return 0

    def _persist_finalized_feedback(self, finalized: Sequence[Dict[str, Any]]) -> None:
        """Append finalized episode evidence without fabricating a repair."""

        if not finalized or not self.site_dir:
            return
        feedback_path = self.site_dir / "skill_feedback_log.jsonl"
        try:
            with open(feedback_path, "a", encoding="utf-8") as handle:
                for item in finalized:
                    handle.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            logger.warning("Failed to persist DRIVE feedback: %s", exc)

    def finalize_episode_feedback(self, final_task_label: Any) -> List[Dict[str, Any]]:
        """Attach ``y_phi``, persist feedback, and update skill statistics."""

        # Both the environment wrapper and the high-level task runner can
        # observe termination.  An episode must be routed exactly once.
        if self.current_episode.get("finalized"):
            return []
        label = self._success_label(final_task_label)
        if not self.pending_feedback:
            # Eq. (7) also covers an episode in which no interaction was
            # invoked: a failed task has no local FAIL evidence and therefore
            # waits for the binary counterfactual rather than being silently
            # dropped or guessed as an interaction failure.
            finalized: List[Dict[str, Any]] = []
            if label == 0:
                item = self._make_feedback_record(
                    skill_type="episode",
                    skill_name="__episode__",
                    descriptor={},
                    task_instruction=str(self.current_episode.get("task_instruction", "")),
                    page_context={},
                    arguments={},
                    local_outcome={"success": False, "invoked": False},
                    execution_log={"environment_step": None},
                )
                item.update(
                    {
                        "final_task_label": 0,
                        "finalized_at": datetime.now(timezone.utc).isoformat(),
                        "maintenance_target": "pending_counterfactual",
                        "counterfactual_required": True,
                        "episode_interaction_statuses": copy_dict(
                            {"items": self.current_episode.get("interaction_statuses", [])}
                        )["items"],
                    }
                )
                finalized.append(item)
            self._persist_finalized_feedback(finalized)
            self.current_episode["finalized"] = True
            return finalized
        interaction_failed = any(
            item.get("skill_type") == "interaction"
            and item.get("invocation_status") == "FAIL"
            for item in self.pending_feedback
        )
        # Equation (7): explicit local execution evidence takes precedence.
        # A task failure without such evidence is deliberately left for the
        # binary counterfactual attributor used by the next offline round.
        maintenance_target = (
            "interaction"
            if interaction_failed
            else "pending_counterfactual"
            if label == 0
            else None
        )
        finalized = []
        for record in self.pending_feedback:
            item = copy_dict(record)
            item["final_task_label"] = label
            item["finalized_at"] = datetime.now(timezone.utc).isoformat()
            item["maintenance_target"] = maintenance_target
            item["counterfactual_required"] = maintenance_target == "pending_counterfactual"
            item["episode_interaction_statuses"] = copy_dict(
                {"items": self.current_episode.get("interaction_statuses", [])}
            )["items"]
            finalized.append(item)
            if item["skill_type"] == "interaction":
                self._apply_interaction_feedback(
                    item, allow_repair=maintenance_target == "interaction"
                )
            elif item["skill_type"] == "reasoning":
                self._apply_reasoning_feedback(item)

        self._persist_finalized_feedback(finalized)

        self._persist_skill_metadata()
        self._persist_reasoning_library()
        self.pending_feedback = []
        self._reasoning_activation_keys.clear()
        self.current_episode["finalized"] = True
        return finalized

    def _apply_interaction_feedback(
        self, feedback: Dict[str, Any], *, allow_repair: bool = False
    ) -> None:
        skill = self.get_skill(feedback["skill_name"])
        if skill is None:
            return
        metadata = skill.metadata
        metadata["invocation_count"] = int(metadata.get("invocation_count", 0)) + 1
        local_success = bool(feedback.get("local_outcome", {}).get("success"))
        if local_success:
            metadata["local_success_count"] = int(metadata.get("local_success_count", 0)) + 1
        if feedback.get("final_task_label") == 1:
            metadata["task_success_count"] = int(metadata.get("task_success_count", 0)) + 1
        metadata["utility_score"] = (
            int(metadata.get("local_success_count", 0)) + 1.0
        ) / (int(metadata.get("invocation_count", 0)) + 2.0)

        trace = feedback.get("execution_log", {}) or {}
        selector_trace = trace.get("selector_trace", []) or []
        localized_trace = [
            item
            for item in selector_trace
            if isinstance(item, dict)
            and item.get("operation") != "goto"
            and item.get("source_selector")
            and item.get("selector")
        ]
        if localized_trace and local_success:
            # A successful fallback is live postcondition evidence, so the
            # selector ordering can be committed as a validated local patch.
            self.patch_interaction_selector_trace(skill.name, localized_trace)
        working = [
            str(item.get("selector"))
            for item in selector_trace
            if isinstance(item, dict)
            and item.get("operation") != "goto"
            and item.get("success")
            and item.get("selector")
        ]
        failed = [
            str(item.get("selector"))
            for item in selector_trace
            if isinstance(item, dict)
            and item.get("operation") != "goto"
            and item.get("success") is False
            and item.get("selector")
        ]
        if not local_success and not failed:
            error_text = _as_error_text(feedback)
            if any(word in error_text for word in ("selector", "element", "locator", "timeout")):
                failed = [str(item) for item in trace.get("selectors", []) if item]
        if (failed or working) and not localized_trace and local_success:
            self.patch_interaction_selectors(skill.name, failed, working)

        error_text = _as_error_text(feedback)
        repair_proposal_id = None
        if allow_repair and not local_success:
            # A failed invocation supplies a repair proposal, not a committed
            # update.  The proposal must later carry replay evidence that its
            # postcondition succeeds in the source-compatible context.
            repair_proposal_id = uuid.uuid4().hex
            proposals = metadata.setdefault("pending_repair_proposals", [])
            proposals.append(
                {
                    "proposal_id": repair_proposal_id,
                    "timestamp": feedback.get("timestamp"),
                    "failed_selectors": failed,
                    "working_selectors": working,
                    "selector_trace": localized_trace,
                    "error": error_text[:500],
                    "required_validation": "replay_postcondition_success",
                }
            )
            metadata["pending_repair_proposals"] = proposals[-20:]

        if (
            allow_repair
            and not local_success
            and any(cue in error_text for cue in ("modal", "dialog", "overlay", "popup", "pop-up"))
        ):
            branches = metadata.setdefault("recovery_branches", [])
            proposal = {
                "trigger": "unexpected_modal",
                "selectors": [
                    'get_by_role("button", name="Close")',
                    "button:has-text('Close')",
                    "[aria-label='Close']",
                ],
                "max_attempts": 1,
                "pending_validation": True,
            }
            if repair_proposal_id:
                proposal["repair_proposal_id"] = repair_proposal_id
            pending_branches = metadata.setdefault("pending_recovery_branches", [])
            if not any(
                isinstance(branch, dict) and branch.get("trigger") == "unexpected_modal"
                for branch in branches + pending_branches
            ):
                pending_branches.append(proposal)

        events = metadata.setdefault("events", [])
        events.append(
            {
                "type": "execution_success" if local_success else "execution_failure",
                "timestamp": feedback.get("timestamp"),
                "url": feedback.get("page_context", {}).get("url"),
                "final_task_label": feedback.get("final_task_label"),
            }
        )
        metadata["events"] = events[-50:]

    def patch_interaction_selectors(
        self,
        skill_name: str,
        failed_selectors: Sequence[str],
        working_selectors: Sequence[str] = (),
        *,
        remove_after: int = 3,
    ) -> Dict[str, List[str]]:
        """Apply the paper's local ordered-selector patch operator.

        Only metadata for selector sets is changed; the whole skill is never
        regenerated.  The runtime consumes these ordered sets through its page
        proxy on the next invocation.
        """

        skill = self.get_skill(skill_name)
        if skill is None:
            raise ValueError(f"Unknown skill '{skill_name}'")
        metadata = skill.metadata
        selector_sets = metadata.setdefault("selector_sets", {})
        failure_counts = metadata.setdefault("selector_failure_counts", {})

        for selector in dict.fromkeys(str(item) for item in failed_selectors if item):
            ordered = list(selector_sets.get(selector, []))
            if not ordered:
                ordered = [selector] + _fallback_selectors(selector)
            failure_counts[selector] = int(failure_counts.get(selector, 0)) + 1
            # Demote a failed selector.  Remove it after repeated local patches.
            ordered = [item for item in ordered if item != selector]
            if failure_counts[selector] < remove_after:
                ordered.append(selector)
            selector_sets[selector] = list(dict.fromkeys(ordered))[:8]
            for template in metadata.get("operation_templates", []):
                if selector in template.get("selectors", []):
                    template["selectors"] = list(selector_sets[selector])

        working_unique = list(dict.fromkeys(str(item) for item in working_selectors if item))
        if working_unique:
            target_sources = [
                str(item)
                for item in dict.fromkeys(failed_selectors)
                if str(item) in selector_sets
            ]
            for source_selector in target_sources:
                ordered = selector_sets[source_selector]
                combined = working_unique + [item for item in ordered if item not in working_unique]
                selector_sets[source_selector] = combined[:8]
                for template in metadata.get("operation_templates", []):
                    if source_selector in template.get("selectors", []):
                        template["selectors"] = list(selector_sets[source_selector])

        if failed_selectors:
            metadata["selector_patch_rounds"] = int(metadata.get("selector_patch_rounds", 0)) + 1
        return selector_sets

    def patch_interaction_selector_trace(
        self,
        skill_name: str,
        selector_trace: Sequence[Dict[str, Any]],
        *,
        remove_after: int = 3,
    ) -> Dict[str, List[str]]:
        """Patch each operation's selector set from its localized runtime trace."""

        skill = self.get_skill(skill_name)
        if skill is None:
            raise ValueError(f"Unknown skill '{skill_name}'")
        metadata = skill.metadata
        selector_sets = metadata.setdefault("selector_sets", {})
        candidate_counts = metadata.setdefault("selector_candidate_failure_counts", {})
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for event in selector_trace:
            source = str(event.get("source_selector", ""))
            if source:
                grouped.setdefault(source, []).append(dict(event))

        patched_failure = False
        for source, events in grouped.items():
            ordered = list(selector_sets.get(source, [])) or [source] + _fallback_selectors(source)
            counts_for_source = candidate_counts.setdefault(source, {})
            failed_candidates = [
                str(event["selector"])
                for event in events
                if event.get("success") is False
            ]
            working_candidates = [
                str(event["selector"])
                for event in events
                if event.get("success") is True
            ]
            for candidate in dict.fromkeys(failed_candidates):
                patched_failure = True
                counts_for_source[candidate] = int(counts_for_source.get(candidate, 0)) + 1
                ordered = [item for item in ordered if item != candidate]
                if counts_for_source[candidate] < remove_after:
                    ordered.append(candidate)
            for candidate in reversed(list(dict.fromkeys(working_candidates))):
                ordered = [candidate] + [item for item in ordered if item != candidate]
            selector_sets[source] = list(dict.fromkeys(ordered))[:8]
            for template in metadata.get("operation_templates", []):
                if (
                    template.get("source_selector") == source
                    or source in template.get("selectors", [])
                ):
                    template["source_selector"] = source
                    template["selectors"] = list(selector_sets[source])
        if patched_failure:
            metadata["selector_patch_rounds"] = int(metadata.get("selector_patch_rounds", 0)) + 1
        return selector_sets

    def commit_validated_interaction_repair(
        self,
        skill_name: str,
        proposal_index: int,
        *,
        replay_postcondition_success: bool,
    ) -> bool:
        """Commit one local repair only after a source-compatible replay.

        This is the persistent counterpart of the paper's interaction
        validator. Runtime failures create proposals; an offline replayer calls
        this method after verifying the patched procedure's local postcondition.
        """

        skill = self.get_skill(skill_name)
        if skill is None:
            raise ValueError(f"Unknown skill '{skill_name}'")
        metadata = skill.metadata
        proposals = list(metadata.get("pending_repair_proposals", []))
        if proposal_index < 0 or proposal_index >= len(proposals):
            raise IndexError("Unknown pending interaction repair proposal")
        if not replay_postcondition_success:
            return False
        proposal = proposals.pop(proposal_index)
        trace = proposal.get("selector_trace", [])
        if trace:
            self.patch_interaction_selector_trace(skill_name, trace)
        else:
            self.patch_interaction_selectors(
                skill_name,
                proposal.get("failed_selectors", []),
                proposal.get("working_selectors", []),
            )
        metadata["pending_repair_proposals"] = proposals
        metadata.setdefault("validated_repair_count", 0)
        metadata["validated_repair_count"] += 1
        proposal_id = proposal.get("proposal_id")
        pending_branches = list(metadata.get("pending_recovery_branches", []))
        approved_branches = []
        remaining_branches = []
        for branch in pending_branches:
            if proposal_id and branch.get("repair_proposal_id") == proposal_id:
                approved = dict(branch)
                approved.pop("pending_validation", None)
                approved["validated_at"] = datetime.now(timezone.utc).isoformat()
                approved_branches.append(approved)
            else:
                remaining_branches.append(branch)
        if approved_branches:
            branches = metadata.setdefault("recovery_branches", [])
            for branch in approved_branches:
                trigger = branch.get("trigger")
                if not any(
                    isinstance(existing, dict) and existing.get("trigger") == trigger
                    for existing in branches
                ):
                    branches.append(branch)
        metadata["pending_recovery_branches"] = remaining_branches
        contract = metadata.get("interaction_contract")
        if isinstance(contract, dict):
            contract["Rec_k"] = list(metadata.get("recovery_branches", []))
        self._persist_skill_metadata()
        return True

    def _apply_reasoning_feedback(self, feedback: Dict[str, Any]) -> None:
        skill_id = str(feedback["skill_name"])
        for index, raw_lesson in enumerate(self.external_task_lessons):
            lesson = normalize_reasoning_skill(raw_lesson, site_hint=self.site_name or "")
            if str(lesson.get("skill_id")) != skill_id:
                continue
            stats = lesson["statistics"]
            stats["N"] = int(stats.get("N", 0)) + 1
            stats["S"] = int(stats.get("S", 0)) + int(feedback.get("final_task_label") == 1)
            smoothing = float(stats.get("lambda", 1.0))
            stats["rho"] = (stats["S"] + smoothing) / (stats["N"] + 2 * smoothing)
            self.external_task_lessons[index] = lesson
            break

    def consolidate_reasoning_skills(
        self,
        *,
        similarity_threshold: float = 0.82,
        min_usage: int = 3,
        min_utility: float = 0.2,
        prune_unused_after_rounds: int = 3,
        merge: bool = True,
        prune: bool = True,
        respect_task_family: bool = True,
    ) -> Dict[str, int]:
        """Maintain reasoning rules with independently switchable operators.

        ``merge`` and ``prune`` default to the historical full-maintenance
        behaviour.  Keeping them separate is important for the RQ4c
        ablation: merge-only, prune-only, and full maintenance must execute
        genuinely different code paths rather than merely receive different
        experiment labels.
        """

        normalized = [
            normalize_reasoning_skill(item, site_hint=self.site_name or "")
            for item in self.external_task_lessons
        ]
        if merge:
            groups: List[List[Dict[str, Any]]] = []
            for lesson in sorted(
                normalized,
                key=lambda item: (
                    float(item["statistics"].get("rho", 0.5)),
                    int(item["statistics"].get("N", 0)),
                ),
                reverse=True,
            ):
                lesson_family = str(
                    lesson["scenario_descriptor"]["W_k"].get("task_family")
                    or lesson.get("task_family", "")
                ).strip()
                group = next(
                    (
                        candidates
                        for candidates in groups
                        if (
                            not respect_task_family
                            or not lesson_family
                            or not str(
                                candidates[0]["scenario_descriptor"]["W_k"].get("task_family")
                                or candidates[0].get("task_family", "")
                            ).strip()
                            or str(
                                candidates[0]["scenario_descriptor"]["W_k"].get("task_family")
                                or candidates[0].get("task_family", "")
                            ).strip() == lesson_family
                        )
                        if reasoning_similarity(candidates[0], lesson) >= similarity_threshold
                    ),
                    None,
                )
                if group is None:
                    groups.append([lesson])
                else:
                    group.append(lesson)

            kept = [
                (
                    merge_reasoning_group(group, site_hint=self.site_name or "")
                    if len(group) > 1
                    else group[0]
                )
                for group in groups
            ]
            merged_count = sum(max(0, len(group) - 1) for group in groups)
        else:
            kept = normalized
            merged_count = 0

        before_prune = len(kept)
        if prune:
            kept = [
                item
                for item in kept
                if not (
                    int(item["statistics"].get("N", 0)) < min_usage
                    and float(item["statistics"].get("rho", 0.5)) < min_utility
                )
            ]
        self.external_task_lessons = kept
        self._persist_reasoning_library()
        return {
            "merged": merged_count,
            "pruned": before_prune - len(kept),
            "remaining": len(kept),
        }

    def prune_invalid_interaction_skills(
        self,
        *,
        min_patch_rounds: int = 3,
        max_utility: float = 0.2,
    ) -> List[str]:
        """Remove skills that still fail after repeated selector-level patches."""

        removed = []
        removed_metadata: Dict[str, Dict[str, Any]] = {}
        for name, skill in list(self.skills.items()):
            metadata = skill.metadata
            if (
                int(metadata.get("selector_patch_rounds", 0)) >= min_patch_rounds
                and float(metadata.get("utility_score", 0.5)) <= max_utility
            ):
                metadata["active"] = False
                metadata["removed_reason"] = "repeated_selector_patch_failure"
                removed.append(name)
                removed_metadata[name] = dict(metadata)
                del self.skills[name]
        self._persist_skill_metadata(extra_metadata=removed_metadata)
        return removed

    def _persist_skill_metadata(
        self,
        *,
        extra_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        if not self.metadata_path:
            return
        try:
            payload = {}
            if self.metadata_path.exists():
                payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            functions = payload.setdefault("functions", {})
            for name, skill in self.skills.items():
                functions.setdefault(name, {}).update(_safe_serialize(skill.metadata))
            for name, updates in (extra_metadata or {}).items():
                functions.setdefault(name, {}).update(updates)
            self.metadata_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
                encoding="utf-8",
            )
        except (OSError, ValueError) as exc:
            logger.warning("Failed to update skill metadata: %s", exc)

    def _persist_reasoning_library(self) -> None:
        if not self.task_lessons_path:
            return
        try:
            self.task_lessons_path.write_text(
                json.dumps(self.external_task_lessons, indent=2, ensure_ascii=False, default=str)
                + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Failed to update reasoning skill library: %s", exc)

    def __len__(self):
        return len(self.skills)

    def get_failure_experiences(self) -> List[Dict[str, Any]]:
        """获取所有任务级经验

        仅返回从外部任务级经验文件加载的经验（task_level_lessons_v3.json）
        这是当前的权威来源，避免重复和冲突
        """
        experiences = []

        # 只使用外部加载的任务级经验
        for lesson_item in self.external_task_lessons:
            skill_name = lesson_item.get("skill_name", "unknown")
            task_lessons = lesson_item.get("task_lessons", {})
            experiences.append({
                "skill_name": skill_name,
                "test_count": 0,  # 外部经验没有 test_count
                "lessons": task_lessons,
                "source": "external"  # 标记为外部来源
            })

        logger.info(f"Loaded {len(experiences)} failure experiences from external task lessons (task_level_lessons_v3.json)")

        return experiences

    def match_failure_experiences(
        self,
        objective: str,
        start_url: str,
        model: str = "gpt-4.1",
        api_base: str = None
    ) -> List[Dict[str, Any]]:
        """
        使用 LLM 匹配当前任务与失败经验

        Args:
            objective: 任务目标描述
            start_url: 任务起始 URL
            model: LLM 模型名称
            api_base: API base URL

        Returns:
            匹配的失败经验列表
        """
        experiences = self.get_failure_experiences()
        print(f"📚 候选失败经验总数: {len(experiences)}")

        if not experiences:
            print(f"⚠️  没有可用的失败经验（reasoning_tips.json 未加载或为空）")
            return []

        # 构建经验摘要供匹配
        exp_summaries = []
        for i, exp in enumerate(experiences):
            lessons = exp["lessons"]

            # 兼容两种格式：task_lessons 和 failure_lessons
            if "failure_summary" in lessons:
                # failure_lessons 格式
                summary = {
                    "index": i,
                    "skill_name": exp["skill_name"],
                    "failure_summary": lessons.get("failure_summary", ""),
                    "common_errors": lessons.get("common_errors", []),
                    "root_causes": lessons.get("root_causes", []),
                    "avoid_tips": lessons.get("avoid_tips", [])
                }
            else:
                # task_lessons 格式（原有格式）
                summary = {
                    "index": i,
                    "skill_name": exp["skill_name"],
                    "scenario_keywords": lessons.get("scenario_keywords", []),
                    "scenario_description": lessons.get("scenario_description", lessons.get("task_pattern", "")),
                    "url_patterns": lessons.get("url_patterns", []),
                    "task_pattern": lessons.get("task_pattern", "")
                }
            exp_summaries.append(summary)

        print(f"🔍 调用 LLM ({model}) 进行场景匹配...")

        # 调用 LLM 进行匹配
        import json
        import os
        try:
            from openai import OpenAI
        except ImportError:
            logger.warning("OpenAI library not available, skipping failure experience matching")
            print(f"⚠️  OpenAI 库不可用，跳过场景匹配")
            return []

        prompt = f"""You are a task matching expert. Determine which failure experiences are relevant to the current task.

# Current Task
Objective: {objective}
Start URL: {start_url}

# Available Failure Experiences
{json.dumps(exp_summaries, indent=2, ensure_ascii=False)}

# Instructions
1. Analyze the current task objective and URL
2. For each failure experience, check:
   - For failure_lessons format: failure_summary, common_errors, root_causes match the task
   - For task_lessons format: scenario_description, scenario_keywords, url_patterns match the task
3. Select experiences that are HIGHLY relevant to the current task (same type of operation, similar errors)
4. Be conservative - only select if there's a clear match with task type and potential failure patterns

Output a JSON array of matching indices, e.g., [0, 2] or [] if no match.
Output ONLY the JSON array, nothing else.
"""

        try:
            base_url = api_base or os.environ.get("LLM_API_BASE_URL", "http://localhost:4141/v1")
            api_key = os.environ.get("LLM_API_KEY", "dummy")

            client = OpenAI(base_url=base_url, api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )

            result_text = response.choices[0].message.content.strip()
            print(f"🤖 LLM 返回: {result_text}")

            # 解析 JSON 数组
            if result_text.startswith("["):
                matched_indices = json.loads(result_text)
            else:
                # 尝试从文本中提取 JSON
                import re
                match = re.search(r'\[[\d,\s]*\]', result_text)
                if match:
                    matched_indices = json.loads(match.group())
                else:
                    matched_indices = []

            # 返回匹配的经验
            matched_experiences = []
            for idx in matched_indices:
                if 0 <= idx < len(experiences):
                    matched_experiences.append(experiences[idx])

            logger.info(f"Matched {len(matched_experiences)} failure experiences for task")
            return matched_experiences

        except Exception as e:
            logger.warning(f"Failed to match failure experiences: {e}")
            return []

    def format_failure_experiences(self, experiences: List[Dict[str, Any]]) -> str:
        """格式化失败经验为 prompt 文本"""
        if not experiences:
            return ""

        lines = []
        lines.append("# ⚠️ RELEVANT FAILURE EXPERIENCES")
        lines.append("")
        lines.append("The following failure patterns are HIGHLY RELEVANT to your current task.")
        lines.append("CAREFULLY read and apply these lessons to AVOID making the same mistakes!")
        lines.append("")

        for i, exp in enumerate(experiences, 1):
            lessons = exp["lessons"]
            lines.append(f"## Experience {i}: {exp['skill_name']}")

            # 兼容两种格式：failure_lessons 和 task_lessons
            if "failure_summary" in lessons:
                # failure_lessons 格式
                if lessons.get("failure_summary"):
                    lines.append(f"📋 Failure Summary: {lessons['failure_summary']}")

                if "common_errors" in lessons and lessons["common_errors"]:
                    lines.append("⚠️ COMMON ERRORS (avoid these):")
                    for error in lessons["common_errors"][:3]:
                        lines.append(f"   • {error}")

                if "root_causes" in lessons and lessons["root_causes"]:
                    lines.append("🔍 ROOT CAUSES:")
                    for cause in lessons["root_causes"][:3]:
                        lines.append(f"   • {cause}")

                if "avoid_tips" in lessons and lessons["avoid_tips"]:
                    lines.append("✅ RECOMMENDED APPROACH:")
                    for tip in lessons["avoid_tips"][:3]:
                        lines.append(f"   • {tip}")

                if "anti_patterns" in lessons and lessons["anti_patterns"]:
                    lines.append("🚨 ANTI-PATTERNS (watch out):")
                    for pattern in lessons["anti_patterns"][:3]:
                        lines.append(f"   • {pattern}")
            else:
                # task_lessons 格式（原有格式）
                if "scenario_description" in lessons:
                    lines.append(f"📋 Scenario: {lessons['scenario_description']}")
                elif "task_pattern" in lessons:
                    lines.append(f"📋 Task Pattern: {lessons['task_pattern']}")

                if "why_failed" in lessons:
                    lines.append(f"⚠️ Why Failed: {lessons['why_failed']}")

                if "better_strategy" in lessons and lessons["better_strategy"]:
                    lines.append("✅ RECOMMENDED STRATEGY:")
                    for strategy in lessons["better_strategy"][:3]:
                        lines.append(f"   • {strategy}")

                if "failure_signals" in lessons and lessons["failure_signals"]:
                    lines.append("🚨 FAILURE SIGNALS (watch out for these):")
                    for signal in lessons["failure_signals"][:2]:
                        lines.append(f"   • {signal}")

                if "task_level_tips" in lessons and lessons["task_level_tips"]:
                    lines.append("💡 TIPS:")
                    for tip in lessons["task_level_tips"][:2]:
                        lines.append(f"   • {tip}")

            lines.append("")

        return "\n".join(lines)

    def record_skill_failure(self, failure_data: Dict[str, Any]) -> None:
        """
        Record a skill execution failure to disk.

        Writes one JSON line to <site_dir>/skill_failure_log.jsonl and updates
        the skill's 'events' list in the metadata JSON with a summary.

        Args:
            failure_data: Dict containing skill_name, error, traceback, url,
                         selectors, skill_args, page_state_snippet, task_id, objective, etc.
        """
        from datetime import datetime

        skill_name = failure_data.get('skill_name', 'unknown')
        timestamp = datetime.now().isoformat()

        # Build the failure record
        record = {
            'timestamp': timestamp,
            'skill_name': skill_name,
            'error': failure_data.get('error', failure_data.get('failure_reason', 'unknown')),
            'traceback': failure_data.get('traceback', ''),
            'url': failure_data.get('url', 'unknown'),
            'selectors': failure_data.get('selectors', []),
            'skill_args': _safe_serialize(failure_data.get('skill_args', {})),
            'page_state_snippet': failure_data.get('page_state_snippet', ''),
            'task_id': failure_data.get('task_id'),
            'objective': failure_data.get('objective'),
        }

        # 1) Append to JSONL file
        if self.site_dir:
            jsonl_path = self.site_dir / 'skill_failure_log.jsonl'
            try:
                with open(jsonl_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
                logger.info(f"Recorded failure for skill '{skill_name}' to {jsonl_path}")
            except Exception as e:
                logger.warning(f"Failed to write failure log: {e}")

        # 2) Update metadata JSON's events field with a summary
        if self.metadata_path and self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                functions = metadata.get('functions', {})
                if skill_name in functions:
                    events = functions[skill_name].setdefault('events', [])
                    event_summary = {
                        'type': 'execution_failure',
                        'timestamp': timestamp,
                        'error_brief': str(record['error'])[:200],
                        'url': record['url'],
                    }
                    events.append(event_summary)

                    # Keep events list bounded (last 50 entries)
                    if len(events) > 50:
                        functions[skill_name]['events'] = events[-50:]

                    with open(self.metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                    logger.info(f"Updated events for skill '{skill_name}' in {self.metadata_path}")
            except Exception as e:
                logger.warning(f"Failed to update metadata events: {e}")

    def __repr__(self):
        site = f"site={self.site_name}" if self.site_name else ""
        return f"<SkillRegistry {len(self.skills)} skills {site}>"


def _safe_serialize(obj):
    """Make an object JSON-serializable by converting non-serializable values to strings."""
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return str(obj)


def copy_dict(value: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-copy a JSON-like feedback record without sharing nested state."""

    import copy

    return copy.deepcopy(value)


def _list_values(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        stripped = value.strip()
        return (
            (stripped.startswith("<") and stripped.endswith(">"))
            or "<actual_" in stripped
            or stripped.startswith("${")
        )
    if isinstance(value, (list, tuple, set)):
        return any(_contains_placeholder(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_placeholder(item) for item in value.values())
    return False


def _as_error_text(feedback: Dict[str, Any]) -> str:
    outcome = feedback.get("local_outcome", {}) or {}
    execution = feedback.get("execution_log", {}) or {}
    return " ".join(
        str(value).lower()
        for value in (
            outcome.get("error"),
            outcome.get("failure_reason"),
            execution.get("exception"),
        )
        if value
    )


def _fallback_selectors(selector: str) -> List[str]:
    """Generate small, local selector alternatives for an ordered set."""

    fallbacks: List[str] = []
    text_match = __import__("re").search(r":has-text\([\"'](.+?)[\"']\)", selector)
    if text_match:
        label = text_match.group(1)
        fallbacks.extend(
            [
                f"text={label}",
                f"[aria-label='{label}']",
                f"[title='{label}']",
            ]
        )
    id_match = __import__("re").search(r"#([A-Za-z][\w-]*)", selector)
    if id_match:
        fallbacks.append(f"[id='{id_match.group(1)}']")
    name_match = __import__("re").search(r"\[name=[\"']?([^\] \"']+)", selector)
    if name_match:
        fallbacks.append(f"[name='{name_match.group(1)}']")
    role_match = __import__("re").search(
        r'(?:get_by_role\(["\']([^"\']+)["\']|role=([^,]+))(?:,\s*name=["\']([^"\']+)["\'])?',
        selector,
    )
    if role_match:
        role = role_match.group(1) or role_match.group(2)
        label = role_match.group(3)
        if label:
            fallbacks.extend([f'[role="{role}"][aria-label="{label}"]', f'text={label}'])
        else:
            fallbacks.append(f'[role="{role}"]')
    label_match = __import__("re").search(r'get_by_label\(["\']([^"\']+)["\']', selector)
    if label_match:
        label = label_match.group(1)
        fallbacks.extend([f'[aria-label="{label}"]', f'label:has-text("{label}")'])
    text_locator_match = __import__("re").search(r'get_by_text\(["\']([^"\']+)["\']', selector)
    if text_locator_match:
        text = text_locator_match.group(1)
        fallbacks.extend([f'text={text}', f':text-is("{text}")'])
    return list(dict.fromkeys(item for item in fallbacks if item != selector))[:5]


# 便捷函数
def load_skills_from_site(site_name: str, skills_base_dir: Path = None,
                          skill_file: str = None, skill_metadata: str = None) -> SkillRegistry:
    """
    便捷函数：从站点名称加载技能库

    Args:
        site_name: 站点名称 (如 'reddit', 'shopping')
        skills_base_dir: skills/ 目录路径，默认为当前项目的 skills/
        skill_file: 技能文件名 (如 'operation_skills.py')
        skill_metadata: 元数据文件名 (如 'operation_skills.json')

    Returns:
        SkillRegistry 实例
    """
    if skills_base_dir is None:
        skills_base_dir = Path(__file__).parent.parent / "skills"

    site_dir = skills_base_dir / site_name
    if not site_dir.exists():
        raise FileNotFoundError(f"站点目录不存在: {site_dir}")

    registry = SkillRegistry()
    registry.load_from_site_dir(site_dir, skill_file=skill_file, skill_metadata=skill_metadata)
    return registry

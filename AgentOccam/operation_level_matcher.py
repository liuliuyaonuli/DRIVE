"""
操作级技能匹配器：根据任务级经验过滤相关的操作级技能

核心功能：
1. 从任务级经验中提取 lesson ID
2. 根据 lesson ID 获取相关的操作级技能
3. 按相关性排序技能
"""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class OperationLevelMatcher:
    """操作级技能匹配器：根据任务级经验过滤相关技能"""

    def __init__(self, skill_registry):
        """
        初始化匹配器

        Args:
            skill_registry: SkillRegistry实例
        """
        self.skill_registry = skill_registry

    def extract_matched_task_families(
        self,
        matched_lessons: List[Dict]
    ) -> List[int]:
        """
        从匹配的lessons中提取任务族ID

        Args:
            matched_lessons: 任务级经验列表（包含兼容字段 cluster_id）

        Returns:
            lesson ID 列表
        """
        task_family_ids = set()
        for lesson in matched_lessons:
            cluster_id = lesson.get("cluster_id")
            if cluster_id is not None:
                task_family_ids.add(cluster_id)

        return sorted(list(task_family_ids))

    def filter_operation_skills(
        self,
        task_intent: str,
        matched_lessons: List[Dict],
        top_k: int = 5,
        only_tested: bool = True,
        only_successful: bool = False
    ) -> Dict[str, Any]:
        """
        根据任务级经验过滤操作级技能

        Args:
            task_intent: 当前任务意图
            matched_lessons: 匹配的任务级经验
            top_k: 返回的最大技能数
            only_tested: 是否只返回已验证的技能（test_count > 0）
            only_successful: 是否只返回测试成功的技能（success_count > 0）
                            如果为 True，则 only_tested 被忽略

        Returns:
            包含以下字段的字典：
            - filtered_skills: List[Skill] - 过滤后的技能
            - related_task_families: List[int] - 相关的任务族ID
            - related_lessons: List[Dict] - 相关的任务级经验
            - skill_count: int - 总的候选技能数
            - tested_count: int - 已验证技能数
            - untested_count: int - 未验证技能数
            - successful_count: int - 测试成功技能数
            - matching_info: str - 匹配信息
        """
        if not matched_lessons:
            return {
                "filtered_skills": [],
                "related_task_families": [],
                "related_lessons": [],
                "skill_count": 0,
                "tested_count": 0,
                "untested_count": 0,
                "successful_count": 0,
                "matching_info": "No matched lessons found"
            }

        # 提取 lesson ID（字段名 cluster_id 为兼容旧 SkillRegistry schema）
        task_family_ids = self.extract_matched_task_families(matched_lessons)

        # 获取这些任务族对应的技能
        candidate_skills = self.skill_registry.get_skills_by_task_families(task_family_ids)

        # 分离技能：成功、已测试、未测试
        successful_skills = []
        tested_skills = []
        untested_skills = []

        for skill in candidate_skills:
            test_count = skill.metadata.get("test_count", 0)
            success_count = skill.metadata.get("success_count", 0)

            if success_count > 0:
                successful_skills.append(skill)
            elif test_count > 0:
                tested_skills.append(skill)
            else:
                untested_skills.append(skill)

        # 按质量排序：success_count 高的优先，然后是 version，然后是 test_count
        def skill_score(skill):
            success_count = skill.metadata.get("success_count", 0)
            test_count = skill.metadata.get("test_count", 0)
            version = skill.metadata.get("version", 0)
            refs_count = len(skill.metadata.get("references", []))

            # 返回优先级元组
            return (
                success_count,
                version,
                test_count,
                refs_count
            )

        successful_skills.sort(key=skill_score, reverse=True)
        tested_skills.sort(key=skill_score, reverse=True)
        untested_skills.sort(key=skill_score, reverse=True)

        # 根据参数决定返回什么
        if only_successful:
            # 最严格：只返回测试成功的技能 (success_count > 0)
            filtered_skills = successful_skills[:top_k]
            matching_info = (
                f"Found {len(successful_skills)} successful skills "
                f"(excluding {len(tested_skills)} tested-only and {len(untested_skills)} untested) from "
                f"{len(task_family_ids)} task families"
            )
        elif only_tested:
            # 中等：返回已测试的技能（包括成功的）
            all_tested = successful_skills + tested_skills
            filtered_skills = all_tested[:top_k]
            matching_info = (
                f"Found {len(successful_skills)} successful + {len(tested_skills)} tested skills "
                f"(excluding {len(untested_skills)} untested) from "
                f"{len(task_family_ids)} task families"
            )
        else:
            # 宽松：优先成功，然后已验证，不足时补充未验证
            all_tested = successful_skills + tested_skills
            filtered_skills = all_tested[:top_k]
            if len(filtered_skills) < top_k:
                needed = top_k - len(filtered_skills)
                filtered_skills.extend(untested_skills[:needed])

            matching_info = (
                f"Found {len(successful_skills)} successful + {len(tested_skills)} tested + "
                f"{len(untested_skills)} untested skills from "
                f"{len(task_family_ids)} task families"
            )

        logger.info(matching_info)

        return {
            "filtered_skills": filtered_skills,
            "related_task_families": task_family_ids,
            "related_lessons": matched_lessons,
            "skill_count": len(candidate_skills),
            "tested_count": len(tested_skills),
            "untested_count": len(untested_skills),
            "successful_count": len(successful_skills),
            "matching_info": matching_info
        }

    def get_skill_details_for_lesson(
        self,
        lesson: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        获取某个lesson相关的技能详细信息

        Args:
            lesson: 单个任务级经验

        Returns:
            包含技能详情的字典
        """
        skill_name = lesson.get("skill_name")
        if not skill_name:
            return {}

        skill = self.skill_registry.get_skill(skill_name)
        if not skill:
            return {}

        return {
            "skill_name": skill.name,
            "skill_summary": skill.get_summary(),
            "is_verified": skill.metadata.get("test_count", 0) > 0,
            "test_count": skill.metadata.get("test_count", 0),
            "version": skill.metadata.get("version", 0),
            "task_family": lesson.get("task_family"),
            "lesson_id": lesson.get("cluster_id")
        }

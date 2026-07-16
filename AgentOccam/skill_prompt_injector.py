"""
Skill Prompt Injector: Dynamically generates prompts based on filtered skills
and injects them into the Agent's decision process.

Core functions:
1. Generate skill overview
2. Generate task-level experience guidance
3. Generate skill usage guide
4. Generate skill invocation strategy
"""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SkillPromptInjector:
    """Module for dynamically injecting skill-related prompts"""

    def __init__(self, operation_level_matcher=None):
        """
        Initialize the injector

        Args:
            operation_level_matcher: OperationLevelMatcher instance (optional)
        """
        self.operation_level_matcher = operation_level_matcher

    def generate_skill_assistance_prompt(
        self,
        filtered_skills: List,
        related_lessons: List[Dict[str, Any]],
        task_intent: str,
        skill_registry=None,
        skill_already_used: bool = False,
        skill_succeeded: bool = False
    ) -> str:
        """
        Generate complete skill assistance prompt for the current task

        Args:
            filtered_skills: List of filtered relevant skills
            related_lessons: Related task-level experiences
            task_intent: Current task intent/goal
            skill_registry: Skill registry (optional)
            skill_already_used: Whether a skill has been attempted
            skill_succeeded: Whether the skill executed successfully

        Returns:
            Complete skill assistance prompt, ready to be inserted into main prompt
        """
        # DRIVE activates at most one skill per level.  A reasoning skill can
        # still guide primitive actions when no interaction skill is applicable.
        filtered_skills = list(filtered_skills[:1])
        related_lessons = list(related_lessons[:1])
        if not filtered_skills and not related_lessons:
            return ""

        prompt_parts = []

        # Part 1: Skill overview
        if filtered_skills:
            prompt_parts.append(self._generate_skill_overview(
                filtered_skills,
                task_intent
            ))

        # Part 2: Task-level experience guidance
        if related_lessons:
            prompt_parts.append(self._generate_lessons_guidance(related_lessons))

        # Part 3: Skill usage guide
        if filtered_skills:
            prompt_parts.append(self._generate_skill_usage_guide(
                filtered_skills,
                skill_registry
            ))

        # Part 4: Invocation strategy (pass skill usage status)
        prompt_parts.append(self._generate_invocation_strategy(
            filtered_skills,
            related_lessons,
            skill_already_used=skill_already_used,
            skill_succeeded=skill_succeeded
        ))

        return "\n".join(filter(None, prompt_parts))

    def _generate_skill_overview(
        self,
        filtered_skills: List,
        task_intent: str
    ) -> str:
        """
        Generate skill overview section

        Tell Agent which skills can be used to complete this task
        """
        lines = []
        lines.append("=" * 70)
        lines.append("DRIVE INTERACTION SKILL (singleton activation)")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"Task: {task_intent}")
        lines.append("")
        lines.append("The following executable procedure is compatible with the current task-page scene:")
        lines.append("")

        for i, skill in enumerate(filtered_skills, 1):
            summary = skill.get_summary()
            test_count = skill.metadata.get("test_count", 0)

            # Add quality indicator
            quality_indicator = "✓ verified" if test_count > 0 else "○ unverified"

            lines.append(f"{i}. {skill.name} [{quality_indicator}]")
            lines.append(f"   {summary}")
            lines.append("")

        lines.append("Invoke only this interaction skill, after checking its arguments and current-page preconditions.")
        lines.append("Call format: use_skill skill_name param=value ...")
        lines.append("If it is not applicable or its required arguments are unknown, use primitive actions instead.")
        lines.append("")

        return "\n".join(lines)

    def _generate_lessons_guidance(
        self,
        related_lessons: List[Dict[str, Any]]
    ) -> str:
        """
        Generate task-level experience guidance

        Extract key suggestions from task_lessons
        """
        lines = []
        lines.append("=" * 70)
        lines.append("DRIVE REASONING SKILL <M, B, V> (singleton activation)")
        lines.append("=" * 70)
        lines.append("")

        for lesson in related_lessons[:1]:
            task_lessons = lesson.get("task_lessons", {})
            task_family = lesson.get("task_family", "Unknown")
            skill_name = lesson.get("skill_name", "Unknown")
            reasoning_skill = lesson.get("reasoning_skill", {})

            lines.append(f"[{task_family}] - Related skill: {skill_name}")
            lines.append("")

            if reasoning_skill:
                lines.append(f"M — prior mistake pattern: {reasoning_skill.get('M', '')}")
                lines.append(f"B — corrected behavior: {reasoning_skill.get('B', '')}")
                lines.append(f"V — verification: {reasoning_skill.get('V', '')}")
                lines.append("")
                continue

            if lesson.get("lesson"):
                lines.append("Recommended lesson:")
                lines.append(f"  {str(lesson['lesson'])[:300]}")
                lines.append("")

            # Compatibility fallback for callers that did not normalize the
            # public legacy artifact through SkillRegistry.
            lines.append(f"M — prior mistake pattern: {task_lessons.get('why_failed', '')}")
            lines.append(f"B — corrected behavior: {' '.join(task_lessons.get('better_strategy', []))}")
            lines.append(f"V — verification: {' '.join(task_lessons.get('verification_strategy', []))}")
            lines.append("")

        return "\n".join(lines)

    def _generate_skill_usage_guide(
        self,
        filtered_skills: List,
        skill_registry=None
    ) -> str:
        """
        Generate skill usage guide (SkillWeaver style)

        Includes complete function signature, docstring and call examples
        """
        lines = []
        lines.append("=" * 70)
        lines.append("SKILL LIBRARY (Knowledge Base Functions)")
        lines.append("=" * 70)
        lines.append("")

        # SkillWeaver style core guidance
        lines.append("You have access to the following skill functions. Use them whenever possible")
        lines.append("to complete the task efficiently. Before calling a skill, carefully check:")
        lines.append("")
        lines.append("  1. **Prerequisites**: Read the skill's docstring to understand required UI state")
        lines.append("  2. **Parameters**: Ensure you have all required parameter values (don't guess!)")
        lines.append("  3. **Call Format**: use_skill skill_name param1=value1 param2=value2 ...")
        lines.append("")
        lines.append("Parameter format examples:")
        lines.append("  • String:  username=\"john_doe\"")
        lines.append("  • List:    forum_names=[\"news\", \"sports\"]")
        lines.append("  • Boolean: flag=True")
        lines.append("  • Integer: count=10")
        lines.append("")
        lines.append("-" * 70)

        # Provide SkillWeaver style complete description for each skill
        for i, skill in enumerate(filtered_skills[:8], 1):
            lines.append("")
            # Use SkillWeaver format skill description
            lines.append(skill.get_skillweaver_format())

            # Call example
            call_example = skill.get_call_example()
            lines.append(f"Call example: {call_example}")
            lines.append("")
            lines.append("-" * 70)

        return "\n".join(lines)

    def _generate_invocation_strategy(
        self,
        filtered_skills: List,
        related_lessons: List[Dict[str, Any]],
        skill_already_used: bool = False,
        skill_succeeded: bool = False
    ) -> str:
        """
        Generate skill invocation strategy (SkillWeaver style)

        Let Agent reason about whether to use skills
        """
        lines = []
        lines.append("=" * 70)
        lines.append("SKILL USAGE GUIDELINES")
        lines.append("=" * 70)
        lines.append("")

        # Retrieval is refreshed at every environment step.  Invocation status
        # from a previous step must not disable future, newly applicable skills.
        if filtered_skills:
            # SkillWeaver style: let Agent reason on its own
            lines.append("**BEFORE CALLING A SKILL**, carefully consider:")
            lines.append("")
            lines.append("If an available skill can fully complete the current task objective, prefer a single use_skill call over primitive click/type actions.")
            lines.append("For ordered tasks such as \"click ONE, then TWO\", use a sequence skill instead of repeating single clicks when the DOM does not visibly change.")
            lines.append("")
            lines.append("1. **Check Skill Scope**: Read the skill's docstring to understand what it does.")
            lines.append("   • What is the skill's PURPOSE? (e.g., navigation, data retrieval, action)")
            lines.append("   • Does it FULLY complete your task, or only PART of it?")
            lines.append("   • If it only navigates to a page, you'll need additional actions after!")
            lines.append("")
            lines.append("2. **Verify Prerequisites**:")
            lines.append("   • You have ALL required parameters with ACTUAL values (not placeholders)")
            lines.append("   • The current UI state matches the skill's prerequisites")
            lines.append("   • If a parameter is unknown, navigate/search to find it first")
            lines.append("")
            lines.append("3. **Plan Post-Skill Actions**:")
            lines.append("   • If skill only navigates → plan to extract info or perform actions after")
            lines.append("   • If skill returns data → check if data answers the task question")
            lines.append("   • If skill performs action → verify the action achieved task goal")
            lines.append("")

            # Show skill prerequisites
            if filtered_skills:
                best_skill = filtered_skills[0]
                lines.append(f"For skill `{best_skill.name}`:")
                lines.append(f"  Signature: {best_skill.get_function_signature()}")

                # Extract prerequisites from docstring
                doc = best_skill.doc or ""
                if "Initial UI State" in doc or "required" in doc.lower():
                    lines.append(f"  Prerequisites: Check the skill's docstring above")

                args_spec = best_skill.get_args_spec()
                if args_spec:
                    lines.append(f"  Required parameters:")
                    for arg_name, arg_type in args_spec.items():
                        lines.append(f"    - {arg_name}: {arg_type}")
                lines.append("")

        elif skill_succeeded:
            # Skill executed successfully: guide to evaluate completion
            lines.append("✅ Skill executed successfully!")
            lines.append("")
            lines.append("**CRITICAL**: Now you MUST compare the skill result with your TASK OBJECTIVE:")
            lines.append("")
            lines.append("📋 TASK COMPLETION CHECKLIST:")
            lines.append("  1. Review your original task objective")
            lines.append("  2. Check what the skill actually accomplished (see SKILL EXECUTION RESULT)")
            lines.append("  3. Ask yourself: \"Does this FULLY answer/complete my task?\"")
            lines.append("")
            lines.append("🔍 COMMON SCENARIOS:")
            lines.append("")
            lines.append("  A) Skill returned the FINAL ANSWER (e.g., count, list, data you need):")
            lines.append("     → Extract the answer and respond: stop [your_answer]")
            lines.append("     → Example: Task asks for count, skill returns {'count': 5} → stop [5]")
            lines.append("")
            lines.append("  B) Skill only NAVIGATED to a page (e.g., opened a forum/user profile):")
            lines.append("     → The task is NOT complete! You're just at the starting point.")
            lines.append("     → Continue with basic actions: click, type, scroll to find information")
            lines.append("     → Example: Task asks for user's comment count, skill opened forum")
            lines.append("       → You still need to: find the user → go to their profile → count comments")
            lines.append("")
            lines.append("  C) Skill performed an ACTION (e.g., posted, voted, subscribed):")
            lines.append("     → Verify the action matches task requirement")
            lines.append("     → If task asks for confirmation, check the page state")
            lines.append("")
            lines.append("⚠️ DO NOT stop [0] or stop with placeholder just because skill succeeded!")
            lines.append("   Only stop when you have the ACTUAL answer to the task question.")
            lines.append("")
            lines.append("📝 BASIC ACTIONS available if you need to continue:")
            lines.append("  • click [id] - Click an element on the page")
            lines.append("  • type [id] [text] [enter] - Type text in an input field")
            lines.append("  • scroll [up/down] - Scroll to see more content")
            lines.append("  • go_back - Navigate back to previous page")
            lines.append("")

        elif skill_already_used:
            # Skill execution failed
            lines.append("❌ Skill execution failed. Please use basic actions to complete the task:")
            lines.append("  • click [id] - Click an element")
            lines.append("  • type [id] [text] [enter] - Type text")
            lines.append("  • scroll [up/down] - Scroll the page")
            lines.append("  • go_back - Go back")
            lines.append("")
            lines.append("Check the SKILL EXECUTION FAILED message above for details.")
            lines.append("")

        # Generate additional strategies based on task-level experience
        if related_lessons:
            lesson = related_lessons[0]
            task_lessons = lesson.get("task_lessons", {})

            if "better_strategy" in task_lessons:
                lines.append("Tips from similar tasks:")
                for strategy in task_lessons["better_strategy"][:2]:
                    lines.append(f"  • {strategy}")
                lines.append("")

        return "\n".join(lines)

    def generate_skill_context_update(
        self,
        current_observation: str,
        filtered_skills: List,
        previous_skill_calls: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate skill context update for each Agent decision step

        Used to refresh which skills are most likely needed before each Agent action

        Args:
            current_observation: Current observation
            filtered_skills: List of filtered skills
            previous_skill_calls: Previous skill call records

        Returns:
            Skill suggestion prompt for the current step
        """
        lines = []

        # Extract key information from current observation, determine most relevant skills
        lines.append("\n[Current Step Skill Suggestions]")
        lines.append("Based on current observation, the following skills may be most useful:")
        lines.append("")

        # Can do more refined matching based on observation content
        for skill in filtered_skills[:3]:  # Suggest at most 3
            lines.append(f"  • {skill.name}: {skill.get_summary()}")

        lines.append("")

        # If there are previous skill call records, show Agent progress
        if previous_skill_calls:
            lines.append(f"[Skill Call History] Called {len(previous_skill_calls)} skills")
            if previous_skill_calls:
                lines.append(f"Last one: {previous_skill_calls[-1].get('skill_name', 'unknown')}")
            lines.append("")

        return "\n".join(lines)

    def generate_light_skill_reminder(
        self,
        filtered_skills: List,
        step: int
    ) -> str:
        """
        Generate lightweight skill reminder (for subsequent steps, reduce prompt length)

        Args:
            filtered_skills: List of filtered skills
            step: Current step number

        Returns:
            Concise skill reminder prompt
        """
        if not filtered_skills:
            return ""

        lines = []
        lines.append(f"\n[Step {step}] Available skills reminder:")
        lines.append("")

        # Only list skill names and brief descriptions
        for i, skill in enumerate(filtered_skills[:3], 1):
            summary = skill.get_summary()
            test_indicator = "✓" if skill.metadata.get("test_count", 0) > 0 else "○"
            lines.append(f"{i}. [{test_indicator}] {skill.name} - {summary}")

        if len(filtered_skills) > 3:
            lines.append(f"... {len(filtered_skills) - 3} more skills available")

        lines.append("")
        lines.append("Use `note` to view complete skill information if needed.")
        lines.append("")

        return "\n".join(lines)


class TaskPatternMemory:
    """Task Pattern Memory Manager - Store and retrieve learned task patterns"""

    def __init__(self, memory_dir: str = "./task_memory"):
        """
        Initialize task memory manager

        Args:
            memory_dir: Memory file save directory
        """
        self.memory_dir = memory_dir
        self._ensure_memory_dir()

    def _ensure_memory_dir(self):
        """Ensure memory directory exists"""
        import os
        os.makedirs(self.memory_dir, exist_ok=True)

    def save_task_pattern(
        self,
        task_family: str,
        matched_lessons: List[Dict[str, Any]],
        filtered_skills: List,
        success_steps: Optional[List[str]] = None
    ) -> str:
        """
        Save learned task pattern to memory

        Args:
            task_family: Task family name
            matched_lessons: Matched task-level experiences
            filtered_skills: Filtered skills
            success_steps: Successful execution steps

        Returns:
            Path to saved file
        """
        import json
        import os
        from datetime import datetime

        pattern_data = {
            "timestamp": datetime.now().isoformat(),
            "task_family": task_family,
            "skill_names": [s.name for s in filtered_skills],
            "skill_count": len(filtered_skills),
            "lessons_count": len(matched_lessons),
            "lesson_ids": [l.get("cluster_id") for l in matched_lessons],
            "success_steps": success_steps or []
        }

        # Use task_family as filename (replace spaces and special characters)
        safe_name = task_family.replace(" ", "_").replace("/", "_").lower()
        file_path = os.path.join(self.memory_dir, f"{safe_name}_pattern.json")

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(pattern_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved task pattern to: {file_path}")
            return file_path
        except Exception as e:
            logger.warning(f"Failed to save task pattern: {e}")
            return ""

    def load_task_pattern(self, task_family: str) -> Optional[Dict[str, Any]]:
        """
        Load saved task pattern

        Args:
            task_family: Task family name

        Returns:
            Task pattern data, or None if not exists
        """
        import json
        import os

        safe_name = task_family.replace(" ", "_").replace("/", "_").lower()
        file_path = os.path.join(self.memory_dir, f"{safe_name}_pattern.json")

        if not os.path.exists(file_path):
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load task pattern: {e}")
            return None

    def get_skill_recall_prompt(self, task_family: str) -> str:
        """
        Get skill recall prompt from memory

        Used to quickly recall learned skills in subsequent steps without re-injecting complete information

        Args:
            task_family: Task family name

        Returns:
            Recall prompt
        """
        pattern = self.load_task_pattern(task_family)

        if not pattern:
            return ""

        lines = []
        lines.append("[Memory Recall] Based on previous learning:")
        lines.append(f"- Task family: {task_family}")
        lines.append(f"- Available skills: {pattern['skill_count']}")

        if pattern.get("skill_names"):
            skills_str = ", ".join(pattern["skill_names"][:3])
            if len(pattern["skill_names"]) > 3:
                skills_str += f", ... ({pattern['skill_count']} total)"
            lines.append(f"- Related skills: {skills_str}")

        if pattern.get("success_steps"):
            lines.append("- Verified successful steps:")
            for step in pattern["success_steps"][:2]:
                lines.append(f"  • {step}")

        lines.append("")
        return "\n".join(lines)

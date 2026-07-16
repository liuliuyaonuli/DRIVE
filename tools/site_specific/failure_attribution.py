"""Shared binary counterfactual attribution for DRIVE failed trajectories."""

from __future__ import annotations

from typing import Any, Dict, Mapping


COUNTERFACTUAL_ATTRIBUTION_INSTRUCTION = """
DRIVE COUNTERFACTUAL ATTRIBUTION (mandatory and binary):
Preserve the agent's high-level intent and ask: would this task have succeeded
if only the interaction procedure had been changed?  Answer YES or NO.
- YES -> failure_level_primary = "operation"
- NO  -> failure_level_primary = "reasoning"
Every failed trajectory must be assigned to exactly one of these two classes;
do not output "mixed".  Include:
"counterfactual": {
  "question": "Would a different interaction procedure succeed while preserving the high-level intent?",
  "answer": "yes|no",
  "rationale": "brief evidence from the trace"
}
"""


def _answer_is_yes(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"yes", "y", "true", "1", "operation", "interaction"}:
        return True
    if normalized in {"no", "n", "false", "0", "reasoning"}:
        return False
    return None


def normalize_failure_attribution(
    result: Mapping[str, Any], *, strict: bool = True
) -> Dict[str, Any]:
    """Enforce the paper's exclusive ``{err_i, err_r}`` attribution."""

    normalized = dict(result)
    counterfactual = normalized.get("counterfactual")
    answer: Any = None
    rationale = ""
    if isinstance(counterfactual, Mapping):
        answer = counterfactual.get(
            "answer",
            counterfactual.get("would_succeed_with_different_interaction"),
        )
        rationale = str(counterfactual.get("rationale", ""))
    elif counterfactual is not None:
        answer = counterfactual

    yes = _answer_is_yes(answer)
    if yes is None:
        # Accept an already normalized binary label, but never silently route
        # an invalid/unknown analysis into the reasoning library.
        existing = str(
            normalized.get("failure_level_primary")
            or normalized.get("failure_level")
            or ""
        ).lower()
        if existing in {"operation", "interaction"}:
            yes = True
        elif existing == "reasoning":
            yes = False
        elif strict:
            raise ValueError(
                "Failure attribution must include an explicit yes/no counterfactual answer"
            )
        else:
            yes = False

    level = "operation" if yes else "reasoning"
    normalized["failure_level"] = level
    normalized["failure_level_primary"] = level
    recommendation = dict(normalized.get("skill_recommendation") or {})
    recommendation["needs_operation_skill"] = level == "operation"
    recommendation["needs_reasoning_guidance"] = level == "reasoning"
    normalized["skill_recommendation"] = recommendation
    normalized["counterfactual"] = {
        "question": (
            "Would a different interaction procedure succeed while preserving "
            "the high-level intent?"
        ),
        "answer": "yes" if yes else "no",
        "rationale": rationale,
    }
    return normalized

#!/usr/bin/env python3
"""Run the batch DRIVE library-management operators for one evolution round."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from AgentOccam.skill_registry import load_skills_from_site


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True)
    parser.add_argument("--skills-root", type=Path, default=PROJECT_ROOT / "skills")
    parser.add_argument("--skill-file", default="operation_skills.py")
    parser.add_argument("--metadata", default="operation_skills.json")
    parser.add_argument("--reasoning-file", default="reasoning_tips.json")
    parser.add_argument("--similarity-threshold", type=float, default=0.82)
    parser.add_argument("--min-reasoning-usage", type=int, default=3)
    parser.add_argument("--min-reasoning-utility", type=float, default=0.20)
    parser.add_argument("--prune-unused-after-rounds", type=int, default=3)
    parser.add_argument("--min-selector-patches", type=int, default=3)
    parser.add_argument("--max-interaction-utility", type=float, default=0.20)
    args = parser.parse_args(argv)

    registry = load_skills_from_site(
        args.site,
        args.skills_root,
        args.skill_file,
        args.metadata,
    )
    reasoning_path = args.skills_root / args.site / args.reasoning_file
    if reasoning_path.exists():
        registry.load_external_task_lessons(reasoning_path)
        reasoning_result = registry.consolidate_reasoning_skills(
            similarity_threshold=args.similarity_threshold,
            min_usage=args.min_reasoning_usage,
            min_utility=args.min_reasoning_utility,
            prune_unused_after_rounds=args.prune_unused_after_rounds,
        )
    else:
        reasoning_result = {"merged": 0, "pruned": 0, "remaining": 0}
    removed = registry.prune_invalid_interaction_skills(
        min_patch_rounds=args.min_selector_patches,
        max_utility=args.max_interaction_utility,
    )
    result = {
        "site": args.site,
        "reasoning": reasoning_result,
        "interaction_removed": removed,
        "interaction_remaining": len(registry),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

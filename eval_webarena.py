import os
import time
import re
import argparse
import json
import random
import shutil
import traceback
import csv

import yaml

from AgentOccam.env import WebArenaEnvironmentWrapper

from AgentOccam.AgentOccam import AgentOccam

from AgentOccam.prompts import AgentOccam_prompt

from AgentOccam.utils import EVALUATOR_DIR, CURRENT_DIR
from AgentOccam.eval_utils import build_error_status
from AgentOccam.skill_registry import load_skills_from_site
from pathlib import Path


class DotDict:
    """Recursive attribute access for YAML dictionaries."""

    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = DotDict(value)
            elif isinstance(value, list):
                value = [DotDict(item) if isinstance(item, dict) else item for item in value]
            setattr(self, key, value)


def log_run(log_file, log_data, summary_file, summary_data):
    """Persist one trajectory and append one evaluator summary row."""

    with open(log_file, "w", encoding="utf-8") as handle:
        json.dump(log_data, handle, ensure_ascii=False, indent=2, default=str)
    exists = os.path.exists(summary_file) and os.path.getsize(summary_file) > 0
    with open(summary_file, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_data.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(summary_data)

def run():
    parser = argparse.ArgumentParser(
        description="Only the config file argument should be passed"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="yaml config file location"
    )
    parser.add_argument(
        "--skill-levels",
        choices=["both", "reasoning", "interaction", "none"],
        default="both",
        help="DRIVE dual-level ablation using the same task/evaluator configuration",
    )
    parser.add_argument(
        "--model",
        help="override actor/critic/judge model (supports Qwen via an OpenAI-compatible endpoint)",
    )
    parser.add_argument(
        "--task-config-dir",
        type=Path,
        required=True,
        help="directory containing external WebArena task JSON files",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-ids", nargs="+", type=int, default=[])
    parser.add_argument("--run-tag", default="")
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = DotDict(yaml.safe_load(file))

    if hasattr(config, "skills"):
        config.skills.use_reasoning_skills = args.skill_levels in {"both", "reasoning"}
        config.skills.use_interaction_skills = args.skill_levels in {"both", "interaction"}
        if args.skill_levels == "none":
            config.skills.use_skills = False
    if args.task_ids:
        config.env.task_ids = args.task_ids
    if args.run_tag:
        config.logname = args.run_tag
        if hasattr(config.agent, "others"):
            config.agent.others.logname = args.run_tag
        if hasattr(config.agent, "actor") and hasattr(config.agent.actor, "others"):
            config.agent.actor.others.logname = args.run_tag
    os.environ["LLM_SEED"] = str(args.seed)
    if args.model and hasattr(config.agent, "actor"):
        config.agent.actor.model = args.model
        config.agent.critic.model = args.model
        config.agent.judge.model = args.model
    
    if config.logging:
        # 始终使用时间戳创建独立的运行文件夹
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        if config.logname:
            # 如果指定了 logname，将其作为前缀
            dstdir = f"{config.logdir}/{config.logname}_{timestamp}"
        else:
            dstdir = f"{config.logdir}/{timestamp}"
        os.makedirs(dstdir, exist_ok=True)
        shutil.copyfile(args.config, os.path.join(dstdir, args.config.split("/")[-1]))
        print(f"结果将保存到: {dstdir}")
    random.seed(args.seed)
    
    config_file_list = []
    
    task_ids = config.env.task_ids
    task_config_dir = args.task_config_dir
    if not task_config_dir.is_dir():
        parser.error("task configuration directory not found")
    if task_ids == "all" or task_ids == ["all"]:
        task_ids = sorted(
            filename[:-len(".json")]
            for filename in os.listdir(task_config_dir)
            if filename.endswith(".json")
        )
    for task_id in task_ids:
        config_file_list.append(str(task_config_dir / f"{task_id}.json"))

    fullpage = config.env.fullpage if hasattr(config.env, "fullpage") else True
    current_viewport_only = not fullpage

    # Copy skills configuration from root to agent config (if it exists)
    if hasattr(config, 'skills') and not hasattr(config.agent, 'skills'):
        config.agent.skills = config.skills
        if config.verbose >= 1:
            print(f"✓ 已将 skills 配置复制到 config.agent")

    if config.agent.type != "AgentOccam":
        raise ValueError("this source release supports AgentOccam only")
    agent_init = lambda: AgentOccam(
        prompt_dict = {k: v for k, v in AgentOccam_prompt.__dict__.items() if isinstance(v, dict)},
        config = config.agent,
    )

    
    for config_file in config_file_list:
        with open(config_file, "r") as f:
            task_config = json.load(f)
            print(f"Task {task_config['task_id']}.")
        if os.path.exists(os.path.join(dstdir, f"{task_config['task_id']}.json")):
            print(f"Skip {task_config['task_id']}.")
            continue
        if task_config['task_id'] in list(range(600, 650))+list(range(681, 689)):
            print("Reddit post task. Sleep 30 mins.")
            time.sleep(1800)

        agent = None
        env = None
        status = None
        skill_registry = None
        try:
            # Create agent first to get skill_registry
            agent = agent_init()

            # Load skill_registry directly from config (agent's skill_registry is in Actor which isn't created yet)
            if hasattr(config, 'skills') and config.skills:
                cfg_sk = config.skills
                # 兼容字典和对象配置
                def get_cfg_value(key, default=None):
                    return cfg_sk.get(key, default) if isinstance(cfg_sk, dict) else getattr(cfg_sk, key, default)

                if get_cfg_value('use_skills', False):
                    skill_site = get_cfg_value('skill_site')
                    if skill_site:
                        try:
                            skill_dir = get_cfg_value('skill_dir')
                            if skill_dir:
                                skills_base_dir = Path(skill_dir)
                            else:
                                skills_base_dir = Path(CURRENT_DIR).parent / "skills"

                            skill_registry = load_skills_from_site(
                                site_name=skill_site,
                                skills_base_dir=skills_base_dir,
                                skill_file=get_cfg_value('skill_file'),
                                skill_metadata=get_cfg_value('skill_metadata'),
                            )
                            task_lessons_path = get_cfg_value('task_lessons_path')
                            if task_lessons_path:
                                skill_registry.load_external_task_lessons(Path(CURRENT_DIR) / task_lessons_path)
                            # Reuse exactly one registry across retrieval,
                            # execution, and feedback; avoid loading divergent
                            # copies for the actor and environment.
                            config.agent.actor.preloaded_skill_registry = skill_registry
                            if config.verbose >= 1:
                                print(f"✓ eval_webarena: 成功加载技能库，共 {len(skill_registry)} 个技能")
                        except Exception as e:
                            print(f"⚠️  eval_webarena: 加载技能库失败: {e}")
                            skill_registry = None

            env = WebArenaEnvironmentWrapper(config_file=config_file,
                                            max_browser_rows=config.env.max_browser_rows,
                                            max_steps=config.max_steps,
                                            slow_mo=1,
                                            observation_type="accessibility_tree",
                                            current_viewport_only=current_viewport_only,
                                            viewport_size={"width": 1920, "height": 1080},
                                            headless=config.env.headless,
                                            global_config=config,
                                            skill_registry=skill_registry)

            objective = env.get_objective()
            status = agent.act(objective=objective, env=env)
        except Exception as e:
            print("=" * 70)
            print(f"[Task ERROR] Task {task_config['task_id']} failed; recording failure and continuing.")
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message: {e}")
            traceback.print_exc()
            print("=" * 70)
            status = build_error_status(e)
        finally:
            if skill_registry is not None:
                final_reward = status.get("reward", 0.0) if isinstance(status, dict) else 0.0
                skill_registry.finalize_episode_feedback(final_reward)
            if env is not None:
                try:
                    env.close()
                except Exception as close_err:
                    print(f"[Warning] Failed to close env for task {task_config['task_id']}: {close_err}")

        if config.logging:
            with open(config_file, "r") as f:
                task_config = json.load(f)
            log_file = os.path.join(dstdir, f"{task_config['task_id']}.json")
            log_data = {
                "task": config_file,
                "id": task_config['task_id'],
                "model": config.agent.actor.model if hasattr(config.agent, "actor") else config.agent.model_name,
                "type": config.agent.type,
                "trajectory": agent.get_trajectory() if agent else [],
            }
            if status and status.get("error"):
                log_data["error"] = status["error"]
            summary_file = os.path.join(dstdir, "summary.csv")
            summary_data = {
                "task": config_file,
                "task_id": task_config['task_id'],
                "model": config.agent.actor.model if hasattr(config.agent, "actor") else config.agent.model_name,
                "type": config.agent.type,
                "logfile": re.search(r"/([^/]+/[^/]+\.json)$", log_file).group(1),
            }
            if status:
                summary_data.update(status)
            log_run(
                log_file=log_file,
                log_data=log_data,
                summary_file=summary_file,
                summary_data=summary_data,
            )
    
if __name__ == "__main__":
    run()

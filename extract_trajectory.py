#!/usr/bin/env python3
"""
提取网页智能体运行轨迹的关键信息
从完整的轨迹JSON文件中提取 objective, plan, reason, action 四个关键字段
并根据任务ID拼接标准答案(reference_answer)

支持根据 summary.csv 过滤失败轨迹
"""

import json
import csv
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import argparse


def load_summary_csv(input_folder: str) -> Dict[int, Dict[str, Any]]:
    """
    从 summary.csv 加载任务成功/失败信息

    Args:
        input_folder: 轨迹文件夹路径

    Returns:
        字典 {task_id: {"success": bool, "reward": float, ...}}
    """
    summary_path = Path(input_folder) / "summary.csv"
    summary_data = {}

    if not summary_path.exists():
        return summary_data

    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                task_id_str = row.get('task_id', '')
                if task_id_str.isdigit():
                    task_id = int(task_id_str)
                else:
                    continue

                # reward == 1.0 才算成功
                success = float(row.get('reward', 0)) == 1.0
                reward = float(row.get('reward', 0))

                summary_data[task_id] = {
                    'success': success,
                    'reward': reward,
                    'num_actions': int(row.get('num_actions', 0)),
                    'done': row.get('done', 'True') == 'True'
                }

        print(f"✓ 从 summary.csv 加载了 {len(summary_data)} 条记录")
        return summary_data

    except Exception as e:
        print(f"⚠️ 读取 summary.csv 失败: {e}")
        return {}


def load_reference_answers(config_dir: str) -> Dict[int, Dict[str, Any]]:
    """
    加载配置文件中的标准答案

    Args:
        config_dir: 配置文件目录路径

    Returns:
        字典 {task_id: {"reference_answer": ..., "must_include": [...], "intent": ...}}
    """
    reference_answers = {}
    config_path = Path(config_dir)

    if not config_path.exists():
        print(f"警告: 配置目录不存在: {config_dir}")
        return reference_answers

    for json_file in config_path.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            task_id = config.get("task_id")
            if task_id is None:
                # 尝试从文件名提取ID (如 "132.json")
                try:
                    task_id = int(json_file.stem)
                except ValueError:
                    continue

            eval_info = config.get("eval") or {}
            ref_answers = eval_info.get("reference_answers") or {}

            # 处理 program_html 类型的评估（从 program_html 提取标准答案）
            program_html = eval_info.get("program_html", [])
            program_html_answers = []
            if program_html:
                for ph in program_html:
                    required = ph.get("required_contents", {})
                    if required:
                        if "exact_match" in required:
                            program_html_answers.append(required["exact_match"])
                        if "must_include" in required:
                            program_html_answers.extend(required["must_include"])

            reference_answers[task_id] = {
                "reference_answer_raw": eval_info.get("reference_answer_raw_annotation", ""),
                "must_include": ref_answers.get("must_include", []),
                "must_exclude": ref_answers.get("must_exclude", []),
                "fuzzy_match": ref_answers.get("fuzzy_match", []),
                "program_html_answers": program_html_answers,  # 新增：从 program_html 提取的答案
                "intent": config.get("intent", ""),
                "eval_types": eval_info.get("eval_types", []),
                "sites": config.get("sites", []),
            }
        except Exception as e:
            print(f"警告: 无法解析配置文件 {json_file.name}: {e}")
            continue

    return reference_answers


def extract_simplified_trajectory(
    trajectory_data: Dict[str, Any],
    reference_answers: Optional[Dict[int, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    从完整轨迹数据中提取简化版本，并附加标准答案

    Args:
        trajectory_data: 完整的轨迹数据
        reference_answers: 标准答案字典 {task_id: {...}}

    Returns:
        简化后的轨迹数据（包含标准答案）
    """
    task_id = trajectory_data.get("id", "")

    simplified = {
        "task": trajectory_data.get("task", ""),
        "id": task_id,
        "model": trajectory_data.get("model", ""),
        "type": trajectory_data.get("type", ""),
        "trajectory": []
    }

    # 添加标准答案信息
    if reference_answers and task_id in reference_answers:
        ref = reference_answers[task_id]
        simplified["reference_answer"] = {
            "answer": ref.get("reference_answer_raw", ""),
            "must_include": ref.get("must_include", []),
            "must_exclude": ref.get("must_exclude", []),
            "fuzzy_match": ref.get("fuzzy_match", []),
            "program_html_answers": ref.get("program_html_answers", []),
            "eval_types": ref.get("eval_types", []),
        }
        # 如果 intent 和 task 中的不同，也保留原始 intent
        if ref.get("intent"):
            simplified["intent"] = ref["intent"]
        if ref.get("sites"):
            simplified["sites"] = ref["sites"]
    else:
        simplified["reference_answer"] = None

    # 提取轨迹中的关键步骤
    for step in trajectory_data.get("trajectory", []):
        simplified_step = {
            "objective": step.get("objective", ""),
            "url": step.get("url", ""),
            "plan": step.get("plan", ""),
            "reason": step.get("reason", ""),
            "action": step.get("action", "")
        }
        simplified["trajectory"].append(simplified_step)

    return simplified


def process_trajectory_folder(
    input_folder: str,
    output_folder: str = None,
    config_dir: str = None,
    merge_to_jsonl: bool = False,
    failures_only: bool = False
) -> None:
    """
    处理轨迹文件夹,提取所有JSON文件的简化版本

    Args:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径(可选,默认为输入文件夹_simplified)
        config_dir: 配置文件目录路径(可选,用于加载标准答案)
        merge_to_jsonl: 是否将所有轨迹合并到一个JSONL文件
        failures_only: 是否只提取失败轨迹（根据 summary.csv）
    """
    input_path = Path(input_folder)

    if not input_path.exists():
        print(f"错误: 输入文件夹不存在: {input_folder}")
        return

    # 设置输出文件夹
    if output_folder is None:
        suffix = "_failures" if failures_only else "_simplified"
        output_folder = str(input_path) + suffix

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # 加载 summary.csv 获取成功/失败信息
    summary_data = load_summary_csv(input_folder)
    failed_task_ids: Set[int] = set()

    if failures_only:
        if not summary_data:
            print("警告: 未找到 summary.csv，无法过滤失败轨迹")
            print("将处理所有轨迹")
        else:
            # 提取失败的任务ID
            for task_id, info in summary_data.items():
                if not info['success']:
                    failed_task_ids.add(task_id)
            print(f"✓ 找到 {len(failed_task_ids)} 个失败任务")

    # 加载标准答案
    reference_answers = {}
    if config_dir:
        print(f"加载标准答案: {config_dir}")
        reference_answers = load_reference_answers(config_dir)
        print(f"已加载 {len(reference_answers)} 个任务的标准答案")
    # 获取所有JSON文件
    json_files = list(input_path.glob("*.json"))
    # 排除 summary 相关文件
    json_files = [f for f in json_files if 'summary' not in f.name.lower()]

    if not json_files:
        print(f"警告: 在 {input_folder} 中没有找到JSON文件")
        return

    print(f"找到 {len(json_files)} 个JSON文件")
    print(f"输出文件夹: {output_folder}")
    if failures_only:
        print(f"模式: 只提取失败轨迹")
    print("-" * 60)

    success_count = 0
    error_count = 0
    skipped_count = 0
    with_answer_count = 0

    # 用于合并的列表
    all_trajectories = []

    # 处理每个JSON文件
    for json_file in sorted(json_files):
        try:
            # 从文件名提取 task_id
            try:
                task_id = int(json_file.stem)
            except ValueError:
                task_id = None

            # 如果只要失败轨迹，检查是否在失败列表中
            if failures_only and failed_task_ids:
                if task_id is None or task_id not in failed_task_ids:
                    skipped_count += 1
                    continue

            # 读取原始数据
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 提取简化版本（包含标准答案）
            simplified_data = extract_simplified_trajectory(data, reference_answers)

            # 添加成功/失败标记
            if task_id and task_id in summary_data:
                simplified_data["success"] = summary_data[task_id]["success"]
                simplified_data["reward"] = summary_data[task_id]["reward"]

            # 保存简化版本
            output_file = output_path / json_file.name
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(simplified_data, f, ensure_ascii=False, indent=2)

            # 收集用于合并
            if merge_to_jsonl:
                all_trajectories.append(simplified_data)

            step_count = len(simplified_data.get("trajectory", []))
            has_answer = simplified_data.get("reference_answer") is not None
            if has_answer:
                with_answer_count += 1

            # 显示状态
            is_success = simplified_data.get("success", None)
            if is_success is True:
                status_mark = "✅"
            elif is_success is False:
                status_mark = "❌"
            else:
                status_mark = "❓"

            answer_mark = "📋" if has_answer else "⚠️"
            print(f"{status_mark} {answer_mark} 处理完成: {json_file.name} ({step_count} 个步骤)")
            success_count += 1

        except Exception as e:
            print(f"✗ 处理失败: {json_file.name} - {str(e)}")
            error_count += 1

    print("-" * 60)
    print(f"处理完成: 成功 {success_count} 个, 跳过 {skipped_count} 个, 失败 {error_count} 个")
    print(f"包含标准答案: {with_answer_count} 个 ({with_answer_count*100//max(success_count,1)}%)")
    print(f"简化版轨迹已保存到: {output_folder}")

    # 合并到JSONL文件
    if merge_to_jsonl and all_trajectories:
        jsonl_name = "failed_trajectories.jsonl" if failures_only else "all_trajectories.jsonl"
        jsonl_file = output_path / jsonl_name
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for traj in all_trajectories:
                f.write(json.dumps(traj, ensure_ascii=False) + "\n")
        print(f"✓ 合并JSONL文件: {jsonl_file} ({len(all_trajectories)} 条记录)")


def main():
    parser = argparse.ArgumentParser(
        description="提取网页智能体运行轨迹的关键信息，并拼接标准答案",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 提取所有轨迹
  python extract_trajectory.py /path/to/trajectories

  # 只提取失败轨迹（根据 summary.csv 的 success 列）
  python extract_trajectory.py /path/to/trajectories --failures-only --merge

  # 指定输出文件夹
  python extract_trajectory.py input_folder -o output_folder

  # 指定配置文件目录（包含标准答案）
  python extract_trajectory.py input_folder -c /path/to/webarena-configs

  # 合并所有轨迹到一个JSONL文件（用于失败分析）
  python extract_trajectory.py input_folder --merge

  # 只提取失败轨迹并合并
  python extract_trajectory.py input_folder --failures-only --merge

输出格式:
  简化后的轨迹JSON包含以下字段:
  - task: 任务配置文件路径
  - id: 任务ID
  - model: 使用的模型
  - type: 智能体类型
  - success: 是否成功（从 summary.csv 读取）
  - reward: 奖励值
  - intent: 任务意图描述
  - sites: 涉及的站点
  - reference_answer: 标准答案信息
    - answer: 原始标准答案
    - must_include: 必须包含的内容
    - must_exclude: 必须排除的内容
    - program_html_answers: 从program_html提取的答案
    - eval_types: 评估类型
  - trajectory: 简化的轨迹步骤列表

合并JSONL文件:
  使用 --merge 参数会在输出目录生成:
  - all_trajectories.jsonl (默认)
  - failed_trajectories.jsonl (使用 --failures-only 时)
        """
    )

    parser.add_argument(
        "input_folder",
        type=str,
        help="包含完整轨迹JSON文件的输入文件夹路径"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="简化版轨迹的输出文件夹路径 (默认: 输入文件夹_simplified 或 _failures)"
    )

    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="配置文件目录路径，包含任务标准答案（可选）"
    )

    parser.add_argument(
        "--merge",
        action="store_true",
        help="将所有轨迹合并到一个JSONL文件"
    )

    parser.add_argument(
        "--failures-only",
        action="store_true",
        help="只提取失败轨迹（根据 summary.csv 的 success 列判断）"
    )

    args = parser.parse_args()

    process_trajectory_folder(
        args.input_folder,
        args.output,
        args.config,
        args.merge,
        args.failures_only
    )


if __name__ == "__main__":
    main()

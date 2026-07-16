#!/usr/bin/env python3
"""
验证生成的技能代码（语法检查 + 签名验证）。

用法:
    # 验证单个文件
    python3 tools/verify_skill_snippets.py \\
        --file /path/to/operation_skills.py

    # 验证并显示详细信息
    python3 tools/verify_skill_snippets.py \\
        --file /path/to/operation_skills.py \\
        --verbose

    # 验证并检查导入依赖
    python3 tools/verify_skill_snippets.py \\
        --file /path/to/operation_skills.py \\
        --check-imports
"""
import importlib.util
import sys
import ast
import inspect
import argparse
import json
import os
import pathlib
import tempfile
from collections import defaultdict


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_env_file(path):
    """读取简单的 KEY=VALUE 环境文件，不覆盖已有环境变量。"""
    path = pathlib.Path(path)
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _operation_skill_check_http_client(httpx_module):
    timeout = float(os.environ.get("OPERATION_SKILL_CHECK_TIMEOUT", "180"))
    return httpx_module.Client(trust_env=False, timeout=timeout)


def check_syntax(src):
    """检查Python语法是否正确"""
    try:
        ast.parse(src)
        return True, None
    except SyntaxError as e:
        return False, f"第{e.lineno}行: {e.msg}"


def check_imports(src):
    """检查必需的导入是否存在"""
    tree = ast.parse(src)
    imports = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])

    # 检查常见依赖
    required = set()
    if 'Page' in src or 'page.' in src:
        required.add('playwright')
    if 'asyncio' in src or 'await asyncio' in src:
        required.add('asyncio')

    missing = required - imports
    return len(missing) == 0, list(missing)


def load_module(file_path):
    """动态加载Python模块"""
    try:
        spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
        if not spec or not spec.loader:
            return None, "无法创建模块spec"

        mod = importlib.util.module_from_spec(spec)
        sys.modules[file_path.stem] = mod
        spec.loader.exec_module(mod)
        return mod, None
    except Exception as e:
        return None, str(e)


def verify_skill_signature(name, func):
    """
    验证技能函数签名

    Returns:
        (is_valid, errors)
    """
    errors = []

    # 检查是否是async函数
    if not inspect.iscoroutinefunction(func):
        errors.append(f"{name}: 必须是 async def")
        return False, errors

    # 检查签名
    try:
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        if not params:
            errors.append(f"{name}: 缺少参数")
            return False, errors

        # 第一个参数必须是 page
        if params[0] != "page":
            errors.append(f"{name}: 第一个参数必须是 'page'，但是 '{params[0]}'")
            return False, errors

    except Exception as e:
        errors.append(f"{name}: 无法解析签名 - {e}")
        return False, errors

    return True, []


def extract_functions(mod):
    """提取模块中的所有函数"""
    functions = {}
    for name, obj in vars(mod).items():
        if inspect.iscoroutinefunction(obj) and not name.startswith('_'):
            functions[name] = obj
    return functions


def _extract_json_object(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return json.JSONDecoder(strict=False).decode(text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        json_text = text[start : end + 1]
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            return json.JSONDecoder(strict=False).decode(json_text)
    raise ValueError("LLM output does not contain a JSON object")


def llm_check_operation_skill(name, source, model, base_url, api_key):
    """使用专用第三方 API 检查操作级技能代码质量。"""
    try:
        from openai import OpenAI
        import httpx
    except ImportError as e:
        raise RuntimeError("openai/httpx 包未安装，无法进行 LLM 检查") from e

    prompt = f"""You are reviewing an operation-level Playwright skill for AgentOccam.

Return exactly one JSON object:
{{
  "verdict": "pass" | "warn" | "fail",
  "issues": ["short concrete issue list"],
  "recommendations": ["short concrete fix suggestions"]
}}

Review criteria:
- The skill must be an async Playwright operation skill whose first argument is page.
- It should perform a concrete reusable browser operation, not only describe or check.
- It should avoid undefined variables, missing imports, nested helper functions, and broad no-op try/except.
- It should wait for navigation/dynamic content when needed.
- It should verify the resulting UI state or extracted value before returning.
- It should raise meaningful exceptions for impossible operations.
- Do not require running a browser; inspect the code statically.

Skill name: {name}

Code:
```python
{source}
```
"""
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=_operation_skill_check_http_client(httpx),
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return _extract_json_object(response.choices[0].message.content)


def llm_repair_operation_skill_file(source, check_results, model, base_url, api_key):
    """使用专用第三方 API 修复操作级技能文件中明确的代码问题。"""
    try:
        from openai import OpenAI
        import httpx
    except ImportError as e:
        raise RuntimeError("openai/httpx 包未安装，无法进行 LLM 修复") from e

    prompt = f"""You are repairing an operation-level Playwright skill file for AgentOccam.

Return exactly one JSON object:
{{
  "repaired_code": "the full repaired Python file",
  "changes": ["short list of concrete edits made"]
}}

Repair rules:
- Only fix clear code-level problems from the review results below.
- Preserve the public async skill functions and their first page argument.
- Preserve imports, exception classes, docstrings, usage logs, and metadata comments unless they are clearly wrong.
- Do not add placeholders, TODOs, nested helper functions, or references to undefined names.
- Keep the file directly loadable as a Python module.
- If no repair is needed, return the original source unchanged and an empty changes list.

Review results:
```json
{json.dumps(check_results, ensure_ascii=False, indent=2)}
```

Current Python file:
```python
{source}
```
"""
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=_operation_skill_check_http_client(httpx),
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    result = _extract_json_object(response.choices[0].message.content)
    repaired_code = result.get("repaired_code")
    if not isinstance(repaired_code, str) or not repaired_code.strip():
        raise ValueError("LLM repair output is missing non-empty repaired_code")
    changes = result.get("changes", [])
    if not isinstance(changes, list):
        changes = [str(changes)]
    return {"repaired_code": repaired_code, "changes": changes}


def validate_skill_source(source, original_path):
    """验证一段技能源代码是否可作为操作级技能文件加载。"""
    errors = []
    syntax_ok, syntax_error = check_syntax(source)
    if not syntax_ok:
        return False, [f"语法错误: {syntax_error}"]

    original_path = pathlib.Path(original_path)
    with tempfile.TemporaryDirectory(prefix="agentoccam_skill_verify_") as tmp_dir:
        temp_path = pathlib.Path(tmp_dir) / original_path.name
        temp_path.write_text(source, encoding="utf-8")
        mod, load_error = load_module(temp_path)
        if not mod:
            return False, [f"加载失败: {load_error}"]

        functions = extract_functions(mod)
        if not functions:
            return False, ["未找到任何 async 技能函数"]

        for name, func in functions.items():
            is_valid, signature_errors = verify_skill_signature(name, func)
            if not is_valid:
                errors.extend(signature_errors)

    return len(errors) == 0, errors


def main():
    _load_env_file(PROJECT_ROOT / ".env.local")

    ap = argparse.ArgumentParser(description="验证技能代码")
    ap.add_argument("--file", required=True, help="技能文件路径")
    ap.add_argument("--verbose", action="store_true", help="显示详细信息")
    ap.add_argument("--check-imports", action="store_true", help="检查导入依赖")
    ap.add_argument("--llm-check", action="store_true", help="使用专用第三方 API 检查操作级技能代码")
    ap.add_argument(
        "--llm-repair",
        action="store_true",
        help="LLM 检查发现 warn/fail 后，修复明显错误并写回技能文件",
    )
    ap.add_argument(
        "--repair-out",
        help="修复结果输出路径；默认覆盖 --file",
    )
    ap.add_argument(
        "--llm-model",
        default=os.environ.get("OPERATION_SKILL_CHECK_MODEL", "claude-sonnet-4-6"),
        help="操作级技能代码检查模型",
    )
    ap.add_argument(
        "--llm-api-base-url",
        default=os.environ.get("OPERATION_SKILL_CHECK_API_BASE_URL", "https://api.vveai.com/v1"),
        help="操作级技能代码检查 API base URL",
    )
    ap.add_argument(
        "--llm-api-key",
        default=os.environ.get("OPERATION_SKILL_CHECK_API_KEY"),
        help="操作级技能代码检查 API key；默认读取 OPERATION_SKILL_CHECK_API_KEY",
    )
    args = ap.parse_args()
    if args.llm_repair:
        args.llm_check = True

    p = pathlib.Path(args.file)

    if not p.exists():
        print(f"错误: 文件不存在: {args.file}")
        sys.exit(1)

    print(f"验证技能文件: {args.file}\n")

    # 1. 读取源代码
    src = p.read_text(encoding="utf-8")

    # 2. 语法检查
    print("[1/4] 语法检查...", end=" ")
    syntax_ok, syntax_error = check_syntax(src)
    if not syntax_ok:
        print(f"✗ 失败")
        print(f"  错误: {syntax_error}")
        sys.exit(1)
    print("✓ 通过")

    # 3. 导入检查（可选）
    if args.check_imports:
        print("[2/4] 导入检查...", end=" ")
        imports_ok, missing = check_imports(src)
        if not imports_ok:
            print(f"⚠️  缺少导入: {', '.join(missing)}")
        else:
            print("✓ 通过")
    else:
        print("[2/4] 导入检查... ⊘ 跳过")

    # 4. 加载模块
    print("[3/4] 加载模块...", end=" ")
    mod, load_error = load_module(p)
    if not mod:
        print(f"✗ 失败")
        print(f"  错误: {load_error}")
        sys.exit(1)
    print("✓ 通过")

    # 5. 签名验证
    print("[4/4] 签名验证...")

    functions = extract_functions(mod)

    if not functions:
        print("  ⚠️  未找到任何async函数")
        sys.exit(0)

    print(f"  找到 {len(functions)} 个技能函数\n")

    all_valid = True
    valid_count = 0
    invalid_count = 0
    errors_by_func = defaultdict(list)

    for name, func in functions.items():
        is_valid, errors = verify_skill_signature(name, func)

        if is_valid:
            valid_count += 1
            if args.verbose:
                # 显示函数签名
                sig = inspect.signature(func)
                print(f"  ✓ {name}{sig}")
        else:
            invalid_count += 1
            all_valid = False
            errors_by_func[name] = errors
            print(f"  ✗ {name}")
            for error in errors:
                print(f"      {error}")

    llm_results = {}
    if args.llm_check:
        if not args.llm_api_key:
            print("\n✗ LLM 检查失败: 未设置 OPERATION_SKILL_CHECK_API_KEY")
            sys.exit(1)

        print(f"\n[LLM] 操作级技能代码检查... model={args.llm_model}")
        for name, func in functions.items():
            try:
                result = llm_check_operation_skill(
                    name,
                    src,
                    args.llm_model,
                    args.llm_api_base_url,
                    args.llm_api_key,
                )
            except Exception as e:
                print(f"  ✗ {name}: LLM 检查失败 - {e}")
                all_valid = False
                llm_results[name] = {"verdict": "fail", "issues": [str(e)], "recommendations": []}
                continue

            verdict = str(result.get("verdict", "fail")).lower()
            issues = result.get("issues", [])
            recommendations = result.get("recommendations", [])
            llm_results[name] = {
                "verdict": verdict,
                "issues": issues,
                "recommendations": recommendations,
            }
            if verdict == "fail":
                all_valid = False
                print(f"  ✗ {name}: fail")
            elif verdict == "warn":
                print(f"  ⚠ {name}: warn")
            else:
                print(f"  ✓ {name}: pass")

            if args.verbose and (issues or recommendations):
                for issue in issues:
                    print(f"      issue: {issue}")
                for rec in recommendations:
                    print(f"      recommendation: {rec}")

        needs_repair = any(
            str(result.get("verdict", "")).lower() in {"warn", "fail"}
            for result in llm_results.values()
        )
        if args.llm_repair and needs_repair:
            print("\n[LLM] 修复操作级技能中的明显问题...", end=" ")
            try:
                repair = llm_repair_operation_skill_file(
                    src,
                    llm_results,
                    args.llm_model,
                    args.llm_api_base_url,
                    args.llm_api_key,
                )
                repaired_src = repair["repaired_code"]
                repair_ok, repair_errors = validate_skill_source(repaired_src, p)
                if not repair_ok:
                    print("✗ 失败")
                    for error in repair_errors:
                        print(f"  错误: {error}")
                    all_valid = False
                else:
                    repair_path = pathlib.Path(args.repair_out) if args.repair_out else p
                    repair_path.write_text(repaired_src, encoding="utf-8")
                    print("✓ 通过")
                    print(f"  已写入: {repair_path}")
                    changes = repair.get("changes", [])
                    if args.verbose and changes:
                        for change in changes:
                            print(f"      change: {change}")
            except Exception as e:
                print(f"✗ 失败 - {e}")
                all_valid = False
        elif args.llm_repair:
            print("\n[LLM] 未发现需要修复的问题")

    # 总结
    print(f"\n{'='*60}")
    if all_valid:
        print(f"✓ 验证通过")
        print(f"{'='*60}")
        print(f"总技能数: {len(functions)}")
        print(f"  有效: {valid_count}")
        if args.llm_check:
            verdict_counts = defaultdict(int)
            for result in llm_results.values():
                verdict_counts[result["verdict"]] += 1
            print(f"  LLM检查: {dict(verdict_counts)}")
        print(f"{'='*60}")
        sys.exit(0)
    else:
        print(f"✗ 验证失败")
        print(f"{'='*60}")
        print(f"总技能数: {len(functions)}")
        print(f"  有效: {valid_count}")
        print(f"  无效: {invalid_count}")
        if args.llm_check:
            verdict_counts = defaultdict(int)
            for result in llm_results.values():
                verdict_counts[result["verdict"]] += 1
            print(f"  LLM检查: {dict(verdict_counts)}")
        print(f"{'='*60}")

        if errors_by_func:
            print(f"\n错误详情:")
            for name, errors in errors_by_func.items():
                print(f"  {name}:")
                for error in errors:
                    print(f"    - {error}")

        sys.exit(2)


if __name__ == "__main__":
    main()

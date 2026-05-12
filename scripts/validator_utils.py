"""Shared utilities for DeepSeek JSON validator scripts."""

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from deepseek_utils import (
    DEFAULT_DEEPSEEK_MODEL,
    create_deepseek_chat_completion,
    get_deepseek_model,
)
from text_utils import clean_json_text


DEFAULT_VALIDATOR_MODEL = DEFAULT_DEEPSEEK_MODEL


def load_project_config(project_arg=None):
    """Load project config from {project}/source_material/config.json."""
    project = project_arg or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not project:
        raise ValueError("Project name required: python script.py PROJECT_NAME")

    base_dir = Path(__file__).resolve().parent.parent
    project_root = base_dir / project
    source_dir = project_root / "source_material"
    config_path = source_dir / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    project_name = config.get("_project_config", {}).get("project_name", project)
    return project_name, source_dir, config


def get_deepseek_client():
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(dotenv_path=repo_root / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Missing DEEPSEEK_API_KEY in .env file")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def get_validator_model(config):
    return os.getenv("DEEPSEEK_VALIDATOR_MODEL") or get_deepseek_model(config)


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def matching_json_end(text, start_index):
    opening = text[start_index]
    closing = "}" if opening == "{" else "]"
    stack = [closing]
    in_string = False
    escape = False

    for index in range(start_index + 1, len(text)):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif stack and char == stack[-1]:
            stack.pop()
            if not stack:
                return index
        elif char == closing and not stack:
            return index
    return -1


def parse_llm_json(content):
    """Parse valid JSON from a model response, accepting fenced JSON blocks."""
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    starts = [i for i, char in enumerate(cleaned) if char in "[{"]
    for start in starts:
        end = matching_json_end(cleaned, start)
        if end == -1:
            continue
        candidate = cleaned[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ValueError(f"DeepSeek did not return valid JSON:\n{content}")


def json_type_name(value):
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    if value is None:
        return "null"
    return type(value).__name__


def assert_same_structure(original, candidate, path="$"):
    """Ensure candidate has exactly the same JSON structure as original."""
    if json_type_name(original) != json_type_name(candidate):
        raise ValueError(
            f"Structure mismatch at {path}: expected {json_type_name(original)}, "
            f"got {json_type_name(candidate)}"
        )

    if isinstance(original, dict):
        original_keys = list(original.keys())
        candidate_keys = list(candidate.keys())
        if set(original_keys) != set(candidate_keys):
            missing = [key for key in original_keys if key not in candidate]
            extra = [key for key in candidate_keys if key not in original]
            raise ValueError(f"Key mismatch at {path}: missing={missing}, extra={extra}")
        for key in original_keys:
            assert_same_structure(original[key], candidate[key], f"{path}.{key}")
        return

    if isinstance(original, list):
        if len(original) != len(candidate):
            raise ValueError(
                f"List length mismatch at {path}: expected {len(original)}, got {len(candidate)}"
            )
        for index, (original_item, candidate_item) in enumerate(zip(original, candidate)):
            assert_same_structure(original_item, candidate_item, f"{path}[{index}]")


def normalize_to_original_order(original, candidate):
    """Return candidate content with dict key order matching original."""
    if isinstance(original, dict):
        return {
            key: normalize_to_original_order(original[key], candidate[key])
            for key in original.keys()
        }
    if isinstance(original, list):
        return [
            normalize_to_original_order(original_item, candidate_item)
            for original_item, candidate_item in zip(original, candidate)
        ]
    return candidate


def backup_json(path, label):
    path = Path(path)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.stem}.before_{label}_{timestamp}{path.suffix}")
    shutil.copy2(path, backup_path)
    return backup_path


def atomic_write_json(path, data):
    path = Path(path)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def validate_json_with_deepseek(json_path, system_prompt, user_prompt, config, label):
    """Validate and rewrite one JSON file, preserving exact structure."""
    json_path = Path(json_path)
    original = read_json(json_path)

    client = get_deepseek_client()
    model = get_validator_model(config)

    response = create_deepseek_chat_completion(
        client,
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content.strip()
    candidate = parse_llm_json(content)
    assert_same_structure(original, candidate)
    normalized = normalize_to_original_order(original, candidate)
    normalized = clean_json_text(normalized)

    backup_path = backup_json(json_path, label)
    atomic_write_json(json_path, normalized)

    return {
        "model": model,
        "json_path": str(json_path),
        "backup_path": str(backup_path),
    }


def validate_json_field_with_deepseek(json_path, field_name, system_prompt, user_prompt, config, label):
    """Validate and rewrite one JSON field, preserving the rest of the file exactly."""
    json_path = Path(json_path)
    original = read_json(json_path)
    if not isinstance(original, dict):
        raise ValueError(f"Expected object at {json_path} so field '{field_name}' can be updated.")
    if field_name not in original:
        raise KeyError(f"Missing field '{field_name}' in {json_path}")

    original_field = original[field_name]
    client = get_deepseek_client()
    model = get_validator_model(config)

    response = create_deepseek_chat_completion(
        client,
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content.strip()
    candidate = parse_llm_json(content)
    assert_same_structure(original_field, candidate, path=f"$.{field_name}")
    normalized_field = normalize_to_original_order(original_field, candidate)

    updated = dict(original)
    updated[field_name] = normalized_field
    updated = clean_json_text(updated)

    backup_path = backup_json(json_path, label)
    atomic_write_json(json_path, updated)

    return {
        "model": model,
        "json_path": str(json_path),
        "backup_path": str(backup_path),
    }

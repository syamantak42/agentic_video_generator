"""validate_narration.py

DeepSeek reviser for generated narration scripts.

Inputs:
    - outputs/output_jsons/narration.json
    - source_material/config.json

Output:
    - outputs/output_jsons/narration.json, rewritten after revision
    - outputs/output_jsons/narration.before_narration_revision_*.json backup

Usage:
    python validate_narration.py <project_name>
"""

import json
import os

from console_utils import configure_utf8_output
from prompt_loader import load_prompt, render_prompt
from validator_utils import (
    atomic_write_json,
    load_project_config,
    read_json,
    validate_json_field_with_deepseek,
)

configure_utf8_output()


def build_prompts(config, narration_data):
    video_title = config.get("video_title", narration_data.get("video_title", ""))
    narration_style = config.get("narration_style", [])
    sections_data = narration_data.get("sections", [])
    n_section = config.get("n_section", narration_data.get("n_sections", len(sections_data)))
    narration_config = config.get("_project_config", {}).get("narration_config", {})
    words_per_section = narration_config.get("words_per_section", 400)
    frames_per_section = narration_config.get("frames_per_section", 2)

    system_prompt = load_prompt("validate_narration_system.txt")
    user_prompt = render_prompt(
        "validate_narration_user.txt",
        video_title=video_title,
        n_section=n_section,
        words_per_section=words_per_section,
        frames_per_section=frames_per_section,
        narration_style_json=json.dumps(narration_style, ensure_ascii=False, indent=2),
        sections_json=json.dumps(sections_data, ensure_ascii=False, indent=2),
    )

    return system_prompt, user_prompt


def validate_rewritten_sections(sections):
    if not isinstance(sections, list) or not sections:
        raise ValueError("Narration revision must return a non-empty JSON array of sections.")

    for index, section in enumerate(sections, start=1):
        if not isinstance(section, dict):
            raise ValueError(f"Narration revision section {index} is not an object.")
        if not str(section.get("section_title", "")).strip():
            raise ValueError(f"Narration revision section {index} is missing section_title.")
        narration_text = section.get("narration_text")
        if not isinstance(narration_text, list) or not narration_text:
            raise ValueError(f"Narration revision section {index} has no narration_text list.")
        if not all(isinstance(item, str) and item.strip() for item in narration_text):
            raise ValueError(f"Narration revision section {index} contains empty narration text.")


def main():
    project, source_dir, config = load_project_config()
    project_root = os.path.dirname(source_dir)
    json_dir = os.path.join(project_root, "outputs", "output_jsons")
    narration_path = os.path.join(json_dir, "narration.json")

    if not os.path.exists(narration_path):
        raise FileNotFoundError(f"Missing narration JSON: {narration_path}")

    narration_data = read_json(narration_path)

    system_prompt, user_prompt = build_prompts(config, narration_data)
    result = validate_json_field_with_deepseek(
        json_path=narration_path,
        field_name="sections",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        config=config,
        label="narration_revision",
        temperature=1.2,
        preserve_structure=False,
        field_validator=validate_rewritten_sections,
    )

    updated_data = read_json(narration_path)
    updated_data["n_sections"] = len(updated_data.get("sections", []))
    atomic_write_json(narration_path, updated_data)

    print(f"[OK] Narration revised with {result['model']}")
    print(f"[OK] Backup saved to {result['backup_path']}")
    print(f"[OK] Updated JSON saved to {result['json_path']}")


if __name__ == "__main__":
    main()

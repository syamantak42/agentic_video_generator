"""validate_outline.py

DeepSeek reviser for generated section outlines.

Inputs:
    - outputs/output_jsons/outline_texts.json
    - source_material/config.json

Output:
    - outputs/output_jsons/outline_texts.json, rewritten with the same JSON structure
      after revision and editing
    - outputs/output_jsons/outline_texts.before_outline_revision_*.json backup

Usage:
    python validate_outline.py <project_name>
"""

import json
import os

from console_utils import configure_utf8_output
from prompt_loader import load_prompt, render_prompt
from validator_utils import load_project_config, read_json, validate_json_with_deepseek

configure_utf8_output()


def build_prompts(config, outline_data):
    video_title = config.get("video_title", "")
    section_guidelines = config.get("section_outlines", [])
    narration_style = config.get("narration_style", [])
    historical_context = config.get("historical_context", "")
    characters = config.get("characters", {})
    n_section = config.get("n_section", len(outline_data) if isinstance(outline_data, list) else "")

    system_prompt = load_prompt("validate_outline_system.txt")
    user_prompt = render_prompt(
        "validate_outline_user.txt",
        video_title=video_title,
        n_section=n_section,
        section_guidelines_json=json.dumps(section_guidelines, ensure_ascii=False, indent=2),
        narration_style_json=json.dumps(narration_style, ensure_ascii=False, indent=2),
        historical_context=historical_context,
        characters_json=json.dumps(characters, ensure_ascii=False, indent=2),
        outline_json=json.dumps(outline_data, ensure_ascii=False, indent=2),
    )

    return system_prompt, user_prompt


def main():
    project, source_dir, config = load_project_config()
    project_root = os.path.dirname(source_dir)
    json_path = os.path.join(project_root, "outputs", "output_jsons", "outline_texts.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing outline JSON: {json_path}")

    outline_data = read_json(json_path)
    system_prompt, user_prompt = build_prompts(config, outline_data)
    result = validate_json_with_deepseek(
        json_path=json_path,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        config=config,
        label="outline_revision",
    )

    print(f"[OK] Outline revised with {result['model']}")
    print(f"[OK] Backup saved to {result['backup_path']}")
    print(f"[OK] Updated JSON saved to {result['json_path']}")


if __name__ == "__main__":
    main()

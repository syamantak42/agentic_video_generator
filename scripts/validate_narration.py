"""validate_narration.py

DeepSeek reviser for generated narration scripts.

Inputs:
    - outputs/output_jsons/narration.json
    - outputs/output_jsons/outline_texts.json, if present
    - source_material/config.json

Output:
    - outputs/output_jsons/narration.json, rewritten with the same JSON structure
      after revision and editing
    - outputs/output_jsons/narration.before_narration_revision_*.json backup

Usage:
    python validate_narration.py <project_name>
"""

import json
import os

from console_utils import configure_utf8_output
from prompt_loader import load_prompt, render_prompt
from validator_utils import load_project_config, read_json, validate_json_field_with_deepseek

configure_utf8_output()


def build_prompts(config, narration_data, outline_data):
    video_title = config.get("video_title", narration_data.get("video_title", ""))
    narration_style = config.get("narration_style", [])
    historical_context = config.get("historical_context", "")
    characters = config.get("characters", {})
    sections_data = narration_data.get("sections", [])
    narration_config = config.get("_project_config", {}).get("narration_config", {})
    words_per_section = narration_config.get("words_per_section", 400)
    frames_per_section = narration_config.get("frames_per_section", 2)

    system_prompt = load_prompt("validate_narration_system.txt")
    user_prompt = render_prompt(
        "validate_narration_user.txt",
        video_title=video_title,
        words_per_section=words_per_section,
        frames_per_section=frames_per_section,
        narration_style_json=json.dumps(narration_style, ensure_ascii=False, indent=2),
        historical_context=historical_context,
        characters_json=json.dumps(characters, ensure_ascii=False, indent=2),
        outline_json=json.dumps(outline_data, ensure_ascii=False, indent=2),
        sections_json=json.dumps(sections_data, ensure_ascii=False, indent=2),
    )

    return system_prompt, user_prompt


def main():
    project, source_dir, config = load_project_config()
    project_root = os.path.dirname(source_dir)
    json_dir = os.path.join(project_root, "outputs", "output_jsons")
    narration_path = os.path.join(json_dir, "narration.json")
    outline_path = os.path.join(json_dir, "outline_texts.json")

    if not os.path.exists(narration_path):
        raise FileNotFoundError(f"Missing narration JSON: {narration_path}")

    narration_data = read_json(narration_path)
    outline_data = read_json(outline_path) if os.path.exists(outline_path) else {}

    system_prompt, user_prompt = build_prompts(config, narration_data, outline_data)
    result = validate_json_field_with_deepseek(
        json_path=narration_path,
        field_name="sections",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        config=config,
        label="narration_revision",
    )

    print(f"[OK] Narration revised with {result['model']}")
    print(f"[OK] Backup saved to {result['backup_path']}")
    print(f"[OK] Updated JSON saved to {result['json_path']}")


if __name__ == "__main__":
    main()

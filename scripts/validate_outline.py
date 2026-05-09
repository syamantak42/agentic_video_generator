"""validate_outline.py

DeepSeek validator for generated section outlines.

Inputs:
    - outputs/output_jsons/outline_texts.json
    - source_material/config.json

Output:
    - outputs/output_jsons/outline_texts.json, rewritten with the same JSON structure
      after validation and editing
    - outputs/output_jsons/outline_texts.before_outline_validation_*.json backup

Usage:
    python validate_outline.py <project_name>
"""

import json
import os

from validator_utils import load_project_config, read_json, validate_json_with_deepseek


def build_prompts(config, outline_data):
    video_title = config.get("video_title", "")
    section_guidelines = config.get("section_outlines", [])
    narration_style = config.get("narration_style", [])
    historical_context = config.get("historical_context", "")
    characters = config.get("characters", {})
    n_section = config.get("n_section", len(outline_data) if isinstance(outline_data, list) else "")

    system_prompt = """
You are a senior YouTube story editor and outline validator.

Your job is to review a generated video outline, silently judge whether it is coherent,
compelling, well-paced, specific, non-repetitive, and aligned with the user's project
requirements, then return an improved JSON version.

Strict rules:
- Return ONLY valid JSON.
- Preserve the original JSON structure exactly.
- Preserve the same top-level type.
- Preserve the same number of sections/items.
- Preserve every existing key exactly.
- Do not add keys.
- Do not remove keys.
- Do not change lists into objects or objects into lists.
- Do not change non-text primitive types.
- Edit only textual values where improvement is needed.
- Keep the outline suitable for a strong YouTube video: clear arc, strong opening,
  distinct section jobs, good transitions, specificity, no filler, no avoidable repetition.
""".strip()

    user_prompt = f"""
PROJECT CONTEXT
Video title:
{video_title}

Expected section count:
{n_section}

Original user section guidelines:
{json.dumps(section_guidelines, ensure_ascii=False, indent=2)}

Narration style requirements:
{json.dumps(narration_style, ensure_ascii=False, indent=2)}

Historical / subject context:
{historical_context}

Character canon:
{json.dumps(characters, ensure_ascii=False, indent=2)}

TASK
Review the outline JSON below as a validator agent.
Ask yourself internally:
- Does the outline form a coherent video from beginning to end?
- Does each section have a distinct purpose?
- Does it satisfy the user's requested subject, tone, structure, and section guidelines?
- Would it make a strong YouTube video rather than a flat encyclopedia summary?
- Are there missing transitions, repeated ideas, vague section descriptions, or weak pacing?

Then edit the outline JSON to make it more coherent, focused, and task-aligned.

Return ONLY the revised JSON. Keep the original JSON structure exactly.

ORIGINAL OUTLINE JSON
{json.dumps(outline_data, ensure_ascii=False, indent=2)}
""".strip()

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
        label="outline_validation",
    )

    print(f"[OK] Outline validated with {result['model']}")
    print(f"[OK] Backup saved to {result['backup_path']}")
    print(f"[OK] Updated JSON saved to {result['json_path']}")


if __name__ == "__main__":
    main()

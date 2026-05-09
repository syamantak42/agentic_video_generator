#!/usr/bin/env python3
"""
Pipeline orchestrator for agentic_video_generator.
Reads project config from {project}/source_material/config.json
Runs stages with user confirmation between steps.
"""

import subprocess
import sys
import os
import json


def load_config(project_name):
    """Load config from config.json"""
    # Scripts are in PROJECT_NAME/scripts/, so go up one level to PROJECT_NAME/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.join(base_dir, project_name)
    config_path = os.path.join(project_root, "source_material", "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config


def run_script(script_name, project_name):
    """Run a script and check for errors."""
    print(f"\nâ–¶ Running {script_name}...")
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
    result = subprocess.run([sys.executable, script_path, project_name])
    
    if result.returncode != 0:
        print(f"âœ— Error in {script_name}. Pipeline stopped.")
        return False
    
    print(f"âœ“ Completed {script_name}.")
    return True


def get_user_confirmation(step_name):
    """Ask user for confirmation before proceeding."""
    print(f"\n{'='*60}")
    print(f"Step: {step_name}")
    print(f"{'='*60}")
    
    while True:
        choice = input("\nOptions: [c]ontinue, [r]erun this step, [s]kip, [q]uit\n> ").strip().lower()
        
        if choice == 'c':
            return 'continue'
        elif choice == 'r':
            return 'rerun'
        elif choice == 's':
            return 'skip'
        elif choice == 'q':
            return 'quit'
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py PROJECT_NAME")
        sys.exit(1)
    
    project_name = sys.argv[1]
    
    try:
        config = load_config(project_name)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Agentic Video Generator - Pipeline")
    print(f"{'='*60}")
    print(f"Project: {project_name}")
    print(f"Title: {config.get('video_title', 'N/A')}")
    
    # Get TTS config
    tts_config = config.get("_project_config", {}).get("tts_config", {})
    tts_model = tts_config.get("model", "kokoro").lower()
    tts_voice = tts_config.get("voice_id", "af")
    print(f"TTS Model: {tts_model} | Voice: {tts_voice}")
    
    # Define pipeline stages
    stages = [
        ("1. Generate Sections", "generate_sections.py"),
        ("2. Generate Script (Narration)", "generate_script.py"),
        ("3. Generate Image Prompts", "generate_image_prompts.py"),
        ("4. Generate Images (Interactive)", "generate_images.py"),
        ("5. Generate Audio (TTS)", f"generate_{tts_model}_voice.py"),
        ("6. Generate Clips", "generate_clips.py"),
        ("7. Compose Video", "make_video.py"),
    ]
    
    completed_stages = []
    current_stage_idx = 0
    
    while current_stage_idx < len(stages):
        stage_name, script_name = stages[current_stage_idx]
        
        response = get_user_confirmation(stage_name)
        
        if response == 'quit':
            print("\nPipeline interrupted by user.")
            sys.exit(0)
        elif response == 'skip':
            print(f"Skipped: {stage_name}")
            current_stage_idx += 1
            continue
        elif response == 'rerun':
            print(f"Rerunning: {stage_name}")
            if run_script(script_name, project_name):
                completed_stages.append(stage_name)
                current_stage_idx += 1
            # If error, loop again to ask for retry
        elif response == 'continue':
            if run_script(script_name, project_name):
                completed_stages.append(stage_name)
                current_stage_idx += 1
            else:
                # Ask if user wants to retry or skip
                choice = input("\nRetry or skip? [r]etry, [s]kip, [q]uit\n> ").strip().lower()
                if choice == 'r':
                    continue
                elif choice == 's':
                    current_stage_idx += 1
                else:
                    sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"âœ“ Pipeline completed successfully!")
    print(f"{'='*60}")
    print(f"Completed stages: {len(completed_stages)}/{len(stages)}")
    print(f"Output location: {project_name}/outputs/output_jsons/")
    print(f"Video output: {project_name}/outputs/videos/{project_name}.mp4")








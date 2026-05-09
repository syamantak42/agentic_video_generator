"""Load prompt templates from scripts/Prompts."""

from pathlib import Path
from string import Template


PROMPT_DIR = Path(__file__).resolve().parent / "Prompts"


def load_prompt(filename):
    path = PROMPT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def render_prompt(filename, **values):
    template = Template(load_prompt(filename))
    string_values = {key: str(value) for key, value in values.items()}
    return template.safe_substitute(string_values).strip()

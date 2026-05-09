"""Text cleanup helpers for generated artifacts."""

import re
import unicodedata


UNICODE_REPLACEMENTS = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u201f": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
    "\u2026": "...",
    "\u00a0": " ",
    "\u200b": "",
    "\u200c": "",
    "\u200d": "",
    "\ufeff": "",
    "\ufffd": "",
}


def clean_text(text):
    """Normalize generated text to plain ASCII while preserving line breaks."""
    if not isinstance(text, str):
        return text

    for old, new in UNICODE_REPLACEMENTS.items():
        text = text.replace(old, new)

    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n"))
    return text.strip()


def clean_json_text(value):
    """Recursively clean strings inside JSON-like structures."""
    if isinstance(value, dict):
        return {key: clean_json_text(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json_text(item) for item in value]
    if isinstance(value, str):
        return clean_text(value)
    return value

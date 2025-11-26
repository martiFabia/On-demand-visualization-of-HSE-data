# services/prompt_loader.py

from pathlib import Path


def load_user_query(path: Path) -> str:
    """
    Load the user query from the default file
    """
    if not path.exists():
        raise FileNotFoundError(f"User prompt file not found: {path}")

    content = path.read_text(encoding="utf-8").strip()

    if not content:
        raise ValueError(f"User prompt file is empty: {path}")

    return content


def load_text_file(path: str) -> str:
    """
    Generic text loader used for system prompts or user prompts.
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {p}")

    text = p.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"Prompt file is empty: {p}")

    return text


# services/file_io.py

from pathlib import Path

DEFAULT_OUTPUT_PATH = Path("../../chart_recommendation/recommendation.txt")

def save_text_file(text: str, output_path: Path = DEFAULT_OUTPUT_PATH) -> None:
    """
    Save the generated text to the default output file.
    The path is internal and not passed as an argument.
    """

    # Ensure parent folder exists
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_path.write_text(text, encoding="utf-8")
    

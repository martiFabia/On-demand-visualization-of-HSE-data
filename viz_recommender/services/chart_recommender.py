# services/chart_recommender.py

from pathlib import Path
from models.llm_client import OpenAILLMClient


def build_full_prompt(
    data_profile_str: str,
    user_query: str,
    system_prompt: str = str,
) -> str:
    """
    Build the full prompt to send to the LLM.
    The OpenAILLMClient.invoke() expects a single string (system+user already combined).
    """
    return f"""{system_prompt}

---

### DATA PROFILE (Use these exact columns):
{data_profile_str}

### USER QUERY:
"{user_query}"
"""


def generate_chart_recommendation(
    llm_client: OpenAILLMClient,
    full_prompt: str,
) -> str:
    """
    Use the LLM client to generate a chart recommendation text.
    """
    return llm_client.invoke(full_prompt)


def save_text_to_file(text: str, output_path: str) -> Path:
    """
    Save the generated text to a file and return the path.
    """
    out_path = Path(output_path)
    out_path.write_text(text, encoding="utf-8")
    return out_path

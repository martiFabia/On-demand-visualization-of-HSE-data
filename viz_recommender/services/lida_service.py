# services/lida_service.py

from typing import Any
import pandas as pd
from lida import Manager, llm


def create_lida_manager(api_key: str) -> Manager:
    """
    Initialize a LIDA Manager using OpenAI as text generator.
    """
    text_gen = llm("openai", api_key=api_key)
    return Manager(text_gen=text_gen)


def load_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    """
    return pd.read_csv(csv_path)


def summarize_dataframe(
    df: pd.DataFrame,
    lida_manager: Manager,
    summary_method: str = "detailed",
) -> str:
    """
    Run LIDA summarization on the given DataFrame and return the summary as string.
    """
    summary = lida_manager.summarize(df, summary_method=summary_method)
    return str(summary)

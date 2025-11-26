import json
from pathlib import Path
import pandas as pd
from typing import Dict, Any

def save_intent_to_file(intent: dict, output_path: str) -> None:
    """
    Save the intent dictionary to a readable .json file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(intent, f, indent=2, ensure_ascii=False)

def load_test_intent(path: str) -> dict:
    """
    Load a test intent or a real intent file.
    This is only an example — replace with your own path.
    """
    intent_path = Path(path)

    if not intent_path.exists():
        raise FileNotFoundError(
            f"Intent file not found. Expected at: {intent_path}"
        )

    with intent_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_sql_results_to_csv(
    results: Dict[str, Dict[str, Any]],
    output_dir: str | Path,
) -> Dict[str, "pd.DataFrame"]:
    """
    Save every SQL generated (ottenuto da execute_sql_on_sqlite) 
    as a separate CSV file in the given folder.

    Parameters:
      - results: dict created by execute_sql_on_sqlite(...)
      - output_dir: output folder (default: aggregate_dataset)
    Output:
      - Creates files like:
        _output_dir_/main_query.csv
        _output_dir_/extra_insight_query_1.csv
        ...
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dfs: Dict[str, pd.DataFrame] = {}

    for key, payload in results.items():
        cols = payload["columns"]
        rows = payload["rows"]

        df = pd.DataFrame(rows, columns=cols)

        # Safe filename (no spaces or odd characters)
        safe_key = key.replace(" ", "_").replace("/", "_").split(":")[0]
        print(f"{safe_key} table created...")

        out_path = output_dir / f"{safe_key}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")

        dfs[safe_key] = df

        print(f"[✓] Saved: {out_path}")

    return dfs
    


    
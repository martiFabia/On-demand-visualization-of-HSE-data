from __future__ import annotations
from typing import Dict, List, Any, Tuple
import sqlite3


def parse_llm_sql_response(response: str) -> Dict[str, str]:
    """
    Parse the SQL string returned by the LLM into separate queries.

    Expected format (as enforced in your prompt):

        -- MAIN QUERY
        SELECT ...

        -- EXTRA INSIGHT QUERY 1
        SELECT ...

        -- EXTRA INSIGHT QUERY 2
        SELECT ...

    Returns a dict mapping a key (e.g. "main_query", "extra_insight_query_1")
    to the corresponding SQL string.
    """
    lines = response.splitlines()

    queries: Dict[str, List[str]] = {}
    current_key: str | None = None

    def normalize_label(label_line: str) -> str:
        # strip leading '--' and spaces
        label = label_line.lstrip("- ").strip().lower()
        # e.g. "main query" -> "main_query"
        label = label.replace(" ", "_")
        # ensure some reasonable key
        return label

    for line in lines:
        stripped = line.strip()

        # Detect label lines starting with "--"
        if stripped.startswith("--"):
            key = normalize_label(stripped)
            current_key = key
            if current_key not in queries:
                queries[current_key] = []
        else:
            # Normal SQL line, append to current query (if any)
            if current_key is None:
                # If the model didn't include labels at all, treat everything as main query
                current_key = "main_query"
                if current_key not in queries:
                    queries[current_key] = []
            queries[current_key].append(line)

    # Join lines into SQL strings and strip empty ones
    sql_queries: Dict[str, str] = {}
    for key, sql_lines in queries.items():
        sql_text = "\n".join(sql_lines).strip()
        if sql_text:
            sql_queries[key] = sql_text

    return sql_queries


def execute_sql_on_sqlite(
    db_path: str,
    sql_response: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Given:
      - db_path: path to your SQLite database file
      - sql_response: the raw string returned by the LLM (with labelled queries)

    1. Parse the response into separate queries.
    2. Execute each query against the SQLite DB.
    3. Return a dict with:
         {
           "main_query": {
               "sql": "<the SQL text>",
               "columns": ["col1", "col2", ...],
               "rows": [(...), (...), ...]
           },
           "extra_insight_query_1": { ... },
           ...
         }

    NOTE: This assumes each block contains a single SELECT statement.
    """

    def _round_numeric_values(row: Tuple[Any, ...]) -> Tuple[Any, ...]:
        """
        Round all numeric values in a row to two decimal places.
        Non-numeric values are returned unchanged.
        """
        rounded_row = []
        for value in row:
            if isinstance(value, (int, float)):
                # Round numeric values to 2 decimal places
                rounded_row.append(round(value, 2))
            else:
                rounded_row.append(value)
        return tuple(rounded_row)

    queries = parse_llm_sql_response(sql_response)

    results: Dict[str, Dict[str, Any]] = {}

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()

        for key, sql in queries.items():
            # Execute the query
            cursor.execute(sql)
            rows = cursor.fetchall()

            col_names: List[str] = (
                [desc[0] for desc in cursor.description] if cursor.description else []
            )

            round_rows = [_round_numeric_values(row) for row in rows]

            # Skip empty result sets (no rows returned)
            if not rows:
                print(f"------- {key.replace(' ', '_').replace('/', '_').split(':')[0]} -------\n")
                print(f">>> Executes \n\t{sql}\n script, 0 rows retrieved. Skipping empty dataset.\n")
                print(f"-----------------------------------------\n")
                continue

            results[key] = {
                "sql": sql,
                "columns": col_names,
                "rows": round_rows,
            }
            print(f"------- {key.replace(' ', '_').replace('/', '_').split(':')[0]} -------\n")
            print(f">>> Executes \n\t{sql}\n script, {len(round_rows)} rows retrieved.\n")
            print(f"-----------------------------------------\n")

    finally:
        conn.close()

    return results


# Optional helper: convert results to pandas DataFrames (if you use pandas)
def results_to_dataframes(
    results: Dict[str, Dict[str, Any]]
) -> Dict[str, "pd.DataFrame"]:  # type: ignore[name-defined]
    """
    Convert the output of execute_sql_on_sqlite into pandas DataFrames.
    Requires pandas installed.
    """
    import pandas as pd

    dfs: Dict[str, pd.DataFrame] = {}

    for key, payload in results.items():
        cols = payload["columns"]
        rows = payload["rows"]
        df = pd.DataFrame(rows, columns=cols)
        dfs[key] = df

    return dfs


    

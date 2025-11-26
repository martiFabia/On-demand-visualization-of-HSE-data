# analytics_table.py

from __future__ import annotations
from typing import List, Dict, Any
import json
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path


def load_assignments(assignments_path: str | Path) -> List[Dict[str, Any]]:
    assignments_path = Path(assignments_path)
    with assignments_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Assignments JSON must be a list of records")

    return data


def build_analytics_dataframe(
    assignments: List[Dict[str, Any]],
    include_raw_index: bool = True,
) -> pd.DataFrame:

    dimension_types = set()
    for rec in assignments:
        assignments_dict = rec.get("assignments", {}) or {}
        for dim_type in assignments_dict.keys():
            dimension_types.add(dim_type)

    dim2col = {dim: dim.lower() for dim in dimension_types}

    rows: List[Dict[str, Any]] = []
    for rec in assignments:
        row: Dict[str, Any] = {}

        if include_raw_index:
            row["row_id"] = rec.get("row_index")

        obs_date = rec.get("observation_date")
        proc_date = rec.get("processed_date")

        row["observation_date"] = obs_date
        row["processed_date"] = proc_date

        for dim_type, col_name in dim2col.items():
            row[col_name] = None

        assignments_dict = rec.get("assignments", {}) or {}
        for dim_type, category in assignments_dict.items():
            col_name = dim2col[dim_type]
            row[col_name] = category

        rows.append(row)

    df = pd.DataFrame(rows)

    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    df["processed_date"] = pd.to_datetime(df["processed_date"], errors="coerce")

    df["processing_time_days"] = (
        (df["processed_date"] - df["observation_date"])
        .dt.total_seconds()
        .div(86400.0)
    )

    df["event_year"] = df["observation_date"].dt.year
    df["event_month"] = df["observation_date"].dt.month

    return df


def save_dataframe_to_sqlite(
    df: pd.DataFrame,
    db_path: str | Path,
    table_name: str = "observations_enriched",
    if_exists: str = "replace",
) -> None:

    db_path = Path(db_path)
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    finally:
        conn.close()


def save_dataframe_to_csv(
    df: pd.DataFrame,
    csv_path: str | Path
) -> None:
    """
    Save the DataFrame also as a CSV file.
    """
    csv_path = Path(csv_path)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Saved CSV to {csv_path}")



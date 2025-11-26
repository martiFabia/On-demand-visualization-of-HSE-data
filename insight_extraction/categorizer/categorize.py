from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional

import json
import pandas as pd

# --- Import from intern modules ---
from insight_extraction.categorizer.embedding.model_loader import load_embedding_model
from insight_extraction.categorizer.my_io.save_json import save_assignment_json

from insight_extraction.categorizer.my_io.data_loader import load_observations_df
from insight_extraction.categorizer.embedding.embedder import embed_texts, embed_categories
from insight_extraction.categorizer.matching.multi_matcher import match_all_dimensions
from insight_extraction.categorizer.analysis import (
    print_category_stats,
    plot_dimension_summary,
    plot_support_vs_mean_score,
    plot_category_support_bar,
    print_cluster_examples)
def build_assignment_json(
    df: pd.DataFrame,
    all_best_idx: Dict[str, Any],
    dim2cat_embs: Dict[str, Dict[str, Any]],
    obs_date_col: str = "Observation_date",
    proc_date_col: str = "Processed_date",
    max_examples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Build a list of JSON-serializable records containing category
    assignments for each row in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with observations (must contain the date columns).
    all_best_idx : Dict[str, np.ndarray]
        For each dimension_type, an array of category indices (or -1).
    dim2cat_embs : Dict[str, Dict[str, np.ndarray]]
        For each dimension_type, a mapping category_name -> embedding.
        We only use the keys to recover the category name.
    obs_date_col : str
        Observation date column name.
    proc_date_col : str
        Processed date column name.
    max_examples : Optional[int]
        If set, limits the maximum number of rows to export.

    Returns
    -------
    List[Dict[str, Any]]
        List of records ready to be saved as JSON.
    """
    records: List[Dict[str, Any]] = []
    n_rows = len(df)

    if max_examples is not None:
        n_rows = min(n_rows, max_examples)

    for i in range(n_rows):
        row = df.iloc[i]

        rec: Dict[str, Any] = {
            "row_index": int(i),
            "observation_date": (
                row[obs_date_col].isoformat()
                if pd.notnull(row[obs_date_col])
                else None
            ),
            "processed_date": (
                row[proc_date_col].isoformat()
                if pd.notnull(row[proc_date_col])
                else None
            ),
            "assignments": {},
        }

        for dim_type, best_idx in all_best_idx.items():
            if i >= len(best_idx):
                continue

            ci = int(best_idx[i])
            if ci == -1:
                # no category assigned
                continue

            cat_names = list(dim2cat_embs[dim_type].keys())
            if ci < 0 or ci >= len(cat_names):
                continue

            rec["assignments"][dim_type] = cat_names[ci]

        records.append(rec)

    return records


def run_pipeline(
    df: pd.DataFrame,
    intent_path: str | Path,
    output_path: str | Path = "assignments.json",
    title_col: str = "Title",
    obs_col: str = "Observation",
    obs_date_col: str = "Observation_date",
    proc_date_col: str = "Processed_date",
    model_name: str = "all-MiniLM-L6-v2",
    expansions_path: Optional[str | Path] = None,
    similarity_threshold: float = 0.4,
    min_support_ratio: float = 0.01,
 
    max_examples: Optional[int] = None,
) -> None:
    """
    Run the full categorization pipeline:

    1. Load observations from Excel.
    2. Load the intent / categories JSON.
    3. Compute embeddings for observations.
    4. Compute embeddings for categories.
    5. Perform matching for all dimensions.
    6. Build JSON assignment records.
    7. Save the output to a file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the observations.
    intent_path : str | Path
        Path to the JSON file with category definitions
        (e.g. structure with "group_by").
    output_path : str | Path
        Where to save the JSON assignments output.
    sheet_name : str
        Excel sheet name to load.
    title_col, obs_col, obs_date_col, proc_date_col : str
        Column names in the Excel file.
    model_name : str
        SentenceTransformer model name to use.
    similarity_threshold : float
        Minimum cosine similarity threshold to consider a category.
    min_support_ratio : float
        Minimum support ratio to keep a category.
    max_examples : Optional[int]
        Optional cap on the number of rows to process.
    """

    intent_path = Path(intent_path)
    output_path = Path(output_path)
    

    # 1. Load observations
    print(f"[1/7] Format Dataset")
    df = load_observations_df(
        df=df,
        title_col=title_col,
        obs_col=obs_col,
        obs_date_col=obs_date_col,
        proc_date_col=proc_date_col,
    )

    # 2. Load intent JSON
    print(f"[2/7] Carico intent JSON da: {intent_path}")
    with intent_path.open("r", encoding="utf-8") as f:
        intent = json.load(f)
    
    # 2b. Load expansions if provided
    expansions = None
    if expansions_path is not None:
        expansions_path = Path(expansions_path)
        print(f"[2b/7] Carico expansions da: {expansions_path}")
        with expansions_path.open("r", encoding="utf-8") as f:
            expansions = json.load(f)
    # 3. Load model
    print(f"[3/7] Carico modello di embedding: {model_name}")
    model = load_embedding_model(model_name=model_name)

    # 4. Embed observations
    print("[4/7] Calcolo embedding delle osservazioni...")
    texts = df["text_for_embedding"].tolist()
    obs_embs = embed_texts(model, texts)

    # 5. Embed categories
    print("[5/7] Calcolo embedding delle categorie...")
    if expansions is None:
        raise ValueError(
            "Hai chiamato embed_categories senza expansions. "
            "Devi passare expansions_path a run_pipeline()."
        )
    dim2cat_embs = embed_categories(model, intent, expansions)

    # 6. Matching for all dimensions
    print("[6/7] Eseguo il matching categorie...")
    all_stats, all_best_idx = match_all_dimensions(
        intent=intent,
        obs_embs=obs_embs,
        dim2cat_embs=dim2cat_embs,
        similarity_threshold=similarity_threshold,
        min_support_ratio=min_support_ratio,
    )
    print_category_stats(all_stats)

    # Summary plot per dimension
    plot_dimension_summary(all_stats)

    # Plot support vs mean_score
    plot_support_vs_mean_score(all_stats)  # tutte le dimensioni insieme
    # or for a specific dimension
    # plot_support_vs_mean_score(all_stats, dimension_type="OBSERVATION_TYPE")

    # Bar chart of top categories for a dimension
    plot_category_support_bar(
        all_stats,
        dimension_type="OBSERVATION_TYPE",
        top_n=10,
        normalize=False,
    )

    # Text clustering (console only)
    print_cluster_examples(
        df=df,
        all_best_idx=all_best_idx,
        dim2cat_embs=dim2cat_embs,
        text_col="text_for_embedding",
        max_examples_per_category=5,
    )

    # (Optional) you can log a small summary of the stats
    for dim, stats in all_stats.items():
        print(f"  - Dimensione '{dim}': {len(stats)} categorie attive")

    # 7. Build records + save JSON
    print("[7/7] Costruisco i record di assegnazione...")
    records = build_assignment_json(
        df=df,
        all_best_idx=all_best_idx,
        dim2cat_embs=dim2cat_embs,
        obs_date_col=obs_date_col,
        proc_date_col=proc_date_col,
        max_examples=max_examples,
    )

    print(f"Salvo {len(records)} record in: {output_path}")
    save_assignment_json(records, str(output_path))

    print("✅ Pipeline completata.")


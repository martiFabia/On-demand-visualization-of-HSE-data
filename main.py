from __future__ import annotations
import os
from datetime import datetime
import json
from pathlib import Path
import time
import pandas as pd

from insight_extraction.categorizer.categorize import run_pipeline
from insight_extraction.semantic_intent.semantic_intent import get_semantic_intent
from insight_extraction.semantic_intent.expander import (
    expand_dimension_categories,
)
from models.llm_client import OpenAILLMClient
from insight_extraction.utils.saving_scripts import save_intent_to_file
from insight_extraction.extraction.extract import define_queries, extract_insights
from from_text_to_streamlit_app.prompts.text_to_json_prompt import get_text_to_json_prompt
from from_text_to_streamlit_app.utils import clean_response, from_csv_to_dict, json_to_streamlit
from viz_recommender.services.chart_recommender import build_full_prompt, generate_chart_recommendation
from viz_recommender.services.file_io import save_text_file
from viz_recommender.services.lida_service import create_lida_manager, load_dataframe, summarize_dataframe
from viz_recommender.services.prompt_loader import load_text_file

DATA_DIR = Path("datasets")
OUT_DIR = Path("output")
USR_PROMPT_DIR = Path("initial_prompts")
RECOMMENDATION_DIR = Path("chart_recommendation")


def main(user_prompt: str, df: pd.DataFrame, run_id: str) -> None:
    
    # Real OpenAI client (assumes OPENAI_API_KEY in the environment)
    llm_client = OpenAILLMClient(
        model_name="gpt-4.1",  # o "gpt-4o", ecc.
        temperature=0.0,
        max_output_tokens=2400,
    )

    print(">>> User question:\t")
    print(user_prompt)
    print("\n>>> Calling LLM for semantic intent...\n")

    # ------------------------------------------------------------------
    # 2. Intent extraction
    # ------------------------------------------------------------------
    print(">>>>>>>>> -------- Intent extraction ------- <<<<<<<<<\n")
    intent = get_semantic_intent(
        user_question=user_prompt,
        llm_client=llm_client,
    )

    print(">>> Parsed intent JSON:")
    print(json.dumps(intent, indent=2, ensure_ascii=False))

    INTENT_DIR = OUT_DIR / "intents"
    os.makedirs(INTENT_DIR, exist_ok=True)

    intent_path = INTENT_DIR / f"intent_{run_id}.json"
    save_intent_to_file(intent, str(intent_path))

    print(f"\n>>> JSON salvato in: {intent_path}")

    # ------------------------------------------------------------------
    # 3. Categories expansion (for dimension_type in group_by)
    # ------------------------------------------------------------------
    print("\n>>>>>>>>> -------- Categories expansion ------- <<<<<<<<<\n")

    EXPANSIONS_DIR = OUT_DIR / "expansions"
    os.makedirs(EXPANSIONS_DIR, exist_ok=True)

    all_expansions: dict[str, dict[str, any]] = {}

    for group in intent.get("group_by", []):
        dim_type = group.get("dimension_type")
        values = list(dict.fromkeys(group.get("values", [])))  # unique

        if not dim_type or not values:
            continue

        print(f"--- Expanding dimension: {dim_type} ({len(values)} values)")

        expanded = expand_dimension_categories(
            dimension_type=dim_type,
            values=values,
            llm_client=llm_client,
            extra_context=(
                "HSE domain: worker safety observations, near misses, "
                "hazards, incidents, environmental and quality issues."
            ),
        )

        all_expansions[dim_type] = expanded

        exp_path = EXPANSIONS_DIR / f"expansion_{dim_type}_{run_id}.json"
        with exp_path.open("w", encoding="utf-8") as f:
            json.dump(expanded, f, indent=2, ensure_ascii=False)

        print(f"Saved expansion for {dim_type} to: {exp_path}\n")

    # single file with all expansions (used by the pipeline)
    expansions_all_path = EXPANSIONS_DIR / f"expansions_all_{run_id}.json"
    with expansions_all_path.open("w", encoding="utf-8") as f:
        json.dump(all_expansions, f, indent=2, ensure_ascii=False)

    print(f">>> All expansions saved to: {expansions_all_path}\n")

    # ------------------------------------------------------------------
    # 4. Categorization pipeline
    # ------------------------------------------------------------------
    print("\n>>>>>>>>> -------- Categorization ------- <<<<<<<<<\n")
    print("Run categorization pipeline...\n")

    allocation_path = OUT_DIR / f"allocation_{run_id}.json"

    run_pipeline(
        df=df,
        intent_path=intent_path,
        output_path=allocation_path,
        model_name="all-MiniLM-L6-v2",
        expansions_path=expansions_all_path,
        similarity_threshold=0.2,
        min_support_ratio=0.01,
    )

    print(f">>> Saved file with categories allocations to: {allocation_path}\n")

    # ------------------------------------------------------------------
    # 5. Insights extraction (SQL)
    # ------------------------------------------------------------------
    print("\n>>>>>>>>> -------- Insights extraction ------- <<<<<<<<<\n")

    DB_DIR = OUT_DIR / "db"
    DB_DIR.mkdir(parents=True, exist_ok=True)

    CSV_DIR = OUT_DIR / "csv"
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    db_path = DB_DIR / f"raw_insights_{run_id}.db"
    csv_path = CSV_DIR / f"raw_insights_{run_id}.csv"

    print(">>> Define and run queries to extract insights...\n")
    sql_code = define_queries(
        llm_client=llm_client,
        allocation_path=allocation_path,
        user_prompt=user_prompt,
        intent=intent,
        db_path=str(db_path),
        csv_path=str(csv_path),
    )

    INSIGHTS_DIR = DATA_DIR / "extracted"
    INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    insights_dfs = extract_insights(
        db_path=str(db_path),
        sql_code=sql_code,
        output_dir=str(INSIGHTS_DIR),
    )

    print(f">>> {len(insights_dfs)} tables generated\n\n")
    insights_name = list(insights_dfs.keys())

    # ------------------------------------------------------------------
    # 6. Chart recommendation
    # ------------------------------------------------------------------
    print(">>>>>>>>>>>> -------- Recommending chart ------- <<<<<<<<<\n")

    system_prompt = load_text_file("viz_recommender/prompts/viz_prompt.txt")

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable not found.")

    lida_manager = create_lida_manager(api_key=api_key)

    os.makedirs(RECOMMENDATION_DIR, exist_ok=True)
    for df_n, file in enumerate(os.listdir(INSIGHTS_DIR)):
        if file.endswith(".csv"):
            csv_path = INSIGHTS_DIR / file

            print(f">>> Generating data profile with LIDA for {file.split('.')[0]} ...")
            df = load_dataframe(str(csv_path))
            data_profile_str = summarize_dataframe(df, lida_manager, summary_method="detailed")

            full_prompt = build_full_prompt(
                data_profile_str=data_profile_str,
                user_query=user_prompt,
                system_prompt=system_prompt,
            )

            print(f">>> Analyzing {insights_name[df_n]} with LLM...\n")
            recommend_survey = generate_chart_recommendation(llm_client, full_prompt)

            recommendation_path = Path(RECOMMENDATION_DIR) / f"{insights_name[df_n]}.txt"
            save_text_file(recommend_survey, recommendation_path)
        else:
            print(f"⚠️⚠️⚠️ Skipping non-CSV file: {file} ⚠️⚠️⚠️\n")
            continue
        
    print("\n>>>>>>>>> -------- Generating Streamlit app ------- <<<<<<<<<\n")
    datasets = from_csv_to_dict()
    prompt = get_text_to_json_prompt(datasets, RECOMMENDATION_DIR)
    response = llm_client.invoke(prompt)
    print(response)

    cleaned_response = clean_response(response)
    
    workflow = json.loads(cleaned_response)
    json_to_streamlit(workflow, data_sources=datasets)


if __name__ == "__main__":
    obs_id = 4

    prompt_path = USR_PROMPT_DIR / f"prompt_{obs_id}.txt"
    df_path = DATA_DIR / f"data_{obs_id}.xlsx"

    with open(prompt_path, "r", encoding="utf-8") as f:
        user_prompt = f.read()

    df = pd.read_excel(df_path, engine="openpyxl")
    
    
    main(user_prompt = user_prompt, df=df, run_id = obs_id)
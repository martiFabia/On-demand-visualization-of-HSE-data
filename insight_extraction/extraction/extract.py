from insight_extraction.extraction.table_creator import *
from insight_extraction.extraction.sql_execute import *
from insight_extraction.extraction.sql_generate import SQLQueryGenerator
from insight_extraction.utils.saving_scripts import save_sql_results_to_csv

def define_queries(llm_client: Any,
                   allocation_path: str, 
                   user_prompt: str, 
                   intent: Dict[str, Any],
                   db_path: str, 
                   csv_path: str, 
                   ) -> None:
    
    assignments = load_assignments(allocation_path)
    df = build_analytics_dataframe(assignments)

    save_dataframe_to_sqlite(df, db_path)
    save_dataframe_to_csv(df, csv_path)
    

    # Example schema (you can adapt it to data_en)
    schema_text = df.columns.tolist()

    generator = SQLQueryGenerator(llm_client=llm_client, sql_dialect="SQLite")

    sql_code = generator.generate_sql(
        user_question=user_prompt,
        json_spec=intent,
        main_table="observations_enriched",  # TODO: adapt with dynamic name
        schema_text=schema_text,

    )

    return sql_code

def extract_insights(db_path: str, sql_code: str, output_dir: str) -> None:
    exec_results = execute_sql_on_sqlite(db_path=db_path, sql_response=sql_code)
    
    return save_sql_results_to_csv(exec_results, output_dir=output_dir)

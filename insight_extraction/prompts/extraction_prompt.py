from __future__ import annotations
from typing import Any, Dict, Optional
import json


def _summarize_categories_from_intent(json_spec: Dict[str, Any]) -> str:
    """
    Costruisce una breve lista testo delle dimensioni e delle categorie
    a partire dall'intent JSON (campo group_by).
    """
    lines = []
    for group in json_spec.get("group_by", []):
        dim = group.get("dimension_type")
        vals = group.get("values", [])
        if not dim or not vals:
            continue
        uniq_vals = list(dict.fromkeys(vals))
        vals_str = ", ".join(uniq_vals)
        lines.append(f"- {dim}: {vals_str}")
    return "\n".join(lines) if lines else "No explicit categories."





def build_extraction_prompt(
    user_question: str,
    json_spec: Dict[str, Any],
   
    main_table: str,
    sql_dialect: str = "SQLite",
    schema_text: Optional[Any] = None,
) -> str:
    """
    Prompt for SQL generation, consistent with the existing parser
    which expects labelled blocks with comments:

        -- MAIN QUERY
        SELECT ...

        -- EXTRA_INSIGHT_QUERY_1
        SELECT ...

    The LLM must return ONLY SQL in this format (no JSON).
    """

    categories_summary = _summarize_categories_from_intent(json_spec)




    prompt = f"""
You are an SQL query generation assistant.

Your task:
- Read the user question.
- Read the semantic intent JSON (metrics, time, filters, group_by).
- Read the table schema and the list of dimensions/categories.
- Optionally, read rich category expansions for more semantic context.
- Generate one or more SQL SELECT queries that answer the question.

IMPORTANT:
- SQL dialect: {sql_dialect}.
- Main table to query: {main_table}.
- Use ONLY the columns that exist in the schema.
- Do NOT invent new column names.
- Focus on the metrics and groupings requested in the intent.
- If multiple metrics are requested, generate MULTIPLE queries.
- Expand the analysis with extra queries for breakdowns / trends / distributions,
  but always consistent with the dataset and the intent.
- All queries must be valid SELECT statements (no DDL / no updates / no deletes).
- DO NOT use forbidden SQL functions (MIN, MAX, window functions, etc.) if your constraints require so.

OUTPUT FORMAT (MANDATORY):
- You MUST output ONLY raw SQL, with one or more labelled blocks.
- Each block MUST be preceded by a comment line starting with `--`.
- Example structure:

  -- MAIN QUERY
  SELECT ...

  -- EXTRA_INSIGHT_QUERY_1
  SELECT ...

  -- EXTRA_INSIGHT_QUERY_2
  SELECT ...

- Do NOT output JSON.
- Do NOT output explanations.
- Do NOT wrap SQL in backticks or markdown.
- Do NOT add any text before the first comment or after the last query.

Guidelines for labelling:
- The primary query answering the main metric(s) MUST be under label:
    -- MAIN QUERY
- Additional useful queries (secondary metrics, per-category breakdowns,
  trend analysis, etc.) should be labelled as:
    -- EXTRA_INSIGHT_QUERY_1
    -- EXTRA_INSIGHT_QUERY_2
    ... and so on.

CATEGORIES (from intent.group_by):
{categories_summary}
SCHEMA COLUMNS THAT YOU MUST FOLLOW FOR GENERATING QUERIES:
{', '.join(schema_text) if schema_text else 'No schema information provided.'}





USER QUESTION:
{user_question}

Now generate the SQL queries ONLY, following the exact labelled-block format.
    """.strip()

    return prompt

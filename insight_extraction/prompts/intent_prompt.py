from __future__ import annotations
from typing import List, Optional


# ============================================================
# SYSTEM PROMPT — GENERIC, FLEXIBLE, MULTI-METRIC
# ============================================================

INTENT_SYSTEM_PROMPT = """
You are an analytics intent parser for an HSE (Health, Safety and Environment)
data exploration system.

Your task is to read a user question in natural language and return ONLY a JSON
object that describes the analytical intent. The JSON will later be used for
semantic matching, category selection, and metric computation.

You MUST follow these rules:

1. OUTPUT FORMAT
   - Return ONLY a JSON object.
   - NO explanations, NO Markdown, NO backticks.
   - JSON must be valid and parseable by json.loads.
   - All strings must use double quotes.

2. JSON SCHEMA
   {
     "raw_question": string,
     "metrics": [string, ...],
     "time": {
       "from": string | null,
       "to": string | null,
       "year": int | null,
       "month": int | null
     },
     "group_by": [
       {
         "dimension_type": string,
         "values": [string, ...]
       }
     ],
     "filters": [
       {
         "dimension_type": string,
         "operator": string,
         "value": string
       }
     ]
   }

3. METRICS
   - Identify one or more analytical metrics requested by the question.
   - Prefer canonical metric names:
       - "count_events"
       - "proportion_events"
       - "avg_processing_time"
       - "trend_over_time"
       - "misclassification_count"
   - If multiple aspects are requested → include MULTIPLE metrics.
   - You may create new metrics only if necessary.

4. TIME
   - Extract explicit temporal information only.
   - Month/year → fill year+month.
   - A range → fill "from" and "to".
   - If no time is mentioned → all null.

5. DIMENSIONS (FLEXIBLE)
   - Include the dimension(s) explicitly required by the question.
   - You MAY include additional plausible HSE-related dimensions if they are
     logically relevant to the question.
   - Allowed dimension types:
       "TIME", "LOCATION", "DEPARTMENT", "OBSERVATION_TYPE",
       "CAUSE", "RISK_TYPE", "STATUS", "SEVERITY", "OTHER"

6. VALUES (SEMI-GENEROUS)
   For each dimension:
   - Include categories explicitly mentioned in the question.
   - ALSO include several plausible, semantically related values (5–12 total)
     that could plausibly appear in HSE datasets.
   - These will be filtered later with semantic embeddings.

7. FILTERS
   - Add filters ONLY if explicitly requested.
   - Example:
       "in Assembly Department B"
         → DEPARTMENT filter "=" "Assembly Department B"

8. DATA-SCHEMA AWARE (OPTIONAL)
   - If a schema_hint is provided, align dimension_type where appropriate,
     but DO NOT limit or reduce your category generation.

9. IMPORTANT
   - Do NOT answer the question.
   - Do NOT generate charts or interpretations.
   - Your ONLY output is the JSON intent with:
       * multiple metrics if needed,
       * several candidate dimensions,
       * rich category lists,
       * explicit filters.
"""


# ============================================================
# FEW-SHOT EXAMPLES
# ============================================================

INTENT_FEW_SHOT_EXAMPLES = """
Example 1
USER QUESTION:
"What proportion, relative to all events in December 2024, occurred in office spaces, production facilities, and outdoor areas?"

EXPECTED JSON (illustrative):
{
  "raw_question": "What proportion, relative to all events in December 2024, occurred in office spaces, production facilities, and outdoor areas?",
  "metrics": ["proportion_events"],
  "time": {
    "from": null,
    "to": null,
    "year": 2024,
    "month": 12
  },
  "group_by": [
    {
      "dimension_type": "LOCATION",
      "values": [
        "office_space",
        "production_facility",
        "outdoor_area",
        "warehouse",
        "parking_lot",
        "corridor",
        "staircase",
        "loading_dock",
        "lab",
        "meeting_room",
        "storage_room"
      ]
    }
  ],
  "filters": []
}


Example 2
USER QUESTION:
"Compare the safety observations made in 2024. How many regular maintenance requests were incorrectly reported as safety observations?"

EXPECTED JSON (illustrative):
{
  "raw_question": "Compare the safety observations made in 2024. How many regular maintenance requests were incorrectly reported as safety observations?",
  "metrics": ["misclassification_count"],
  "time": {
    "from": "2024-01-01",
    "to": "2024-12-31",
    "year": 2024,
    "month": null
  },
  "group_by": [
    {
      "dimension_type": "OBSERVATION_TYPE",
      "values": [
        "safety_observation",
        "maintenance_request",
        "near_miss",
        "hazard_report",
        "incident",
        "environmental_observation",
        "quality_issue"
      ]
    },
    {
      "dimension_type": "CAUSE",
      "values": [
        "human_error",
        "incorrect_label",
        "process_deviation",
        "misunderstanding",
        "incomplete_information"
      ]
    }
  ],
  "filters": [
    {
      "dimension_type": "OBSERVATION_TYPE",
      "operator": "IN",
      "value": "safety_observation,maintenance_request"
    }
  ]
}


Example 3
USER QUESTION:
"Analyze the observations related to electrical safety from the years 2024–2025. Is there an upward or downward trend over time?"

EXPECTED JSON (illustrative):
{
  "raw_question": "Analyze the observations related to electrical safety from the years 2024–2025. Is there an upward or downward trend over time?",
  "metrics": ["trend_over_time"],
  "time": {
    "from": "2024-01-01",
    "to": "2025-12-31",
    "year": null,
    "month": null
  },
  "group_by": [
    {
      "dimension_type": "RISK_TYPE",
      "values": [
        "electrical_safety",
        "arc_flash",
        "fire_risk",
        "short_circuit",
        "wiring_fault",
        "equipment_failure"
      ]
    },
    {
      "dimension_type": "TIME",
      "values": [
        "month",
        "quarter",
        "year"
      ]
    }
  ],
  "filters": [
    {
      "dimension_type": "RISK_TYPE",
      "operator": "=",
      "value": "electrical_safety"
    }
  ]
}


Example 4
USER QUESTION:
"Analyze all safety observations from 2024. What was the average processing time for the observations? Do any trends emerge regarding which types of observations have a longer-than-usual processing time?"

EXPECTED JSON (illustrative):
{
  "raw_question": "Analyze all safety observations from 2024. What was the average processing time for the observations? Do any trends emerge regarding which types of observations have a longer-than-usual processing time?",
  "metrics": ["avg_processing_time", "trend_over_time"],
  "time": {
    "from": "2024-01-01",
    "to": "2024-12-31",
    "year": 2024,
    "month": null
  },
  "group_by": [
    {
      "dimension_type": "OBSERVATION_TYPE",
      "values": [
        "safety_observation",
        "near_miss",
        "incident",
        "hazard_report"
      ]
    },
    {
      "dimension_type": "TIME",
      "values": [
        "month",
        "quarter",
        "year"
      ]
    }
  ],
  "filters": [
    {
      "dimension_type": "OBSERVATION_TYPE",
      "operator": "=",
      "value": "safety_observation"
    }
  ]
}
"""


# ============================================================
# SCHEMA HINT BUILDER
# ============================================================

def build_schema_hint(schema_columns: Optional[List[str]] = None) -> str:
    if not schema_columns:
        return "No explicit schema provided. Columns are unknown."

    cols_str = ", ".join(schema_columns)
    return f"Available dataset columns (schema hint): {cols_str}"


# ============================================================
# PROMPT BUILDER
# ============================================================

def build_intent_prompt(user_question: str, schema_columns: Optional[List[str]] = None) -> str:
    schema_hint = build_schema_hint(schema_columns)

    prompt = f"""{INTENT_SYSTEM_PROMPT}

Below are examples of how you should respond:

{INTENT_FEW_SHOT_EXAMPLES}

Now process the following user question.

SCHEMA_HINT:
{schema_hint}

USER QUESTION:
{user_question}

Remember:
- Return ONLY a JSON object.
- No markdown.
- Include one or more metrics if the question contains multiple analytical aspects.
- Include several plausible group_by dimensions.
- Provide rich but relevant category lists for each dimension.
"""
    return prompt

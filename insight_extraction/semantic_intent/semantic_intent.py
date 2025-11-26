from __future__ import annotations
import json
from typing import Any, Dict, List, Optional
from ..prompts.intent_prompt import build_intent_prompt

# ---------------------------------------------------------------------
# Constants: logical schema and system prompt
# ---------------------------------------------------------------------



def parse_intent_response(raw_response: str) -> Dict[str, Any]:
    """
    Try to parse the model response as JSON.
    If there are extra characters before/after, attempt a conservative cleanup.
    In production you may add logging and more robust fallbacks here.
    """
    raw = raw_response.strip()

    # Ideal case: it's already pure JSON
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Simple fallback: try to extract the first complete {...}
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = raw[first_brace:last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # If it still fails, raise (in production you might retry with a fresh prompt)
    raise ValueError(f"LLM response is not valid JSON: {raw_response}")


# ---------------------------------------------------------------------
# Main function: abstract API call to the LLM
# ---------------------------------------------------------------------

def get_semantic_intent(
    user_question: str,
    llm_client: Any,
) -> Dict[str, Any]:
    """
    Main function of the first block.

    - Build the prompt for the LLM.
    - Ask the model to return ONLY JSON with:
        raw_question, metric, time, group_by, filters, focus_topics.
    - Parse the JSON and return it as a Python dict.

    Parameters
    ----------
    user_question : str
        Natural language question from the user (e.g. "What proportion ... ?").
    llm_client : Any
        Object encapsulating the LLM call. It should expose `.invoke(prompt: str) -> str`
        or `.generate(prompt: str) -> str`.
        Adapt this wrapper to your infrastructure (OpenAI, Bedrock, etc.).
    schema_columns : Optional[List[str]]
        List of available column names in the dataframe/SQL result
        (e.g. ["Created", "Status", "Division", "ObservationCause", ...]).
        Used as a hint to make the intent more data-aware.

    Returns
    -------
    intent : Dict[str, Any]
        Dictionary with keys:
          - "raw_question"
          - "metric"
          - "time"
          - "group_by"
          - "filters"
          - "focus_topics"
    """
    # dataframe structure extraction
    schema_columns = [
        "Created",
        "Status",
        "Division",
        "ObservationCause",
        "Location",
        "ProcessingTimeDays",
        "ObservationType",
        "Department",
        "RiskType",
    ]
    prompt = build_intent_prompt(user_question, schema_columns)

    # Adapt this part to your LLM implementation.
    # Generic example:
    if hasattr(llm_client, "invoke"):
        raw_response = llm_client.invoke(prompt)
    elif hasattr(llm_client, "generate"):
        raw_response = llm_client.generate(prompt)
    else:
        raise TypeError(
            "llm_client must expose an 'invoke(prompt: str) -> str' or "
            "'generate(prompt: str) -> str' method."
        )

    # In some SDKs the response is a complex object: assume a string here.
    if not isinstance(raw_response, str):
        # Try to extract text from common structures
        try:
            raw_response = raw_response["content"]
        except Exception:
            raw_response = str(raw_response)

    intent = parse_intent_response(raw_response)
    return intent


# ---------------------------------------------------------------------
# Usage example (remove or keep as a manual test)
# ---------------------------------------------------------------------



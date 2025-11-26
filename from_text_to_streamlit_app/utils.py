import streamlit as st
import pandas as pd
from pathlib import Path


STREAMLIT_FRIENDLY_JSON_SCHEMA = """
        {
          "components": [
            {
              "id": "unique string",
              "type": "Streamlit API element (e.g. markdown, metric, line_chart)",
              "args": {
                "data": ["list of positional args"],
                "config": {"keyword": "value", ...}
              },
              "dependencies": {
                "inputs": ["variables from previous components or datasets"],
                "outputs": ["variables this component produces"]
              },
              "layout": {
                "area": "main | sidebar | column",
                "column": "integer (optional)",
                "expander": "string (optional)"
              }
            }
          ]
        }
        """

RECOMMENDATION_OUTPUT_PATH = "chart_recommendation/recommendation.txt"

EXTRACTED_DATASETS_PATH = "datasets/extracted"

def from_csv_to_dict(datasets_path = EXTRACTED_DATASETS_PATH):
    datasets_path = Path(datasets_path)
    datasets = {}

    for csv_file in datasets_path.glob("*.csv"):
        dataset_name = csv_file.stem
        datasets[dataset_name] = pd.read_csv(csv_file)

    return datasets


def extract_best_visualization(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
    part2_started = False
    result_lines = []
    
    for line in lines:
        if "PART 2:" in line:
            part2_started = True
        if part2_started:
            result_lines.append(line)
    
    return ''.join(result_lines)


def resolve_data(json_component, workflow_state):
    '''
    Resolves data references in the component's args against the current workflow state.
    Returns a list of resolved data objects, or None if any reference is invalid.
    '''
    args = json_component.get("args", {})
    
    if isinstance(args, dict):
        current_data = args.get("data", [])
    elif isinstance(args, list):
        current_data = args
    else:
        current_data = []

    resolved_data = []
    for d in current_data:
        if isinstance(d, str):
            if d in workflow_state:
                resolved_data.append(workflow_state[d])
            else:
                # invalid string → reject the whole component
                return None
        else:
            resolved_data.append(d)
    return resolved_data


def render_advanced_chart(json_component, workflow_state):
    '''
    Renders advanced chart types that require specific Streamlit methods.
    Supported types: plotly_chart, altair_chart, vega_lite_chart, graphviz_chart, pydeck_chart
    '''
    comp_type = json_component.get("type")
    cfg = json_component.get("args", {}).get("config", {})

    # resolve data if needed
    resolved_data = resolve_data(json_component, workflow_state)
    if resolved_data is None:
        return

    if comp_type == "plotly_chart":
        fig = cfg.get("figure_or_data")  # must be a plotly Figure object
        if fig is not None:
            st.plotly_chart(fig, config=cfg.get("config"), width=cfg.get("width"), theme=cfg.get("theme"))

    elif comp_type == "altair_chart":
        chart = cfg.get("altair_chart")  # must be an Altair Chart object
        if chart is not None:
            st.altair_chart(chart, width=cfg.get("width"), height=cfg.get("height"), theme=cfg.get("theme"))

    elif comp_type == "vega_lite_chart":
        data = cfg.get("data")          # dataframe or list of dicts
        spec = cfg.get("spec")          # Vega-Lite spec
        if data is not None or spec is not None:
            st.vega_lite_chart(data, spec=spec, width=cfg.get("width"), height=cfg.get("height"), theme=cfg.get("theme"))

    elif comp_type == "graphviz_chart":
        figure_or_dot = cfg.get("figure_or_dot")  # dot string or Graphviz figure
        if figure_or_dot is not None:
            st.graphviz_chart(figure_or_dot, width=cfg.get("width"), height=cfg.get("height"))

    elif comp_type == "pydeck_chart":
        deck = cfg.get("pydeck_obj")  # must be a pydeck.Deck object
        if deck is not None:
            st.pydeck_chart(deck, width=cfg.get("width"), height=cfg.get("height"))


def render_component(json_component, workflow_state, columns_map=None):
    # resolve layout
    layout = json_component.get("layout")
    current_streamlit_element = st

    if layout.get("area") == "sidebar":
        current_streamlit_element = st.sidebar
    
    col_idx = layout.get("column")
    if col_idx is not None and columns_map:
        current_streamlit_element = columns_map.get(col_idx, current_streamlit_element)
    
    # resolve current visualization element
    type = json_component.get("type", "")
    if type in ["plotly_chart", "altair_chart", "vega_lite_chart", "pydeck_chart", "graphviz_chart"]:
        render_advanced_chart(type, json_component, workflow_state)
    attribute = current_streamlit_element
    attribute = getattr(attribute, type)

    # extract and validate data references
    resolved_data = resolve_data(json_component, workflow_state)
    if resolved_data is None:
        # skip component entirely if it references nonexistent datasets
        return
        
    # extract configuration for current element
    current_config = json_component.get("args", {}).get("config", {})
    resolved_config = {k: workflow_state.get(v, v) if isinstance(v, str) else v for k, v in current_config.items()}

    # add workflow dependencies if any
    for dependency in json_component["dependencies"].get("inputs", []):
        if dependency in workflow_state:
            current_config[dependency] = workflow_state[dependency]

    def call_attribute():
        try:
            if json_component["type"] in ["markdown", "caption"]:
                # For markdown/caption, only pass the config body
                return attribute(resolved_config.get("body", ""))
            else:
                return attribute(*resolved_data, **resolved_config)
        except TypeError:
            # fallback: try only first positional argument
            if resolved_data:
                return attribute(resolved_data[0], **resolved_config)
            else:
                return attribute(**resolved_config)
            
    # call the expander function if needed
    expander_title = layout.get("expander")
    if expander_title:
        with current_streamlit_element.expander(expander_title):
            output = call_attribute()
    else:
        output = call_attribute()

    # store element outputs if any (future dependency inputs)
    for var in json_component["dependencies"].get("outputs", []):
        workflow_state[var] = output


def json_to_streamlit(workflow, data_sources: dict[str, pd.DataFrame] = None):
    '''
    Dynamically executes the LLM-generated JSON workflow in the Streamlit app,
    without providing readable Python code
    '''
    if data_sources is None:
        data_sources = {}

    workflow_state = {**data_sources}

    # preprocess columns if any component uses the "column" layout
    max_col_idx = max(
        (comp.get("layout", {}).get("column", -1) for comp in workflow.get("components", [])),
        default=-1,
    )
    columns_map = {i: col for i, col in enumerate(st.columns(max_col_idx + 1))} if max_col_idx >= 0 else {}

    print("Choosing the best components to render ...")
    # render each component
    for comp in workflow.get("components", []):
        render_component(comp, workflow_state, columns_map)

    print("Visualization elements JSON successfully created!")
    return workflow_state


def clean_response(response):
    '''
    Cleans the LLM response to extract valid JSON content.
    '''
    raw = response.strip()
    # Hard-clean markdown fences if present
    if raw.startswith("```"):
        raw = raw.strip("`")
        # Remove potential "json" specifier
        if raw.startswith("json"):
            raw = raw[len("json"):].strip()
    return raw
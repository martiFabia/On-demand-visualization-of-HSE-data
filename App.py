# app.py
import streamlit as st
import pandas as pd
import main

# ============================
# Placeholder: your LLM logic
# ============================
def get_chart_recommendations(user_query: str, df: pd.DataFrame) -> str:
    main.main(user_prompt=user_query, df=df, run_id=0)  # You can modify this to pass the DataFrame directly
    return ()


# ==============
# Streamlit app
# ==============
st.set_page_config(page_title="HSE Chart Recommender", layout="centered")

st.title("📊 Chart Recommendation Demo")

# -----------------------------------------------------------
# SHORT DESCRIPTION OF THE FUNCTIONALITIES
# -----------------------------------------------------------
st.markdown(
    """
### 📝 What this application does

This simple interface allows you to:

1. **Upload a CSV dataset**  
   (e.g., the dataset retrieved by your HSE RAG application)

2. **Write a natural-language request**  
   describing what you want to analyse or visualize  
   (e.g., *"Show the trend of safety observations in 2024"*).

3. **Generate chart recommendations**  
   The system processes the prompt + dataset and proposes  
   the most suitable visualization types for your analysis.

Use this tool to test your LLM-based visualization recommendation pipeline.
"""
)

# --- Upload CSV ---
uploaded_file = st.file_uploader(
    "Upload your dataset (.csv or .xlsx)",
    type=["csv", "xlsx"],
    help="Select the CSV or Excel file returned by your RAG / HSE system",
)

# --- User request ---
user_query = st.text_area(
    "User request",
    placeholder='e.g. "What proportion, relative to all events in December 2024, occurred in office spaces, production facilities, and outdoor areas?"',
    height=100,
)

run_button = st.button("Run recommendation")

if run_button:
    df = None
    if uploaded_file is None:
        st.error("Please upload a CSV file first.")
    elif not user_query.strip():
        st.error("Please write a request in the text box.")
    else:
        # Read CSV as DataFrame
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file, engine="openpyxl")
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            st.stop()

        st.success("✅ File loaded successfully!")

        # Show a small preview of the dataset
        st.subheader("Dataset preview")
        st.dataframe(df.head())

        st.success("Data loaded! Wait for data visualization.")

        with st.spinner("Generating chart recommendations..."):
            recommendation_text = get_chart_recommendations(user_query, df)


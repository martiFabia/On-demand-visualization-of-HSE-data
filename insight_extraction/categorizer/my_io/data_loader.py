import pandas as pd
from typing import Optional

def load_observations_df(df: pd.DataFrame,
                         title_col: str = "Title",
                         obs_col: str = "Observation",
                         obs_date_col: str = "Observation_date",
                         proc_date_col: str = "Processed_date",
                         ) -> pd.DataFrame:

    # make a copy to avoid mutating the original df passed as input
    df = df.copy()

    df["text_for_embedding"] = (
        df[title_col].fillna("").astype(str) + " " +
        df[obs_col].fillna("").astype(str)
    ).str.strip()

    df[obs_date_col] = pd.to_datetime(df[obs_date_col], errors="coerce")
    df[proc_date_col] = pd.to_datetime(df[proc_date_col], errors="coerce")

    return df


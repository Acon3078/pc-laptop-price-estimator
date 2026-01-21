import pandas as pd
import streamlit as st

from .config import DATA_PATH


@st.cache_data
def load_dataset():
    """Load the cleaned dataset if present."""
    if not DATA_PATH.exists():
        return None
    try:
        return pd.read_parquet(DATA_PATH)
    except Exception:
        return None


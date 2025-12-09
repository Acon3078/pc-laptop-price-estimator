from __future__ import annotations

import pandas as pd
import joblib
import streamlit as st

from .config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    MODEL_PATH,
    NUMERIC_FEATURES,
    PREPROCESSOR_PATH,
)


@st.cache_resource
def load_pipeline():
    """Load the persisted sklearn pipeline."""
    if not MODEL_PATH.exists():
        st.error(
            f"Model file not found: {MODEL_PATH}. "
            "Run the training notebook to export price_prediction_pipeline.joblib."
        )
        st.stop()
    pipeline = joblib.load(MODEL_PATH)

    # Backward-compatibility for sklearn<=1.4.2
    preprocess = getattr(pipeline, "named_steps", {}).get("preprocess")
    if preprocess is not None and not hasattr(preprocess, "force_int_remainder_cols"):
        preprocess.force_int_remainder_cols = False

    return pipeline


@st.cache_resource
def load_preprocessor():
    """Load the preprocessing pipeline (either standalone or from the full pipeline)."""
    if PREPROCESSOR_PATH.exists():
        pre = joblib.load(PREPROCESSOR_PATH)
        if not hasattr(pre, "force_int_remainder_cols"):
            pre.force_int_remainder_cols = False
        return pre

    pipeline = load_pipeline()
    pre = pipeline.named_steps.get("preprocess", None)
    if pre is not None and not hasattr(pre, "force_int_remainder_cols"):
        pre.force_int_remainder_cols = False
    return pre


def get_feature_defaults(preprocessor) -> dict:
    """
    Use the SimpleImputer statistics inside the numeric/categorical
    pipelines as default values for the UI.
    """
    defaults = {}

    if preprocessor is None:
        # fallback generic defaults if preprocessor not available
        for f in NUMERIC_FEATURES:
            defaults[f] = 0.0
        for f in CATEGORICAL_FEATURES:
            defaults[f] = ""
        return defaults

    transformers = getattr(preprocessor, "named_transformers_", {})

    # Numeric defaults
    num_pipeline = transformers.get("num")
    if num_pipeline is not None:
        num_imputer = num_pipeline.named_steps.get("imputer")
        if hasattr(num_imputer, "statistics_"):
            for feature, value in zip(NUMERIC_FEATURES, num_imputer.statistics_):
                defaults[feature] = float(value)

    # Categorical defaults
    cat_pipeline = transformers.get("cat")
    if cat_pipeline is not None:
        cat_imputer = cat_pipeline.named_steps.get("imputer")
        if hasattr(cat_imputer, "statistics_"):
            for feature, value in zip(CATEGORICAL_FEATURES, cat_imputer.statistics_):
                defaults[feature] = value

    # Fallbacks if anything missing
    for f in NUMERIC_FEATURES:
        defaults.setdefault(f, 0.0)
    for f in CATEGORICAL_FEATURES:
        defaults.setdefault(f, "")

    return defaults


def get_categorical_options(preprocessor) -> dict:
    """
    Read the OneHotEncoder categories_ from the categorical pipeline
    so the selectboxes show exactly the categories seen during training.
    """
    options = {f: [] for f in CATEGORICAL_FEATURES}

    if preprocessor is None:
        return options

    transformers = getattr(preprocessor, "named_transformers_", {})
    cat_pipeline = transformers.get("cat")
    if cat_pipeline is None:
        return options

    encoder = cat_pipeline.named_steps.get("onehot")
    if encoder is None or not hasattr(encoder, "categories_"):
        return options

    for feature, cats in zip(CATEGORICAL_FEATURES, encoder.categories_):
        options[feature] = list(cats)

    return options


def build_input_row(numeric_inputs: dict, categorical_inputs: dict) -> pd.DataFrame:
    """Construct a single-row DataFrame in the correct column order."""
    data = {}
    data.update(numeric_inputs)
    data.update(categorical_inputs)

    # Ensure all features present, with None if missing
    for f in ALL_FEATURES:
        data.setdefault(f, None)

    return pd.DataFrame([data], columns=ALL_FEATURES)


def predict_price(pipeline, row_df: pd.DataFrame) -> float:
    """Use the final unified pipeline from the notebook (expects preprocessing inside)."""
    preds = pipeline.predict(row_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    return float(preds[0])


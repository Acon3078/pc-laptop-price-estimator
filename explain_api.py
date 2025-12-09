# explain_api.py – SHAP-based local explanation for the Streamlit app

import joblib
import numpy as np
import pandas as pd
import shap

from app_core.config import (
    ALL_FEATURES,
    MODEL_PATH,
    NUMERIC_FEATURES,
    SHAP_BACKGROUND_PATH,
)

# -------------------------------------------------------------------
# -------------------------------------------------------------------
# LOAD PIPELINE + BACKGROUND DATA + BUILD SHAP EXPLAINER
# -------------------------------------------------------------------
_pipeline = joblib.load(MODEL_PATH)

# Extract preprocessor + model from the pipeline
_preprocess = _pipeline.named_steps["preprocess"]
_model = _pipeline.named_steps["model"]

# Background sample for SHAP (saved from the notebook)
X_background = joblib.load(SHAP_BACKGROUND_PATH)
X_background = X_background[ALL_FEATURES]
X_background_transformed = _preprocess.transform(X_background)

# Names of transformed features (after one-hot, scaling, etc.)
_feature_names_transformed = _preprocess.get_feature_names_out()

# SHAP TreeExplainer (for tree-based model; works fine for RF / XGB)
_explainer = shap.TreeExplainer(_model, X_background_transformed)


# -------------------------------------------------------------------
# HELPER: FRIENDLY LABELS FOR FEATURES
# -------------------------------------------------------------------
FRIENDLY_NAMES = {
    "num__cpu_bench_value": "CPU market value",
    "num__cpu_bench_mark": "CPU performance score",
    "num__cpu_bench_rank": "CPU performance rank",

    "num__gpu_bench_mark": "GPU performance score",
    "num__gpu_bench_value": "GPU market value",

    "num__ram_gb": "RAM (GB)",
    "num__total_storage_gb": "Total storage (GB)",
    "num__total_ssd_gb": "SSD capacity (GB)",
    "num__total_hdd_gb": "HDD capacity (GB)",

    "num__screen_inches": "Screen size (inches)",
    "num__screen_refresh_hz": "Screen refresh rate (Hz)",
}

def _friendly_label(transformed_name: str) -> str:
    """Map transformed feature names to something readable for humans."""
    if transformed_name in FRIENDLY_NAMES:
        return FRIENDLY_NAMES[transformed_name]

    if transformed_name.startswith("cat__"):
        # cat__Tipo de producto_Portátil multimedia -> "Tipo de producto: Portátil multimedia"
        raw = transformed_name.replace("cat__", "")
        raw = raw.replace("_", " ")
        col, _, val = raw.partition(" ")
        return f"{col}: {val}"

    if transformed_name.startswith("num__"):
        return transformed_name.replace("num__", "")

    return transformed_name


# -------------------------------------------------------------------
# CORE: EXPLAIN A SINGLE PREDICTION
# -------------------------------------------------------------------
def _explain_single_prediction(row_df: pd.DataFrame, top_k: int = 8) -> dict:
    """
    row_df: single-row DataFrame with the same columns as the training features.
    Returns:
        - predicted_price
        - base_price (model baseline)
        - contributions: DataFrame sorted by |SHAP value|
    """
    # 1) Predict price with the full pipeline
    pred_price = float(_pipeline.predict(row_df)[0])

    # 2) Transform row using the preprocessor
    X_row_transformed = _preprocess.transform(row_df)

    # 3) Compute SHAP values
    shap_values = _explainer(X_row_transformed)
    shap_vals_row = shap_values.values[0]

    # Expected (baseline) value of the model
    expected = _explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        base_price = float(np.array(expected).mean())
    else:
        base_price = float(expected)

    # 4) Build contributions DataFrame
    contrib_df = pd.DataFrame(
        {
            "feature_transformed": _feature_names_transformed,
            "shap_value": shap_vals_row,
        }
    )
    contrib_df["abs_shap"] = contrib_df["shap_value"].abs()
    contrib_df = contrib_df.sort_values("abs_shap", ascending=False).head(top_k)

    # Direction of effect: raises or lowers price
    contrib_df["effect_direction"] = np.where(
        contrib_df["shap_value"] > 0, "raises", "lowers"
    )

    # Friendly labels for UI / report
    contrib_df["feature_friendly"] = contrib_df["feature_transformed"].apply(
        _friendly_label
    )

    return {
        "predicted_price": pred_price,
        "base_price": base_price,
        "contributions": contrib_df.reset_index(drop=True),
    }


def _summarize_explanation(explanation: dict, n: int = 3) -> str:
    """
    Build a short text summary from the top-n SHAP contributions.
    """
    df = explanation["contributions"].head(n)
    sentences = []
    for _, row in df.iterrows():
        delta = abs(row["shap_value"])
        sentences.append(
            f"{row['feature_friendly']} {row['effect_direction']} the price by about {delta:.0f} €."
        )
    return " ".join(sentences)


# -------------------------------------------------------------------
# PUBLIC API FOR THE WEB APP
# -------------------------------------------------------------------
def predict_and_explain(row_df: pd.DataFrame, top_k: int = 8) -> dict:
    """
    Main function for the Streamlit app.

    Parameters
    ----------
    row_df : pd.DataFrame (one row)
        Must contain the same columns as used for training:
        NUMERIC_FEATURES + CATEGORICAL_FEATURES.
    top_k : int
        Number of feature contributions to keep.

    Returns
    -------
    dict with keys:
        - 'predicted_price' : float
        - 'base_price'      : float  (model baseline)
        - 'contributions'   : pd.DataFrame (top-k features)
        - 'summary'         : short natural-language explanation
    """
    # Ensure correct column order
    row_df = row_df[ALL_FEATURES]

    explanation = _explain_single_prediction(row_df, top_k=top_k)
    explanation["summary"] = _summarize_explanation(explanation)
    return explanation

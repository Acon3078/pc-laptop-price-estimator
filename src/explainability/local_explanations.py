"""
Local explanation utilities for SHAP-based model interpretability.

This module provides functions for generating human-readable explanations of
individual model predictions using SHAP (SHapley Additive exPlanations) values.
It includes feature name mapping, SHAP value computation, and natural-language
summarization of feature contributions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union


# Mapping of transformed feature names to human-readable labels
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


def friendly_label(transformed_name: str) -> str:
    """
    Map transformed feature names to human-readable labels.

    Converts technical feature names (e.g., `num__cpu_bench_value`) into
    user-friendly labels (e.g., "CPU market value") for display in UI or reports.
    Handles three naming patterns:
    1. Predefined mappings in FRIENDLY_NAMES dictionary
    2. Categorical one-hot encoded features: `cat__ColumnName_Value` -> "ColumnName: Value"
    3. Numeric features: `num__feature_name` -> "feature_name"

    Args:
        transformed_name (str): Technical feature name from the preprocessed pipeline,
            typically prefixed with `num__` or `cat__`.

    Returns:
        str: Human-readable label for the feature. If no mapping is found, returns
            the input string as-is.

    Error-handling:
        Returns the input string unchanged if it doesn't match any known pattern.
        No exceptions are raised for invalid inputs.
    """
    if transformed_name in FRIENDLY_NAMES:
        return FRIENDLY_NAMES[transformed_name]

    if transformed_name.startswith("cat__"):
        # Categorical one-hot pattern: cat__ColumnName_Value -> "ColumnName: Value"
        # Example: cat__Tipo de producto_Gaming Laptop -> "Tipo de producto: Gaming Laptop"
        # partition(" ") handles multi-word column names correctly (splits on first space)
        raw = transformed_name.replace("cat__", "")
        raw = raw.replace("_", " ")
        col, _, val = raw.partition(" ")
        return f"{col}: {val}"

    if transformed_name.startswith("num__"):
        # Numeric features: strip num__ prefix
        return transformed_name.replace("num__", "")

    # Fallback: return as-is for unknown patterns
    return transformed_name


def explain_single_prediction(
    row_df: pd.DataFrame,
    pipeline: Any,
    preprocess: Any,
    explainer: Any,
    feature_names: np.ndarray,
    top_k: int = 8,
) -> Dict[str, Any]:
    """
    Compute SHAP values for a single prediction and return structured explanation.

    This function generates a local explanation for an individual prediction by:
    1. Predicting the price using the full pipeline
    2. Transforming the input through the preprocessor
    3. Computing SHAP values using the TreeExplainer
    4. Building a DataFrame of feature contributions sorted by importance

    SHAP values have the additive property: sum(shap_values) = predicted_price - base_price.
    Positive SHAP values indicate features that raise the price above the baseline;
    negative values indicate features that lower the price below the baseline.

    Args:
        row_df (pd.DataFrame): Single-row DataFrame containing the same columns as
            the training features. Must match the feature set used during model training.
        pipeline (Any): Trained prediction pipeline (e.g., sklearn Pipeline) with
            `predict()` method that accepts a DataFrame and returns price predictions.
        preprocess (Any): Preprocessor component (e.g., ColumnTransformer) with
            `transform()` method that transforms raw features to model-ready format.
        explainer (Any): SHAP TreeExplainer instance initialized with the trained model
            and background dataset. Must have `__call__()` method and `expected_value`
            attribute.
        feature_names (np.ndarray): Array of transformed feature names (after preprocessing),
            typically obtained via `preprocess.get_feature_names_out()`. Used to label
            SHAP contributions in the output DataFrame.
        top_k (int, optional): Number of top features to return, ranked by absolute
            SHAP value. Defaults to 8. Limits output to most important features for
            readability.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - `predicted_price` (float): The model's predicted price for the input row.
            - `base_price` (float): The model's baseline/expected value (average prediction
                over the background dataset).
            - `contributions` (pd.DataFrame): DataFrame with columns:
                - `feature_transformed`: Technical feature name
                - `shap_value`: SHAP contribution value (can be positive or negative)
                - `abs_shap`: Absolute value of SHAP contribution
                - `effect_direction`: "raises" or "lowers" (price relative to baseline)
                - `feature_friendly`: Human-readable feature label
                Rows are sorted by `abs_shap` in descending order, limited to top_k.

    Error-handling:
        Raises KeyError if `row_df` is missing required columns.
        Raises AttributeError if `pipeline`, `preprocess`, or `explainer` don't have
            the expected methods (`predict`, `transform`, `__call__`).
        Raises IndexError if `row_df` is empty or doesn't have exactly one row.
        Raises ValueError if `feature_names` length doesn't match SHAP values length.
    """
    # 1) Predict price
    pred_price = float(pipeline.predict(row_df)[0])

    # 2) Transform row using the preprocessor
    X_row_transformed = preprocess.transform(row_df)

    # 3) Compute SHAP values
    shap_values = explainer(X_row_transformed)
    shap_vals_row = shap_values.values[0]

    # Expected (baseline) value of the model
    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        base_price = float(np.array(expected).mean())
    else:
        base_price = float(expected)

    # 4) Build contributions DataFrame
    contrib_df = pd.DataFrame(
        {
            "feature_transformed": feature_names,
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
        friendly_label
    )

    return {
        "predicted_price": pred_price,
        "base_price": base_price,
        "contributions": contrib_df.reset_index(drop=True),
    }


def summarize_explanation(explanation: Dict[str, Any], n: int = 3) -> str:
    """
    Build a short natural-language summary from the top-n SHAP contributions.

    Generates a human-readable text description explaining how the top features
    affect the predicted price. Each feature's contribution is described as either
    "raises" or "lowers" the price by a specific amount (in euros).

    Args:
        explanation (Dict[str, Any]): Explanation dictionary returned by
            `explain_single_prediction()`. Must contain a `contributions` key with
            a DataFrame that has columns:
            - `feature_friendly`: Human-readable feature name
            - `effect_direction`: "raises" or "lowers"
            - `shap_value`: SHAP contribution value
        n (int, optional): Number of top features to include in the summary.
            Defaults to 3. Features are taken from the top of the contributions
            DataFrame (already sorted by importance).

    Returns:
        str: Natural-language summary sentence(s) describing how the top-n features
            affect the predicted price. Format: "Feature1 raises the price by about X €.
            Feature2 lowers the price by about Y €. ..." Sentences are joined with
            spaces.

    Error-handling:
        Raises KeyError if `explanation` doesn't contain a `contributions` key.
        Raises KeyError if the contributions DataFrame is missing required columns
            (`feature_friendly`, `effect_direction`, `shap_value`).
        Returns empty string if `n` is 0 or if contributions DataFrame is empty.
    """
    df = explanation["contributions"].head(n)
    sentences = []
    for _, row in df.iterrows():
        delta = abs(row["shap_value"])
        sentences.append(
            f"{row['feature_friendly']} {row['effect_direction']} the price by about {delta:.0f} €."
        )
    return " ".join(sentences)


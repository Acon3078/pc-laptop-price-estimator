"""
Explainability utilities for model interpretability.

This package provides functions for generating local (instance-level) explanations
of model predictions using SHAP values, including feature name mapping and
natural-language summarization.
"""

from .local_explanations import (
    FRIENDLY_NAMES,
    friendly_label,
    explain_single_prediction,
    summarize_explanation,
)

__all__ = [
    "FRIENDLY_NAMES",
    "friendly_label",
    "explain_single_prediction",
    "summarize_explanation",
]


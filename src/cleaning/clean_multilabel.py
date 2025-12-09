"""
Clean multilabel fields by taking the first label.

This module handles fields containing multiple values
separated by "/", "+", ",", or "|".
"""

import re
import pandas as pd
import numpy as np


def take_first_label(value, separators=None):
    """
    Extract the first label from a multilabel field.
    
    Parameters:
    -----------
    value : str or None
        Multilabel string
    separators : list of str, optional
        Separator characters (default: ["/", "+", ",", "|"])
    
    Returns:
    --------
    str or None
        First label, cleaned
    """
    if pd.isna(value) or value is None:
        return None
    
    if isinstance(value, (int, float)):
        return str(value)
    
    value_str = str(value).strip()
    
    if not value_str:
        return None
    
    if separators is None:
        separators = ["/", "+", ",", "|"]
    
    # Split by any separator
    pattern = '|'.join(re.escape(sep) for sep in separators)
    parts = re.split(pattern, value_str)
    
    # Take first part and clean it
    first_part = parts[0].strip() if parts else None
    
    return first_part if first_part else None


def clean_multilabel_series(series, separators=None):
    """
    Clean a pandas Series of multilabel fields.
    
    Parameters:
    -----------
    series : pd.Series
        Series containing multilabel data
    separators : list of str, optional
        Separator characters
    
    Returns:
    --------
    pd.Series
        Series with first label only
    """
    return series.apply(lambda x: take_first_label(x, separators))


def identify_multilabel_columns(df, separators=None):
    """
    Identify columns that contain multilabel data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    separators : list of str, optional
        Separator characters to check for
    
    Returns:
    --------
    list
        List of column names containing multilabel data
    """
    if separators is None:
        separators = ["/", "+", ",", "|"]
    
    multilabel_cols = []
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if any value contains separators
            sample = df[col].dropna().astype(str)
            if len(sample) > 0:
                pattern = '|'.join(re.escape(sep) for sep in separators)
                has_multilabel = sample.str.contains(pattern, regex=True, na=False).any()
                if has_multilabel:
                    multilabel_cols.append(col)
    
    return multilabel_cols


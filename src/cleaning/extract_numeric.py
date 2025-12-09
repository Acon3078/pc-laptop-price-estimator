"""
Extract numeric values from text fields containing units and labels.

This module provides functions to extract numeric values from strings
that contain text descriptors, units, and formatting characters.
"""

import re
import pandas as pd
import numpy as np


def extract_numeric(value, unit_patterns=None):
    """
    Extract numeric value from string containing text and units.
    
    Parameters:
    -----------
    value : str, float, int, or None
        Input value that may contain text and numeric data
    unit_patterns : list of str, optional
        Additional unit patterns to remove (default: common units)
    
    Returns:
    --------
    float or None
        Extracted numeric value, or None if extraction fails
    """
    if pd.isna(value) or value is None:
        return None
    
    # If already numeric, return as float
    if isinstance(value, (int, float)):
        return float(value)
    
    # Convert to string
    value_str = str(value).strip()
    
    if not value_str or value_str.lower() in ['ninguno', 'none', 'nan', '']:
        return None
    
    # Default unit patterns to remove
    if unit_patterns is None:
        unit_patterns = [
            r'pulgadas?', r'pulg', r'inches?', r'"',
            r'ghz', r'mhz', r'gb', r'tb', r'mb', r'kb',
            r'wh', r'w', r'v', r'a', r'amp',
            r'cd/m²', r'cd/m2', r'nits?',
            r'ppp', r'dpi', r'píxeles?', r'pixels?',
            r'kg', r'g', r'cm', r'mm', r'm',
            r'h', r'horas?', r'hours?',
            r'€', r'eur', r'usd', r'\$',
            r'mpx', r'megapixel',
            r'celdas?', r'cells?',
            r'mah', r'mah',
        ]
    
    # Remove common Spanish text
    spanish_removals = [
        r'procesador', r'procesadores?',
        r'memoria', r'compartida',
        r'integrada', r'integrado',
    ]
    
    # Combine all patterns
    all_patterns = unit_patterns + spanish_removals
    
    # Remove unit patterns (case insensitive)
    cleaned = value_str
    for pattern in all_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Replace comma with period for European number format
    cleaned = cleaned.replace(',', '.')
    
    # Remove all non-numeric characters except decimal point and minus sign
    # Keep first decimal point and minus sign
    cleaned = re.sub(r'[^\d\.\-]', '', cleaned)
    
    # Extract first number (handles negative numbers)
    match = re.search(r'-?\d+\.?\d*', cleaned)
    if match:
        try:
            return float(match.group())
        except (ValueError, AttributeError):
            return None
    
    return None


def extract_numeric_series(series, unit_patterns=None):
    """
    Extract numeric values from a pandas Series.
    
    Parameters:
    -----------
    series : pd.Series
        Series containing mixed text/numeric data
    unit_patterns : list of str, optional
        Additional unit patterns to remove
    
    Returns:
    --------
    pd.Series
        Series with extracted numeric values
    """
    return series.apply(lambda x: extract_numeric(x, unit_patterns))


def convert_tb_to_gb(value):
    """
    Convert TB to GB (using 1024 GB = 1 TB).
    
    Parameters:
    -----------
    value : float or None
        Value in TB
    
    Returns:
    --------
    float or None
        Value in GB
    """
    if pd.isna(value) or value is None:
        return None
    return float(value) * 1024


def normalize_decimal_separator(value):
    """
    Normalize decimal separator (comma to period).
    
    Parameters:
    -----------
    value : str, float, or None
        Input value
    
    Returns:
    --------
    str or None
        Value with normalized decimal separator
    """
    if pd.isna(value) or value is None:
        return None
    
    if isinstance(value, (int, float)):
        return str(value)
    
    value_str = str(value).strip()
    # Replace comma with period
    value_str = value_str.replace(',', '.')
    return value_str


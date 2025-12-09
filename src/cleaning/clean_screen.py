"""
Clean screen size and resolution fields.

This module handles extraction of screen diagonal in inches
and parsing of resolution into width and height.
"""

import re
import pandas as pd
import numpy as np
from .extract_numeric import extract_numeric


def extract_screen_size_inches(value):
    """
    Extract screen diagonal in inches from string.
    
    Handles formats like:
    - "15.6 pulgadas"
    - "15,6 pulgadas"
    - "39.624 cm" (converts to inches)
    - "15.6"
    
    Parameters:
    -----------
    value : str, float, or None
        Screen size value
    
    Returns:
    --------
    float or None
        Screen size in inches
    """
    if pd.isna(value) or value is None:
        return None
    
    if isinstance(value, (int, float)):
        # If already numeric, assume inches
        return float(value)
    
    value_str = str(value).strip()
    
    if not value_str:
        return None
    
    # Check if in cm and convert to inches
    if 'cm' in value_str.lower():
        cm_value = extract_numeric(value_str)
        if cm_value is not None:
            return cm_value / 2.54  # Convert cm to inches
    
    # Extract numeric value (handles "pulgadas" text)
    inches = extract_numeric(value_str, unit_patterns=['pulgadas', 'pulg', 'inches', '"'])
    
    return inches


def parse_resolution(value):
    """
    Parse resolution string into width and height.
    
    Handles formats like:
    - "1920 x 1080 píxeles"
    - "1920x1080"
    - "3.024 x 1.964 píxeles"
    - "Full HD"
    
    Parameters:
    -----------
    value : str or None
        Resolution string
    
    Returns:
    --------
    dict
        Dictionary with keys: 'resolution_x', 'resolution_y'
    """
    result = {
        'resolution_x': None,
        'resolution_y': None
    }
    
    if pd.isna(value) or value is None:
        return result
    
    value_str = str(value).strip()
    
    if not value_str:
        return result
    
    # Common resolution mappings
    resolution_map = {
        'full hd': (1920, 1080),
        'fhd': (1920, 1080),
        'hd': (1280, 720),
        'qhd': (2560, 1440),
        '4k': (3840, 2160),
        'uhd': (3840, 2160),
        '8k': (7680, 4320),
    }
    
    # Check for named resolutions
    value_lower = value_str.lower()
    for name, (x, y) in resolution_map.items():
        if name in value_lower:
            result['resolution_x'] = x
            result['resolution_y'] = y
            return result
    
    # Try to extract two numbers separated by x
    # Pattern: number x number (with optional spaces and text)
    pattern = r'(\d+[.,]?\d*)\s*[x×]\s*(\d+[.,]?\d*)'
    match = re.search(pattern, value_str, re.IGNORECASE)
    
    if match:
        try:
            x = float(match.group(1).replace(',', '.'))
            y = float(match.group(2).replace(',', '.'))
            result['resolution_x'] = int(x)
            result['resolution_y'] = int(y)
        except (ValueError, AttributeError):
            pass
    
    return result


def clean_screen_fields(df, size_col=None, resolution_col=None):
    """
    Clean screen-related columns in a dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    size_col : str, optional
        Column name for screen size
    resolution_col : str, optional
        Column name for resolution
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with cleaned screen fields
    """
    df = df.copy()
    
    # Create screen columns if they don't exist
    if 'screen_size_inches' not in df.columns:
        df['screen_size_inches'] = None
    if 'resolution_x' not in df.columns:
        df['resolution_x'] = None
    if 'resolution_y' not in df.columns:
        df['resolution_y'] = None
    
    # Process screen size
    if size_col and size_col in df.columns:
        df['screen_size_inches'] = df[size_col].apply(extract_screen_size_inches)
    
    # Process resolution
    if resolution_col and resolution_col in df.columns:
        resolution_results = df[resolution_col].apply(parse_resolution)
        df['resolution_x'] = resolution_results.apply(lambda x: x['resolution_x'])
        df['resolution_y'] = resolution_results.apply(lambda x: x['resolution_y'])
    
    return df


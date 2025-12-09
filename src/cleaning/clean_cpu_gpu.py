"""
Clean CPU and GPU name fields for fuzzy matching.

This module provides functions to normalize CPU and GPU names
by removing trademarks, extra whitespace, and special characters.
"""

import re
import pandas as pd
import numpy as np


def clean_cpu_gpu_name(value):
    """
    Clean CPU or GPU name for fuzzy matching.
    
    Rules:
    - Lowercase
    - Strip (R), (TM), procesador, and repeated whitespace
    - Remove trailing symbols and special characters
    
    Parameters:
    -----------
    value : str or None
        CPU or GPU name
    
    Returns:
    --------
    str or None
        Cleaned name
    """
    if pd.isna(value) or value is None:
        return None
    
    value_str = str(value).strip()
    
    if not value_str or value_str.lower() in ['ninguno', 'none', 'nan', '']:
        return None
    
    # Convert to lowercase
    cleaned = value_str.lower()
    
    # Remove trademark symbols
    cleaned = re.sub(r'\(r\)', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\(tm\)', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'®', '', cleaned)
    cleaned = re.sub(r'™', '', cleaned)
    
    # Remove Spanish processor-related terms
    cleaned = re.sub(r'\bprocesador\b', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bprocesadores\b', '', cleaned, flags=re.IGNORECASE)
    
    # Remove extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Remove leading/trailing whitespace
    cleaned = cleaned.strip()
    
    # Remove trailing symbols and special characters (but keep internal ones)
    # Keep alphanumeric, spaces, dashes, and periods
    cleaned = re.sub(r'[^\w\s\-\.]+$', '', cleaned)
    cleaned = re.sub(r'^[^\w\s\-\.]+', '', cleaned)
    
    # Final cleanup of multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned if cleaned else None


def clean_cpu_gpu_series(series):
    """
    Clean a pandas Series of CPU or GPU names.
    
    Parameters:
    -----------
    series : pd.Series
        Series containing CPU or GPU names
    
    Returns:
    --------
    pd.Series
        Series with cleaned names
    """
    return series.apply(clean_cpu_gpu_name)


def create_cleaned_cpu_gpu_columns(df, cpu_col, gpu_col=None):
    """
    Create cleaned CPU and GPU columns while preserving originals.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    cpu_col : str
        Column name for CPU
    gpu_col : str, optional
        Column name for GPU
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with added cleaned columns
    """
    df = df.copy()
    
    # Clean CPU
    if cpu_col in df.columns:
        df['cpu_clean'] = clean_cpu_gpu_series(df[cpu_col])
    
    # Clean GPU
    if gpu_col and gpu_col in df.columns:
        df['gpu_clean'] = clean_cpu_gpu_series(df[gpu_col])
    
    return df


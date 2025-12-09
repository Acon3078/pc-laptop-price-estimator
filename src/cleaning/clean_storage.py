"""
Clean and standardize storage-related fields.

This module handles parsing of storage capacity fields,
extracting SSD and HDD sizes, and converting units.
"""

import re
import pandas as pd
import numpy as np
from .extract_numeric import extract_numeric, convert_tb_to_gb


def parse_storage_capacity(value):
    """
    Parse storage capacity string to extract numeric value in GB.
    
    Handles formats like:
    - "512 GB"
    - "1 TB"
    - "1.000 GB"
    - "512GB SSD"
    - "1TB HDD"
    
    Parameters:
    -----------
    value : str, float, or None
        Storage capacity string
    
    Returns:
    --------
    float or None
        Storage capacity in GB
    """
    if pd.isna(value) or value is None:
        return None
    
    if isinstance(value, (int, float)):
        # Assume it's already in GB if numeric
        return float(value)
    
    value_str = str(value).strip().upper()
    
    if not value_str or value_str in ['NINGUNO', 'NONE', 'NAN', '']:
        return None
    
    # Check for TB first (multiply by 1024)
    tb_match = re.search(r'(\d+[.,]?\d*)\s*TB', value_str)
    if tb_match:
        tb_value = float(tb_match.group(1).replace(',', '.'))
        return tb_value * 1024
    
    # Check for GB
    gb_match = re.search(r'(\d+[.,]?\d*)\s*GB', value_str)
    if gb_match:
        return float(gb_match.group(1).replace(',', '.'))
    
    # Check for MB (divide by 1024)
    mb_match = re.search(r'(\d+[.,]?\d*)\s*MB', value_str)
    if mb_match:
        mb_value = float(mb_match.group(1).replace(',', '.'))
        return mb_value / 1024
    
    # Try to extract any number and assume GB
    numeric = extract_numeric(value_str)
    if numeric is not None:
        return numeric
    
    return None


def parse_combined_storage(value):
    """
    Parse combined storage strings like "512GB SSD + 1TB HDD".
    
    Parameters:
    -----------
    value : str or None
        Combined storage string
    
    Returns:
    --------
    dict
        Dictionary with keys: 'ssd_gb', 'hdd_gb', 'total_gb'
    """
    result = {
        'ssd_gb': None,
        'hdd_gb': None,
        'total_gb': None
    }
    
    if pd.isna(value) or value is None:
        return result
    
    value_str = str(value).strip().upper()
    
    if not value_str:
        return result
    
    # Split by common separators
    parts = re.split(r'[+\+/|,]', value_str)
    
    ssd_total = 0
    hdd_total = 0
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Check if SSD
        if 'SSD' in part or 'SOLID' in part:
            capacity = parse_storage_capacity(part)
            if capacity is not None:
                ssd_total += capacity
        # Check if HDD
        elif 'HDD' in part or 'HARD' in part or 'DISCO' in part:
            capacity = parse_storage_capacity(part)
            if capacity is not None:
                hdd_total += capacity
        else:
            # Try to parse as generic storage
            capacity = parse_storage_capacity(part)
            if capacity is not None:
                # If no type specified, assume it's the main storage
                if ssd_total == 0 and hdd_total == 0:
                    ssd_total = capacity
    
    result['ssd_gb'] = ssd_total if ssd_total > 0 else None
    result['hdd_gb'] = hdd_total if hdd_total > 0 else None
    
    # Calculate total
    total = 0
    if result['ssd_gb'] is not None:
        total += result['ssd_gb']
    if result['hdd_gb'] is not None:
        total += result['hdd_gb']
    
    result['total_gb'] = total if total > 0 else None
    
    return result


def clean_storage_fields(df, ssd_col=None, hdd_col=None, storage_type_col=None):
    """
    Clean storage-related columns in a dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    ssd_col : str, optional
        Column name for SSD capacity
    hdd_col : str, optional
        Column name for HDD capacity
    storage_type_col : str, optional
        Column name for storage type
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with cleaned storage fields
    """
    df = df.copy()
    
    # Create storage columns if they don't exist
    if 'storage_total_gb' not in df.columns:
        df['storage_total_gb'] = None
    if 'ssd_gb' not in df.columns:
        df['ssd_gb'] = None
    if 'hdd_gb' not in df.columns:
        df['hdd_gb'] = None
    
    # Process SSD column
    if ssd_col and ssd_col in df.columns:
        df['ssd_gb'] = df[ssd_col].apply(parse_storage_capacity)
    
    # Process HDD column
    if hdd_col and hdd_col in df.columns:
        df['hdd_gb'] = df[hdd_col].apply(parse_storage_capacity)
    
    # Process storage type column for combined storage
    if storage_type_col and storage_type_col in df.columns:
        combined_results = df[storage_type_col].apply(parse_combined_storage)
        df['ssd_gb'] = combined_results.apply(lambda x: x['ssd_gb'])
        df['hdd_gb'] = combined_results.apply(lambda x: x['hdd_gb'])
        df['storage_total_gb'] = combined_results.apply(lambda x: x['total_gb'])
    
    # Calculate total if not already set
    if df['storage_total_gb'].isna().all():
        df['storage_total_gb'] = (
            df['ssd_gb'].fillna(0) + df['hdd_gb'].fillna(0)
        ).replace(0, np.nan)
    
    return df


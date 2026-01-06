"""
Clean and standardize storage-related fields.

This module handles parsing of storage capacity fields,
extracting SSD and HDD sizes, and converting units.
"""

import re
import pandas as pd
import numpy as np
from .extract_numeric import extract_numeric, convert_tb_to_gb


def _parse_european_number(num_str):
    """
    Parse a number string that may use European format (period as thousands separator).
    
    Handles:
    - "1.000" -> 1000 (period as thousands separator)
    - "1,000" -> 1000 (comma as thousands separator)
    - "1.5" -> 1.5 (period as decimal separator)
    - "1,5" -> 1.5 (comma as decimal separator)
    
    Parameters:
    -----------
    num_str : str
        Number string that may contain period or comma
    
    Returns:
    --------
    float
        Parsed numeric value
    """
    # Check if period is used as thousands separator (exactly 3 digits after period)
    if '.' in num_str:
        parts = num_str.split('.')
        if len(parts) == 2 and len(parts[1]) == 3 and parts[1].isdigit():
            # Period is thousands separator, remove it
            return float(parts[0] + parts[1])
        else:
            # Period is decimal separator, keep as is
            return float(num_str)
    
    # Check if comma is used as thousands separator (exactly 3 digits after comma)
    if ',' in num_str:
        parts = num_str.split(',')
        if len(parts) == 2 and len(parts[1]) == 3 and parts[1].isdigit():
            # Comma is thousands separator, remove it
            return float(parts[0] + parts[1])
        else:
            # Comma is decimal separator, replace with period
            return float(num_str.replace(',', '.'))
    
    # No separator, just convert to float
    return float(num_str)


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
        tb_value = _parse_european_number(tb_match.group(1))
        return tb_value * 1024
    
    # Check for GB
    gb_match = re.search(r'(\d+[.,]?\d*)\s*GB', value_str)
    if gb_match:
        return _parse_european_number(gb_match.group(1))
    
    # Check for MB (divide by 1024)
    mb_match = re.search(r'(\d+[.,]?\d*)\s*MB', value_str)
    if mb_match:
        mb_value = _parse_european_number(mb_match.group(1))
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
    
    # Initialize storage columns as empty (will be filled from specific columns first)
    if 'storage_total_gb' not in df.columns:
        df['storage_total_gb'] = np.nan
    if 'ssd_gb' not in df.columns:
        df['ssd_gb'] = np.nan
    if 'hdd_gb' not in df.columns:
        df['hdd_gb'] = np.nan
    
    # Process SSD column - this takes priority
    if ssd_col and ssd_col in df.columns:
        ssd_parsed = df[ssd_col].apply(parse_storage_capacity)
        # Only update where we got a valid result
        df.loc[ssd_parsed.notna(), 'ssd_gb'] = ssd_parsed[ssd_parsed.notna()]
    
    # Process HDD column - this takes priority
    if hdd_col and hdd_col in df.columns:
        hdd_parsed = df[hdd_col].apply(parse_storage_capacity)
        # Only update where we got a valid result
        df.loc[hdd_parsed.notna(), 'hdd_gb'] = hdd_parsed[hdd_parsed.notna()]
    
    # Process storage type column for combined storage
    # Only use this to fill missing values (where ssd_col/hdd_col didn't provide values)
    if storage_type_col and storage_type_col in df.columns:
        combined_results = df[storage_type_col].apply(parse_combined_storage)
        
        # Only fill where ssd_gb is still NaN
        ssd_from_type = combined_results.apply(lambda x: x['ssd_gb'])
        mask_ssd_missing = df['ssd_gb'].isna()
        df.loc[mask_ssd_missing & ssd_from_type.notna(), 'ssd_gb'] = ssd_from_type[mask_ssd_missing & ssd_from_type.notna()]
        
        # Only fill where hdd_gb is still NaN
        hdd_from_type = combined_results.apply(lambda x: x['hdd_gb'])
        mask_hdd_missing = df['hdd_gb'].isna()
        df.loc[mask_hdd_missing & hdd_from_type.notna(), 'hdd_gb'] = hdd_from_type[mask_hdd_missing & hdd_from_type.notna()]
    
    # Always calculate storage_total_gb as the sum of ssd_gb and hdd_gb
    # This ensures accuracy regardless of what storage_type_col provided
    df['storage_total_gb'] = (
        df['ssd_gb'].fillna(0) + df['hdd_gb'].fillna(0)
    ).replace(0, np.nan)
    
    return df


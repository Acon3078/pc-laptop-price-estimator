"""
Main cleaning pipeline that orchestrates all cleaning functions.

This module imports and coordinates all cleaning transformations.
"""

import pandas as pd
import numpy as np
from .extract_numeric import extract_numeric_series, convert_tb_to_gb
from .clean_storage import clean_storage_fields, parse_storage_capacity
from .clean_cpu_gpu import create_cleaned_cpu_gpu_columns
from .clean_screen import clean_screen_fields
from .clean_multilabel import clean_multilabel_series, identify_multilabel_columns


def clean_dataframe(df, config=None):
    """
    Apply complete cleaning pipeline to dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    config : dict, optional
        Configuration dictionary with column mappings
    
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    if config is None:
        config = {}
    
    df = df.copy()
    
    # Step 1: Identify and clean multilabel fields
    multilabel_cols = identify_multilabel_columns(df)
    for col in multilabel_cols:
        if col not in config.get('preserve_multilabel', []):
            df[col] = clean_multilabel_series(df[col])
    
    # Step 2: Clean storage fields
    storage_config = config.get('storage', {})
    df = clean_storage_fields(
        df,
        ssd_col=storage_config.get('ssd_col'),
        hdd_col=storage_config.get('hdd_col'),
        storage_type_col=storage_config.get('storage_type_col')
    )
    
    # Step 3: Clean screen fields
    screen_config = config.get('screen', {})
    df = clean_screen_fields(
        df,
        size_col=screen_config.get('size_col'),
        resolution_col=screen_config.get('resolution_col')
    )
    
    # Step 4: Clean CPU/GPU fields
    cpu_gpu_config = config.get('cpu_gpu', {})
    df = create_cleaned_cpu_gpu_columns(
        df,
        cpu_col=cpu_gpu_config.get('cpu_col'),
        gpu_col=cpu_gpu_config.get('gpu_col')
    )
    
    # Step 5: Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            if pd.notna(median_val):
                df[col].fillna(median_val, inplace=True)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isna().any():
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col].fillna(mode_val[0], inplace=True)
            else:
                df[col].fillna('Unknown', inplace=True)
    
    return df


def get_default_config():
    """
    Get default configuration for cleaning pipeline.
    
    Returns:
    --------
    dict
        Default configuration dictionary
    """
    return {
        'storage': {
            'ssd_col': 'Disco duro_Capacidad de memoria SSD',
            'hdd_col': 'Disco duro_Capacidad del disco duro',
            'storage_type_col': 'Disco duro_Tipo de disco duro'
        },
        'screen': {
            'size_col': 'Pantalla_Tamaño de la pantalla',
            'resolution_col': 'Pantalla_Resolución de pantalla'
        },
        'cpu_gpu': {
            'cpu_col': 'Procesador_Procesador',
            'gpu_col': 'Gráfica_Tarjeta gráfica'
        },
        'preserve_multilabel': []
    }


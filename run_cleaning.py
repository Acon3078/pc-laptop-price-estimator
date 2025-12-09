#!/usr/bin/env python3
"""
Execute the complete data cleaning pipeline.

This script runs all cleaning transformations and exports the cleaned dataset.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from cleaning import (
    extract_numeric_series,
    clean_storage_fields,
    clean_screen_fields,
    create_cleaned_cpu_gpu_columns,
    clean_multilabel_series,
    identify_multilabel_columns,
    get_default_config
)

def main():
    """Execute the complete cleaning pipeline."""
    
    print("=" * 80)
    print("DATA CLEANING PIPELINE")
    print("=" * 80)
    
    # Step 1: Load data
    print("\n[1/7] Loading data...")
    data_path = os.path.join(project_root, 'data')
    df_computers = pd.read_csv(os.path.join(data_path, 'db_computers_2025_raw.csv'))
    print(f"  ✓ Loaded {df_computers.shape[0]} rows, {df_computers.shape[1]} columns")
    
    # Step 2: Start cleaning
    print("\n[2/7] Starting cleaning process...")
    df_clean = df_computers.copy()
    print(f"  Initial shape: {df_clean.shape}")
    
    # Step 3: Clean multilabel fields
    print("\n[3/7] Cleaning multilabel fields...")
    multilabel_cols = identify_multilabel_columns(df_clean)
    preserve_cols = ['Ofertas']  # Preserve some columns if needed
    cleaned_count = 0
    for col in multilabel_cols:
        if col not in preserve_cols:
            df_clean[col] = clean_multilabel_series(df_clean[col])
            cleaned_count += 1
    print(f"  ✓ Cleaned {cleaned_count} multilabel columns")
    
    # Step 4: Clean storage fields
    print("\n[4/7] Cleaning storage fields...")
    config = get_default_config()
    df_clean = clean_storage_fields(
        df_clean,
        ssd_col=config['storage']['ssd_col'],
        hdd_col=config['storage']['hdd_col'],
        storage_type_col=config['storage']['storage_type_col']
    )
    print(f"  ✓ Created storage_total_gb, ssd_gb, hdd_gb columns")
    
    # Step 5: Clean screen fields
    print("\n[5/7] Cleaning screen fields...")
    df_clean = clean_screen_fields(
        df_clean,
        size_col=config['screen']['size_col'],
        resolution_col=config['screen']['resolution_col']
    )
    print(f"  ✓ Created screen_size_inches, resolution_x, resolution_y columns")
    
    # Step 6: Clean CPU/GPU fields
    print("\n[6/7] Cleaning CPU/GPU fields...")
    df_clean = create_cleaned_cpu_gpu_columns(
        df_clean,
        cpu_col=config['cpu_gpu']['cpu_col'],
        gpu_col=config['cpu_gpu']['gpu_col']
    )
    print(f"  ✓ Created cpu_clean, gpu_clean columns")
    
    # Step 7: Extract numeric from other fields
    print("\n[7/7] Extracting numeric values from other fields...")
    numeric_fields = [
        ('RAM_Memoria RAM', 'ram_gb'),
        ('Procesador_Frecuencia de reloj', 'cpu_freq_ghz'),
        ('Medidas y peso_Peso', 'weight_kg'),
        ('Alimentación_Vatios-hora', 'battery_wh'),
    ]
    
    for col, new_col in numeric_fields:
        if col in df_clean.columns:
            df_clean[new_col] = extract_numeric_series(df_clean[col])
            print(f"  ✓ Created {new_col} from {col}")
    
    # Step 8: Missing value imputation
    print("\n[8/8] Imputing missing values...")
    # Numeric columns: median imputation
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    imputed_numeric = 0
    for col in numeric_cols:
        if df_clean[col].isna().any():
            median_val = df_clean[col].median()
            if pd.notna(median_val):
                missing_count = df_clean[col].isna().sum()
                df_clean[col].fillna(median_val, inplace=True)
                imputed_numeric += missing_count
    
    # Categorical columns: mode imputation or "Unknown"
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    imputed_categorical = 0
    for col in categorical_cols:
        if df_clean[col].isna().any():
            mode_val = df_clean[col].mode()
            if len(mode_val) > 0:
                missing_count = df_clean[col].isna().sum()
                df_clean[col].fillna(mode_val[0], inplace=True)
                imputed_categorical += missing_count
            else:
                missing_count = df_clean[col].isna().sum()
                df_clean[col].fillna('Unknown', inplace=True)
                imputed_categorical += missing_count
    
    print(f"  ✓ Imputed {imputed_numeric} numeric values (median)")
    print(f"  ✓ Imputed {imputed_categorical} categorical values (mode/Unknown)")
    
    # Step 9: Export cleaned dataset
    print("\n[9/9] Exporting cleaned dataset...")
    output_dir = os.path.join(data_path, 'clean')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'db_computers_cleaned.csv')
    df_clean.to_csv(output_path, index=False)
    
    # Summary
    print("\n" + "=" * 80)
    print("CLEANING COMPLETE!")
    print("=" * 80)
    print(f"\nOriginal dataset:")
    print(f"  Rows: {df_computers.shape[0]}")
    print(f"  Columns: {df_computers.shape[1]}")
    print(f"  Missing values: {df_computers.isnull().sum().sum()}")
    
    print(f"\nCleaned dataset:")
    print(f"  Rows: {df_clean.shape[0]}")
    print(f"  Columns: {df_clean.shape[1]}")
    print(f"  Missing values: {df_clean.isnull().sum().sum()}")
    
    print(f"\nNew columns created:")
    new_cols = [col for col in df_clean.columns if col not in df_computers.columns]
    for col in new_cols:
        print(f"  - {col}")
    
    print(f"\n✓ Cleaned dataset saved to: {output_path}")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Standalone script that installs dependencies and runs the cleaning pipeline.
No terminal required - just run this file in any Python IDE!

Usage:
- Double-click this file (if Python is set as default)
- Or open in VS Code/PyCharm/Spyder and press Run
- Or right-click → "Run with Python"
"""

import subprocess
import sys
import os

def install_packages():
    """Install required packages if not already installed."""
    required_packages = {
        'pandas': 'pandas>=1.5.0',
        'numpy': 'numpy>=1.23.0',
        'matplotlib': 'matplotlib>=3.6.0',
        'seaborn': 'seaborn>=0.12.0'
    }
    
    print("=" * 80)
    print("CHECKING AND INSTALLING DEPENDENCIES")
    print("=" * 80)
    
    missing_packages = []
    for package_name, package_spec in required_packages.items():
        try:
            __import__(package_name)
            print(f"✓ {package_name} is already installed")
        except ImportError:
            print(f"✗ {package_name} is missing - installing...")
            missing_packages.append(package_spec)
    
    if missing_packages:
        print(f"\nInstalling {len(missing_packages)} missing package(s)...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--quiet"
            ] + missing_packages)
            print("✓ All packages installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error installing packages: {e}")
            print("\nPlease install manually using:")
            print(f"  pip install {' '.join(missing_packages)}")
            return False
    else:
        print("\n✓ All required packages are installed!")
    
    return True

def run_cleaning():
    """Run the cleaning pipeline."""
    print("\n" + "=" * 80)
    print("RUNNING DATA CLEANING PIPELINE")
    print("=" * 80)
    
    # Import after ensuring packages are installed
    import pandas as pd
    import numpy as np
    
    # Add src to path
    project_root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    try:
        from cleaning import (
            extract_numeric_series,
            clean_storage_fields,
            clean_screen_fields,
            create_cleaned_cpu_gpu_columns,
            clean_multilabel_series,
            identify_multilabel_columns,
            get_default_config
        )
    except ImportError as e:
        print(f"✗ Error importing cleaning modules: {e}")
        print("Make sure the 'src/cleaning' folder exists with all Python files.")
        return False
    
    # Step 1: Load data
    print("\n[1/7] Loading data...")
    data_path = os.path.join(project_root, 'data')
    csv_path = os.path.join(data_path, 'db_computers_2025_raw.csv')
    
    if not os.path.exists(csv_path):
        print(f"✗ Error: Cannot find {csv_path}")
        print("Make sure the data file exists in the 'data' folder.")
        return False
    
    try:
        df_computers = pd.read_csv(csv_path, low_memory=False)
        print(f"  ✓ Loaded {df_computers.shape[0]} rows, {df_computers.shape[1]} columns")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False
    
    # Step 2: Start cleaning
    print("\n[2/7] Starting cleaning process...")
    df_clean = df_computers.copy()
    print(f"  Initial shape: {df_clean.shape}")
    
    # Step 3: Clean multilabel fields
    print("\n[3/7] Cleaning multilabel fields...")
    try:
        multilabel_cols = identify_multilabel_columns(df_clean)
        preserve_cols = ['Ofertas']
        cleaned_count = 0
        for col in multilabel_cols:
            if col not in preserve_cols:
                df_clean[col] = clean_multilabel_series(df_clean[col])
                cleaned_count += 1
        print(f"  ✓ Cleaned {cleaned_count} multilabel columns")
    except Exception as e:
        print(f"  ⚠ Warning in multilabel cleaning: {e}")
    
    # Step 4: Clean storage fields
    print("\n[4/7] Cleaning storage fields...")
    try:
        config = get_default_config()
        df_clean = clean_storage_fields(
            df_clean,
            ssd_col=config['storage']['ssd_col'],
            hdd_col=config['storage']['hdd_col'],
            storage_type_col=config['storage']['storage_type_col']
        )
        print(f"  ✓ Created storage_total_gb, ssd_gb, hdd_gb columns")
    except Exception as e:
        print(f"  ⚠ Warning in storage cleaning: {e}")
    
    # Step 5: Clean screen fields
    print("\n[5/7] Cleaning screen fields...")
    try:
        df_clean = clean_screen_fields(
            df_clean,
            size_col=config['screen']['size_col'],
            resolution_col=config['screen']['resolution_col']
        )
        print(f"  ✓ Created screen_size_inches, resolution_x, resolution_y columns")
    except Exception as e:
        print(f"  ⚠ Warning in screen cleaning: {e}")
    
    # Step 6: Clean CPU/GPU fields
    print("\n[6/7] Cleaning CPU/GPU fields...")
    try:
        df_clean = create_cleaned_cpu_gpu_columns(
            df_clean,
            cpu_col=config['cpu_gpu']['cpu_col'],
            gpu_col=config['cpu_gpu']['gpu_col']
        )
        print(f"  ✓ Created cpu_clean, gpu_clean columns")
    except Exception as e:
        print(f"  ⚠ Warning in CPU/GPU cleaning: {e}")
    
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
            try:
                df_clean[new_col] = extract_numeric_series(df_clean[col])
                print(f"  ✓ Created {new_col} from {col}")
            except Exception as e:
                print(f"  ⚠ Warning creating {new_col}: {e}")
    
    # Step 8: Missing value imputation
    print("\n[8/8] Imputing missing values...")
    try:
        # Numeric columns: median imputation
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        imputed_numeric = 0
        for col in numeric_cols:
            if df_clean[col].isna().any():
                median_val = df_clean[col].median()
                if pd.notna(median_val):
                    missing_count = df_clean[col].isna().sum()
                    df_clean[col] = df_clean[col].fillna(median_val)
                    imputed_numeric += missing_count
        
        # Categorical columns: mode imputation or "Unknown"
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        imputed_categorical = 0
        for col in categorical_cols:
            if df_clean[col].isna().any():
                mode_val = df_clean[col].mode()
                if len(mode_val) > 0:
                    missing_count = df_clean[col].isna().sum()
                    df_clean[col] = df_clean[col].fillna(mode_val[0])
                    imputed_categorical += missing_count
                else:
                    missing_count = df_clean[col].isna().sum()
                    df_clean[col] = df_clean[col].fillna('Unknown')
                    imputed_categorical += missing_count
        
        print(f"  ✓ Imputed {imputed_numeric} numeric values (median)")
        print(f"  ✓ Imputed {imputed_categorical} categorical values (mode/Unknown)")
    except Exception as e:
        print(f"  ⚠ Warning in missing value imputation: {e}")
    
    # Step 9: Export cleaned dataset
    print("\n[9/9] Exporting cleaned dataset...")
    try:
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
        return True
    except Exception as e:
        print(f"✗ Error exporting dataset: {e}")
        return False

def main():
    """Main function - installs packages and runs cleaning."""
    print("\n" + "=" * 80)
    print("DATA CLEANING PROJECT - AUTOMATED RUNNER")
    print("=" * 80)
    print("\nThis script will:")
    print("  1. Check and install required packages")
    print("  2. Run the complete data cleaning pipeline")
    print("  3. Generate the cleaned dataset")
    print("\nNo terminal required - just run this file!")
    print("=" * 80)
    
    # Install packages
    if not install_packages():
        print("\n✗ Failed to install required packages.")
        print("Please install manually or check your Python/pip installation.")
        input("\nPress Enter to exit...")
        return
    
    # Run cleaning
    success = run_cleaning()
    
    if success:
        print("\n✅ SUCCESS! All done!")
    else:
        print("\n⚠ Some errors occurred. Check the messages above.")
    
    # Keep window open if double-clicked
    try:
        input("\nPress Enter to exit...")
    except:
        pass

if __name__ == "__main__":
    main()


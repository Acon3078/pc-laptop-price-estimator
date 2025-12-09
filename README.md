# Computer Price Estimator (Streamlit)

Interactive Streamlit app that predicts PC / laptop prices from hardware specs, with bilingual UI (EN/ES), optional SHAP explanations, and an OpenAI-powered chatbot that can query the cleaned dataset.

## What’s here
- `app.py`: Streamlit entrypoint (tabs for prediction, debug, chatbot).
- `app_core/`: Shared config, feature lists, pipeline loaders, and dataset helper.
- `chatbot.py`: OpenAI-backed chat with dataset querying and synthetic fallback.
- `explain_api.py`: SHAP-based local explanation for predictions.
- `data/clean/db_computers_cleaned.csv`: Cleaned dataset for the chatbot/context.
- `price_prediction_pipeline.joblib`, `price_preprocessor.joblib`, `shap_background.joblib`: Artifacts exported from the training notebook.
- `notebooks/EDA_and_Cleaning.ipynb` + `src/cleaning/`: Full EDA/cleaning workflow (optional).

## Quickstart (local)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Then open the URL Streamlit prints (e.g., http://localhost:8501).

## Configuration
- Model/data artifacts: keep the `price_prediction_pipeline.joblib`, `price_preprocessor.joblib`, `shap_background.joblib`, and `data/clean/db_computers_cleaned.csv` alongside the app. If you host them elsewhere, add a download step before app startup.
- Chatbot: set `OPENAI_API_KEY` in a `.env` file or environment variable to enable LLM responses; otherwise it falls back to an echo reply. Secrets should live in `.streamlit/secrets.toml` when deploying to Streamlit Cloud.

## Deployment (Streamlit Community Cloud)
1. Push the repo to GitHub (excluding raw data; keep cleaned data/models or add a download hook).
2. In Streamlit Cloud, create a new app pointing to `app.py`.
3. Add `OPENAI_API_KEY` in Secrets (and any artifact URLs if downloading on startup).
4. Deploy and verify price prediction, SHAP summary, and chatbot modes.

## Testing
- Smoke test locally: fill the prediction form and submit; check history table.
- Chatbot: ask for a hardware combo; without an API key it echoes, with a key it queries the dataset and returns model-based prices.
- Optional sanity check: `python -m compileall app_core app.py chatbot.py explain_api.py`.

## Project structure (clean vs. raw data)
- Keep raw CSVs out of version control; retain the cleaned dataset and model artifacts needed to run the app.
- If size is an issue, store artifacts in object storage/Git LFS and add a small bootstrap download in `app.py`.

## EDA / data cleaning (optional)
- Run `notebooks/EDA_and_Cleaning.ipynb` for the full exploration, or `run_everything.py` / `run_cleaning.py` for scripted cleaning. Outputs land in `data/clean/` and feed the app’s artifacts.
- Open Terminal/Command Prompt
- Type: `python --version` or `python3 --version`
- Should show: `Python 3.8.x` or higher

**If Python is not installed:**
1. Go to https://www.python.org/downloads/
2. Download Python 3.8 or higher
3. Install it (check "Add Python to PATH" during installation)

---

### **Step 4: Install Required Packages**

**Option A - Using pip in Terminal:**
```bash
pip install pandas numpy matplotlib seaborn
```
or
```bash
pip3 install pandas numpy matplotlib seaborn
```

**Option B - Using pip in Python:**
```python
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "numpy", "matplotlib", "seaborn"])
```

**Option C - Install Jupyter (if using notebook):**
```bash
pip install jupyter
```

**Verify installation:**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
print("All packages installed successfully!")
```

---

### **Step 5: Run the EDA and Cleaning Process**

#### **If Using Method A (Automated Script):**

1. **Open** `run_everything.py` in any text editor or IDE
2. **Run** the script:
   - **VS Code**: Press `F5` or click "Run"
   - **PyCharm**: Right-click → "Run 'run_everything'"
   - **Spyder**: Press `F5`
   - **Double-click**: If configured to run Python files
3. **Wait** for the script to complete (may take a few minutes)
4. **Check** output: `data/clean/db_computers_cleaned.csv` should be created

**What happens:**
- Script installs missing packages automatically
- Runs all cleaning steps
- Generates cleaned dataset
- Shows summary statistics

---

#### **If Using Method B (Jupyter Notebook - RECOMMENDED):**

**5.1: Open Jupyter Notebook**

**Option 1 - Using Jupyter:**
1. Open Terminal/Command Prompt
2. Navigate to project folder: `cd "path/to/AI EDA FIXED"`
3. Type: `jupyter notebook`
4. Browser opens automatically
5. Click on `notebooks/EDA_and_Cleaning.ipynb`

**Option 2 - Using VS Code:**
1. Open VS Code
2. Open the project folder
3. Click on `notebooks/EDA_and_Cleaning.ipynb`
4. VS Code will open it as a notebook

**Option 3 - Using PyCharm:**
1. Open PyCharm
2. Open the project folder
3. Right-click `notebooks/EDA_and_Cleaning.ipynb`
4. Select "Open in Jupyter"

---

**5.2: Run the Notebook Cells**

The notebook is organized into sections. Run cells in order:

**Section 1: Setup (Cell 1)**
- Click on the first code cell
- Press `Shift + Enter` to run
- This imports all libraries and sets up paths
- **Expected output**: "Libraries imported successfully!"

**Section 2: Load Data (Cell 3)**
- Run the cell to load all three datasets
- **Expected output**: Shows dataset shapes (rows, columns)

**Section 3: Initial Inspection (Cells 5-7)**
- **Cell 5**: Shows dataset shape, data types, first few rows
- **Cell 6**: Missing values analysis - shows which columns have missing data
- **Cell 7**: Duplicate rows check
- **What to look for**: Note how many missing values and duplicates exist

**Section 4: Missing Values Visualization (Cell 9)**
- Creates a heatmap showing missing data patterns
- **Expected output**: Color-coded heatmap visualization
- **Saved to**: `docs/missing_values_heatmap.png`

**Section 5: Numeric Fields Analysis (Cell 11)**
- Shows sample values from columns that should be numeric but are stored as text
- **What to look for**: Notice inconsistent formats (e.g., "16 GB", "1 TB", "15,6 pulgadas")

**Section 6: Categorical Fields Analysis (Cell 13)**
- Shows value counts for important categorical columns
- **What to look for**: See most common processors, GPUs, operating systems

**Section 7: Data Quality Issues (Cells 15-18)**
- **Cell 15**: Identifies multilabel fields (multiple values in one cell)
- **Cell 16**: Shows storage format issues
- **Cell 17**: Shows screen size format issues
- **Cell 18**: Shows CPU/GPU naming inconsistencies
- **What to look for**: Examples of data that needs cleaning

**Section 8: Data Cleaning (Cells 20-32)**
- **Cell 20**: Creates a copy of the dataframe for cleaning
- **Cell 22**: Cleans multilabel fields (extracts first value)
- **Cell 24**: Cleans storage fields (standardizes to GB, separates SSD/HDD)
- **Cell 26**: Cleans screen fields (converts to inches, parses resolution)
- **Cell 28**: Cleans CPU/GPU names (removes special characters, lowercases)
- **Cell 30**: Extracts numeric values from text fields (RAM, CPU freq, weight, battery)
- **Cell 32**: Imputes missing values (median for numeric, mode for categorical)
- **What to look for**: See "Before" and "After" examples for each cleaning step

**Section 9: Visualizations (Cells 34-36)**
- **Cell 34**: Storage distribution histogram
- **Cell 35**: Screen size distribution histogram
- **Cell 36**: RAM distribution histogram
- **Expected output**: Three distribution plots
- **Saved to**: `docs/` folder

**Section 10: Export Cleaned Data (Cell 38)**
- Saves the cleaned dataset to CSV
- **Expected output**: "CLEANED DATASET EXPORTED"
- **Location**: `data/clean/db_computers_cleaned.csv`

**Section 11: Summary Statistics (Cells 40-41)**
- **Cell 40**: Shows summary statistics for numeric fields
- **Cell 41**: Shows data quality summary (before vs. after)
- **What to look for**: Compare missing values before and after cleaning

**Quick Run All:**
- Instead of running cells one by one, you can:
  - **Jupyter**: Menu → "Run All" or "Cell" → "Run All"
  - **VS Code**: Click "Run All" button at the top
  - **PyCharm**: Right-click → "Run All Cells"

---

#### **If Using Method C (Python Script):**

1. **Open** `run_cleaning.py` (if it exists) or create a new Python file
2. **Copy** the code from the "Quick Code Examples" section below
3. **Run** the script in your IDE
4. **Check** output: `data/clean/db_computers_cleaned.csv`

---

### **Step 6: Verify Results**

After running the process, verify everything worked:

**Check 1: Cleaned Dataset Exists**
- Location: `data/clean/db_computers_cleaned.csv`
- Should have same number of rows as original
- Should have more columns (new standardized columns added)

**Check 2: Visualizations Created (if using notebook)**
- `docs/missing_values_heatmap.png`
- `docs/storage_distribution.png`
- `docs/screen_size_distribution.png`
- `docs/ram_distribution.png`

**Check 3: No Missing Values**
- Open the cleaned CSV file
- Verify no empty cells (or very few)
- All missing values should be imputed

**Check 4: New Columns Present**
- Look for these new columns in the cleaned dataset:
  - `storage_total_gb`, `ssd_gb`, `hdd_gb`
  - `screen_size_inches`, `resolution_x`, `resolution_y`
  - `cpu_clean`, `gpu_clean`
  - `ram_gb`, `cpu_freq_ghz`, `weight_kg`, `battery_wh`

---

### **Step 7: Review the Results**

**What to Review:**

1. **Summary Statistics** (from notebook Cell 40):
   - Mean, median, min, max for numeric fields
   - Check if values make sense (e.g., screen sizes between 10-20 inches)

2. **Data Quality Summary** (from notebook Cell 41):
   - Compare missing values before vs. after
   - Should see significant reduction in missing values

3. **Visualizations**:
   - Check distributions look reasonable
   - Look for any unexpected patterns or outliers

4. **Sample Rows**:
   - Open the cleaned CSV
   - Check a few rows to verify cleaning worked correctly
   - Compare with original data if needed

---

### **Step 8: Use the Cleaned Dataset**

Your cleaned dataset is now ready for:
- Machine learning models
- Statistical analysis
- Further data processing
- Business intelligence tools

**Location**: `data/clean/db_computers_cleaned.csv`

---

## 🚀 Quick Start - No Terminal Required!

### Option 1: Run Everything Automatically (Easiest - Just Double-Click!)

1. **Double-click** the file: `run_everything.py`
   - Or open it in any Python IDE (VS Code, PyCharm, Spyder, etc.)
   - Press `F5` or click "Run"

2. **That's it!** The script will:
   - Automatically install any missing packages
   - Run the complete cleaning pipeline
   - Generate the cleaned dataset
   - Show you a summary

3. **Find your cleaned dataset**:
   - Location: `data/clean/db_computers_cleaned.csv`

**No terminal, no commands, no setup needed!**

---

### Option 2: Run the Jupyter Notebook (Interactive - Recommended for Exploration)

1. **Open the notebook** in Jupyter, VS Code, or any Python IDE:
   - Navigate to: `notebooks/EDA_and_Cleaning.ipynb`
   - Double-click to open

2. **Install packages** (if needed):
   - In the first cell, you can run: `!pip install pandas numpy matplotlib seaborn`
   - Or install via your IDE's package manager

3. **Run all cells**:
   - **Jupyter**: Click "Run All" in the menu (or press `Shift + Enter` in each cell)
   - **VS Code**: Click "Run All" button at the top
   - **PyCharm**: Right-click → "Run All Cells"

4. **What you'll see**:
   - Complete EDA analysis (missing values, distributions, data quality issues)
   - Step-by-step data cleaning process
   - Visualizations (heatmaps, distributions)
   - Summary statistics

5. **Done!** The cleaned dataset will be saved to `data/clean/db_computers_cleaned.csv`

---

### Option 2: Run Python Script Directly

1. **Open the Python script** in your IDE:
   - File: `run_cleaning.py`
   - Open it in VS Code, PyCharm, Spyder, or any Python IDE

2. **Install packages** (if needed):
   - In your IDE, install: `pandas`, `numpy`, `matplotlib`, `seaborn`
   - Or add this at the top of the script temporarily:
     ```python
     import subprocess
     import sys
     subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "numpy", "matplotlib", "seaborn"])
     ```

3. **Run the script**:
   - **VS Code**: Press `F5` or click the "Run" button
   - **PyCharm**: Right-click → "Run 'run_cleaning'"
   - **Spyder**: Press `F5`
   - **Any IDE**: Use the "Run" button or `Ctrl+F5` / `Cmd+F5`

4. **Done!** The cleaned dataset will be saved to `data/clean/db_computers_cleaned.csv`

---

### Option 3: Run in Python Interactive Mode

1. **Open Python** (IDLE, IPython, or Python console in your IDE)

2. **Copy and paste this code**:

```python
# Install packages (run once)
import subprocess
import sys
packages = ['pandas', 'numpy', 'matplotlib', 'seaborn']
for package in packages:
    try:
        __import__(package)
        print(f"✓ {package} already installed")
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Now run the cleaning
exec(open('run_cleaning.py').read())
```

3. **Press Enter** to execute

---

## 📁 Project Structure

```
.
├── data/
│   ├── db_computers_2025_raw.csv      # Raw input data
│   ├── db_cpu_raw.csv                 # CPU reference data
│   ├── db_gpu_raw.csv                 # GPU reference data
│   └── clean/
│       └── db_computers_cleaned.csv   # Output: Cleaned dataset (generated)
├── notebooks/
│   └── EDA_and_Cleaning.ipynb         # Interactive EDA notebook ⭐
├── src/
│   └── cleaning/                      # Cleaning modules (required)
│       ├── __init__.py
│       ├── extract_numeric.py
│       ├── clean_storage.py
│       ├── clean_cpu_gpu.py
│       ├── clean_screen.py
│       ├── clean_multilabel.py
│       └── cleaning_pipeline.py
├── docs/
│   └── data_cleaning_notes.md         # Documentation
├── run_cleaning.py                    # Standalone cleaning script ⭐
└── requirements.txt                   # Python dependencies
```

## 📊 Exploratory Data Analysis (EDA) - Step by Step

The EDA process in the notebook follows these comprehensive steps:

### 1. Data Loading
- Loads three datasets:
  - `db_computers_2025_raw.csv` - Main computer dataset
  - `db_cpu_raw.csv` - CPU reference data
  - `db_gpu_raw.csv` - GPU reference data
- Displays initial shape and column counts

### 2. Initial Data Inspection
- **2.1 Dataset Shape and Types**
  - Examines rows, columns, and data types
  - Shows first few rows for initial inspection
  
- **2.2 Missing Values Analysis**
  - Calculates missing value counts and percentages per column
  - Identifies top 20 columns with most missing values
  - Creates a missing values heatmap visualization (top 30 columns)
  
- **2.3 Duplicate Rows**
  - Counts and reports duplicate rows
  - Calculates duplicate percentage

### 3. Missing Values Visualization
- Generates a heatmap showing missing value patterns across columns
- Helps identify columns with systematic missing data
- Saves visualization to `docs/missing_values_heatmap.png`

### 4. Numeric Fields Analysis
- **4.1 Numeric Fields Stored as Strings**
  - Identifies columns that contain numeric data but are stored as strings:
    - Screen size and diagonal
    - RAM memory
    - Storage capacity (SSD/HDD)
    - CPU frequency
    - Weight
    - Battery capacity (Wh)
  - Shows sample values to understand format inconsistencies

### 5. Categorical Fields Analysis
- **5.1 Value Counts for Key Categorical Fields**
  - Analyzes distribution of values in important categorical columns:
    - Processor/CPU names
    - Graphics card/GPU names
    - Operating system
    - Product series
    - Product type
  - Shows unique value counts and top 10 most frequent values per column

### 6. Data Quality Issues Identification
- **6.1 Multilabel Fields**
  - Identifies columns containing multiple values separated by commas
  - Shows examples of multilabel data that needs cleaning
  
- **6.2 Storage Format Issues**
  - Examines storage capacity fields for format inconsistencies
  - Identifies mixed units (GB, TB) and combined SSD/HDD entries
  
- **6.3 Screen Size Issues**
  - Analyzes screen size fields for unit inconsistencies
  - Identifies different formats (inches, centimeters, etc.)
  
- **6.4 CPU/GPU Naming Issues**
  - Examines processor and graphics card names
  - Identifies inconsistencies in naming conventions (brands, model numbers, special characters)

### 7. Data Cleaning (Applied Step by Step)
See "What Gets Cleaned?" section below for details.

### 8. Visualizations
- **Distribution plots** for key numeric fields:
  - Total storage (GB) distribution
  - Screen size (inches) distribution
  - RAM (GB) distribution
- All visualizations saved to `docs/` folder

### 9. Summary Statistics
- Final summary statistics for cleaned numeric fields
- Data quality comparison (before vs. after cleaning)
- Summary of all cleaning operations performed

---

## 🎯 What Gets Cleaned?

The cleaning pipeline performs the following transformations:

1. **Multilabel Fields**: Extracts first value from fields with multiple values
   - Example: `"wifi, Bluetooth"` → `"wifi"`

2. **Storage**: Standardizes to GB, separates SSD/HDD
   - Example: `"1 TB"` → `1024 GB`, `"512GB SSD + 1TB HDD"` → `ssd_gb=512, hdd_gb=1024`

3. **Screen Size**: Converts to inches
   - Example: `"15,6 pulgadas"` → `15.6 inches`

4. **Resolution**: Parses into width × height
   - Example: `"1920 x 1080 píxeles"` → `resolution_x=1920, resolution_y=1080`

5. **CPU/GPU Names**: Cleans for fuzzy matching
   - Example: `"Intel Core i7-13700H (R)"` → `"intel core i7-13700h"`

6. **Numeric Fields**: Extracts numbers from text
   - Example: `"16 GB RAM"` → `16`

7. **Missing Values**: Imputes using median (numeric) or mode (categorical)

## 📊 Output

The cleaned dataset (`data/clean/db_computers_cleaned.csv`) includes:

- **All original columns** (cleaned and standardized)
- **New standardized columns**:
  - `storage_total_gb`, `ssd_gb`, `hdd_gb` - Storage in GB
  - `screen_size_inches`, `resolution_x`, `resolution_y` - Screen dimensions
  - `cpu_clean`, `gpu_clean` - Cleaned names for matching
  - `ram_gb`, `cpu_freq_ghz`, `weight_kg`, `battery_wh` - Extracted numeric values

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'pandas'"

**Solution**: Install packages using one of these methods:

**Method 1 - In Python code:**
```python
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "numpy", "matplotlib", "seaborn"])
```

**Method 2 - In Jupyter notebook:**
```python
!pip install pandas numpy matplotlib seaborn
```

**Method 3 - In VS Code:**
- Open Command Palette (`Cmd+Shift+P` / `Ctrl+Shift+P`)
- Type "Python: Select Interpreter"
- Then use the package manager or terminal in VS Code

### "ModuleNotFoundError: No module named 'cleaning'"

**Solution**: This happens when the notebook can't find the `src/cleaning/` directory. The notebook now has improved path detection, but if you still get this error:

1. **Make sure you're running the notebook from the project root or notebooks folder**
   - The notebook should be at: `notebooks/EDA_and_Cleaning.ipynb`
   - The src folder should be at: `src/cleaning/`

2. **Check the current working directory**:
   - In Jupyter, the notebook will show the current directory in the error message
   - Make sure you're in the right location

3. **If using Jupyter Notebook/Lab**:
   - Navigate to the project folder first: `cd "path/to/AI EDA FIXED"`
   - Then start Jupyter: `jupyter notebook`

4. **If using VS Code**:
   - Open the entire project folder (not just the notebook)
   - The notebook should automatically detect the correct paths

5. **Manual fix** (if needed):
   - The notebook will print the paths it's trying
   - If they're wrong, you can manually set them in the first cell:
   ```python
   import sys
   import os
   # Manually set the project root
   project_root = '/Users/haidar/Downloads/AI EDA FIXED'  # Change to your path
   src_path = os.path.join(project_root, 'src')
   sys.path.insert(0, src_path)
   ```

### "ImportError: cannot import name 'X' from 'cleaning'"

**Solution**: Make sure the `src/cleaning/` folder exists with all Python files:
- `__init__.py`
- `extract_numeric.py`
- `clean_storage.py`
- `clean_cpu_gpu.py`
- `clean_screen.py`
- `clean_multilabel.py`
- `cleaning_pipeline.py`

### "FileNotFoundError: db_computers_2025_raw.csv"

**Solution**: Make sure you're running the script from the project root directory, or update the path in the code:
```python
# In run_cleaning.py or notebook, change:
data_path = '../data/'  # If running from notebooks folder
# OR
data_path = 'data/'     # If running from project root
```

## 📝 Requirements

- **Python 3.8 or higher**
- **Required packages**:
  - pandas >= 1.5.0
  - numpy >= 1.23.0
  - matplotlib >= 3.6.0
  - seaborn >= 0.12.0

## 📚 Documentation

For detailed information about the EDA and cleaning process, see:
- **`notebooks/EDA_and_Cleaning.ipynb`** - Interactive notebook with complete EDA and cleaning steps (run all cells to see full analysis)
- **`docs/data_cleaning_notes.md`** - Complete documentation of all cleaning steps, issues found, and solutions
- **`docs/missing_values_heatmap.png`** - Visualization of missing data patterns
- **`docs/storage_distribution.png`** - Distribution of storage capacity
- **`docs/screen_size_distribution.png`** - Distribution of screen sizes
- **`docs/ram_distribution.png`** - Distribution of RAM capacity

## 💡 Quick Code Examples

### Run cleaning from Python script:

```python
# Just open and run run_cleaning.py in your IDE
# Or copy this into a new Python file:

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cleaning import clean_dataframe, get_default_config
import pandas as pd

# Load data
df = pd.read_csv('data/db_computers_2025_raw.csv')

# Get config and clean
config = get_default_config()
df_clean = clean_dataframe(df, config)

# Export
df_clean.to_csv('data/clean/db_computers_cleaned.csv', index=False)
print("Done! Cleaned dataset saved.")
```

### Use individual cleaning functions:

```python
from cleaning import extract_numeric_series, clean_storage_fields

# Extract numeric from string
df['ram_gb'] = extract_numeric_series(df['RAM_Memoria RAM'])

# Clean storage
df = clean_storage_fields(df, ssd_col='Disco duro_Capacidad de memoria SSD')
```

## ✅ Checklist Before Submission

- [ ] Notebook runs successfully: `notebooks/EDA_and_Cleaning.ipynb`
- [ ] Cleaned dataset generated: `data/clean/db_computers_cleaned.csv`
- [ ] Documentation complete: `docs/data_cleaning_notes.md`
- [ ] All Python modules included: `src/cleaning/*.py`
- [ ] All cells in notebook execute without errors
- [ ] Visualizations generate correctly

---

**Note**: The cleaned dataset is already generated and ready to use. You can either:
- Use the existing `data/clean/db_computers_cleaned.csv` file
- Re-run the notebook/script to regenerate it

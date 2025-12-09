from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent

# Model and Data Artifacts
MODEL_PATH = BASE_DIR / "artifacts" / "price_prediction_pipeline.joblib"
PREPROCESSOR_PATH = BASE_DIR / "artifacts" / "price_preprocessor.joblib"
SHAP_BACKGROUND_PATH = BASE_DIR / "artifacts" / "shap_background.joblib"
DATA_PATH = BASE_DIR / "data" / "clean" / "db_computers_cleaned.csv"

# Feature Lists (kept in one place to avoid drift across modules)
NUMERIC_FEATURES = [
    "cpu_bench_mark",
    "cpu_bench_rank",
    "cpu_bench_value",
    "gpu_bench_mark",
    "gpu_bench_rank",
    "gpu_bench_value",
    "ram_gb",
    "total_storage_gb",
    "total_ssd_gb",
    "total_hdd_gb",
    "screen_inches",
    "screen_refresh_hz",
    "screen_width_px",
    "screen_height_px",
    "screen_pixel_density",
    "combined_perf_index",
    "has_ssd",
    "is_high_refresh",
    "is_high_ram",
    "is_gaming_ready",
]

CATEGORICAL_FEATURES = [
    "Tipo de producto",
    "cpu_brand",
    "cpu_family",
    "gpu_brand",
    "gpu_series",
    "screen_resolution_category",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


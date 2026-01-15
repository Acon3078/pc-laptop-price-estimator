# Computer Price Estimator (Streamlit)

## 1) Project Overview — One-Sentence Value Proposition
End-to-end ML web app that cleans and enriches a large computer-market dataset, trains a regression pipeline, and serves **price predictions with local explainability (SHAP)** via both a structured UI and an optional natural-language interface.

## 2) Scope
- Price prediction from partially specified computer specs via a trained regression pipeline
- Per-prediction explainability (SHAP contribution breakdown) plus global feature importances
- Two interaction modes: structured form inputs + optional natural-language querying (API-key gated fallback)

## 3) Key Features / What It Does
- **EDA + Data Quality Profiling:** visualizes distributions and missingness (133/136 columns have missing values; many are sparse).
- **Data Cleaning & Parsing (multilingual/unstructured):**
  - Normalizes CPU/GPU strings (removes trademarks/Spanish terms) for benchmark matching.
  - Parses storage specs like `512GB SSD + 1TB HDD` → numeric features (total/SSD/HDD in GB).
  - Extracts screen size/resolution from Spanish formats (e.g., `15.6 pulgadas`, `1920 x 1080`) and derives pixel density.
  - Simplifies multilabel fields by taking the first label across ~49 columns (e.g., `/`, `+`, `,`, `|` separators).
- **External Data Enrichment:** fuzzy matches CPU/GPU models to benchmark tables (PassMark-style datasets) to add `mark/rank/value` features.
- **Handles Incomplete Inputs:** supports partially specified configurations by using sensible defaults + pipeline-based imputation (preprocessor fit on training data to avoid leakage).
- **Predictive Price Estimation:** scikit-learn `Pipeline` + `ColumnTransformer` with:
  - numeric: median imputation → scaling
  - categorical: most-frequent imputation → one-hot encoding (`handle_unknown="ignore"`)
- **Explainability:** permutation importance + **per-prediction SHAP contribution breakdown** served in the Streamlit app.
- **Dual User Interfaces:**
  - Standard form inputs (sliders/dropdowns) with sensible defaults derived from training stats.
  - Natural-language chatbot that (when enabled) extracts constraints and queries the dataset + model to answer conversationally.
- **Prediction History (per session):** stores user prediction history in Streamlit `session_state` (not yet persisted).

## 4) Results (Validation Split)
Trained on **8,064 listings** with an **80/20 split** (random_state=42).
- **Validation RMSE:** 278.47  
- **Validation MAE:** 191.99  
- **Best CV RMSE (tuning run):** 269.96 (5-fold CV)

## 5) Tech Stack & Skills Demonstrated (Concrete)
**Core ML & Data**
- Dataset profiling + missingness analysis on a sparse real-world dataset (8,064 × 136 raw features).
- Robust parsing of multilingual specs (storage, screen size/resolution, numeric extraction with European separators).
- Feature engineering: `combined_perf_index` (weighted log CPU/GPU benchmarks), storage totals, pixel density, and binary capability flags (e.g., high refresh / high RAM / gaming-ready).
- External enrichment via fuzzy matching of CPU/GPU names to benchmark tables.

**Modeling**
- Baselines compared with **5-fold KFold CV**: Linear Regression, Random Forest, Gradient Boosting (and XGBoost if available).
- Final pipeline: `ColumnTransformer` + `GradientBoostingRegressor`.
- Tuning via `RandomizedSearchCV` (scoring: neg-RMSE) for Random Forest search space.

**Explainability**
- Global importance via `permutation_importance`.
- Local explanations via **SHAP TreeExplainer** using a fixed background sample of 1,000 training rows.

**Deployment (Streamlit)**
- Artifact-driven inference (`joblib`): pipeline, preprocessor, SHAP background.
- Caching: `@st.cache_resource` for pipeline/preprocessor and `@st.cache_data` for dataset.
- Modular app structure (`app_core/`, `src/cleaning/`) + bilingual UI toggle (EN/ES).

**LLM Interface (Optional)**
- Uses OpenAI client when `OPENAI_API_KEY` is set; otherwise runs in echo/fallback mode.
- Two-step flow: LLM extracts filters → app queries dataset and predicts using the same model pipeline.
- Guardrails + error handling for quota/auth failures with graceful fallback.

## 6) How It Works (High-Level Flow)
- **Data** → cleaned/enriched in `Model_Training.ipynb` + `src/cleaning/` → saved to `data/clean/db_computers_cleaned.csv` (and feature-engineered `db_computers_final.csv`).
- **Modeling** → pipeline exported to:
  - `artifacts/price_prediction_pipeline.joblib`
  - `artifacts/price_preprocessor.joblib`
  - `artifacts/shap_background.joblib`
- **App** → `app.py` loads artifacts, provides tabs for prediction / debug / chatbot, and renders SHAP-based explanations when available.
- **Chatbot** → `chatbot.py` answers dataset/model questions (requires `OPENAI_API_KEY`; otherwise echo mode).

## 7) Quick Start / How to Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Then open the URL Streamlit prints (e.g., http://localhost:8501).

**Notes**
- Keep `artifacts/` and `data/clean/db_computers_cleaned.csv` alongside `app.py`.
- Set `OPENAI_API_KEY` (env or `.env`) to enable the chatbot’s “smart” mode; without it, the chatbot echoes.

## 8) Roadmap / Next Steps
- Persist prediction history to disk/DB (the notebook includes a `log_interaction()` JSONL helper; not currently wired into the app).
- Add a dedicated “Model Insights” view to surface global importances already exported from training.
- Optional: host artifacts remotely and add a startup download hook.

## Connect with me
<p>
  <a href="https://www.linkedin.com/in/adrian-con">
    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"/>
  </a>
  <a href="mailto:adriancongarcia10@gmail.com">
    <img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Gmail"/>
  </a>
  <a href="https://medium.com/@adriancongarcia10">
    <img src="https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white" alt="Medium"/>
  </a>
  <a href="https://github.com/Acon3078">
    <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/>
  </a>
</p>

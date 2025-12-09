# Architecture Overview

## Components
- `app.py`: Streamlit entrypoint; renders tabs (prediction, debug, chatbot), wires UI to the shared pipeline helpers, and manages session history/state.
- `app_core/`
  - `config.py`: Central feature lists and artifact paths (model, preprocessor, SHAP background, cleaned dataset).
  - `pipeline.py`: Loading/caching of sklearn pipeline & preprocessor, default/option extraction, input row builder, and core prediction helper.
  - `data.py`: Cached loader for the cleaned dataset used by the chatbot.
- `explainability.py`: SHAP explainer that reuses the shared feature list and artifacts to return price + top-k contributions + summary.
- `chatbot.py`: OpenAI-backed chat flow. Extracts filters, queries the cleaned dataset, falls back to synthetic rows if needed, and formats responses; echoes if no API key.
- `Model_Training.ipynb` and `src/cleaning/`: Data prep pipeline that produces the cleaned dataset and the model artifacts used by the app.

## Data & model artifacts
- Required at runtime: `artifacts/price_prediction_pipeline.joblib`, `artifacts/price_preprocessor.joblib`, `artifacts/shap_background.joblib`, and `data/clean/db_computers_cleaned.csv`.
- Optional hosting: If these are large, host externally (e.g., object storage or Git LFS) and add a small download-on-start hook before loading the pipeline.

## Caching & performance
- Streamlit caching is used for the pipeline, preprocessor, dataset, and feature defaults/options to avoid repeated loads.

## Deployment notes
- Streamlit Community Cloud entrypoint: `app.py`.
- Secrets (`OPENAI_API_KEY`, artifact URLs if using remote storage) go in `.streamlit/secrets.toml` or workspace env vars.
- For local dev: use `.env` to set `OPENAI_API_KEY`; keep secrets out of version control.

## Extensibility
- Feature definitions live in `app_core/config.py` to prevent drift across app, chatbot, and explainability modules.
- Additional tabs or APIs can import helpers from `app_core` without duplicating feature lists or loaders.


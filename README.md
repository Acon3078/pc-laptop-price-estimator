# Computer Price Estimator (Streamlit)

## 1) Project Overview — One-Sentence Value Proposition
Predict realistic laptop/PC prices from hardware specs in seconds, with a bilingual Streamlit UI, explainability, and an optional AI assistant to explore the dataset.

## 2) Key Features / What It Does
- Price prediction form for laptops/PCs (EN/ES UI).
- SHAP-based explanations to show which specs drive the price.
- OpenAI-powered chatbot that can query the cleaned dataset (falls back to echo if no key).
- Cleaned reference dataset and reusable model artifacts ready for deployment.

## 3) Why This Project Matters (Impact / Use Cases)
- Helps shoppers and resellers benchmark fair prices before buying or listing hardware.
- Assists procurement or support teams in giving consistent, data-backed quotes.
- Demonstrates an end-to-end ML workflow: cleaning, modeling, explainability, and UX.

## 4) Tech Stack & Skills Demonstrated
- Streamlit (multi-tab app, bilingual UX), Python
- scikit-learn pipelines + joblib artifacts
- SHAP for local model explanations
- OpenAI API integration for dataset Q&A
- pandas/numpy data cleaning and feature engineering (see `src/cleaning/`, `Model_Training.ipynb`)

## 5) How It Works (High-Level Flow)
- Data → Cleaned via the notebook and `src/cleaning/` utilities → saved to `data/clean/db_computers_cleaned.csv`.
- Modeling → Trained pipelines exported to `artifacts/price_prediction_pipeline.joblib`, `price_preprocessor.joblib`, `shap_background.joblib`.
- App → `app.py` loads artifacts, renders tabs for prediction/debug/chatbot, and serves SHAP explanations.
- Chatbot → `chatbot.py` answers dataset questions and can synthesize estimates; uses `OPENAI_API_KEY` when available.

## 6) Quick Start / How to Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Then open the URL Streamlit prints (e.g., http://localhost:8501).

Notes:
- Keep `artifacts/` and `data/clean/db_computers_cleaned.csv` alongside `app.py`. If you host them elsewhere, add a small download step before startup.
- Set `OPENAI_API_KEY` in your environment or `.env` to enable chatbot answers; without it, the chatbot echoes.
- For a deeper dive or to regenerate artifacts, run `Model_Training.ipynb` (full EDA/cleaning/model training) or use the functions in `src/cleaning/`.

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

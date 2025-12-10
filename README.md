# Computer Price Estimator (Streamlit)

## 1) Project Overview — One-Sentence Value Proposition
This project delivers an end-to-end machine-learning web application that analyzes a large computer-market dataset to provide accurate and explainable price predictions through both traditional and natural-language interfaces.

## 2) Key Features / What It Does
- **EDA and Cleaning**: Examining distributions and missing values to guide appropriate imputation and normalization for handling sparsity.
- **Predictive Price Estimation**: Model-based prediction of computer prices based on user-defined configurations or provided product features.
- **Feature Importance & Explainability**: Clear breakdown of which components most influence the predicted price.
- **Dual User Interfaces**:
    - Standard input fields (sliders, dropdowns, checkboxes) for structured data entry.
    - A lightweight natural-language interface enabling users to request predictions conversationally.
- **Feedback Capture**: A simple in-app mechanism that records user prediction history per session, with planned future use for evaluating outputs and monitoring quality.

## 3) Why This Project Matters (Impact / Use Cases)
- Supports informed purchasing decisions by transforming complex product specifications into interpretable insights and price expectations.
- Improves market transparency through data-driven exploration of thousands of real marketplace listings.
- Demonstrates practical ML deployment by connecting model development, evaluation, and serving in a real web application.
- Bridges human–AI interaction with both structured UI components and conversational agent capabilities.

## 4) Tech Stack & Skills Demonstrated
- **Tools & Libraries**: Python, pandas, numpy, scikit-learn pipelines + joblib artifacts, Streamlit, matplotlib/seaborn, OpenAI-powered chatbot (echo fallback without key).

- **Data Skills**: EDA, cleaning unstructured and multilingual fields, handling missing data, feature engineering, semantic alignment.

- **Modeling Skills**: Regression modeling, feature selection, model training/validation, hyperparameter tuning, performance estimation, explainability.

- **ML Workflow Competencies**: Model serving, agent-based NLP interfaces, bilingual UI integration, and end-to-end ML pipeline structuring.

- **Software Practices:** Code organization, modularization, testing, documentation, and collection of user feedback for future iterative improvement.

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

## 7) Roadmap / Next Steps
- Persist prediction history + optional user feedback (via the `log_interaction` helper in `Model_Training.ipynb`) to monitor usage and improve the model.
- Add an in-app “Model Insights” view for global/importances already exported from the notebook.
- Optional: Host artifacts remotely and add a startup download hook for lighter deployments.

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

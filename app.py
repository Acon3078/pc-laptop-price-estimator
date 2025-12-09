import pandas as pd
import streamlit as st
import chatbot
from app_core.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from app_core.data import load_dataset
from app_core.pipeline import (
    build_input_row,
    get_categorical_options,
    get_feature_defaults,
    load_pipeline,
    load_preprocessor,
    predict_price,
)

# -------------------------------------------------------------------
# STREAMLIT PAGE CONFIG (must be first Streamlit call)
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Computer Price Estimator",
    page_icon="💻",
    layout="wide",
)

# -------------------------------------------------------------------
# TRY TO LOAD SHAP EXPLAINABILITY API
# -------------------------------------------------------------------
try:
    from explainability import predict_and_explain
except Exception:
    # If explainability.py is missing or broken, fall back to None
    predict_and_explain = None

# -------------------------------------------------------------------
# LANGUAGE SUPPORT
# -------------------------------------------------------------------
LANGUAGES = {
    "en": "English",
    "es": "Español",
}

TEXT = {
    "en": {
        "app_title": "💻 Computer Price Estimator",
        "section1": "1. Describe the computer configuration",
        "core_hw": "Core hardware",
        "display_cfg": "Display & configuration",
        "cpu_brand": "CPU brand",
        "cpu_family": "CPU family / series (e.g. Core i7, Ryzen 5)",
        "cpu_bench_mark": "CPU benchmark score",
        "cpu_bench_rank": "CPU benchmark rank (lower is better)",
        "cpu_bench_value": "CPU value index",

        "gpu_title": "GPU",
        "gpu_brand": "GPU brand",
        "gpu_series": "GPU series / model (e.g. RTX 4060, RX 6800)",
        "gpu_bench_mark": "GPU benchmark score",
        "gpu_bench_rank": "GPU benchmark rank (lower is better)",
        "gpu_bench_value": "GPU value index",

        "mem_storage": "Memory & storage",
        "ram_gb": "RAM (GB)",
        "total_storage_gb": "Total storage (GB)",
        "total_ssd_gb": "SSD capacity (GB)",
        "total_hdd_gb": "HDD capacity (GB)",

        "product_type": "Product type",
        "screen_inches": "Screen size (inches)",
        "screen_refresh_hz": "Refresh rate (Hz)",
        "screen_width_px": "Screen width (pixels)",
        "screen_height_px": "Screen height (pixels)",
        "screen_ppi": "Pixel density (ppi)",
        "screen_res_cat": "Resolution category",

        "summary_flags": "Summary flags",
        "has_ssd": "Has SSD",
        "is_high_refresh": "High refresh (≥120 Hz)",
        "is_high_ram": "High RAM (≥16 GB)",
        "is_gaming_ready": "Gaming ready",

        "combined_perf": "Combined performance index (0–100)",
        "predict_button": "Predict price 💰",
        "short_expl": "Short explanation",
        "predicted_title": "2. Predicted price",
        "metric_label": "Estimated price (model)",
        "see_raw": "See raw feature vector",

        "history_title": "3. Prediction history",
        "clear_history": "Clear history 🧹",

        "tab_prediction": "🔮 Price prediction",
        "tab_debug": "🛠 Advanced / Debug",
        "tab_chat": "💬 Chatbot",
    },

    "es": {
        "app_title": "💻 Estimador de Precio de Ordenadores",
        "section1": "1. Describe la configuración del ordenador",
        "core_hw": "Componentes principales",
        "display_cfg": "Pantalla y configuración",
        "cpu_brand": "Marca de CPU",
        "cpu_family": "Familia / serie de CPU (p. ej. Core i7, Ryzen 5)",
        "cpu_bench_mark": "Puntuación de CPU",
        "cpu_bench_rank": "Rango de CPU (más bajo es mejor)",
        "cpu_bench_value": "Índice de valor de CPU",

        "gpu_title": "GPU",
        "gpu_brand": "Marca de GPU",
        "gpu_series": "Serie / modelo de GPU (p. ej. RTX 4060, RX 6800)",
        "gpu_bench_mark": "Puntuación de GPU",
        "gpu_bench_rank": "Rango de GPU (más bajo es mejor)",
        "gpu_bench_value": "Índice de valor de GPU",

        "mem_storage": "Memoria y almacenamiento",
        "ram_gb": "RAM (GB)",
        "total_storage_gb": "Almacenamiento total (GB)",
        "total_ssd_gb": "Capacidad SSD (GB)",
        "total_hdd_gb": "Capacidad HDD (GB)",

        "product_type": "Tipo de producto",
        "screen_inches": "Tamaño de pantalla (pulgadas)",
        "screen_refresh_hz": "Tasa de refresco (Hz)",
        "screen_width_px": "Ancho de pantalla (píxeles)",
        "screen_height_px": "Alto de pantalla (píxeles)",
        "screen_ppi": "Densidad de píxeles (ppi)",
        "screen_res_cat": "Categoría de resolución",

        "summary_flags": "Indicadores resumen",
        "has_ssd": "Incluye SSD",
        "is_high_refresh": "Alta frecuencia (≥120 Hz)",
        "is_high_ram": "Mucha RAM (≥16 GB)",
        "is_gaming_ready": "Listo para gaming",

        "combined_perf": "Índice de rendimiento combinado (0–100)",
        "predict_button": "Predecir precio 💰",
        "short_expl": "Explicación breve",
        "predicted_title": "2. Precio predicho",
        "metric_label": "Precio estimado (modelo)",
        "see_raw": "Ver vector de características",

        "history_title": "3. Historial de predicciones",
        "clear_history": "Borrar historial 🧹",

        "tab_prediction": "🔮 Predicción de precio",
        "tab_debug": "🛠 Avanzado / Debug",
        "tab_chat": "💬 Chatbot",
    },
}

# Session state for language
if "lang" not in st.session_state:
    st.session_state["lang"] = "en"


def t(key: str) -> str:
    """Translate key according to current language."""
    lang = st.session_state.get("lang", "en")
    return TEXT.get(lang, TEXT["en"]).get(key, key)


# -------------------------------------------------------------------
# TOP BAR: TITLE + LANGUAGE TOGGLE
# -------------------------------------------------------------------
col_title, col_lang = st.columns([3, 1])

with col_title:
    st.title(t("app_title"))

with col_lang:
    # Right-aligned label
    st.write("")
    st.write("")
    st.session_state["lang"] = st.radio(
        "Language / Idioma",
        options=list(LANGUAGES.keys()),
        format_func=lambda x: LANGUAGES[x],
        horizontal=True,
        index=list(LANGUAGES.keys()).index(st.session_state["lang"]),
    )

# -------------------------------------------------------------------
# INTRO TEXT (EN / ES)
# -------------------------------------------------------------------
INTRO_EN = """
Welcome to the **Computer Price Estimator** — a smart tool that predicts the price of a PC or laptop 
based on its technical characteristics.

This application allows users to:

### 1. Build a computer configuration  
Choose CPU, GPU, RAM, storage, screen characteristics, and other specifications.  
Default values are automatically suggested when you’re unsure.

### 2. Get an instant price estimate  
Once the configuration is complete, the model predicts the **market price** based on thousands of real products.

### 3. Understand *why* that price was predicted  
The estimator provides a short explanation that highlights the most important components influencing the price.  
(This module can be enhanced with Toni’s full SHAP explainability.)

---

This tool is part of the **DAI Group Project**, specifically supporting:

- **Application UI**  
- **Explainability**  

---

### Get started  
Go to the **Price prediction** tab below, configure your device, and click **"Predict price 💰"**!
"""

INTRO_ES = """
Bienvenido al **Estimador de Precio de Ordenadores** — una herramienta inteligente que predice el precio de un PC
o portátil a partir de sus características técnicas.

Esta aplicación te permite:

### 1. Construir una configuración de ordenador  
Elige CPU, GPU, RAM, almacenamiento, pantalla y otras especificaciones.  
Se proponen valores por defecto automáticamente cuando no estás seguro.

### 2. Obtener una estimación de precio al instante  
Cuando la configuración está completa, el modelo predice el **precio de mercado** usando miles de productos reales.

### 3. Entender *por qué* se ha predicho ese precio  
El estimador genera una breve explicación con los componentes que más influyen en el precio.  
(Este módulo se puede ampliar con toda la explicabilidad SHAP de Toni.)

---

Esta herramienta forma parte del **DAI Group Project**, apoyando:

- **Interfaz de aplicación (Application UI)**  
- **Explicabilidad (Explainability)**  

---

### Empezar  
Ve a la pestaña **Predicción de precio**, configura tu equipo y pulsa **"Predecir precio 💰"**.
"""

st.markdown(INTRO_EN if st.session_state["lang"] == "en" else INTRO_ES)

# Session state for prediction history
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []


# -------------------------------------------------------------------
# LOAD PIPELINE OBJECTS
# -------------------------------------------------------------------
pipeline = load_pipeline()
preprocessor = load_preprocessor()
defaults = get_feature_defaults(preprocessor)
cat_options = get_categorical_options(preprocessor)

# -------------------------------------------------------------------
# TABS
# -------------------------------------------------------------------
tab_form, tab_advanced, tab_chat = st.tabs(
    [t("tab_prediction"), t("tab_debug"), t("tab_chat")]
)

# -------------------------------------------------------------------
# TAB 1 – MAIN PREDICTION FORM
# -------------------------------------------------------------------
with tab_form:
    st.subheader(t("section1"))

    with st.form("prediction_form"):
        col_left, col_right = st.columns(2)

        # --------- LEFT COLUMN: CPU / GPU / RAM / STORAGE ----------
        with col_left:
            st.markdown(f"### {t('core_hw')}")
            st.markdown("**CPU**")

            cpu_brand_options = cat_options.get("cpu_brand") or ["Intel", "AMD", "Apple", "Other"]
            cpu_brand_default = str(defaults.get("cpu_brand", cpu_brand_options[0]))
            cpu_brand_index = cpu_brand_options.index(cpu_brand_default) if cpu_brand_default in cpu_brand_options else 0

            cpu_brand = st.selectbox(
                t("cpu_brand"),
                options=cpu_brand_options,
                index=cpu_brand_index,
                key="cpu_brand",
            )

            cpu_family = st.text_input(
                t("cpu_family"),
                value=str(defaults.get("cpu_family", "")),
                key="cpu_family",
            )

            cpu_bench_mark = st.number_input(
                t("cpu_bench_mark"),
                min_value=0.0,
                max_value=100000.0,
                value=float(defaults.get("cpu_bench_mark", 20000.0)),
                step=100.0,
                help="Higher is better (e.g. PassMark CPU Mark).",
            )
            cpu_bench_rank = st.number_input(
                t("cpu_bench_rank"),
                min_value=0.0,
                max_value=5000.0,
                value=float(defaults.get("cpu_bench_rank", 1000.0)),
                step=10.0,
            )
            cpu_bench_value = st.number_input(
                t("cpu_bench_value"),
                min_value=0.0,
                max_value=500.0,
                value=float(defaults.get("cpu_bench_value", 50.0)),
                step=1.0,
                help="Performance per price index.",
            )

            st.markdown("---")
            st.markdown(f"**{t('gpu_title')}**")

            gpu_brand_options = cat_options.get("gpu_brand") or ["NVIDIA", "AMD", "Intel", "Other"]
            gpu_brand_default = str(defaults.get("gpu_brand", gpu_brand_options[0]))
            gpu_brand_index = gpu_brand_options.index(gpu_brand_default) if gpu_brand_default in gpu_brand_options else 0

            gpu_brand = st.selectbox(
                t("gpu_brand"),
                options=gpu_brand_options,
                index=gpu_brand_index,
                key="gpu_brand",
            )

            gpu_series = st.text_input(
                t("gpu_series"),
                value=str(defaults.get("gpu_series", "")),
                key="gpu_series",
            )

            gpu_bench_mark = st.number_input(
                t("gpu_bench_mark"),
                min_value=0.0,
                max_value=100000.0,
                value=float(defaults.get("gpu_bench_mark", 15000.0)),
                step=100.0,
                help="Higher is better (e.g. PassMark G3D Mark).",
            )
            gpu_bench_rank = st.number_input(
                t("gpu_bench_rank"),
                min_value=0.0,
                max_value=10000.0,
                value=float(defaults.get("gpu_bench_rank", 2000.0)),
                step=10.0,
            )
            gpu_bench_value = st.number_input(
                t("gpu_bench_value"),
                min_value=0.0,
                max_value=500.0,
                value=float(defaults.get("gpu_bench_value", 40.0)),
                step=1.0,
            )

            st.markdown("---")
            st.markdown(f"**{t('mem_storage')}**")

            ram_gb = st.number_input(
                t("ram_gb"),
                min_value=2.0,
                max_value=256.0,
                value=float(defaults.get("ram_gb", 16.0)),
                step=2.0,
            )
            total_storage_gb = st.number_input(
                t("total_storage_gb"),
                min_value=64.0,
                max_value=8192.0,
                value=float(defaults.get("total_storage_gb", 512.0)),
                step=64.0,
            )
            total_ssd_gb = st.number_input(
                t("total_ssd_gb"),
                min_value=0.0,
                max_value=8192.0,
                value=float(defaults.get("total_ssd_gb", 512.0)),
                step=64.0,
            )
            total_hdd_gb = st.number_input(
                t("total_hdd_gb"),
                min_value=0.0,
                max_value=8192.0,
                value=float(defaults.get("total_hdd_gb", 0.0)),
                step=128.0,
            )

        # --------- RIGHT COLUMN: SCREEN / FLAGS / TYPE ----------
        with col_right:
            st.markdown(f"### {t('display_cfg')}")

            tipo_options = cat_options.get("Tipo de producto") or ["Portátil", "Sobremesa", "All in One", "Mini PC"]
            tipo_default = str(defaults.get("Tipo de producto", tipo_options[0]))
            tipo_index = tipo_options.index(tipo_default) if tipo_default in tipo_options else 0

            tipo_producto = st.selectbox(
                t("product_type"),
                options=tipo_options,
                index=tipo_index,
                key="tipo_prod",
            )

            screen_inches = st.number_input(
                t("screen_inches"),
                min_value=10.0,
                max_value=40.0,
                value=float(defaults.get("screen_inches", 15.6)),
                step=0.1,
            )
            screen_refresh_hz = st.number_input(
                t("screen_refresh_hz"),
                min_value=30.0,
                max_value=360.0,
                value=float(defaults.get("screen_refresh_hz", 60.0)),
                step=10.0,
            )

            default_width = float(defaults.get("screen_width_px", 1920.0))
            default_height = float(defaults.get("screen_height_px", 1080.0))

            screen_width_px = st.number_input(
                t("screen_width_px"),
                min_value=800.0,
                max_value=5120.0,
                value=max(default_width, 800.0),
                step=80.0,
            )

            screen_height_px = st.number_input(
                t("screen_height_px"),
                min_value=600.0,
                max_value=2880.0,
                value=max(default_height, 600.0),
                step=60.0,
            )

            default_ppi = float(defaults.get("screen_pixel_density", 140.0))

            screen_pixel_density = st.number_input(
                t("screen_ppi"),
                min_value=70.0,
                max_value=400.0,
                value=max(70.0, min(default_ppi, 400.0)),
                step=5.0,
            )

            res_options = cat_options.get("screen_resolution_category") or ["HD", "Full HD", "2K", "4K"]
            res_default = str(defaults.get("screen_resolution_category", res_options[0]))
            res_index = res_options.index(res_default) if res_default in res_options else 0

            screen_resolution_category = st.selectbox(
                t("screen_res_cat"),
                options=res_options,
                index=res_index,
                key="screen_res_cat",
            )

            st.markdown("---")
            st.markdown(f"**{t('summary_flags')}**")

            has_ssd = st.checkbox(
                t("has_ssd"),
                value=bool(defaults.get("has_ssd", True)),
            )
            is_high_refresh = st.checkbox(
                t("is_high_refresh"),
                value=bool(defaults.get("is_high_refresh", screen_refresh_hz >= 120)),
            )
            is_high_ram = st.checkbox(
                t("is_high_ram"),
                value=bool(defaults.get("is_high_ram", ram_gb >= 16)),
            )
            is_gaming_ready = st.checkbox(
                t("is_gaming_ready"),
                value=bool(defaults.get("is_gaming_ready", False)),
            )

            combined_perf_index = st.number_input(
                t("combined_perf"),
                min_value=0.0,
                max_value=100.0,
                value=float(defaults.get("combined_perf_index", 50.0)),
                step=1.0,
                help="Optional global score combining CPU, GPU, RAM, etc.",
            )

        submitted = st.form_submit_button(t("predict_button"), use_container_width=True)

    # --- Handle prediction after form submit ---
    if submitted:
        num_inputs = {
            "cpu_bench_mark": cpu_bench_mark,
            "cpu_bench_rank": cpu_bench_rank,
            "cpu_bench_value": cpu_bench_value,
            "gpu_bench_mark": gpu_bench_mark,
            "gpu_bench_rank": gpu_bench_rank,
            "gpu_bench_value": gpu_bench_value,
            "ram_gb": ram_gb,
            "total_storage_gb": total_storage_gb,
            "total_ssd_gb": total_ssd_gb,
            "total_hdd_gb": total_hdd_gb,
            "screen_inches": screen_inches,
            "screen_refresh_hz": screen_refresh_hz,
            "screen_width_px": screen_width_px,
            "screen_height_px": screen_height_px,
            "screen_pixel_density": screen_pixel_density,
            "combined_perf_index": combined_perf_index,
            "has_ssd": int(has_ssd),
            "is_high_refresh": int(is_high_refresh),
            "is_high_ram": int(is_high_ram),
            "is_gaming_ready": int(is_gaming_ready),
        }

        cat_inputs = {
            "Tipo de producto": tipo_producto,
            "cpu_brand": cpu_brand,
            "cpu_family": cpu_family,
            "gpu_brand": gpu_brand,
            "gpu_series": gpu_series,
            "screen_resolution_category": screen_resolution_category,
        }

        row_df = build_input_row(num_inputs, cat_inputs)

        # If explainability API is available, use it, otherwise plain prediction
        if predict_and_explain is not None:
            result = predict_and_explain(row_df)
            predicted_price = float(result["predicted_price"])
            summary_text = result.get("summary", "")
        else:
            predicted_price = predict_price(pipeline, row_df)
            summary_text = (
                "Explainability module not connected yet. "
                "Toni can plug in `predict_and_explain(row_df)` via explainability.py."
            )

        # Save to history
        st.session_state["prediction_history"].append(
            {
                "Product type": tipo_producto,
                "CPU": f"{cpu_brand} {cpu_family}",
                "GPU": f"{gpu_brand} {gpu_series}",
                "RAM (GB)": ram_gb,
                "Storage (GB)": total_storage_gb,
                "Predicted price (€)": round(predicted_price, 2),
            }
        )

        st.markdown(f"### {t('predicted_title')}")
        col_price, col_summary = st.columns([1, 2])

        with col_price:
            st.metric(t("metric_label"), f"{predicted_price:,.0f} €")

        with col_summary:
            st.write(f"**{t('short_expl')}**")
            st.write(summary_text)

        with st.expander(t("see_raw")):
            st.json({"numeric": num_inputs, "categorical": cat_inputs})


# -------------------------------------------------------------------
# TAB 2 – ADVANCED / DEBUG
# -------------------------------------------------------------------
with tab_advanced:
    st.subheader("Model & feature debug")

    st.markdown("**Features used by the model**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Numeric features**")
        st.write(NUMERIC_FEATURES)
    with col2:
        st.markdown("**Categorical features**")
        st.write(CATEGORICAL_FEATURES)

    st.markdown("---")

    st.markdown("**Pipeline structure**")
    st.text(str(pipeline))

    if st.checkbox("Show preprocessor details"):
        st.write(preprocessor)

    st.markdown("---")

# -------------------------------------------------------------------
# TAB 3 – CHATBOT
# -------------------------------------------------------------------
with tab_chat:
    col_title, col_clear = st.columns([3, 1])
    with col_title:
        st.subheader("Chatbot")
    with col_clear:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    prompt = st.chat_input(
        "Ask about prices, hardware comparisons, or best-buy opportunities"
    )

    if not chatbot.is_ready():
        st.info(
            "💡 To enable AI responses, set OPENAI_API_KEY in .env file. "
            "Using basic echo mode for now."
        )

    if prompt:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Show loading indicator while processing
        with st.spinner("🤔 Analyzing your question and querying the dataset..."):
            # Load dataset and pass to chatbot
            dataset_df = load_dataset()
            reply = chatbot.respond(prompt, st.session_state.chat_history, dataset_df=dataset_df, pipeline=pipeline, preprocessor=preprocessor)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
        
        # Rerun to display the new messages
        st.rerun()

# -------------------------------------------------------------------
# GLOBAL PREDICTION HISTORY (UNDER TABS)
# -------------------------------------------------------------------
if st.session_state["prediction_history"]:
    st.markdown(f"### {t('history_title')}")

    hist_df = pd.DataFrame(st.session_state["prediction_history"])
    st.dataframe(hist_df, use_container_width=True)

    if st.button(t("clear_history")):
        st.session_state["prediction_history"].clear()
        st.rerun()

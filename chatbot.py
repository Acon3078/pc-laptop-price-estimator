"""
Lightweight chatbot helper.

- Uses OpenAI if API key is set (via .env file or OPENAI_API_KEY env var).
- Falls back to a safe echo response when the key/client is missing or errors.
"""
import os
from typing import List, Dict

from app_core.config import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, will use system env vars only
    pass

try:
    import streamlit as st
    _has_streamlit = True
except Exception:
    _has_streamlit = False

try:
    from openai import OpenAI
except Exception:  # package missing or import error
    OpenAI = None

_client = None


def _get_client():
    """Lazy initialization of OpenAI client - loads from .env or environment variable."""
    global _client
    if _client is not None:
        return _client
    
    if OpenAI is None:
        return None
    
    # Ensure dotenv is loaded (in case it wasn't loaded at import time)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    # Load from environment variable (loaded from .env file by dotenv)
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        return None
    
    try:
        # Ensure it's a string and strip whitespace
        api_key_str = str(api_key).strip()
        if not api_key_str:
            return None
        
        _client = OpenAI(api_key=api_key_str)
        return _client
    except Exception:
        return None


def is_ready() -> bool:
    """Return True if an LLM client is configured."""
    return _get_client() is not None


def get_system_prompt() -> str:
    return (
        "You are a helpful assistant for a laptop/PC pricing application. "
        "Your responses must be based ONLY on the dataset and predictions provided by this application. "
        "Do not use external knowledge or general information about hardware - only reference data "
        "from the application's dataset and model predictions.\n\n"
        "IMPORTANT FORMATTING RULES:\n"
        "- Format ALL numbers as plain text (e.g., '1,500 euros' not '1500€' or code blocks)\n"
        "- Do not use markdown code blocks for numbers\n"
        "- Use consistent number formatting throughout (e.g., '1,234.56' for prices)\n"
        "- Keep answers concise (<=120 words)\n\n"
        "If you don't have specific data from the dataset to answer a question, "
        "be transparent and state that the information is not available in the current dataset."
    )


def _build_messages(user_msg: str, history: List[Dict[str, str]]) -> list:
    """Convert Streamlit chat history to OpenAI chat messages."""
    messages = [{"role": "system", "content": get_system_prompt()}]
    # Keep last few turns to control token use
    for turn in history[-4:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_msg})
    return messages


def query_dataset_and_predict(
    filters: dict,
    dataset_df,
    pipeline,
    max_results: int = 10
) -> dict:
    """
    Query the dataset based on filters and run predictions.
    
    Args:
        filters: Dict with filter criteria (e.g., {'Tipo de producto': ['Desktop', 'Mini-PC'], 
                 'cpu_clean': 'core ultra 7', 'gpu_brand': 'NVIDIA'})
        dataset_df: The cleaned dataset DataFrame
        pipeline: The prediction pipeline
        max_results: Maximum number of results to return
    
    Returns:
        Dict with 'results' (DataFrame), 'count' (total matches), 'predictions' (array)
    """
    if dataset_df is None or pipeline is None:
        return {"results": None, "count": 0, "predictions": None}
    
    df = dataset_df.copy()
    
    # Apply filters
    for key, value in filters.items():
        if key not in df.columns:
            continue
        
        if isinstance(value, list):
            # Multiple values (OR condition)
            df = df[df[key].isin(value)]
        elif isinstance(value, str):
            # String contains (case-insensitive)
            df = df[df[key].astype(str).str.contains(value, case=False, na=False)]
        elif isinstance(value, (int, float)):
            # Exact numeric match
            df = df[df[key] == value]
    
    if len(df) == 0:
        return {"results": None, "count": 0, "predictions": None}
    
    # Limit results
    df_results = df.head(max_results).copy()
    
    # Run predictions if we have the required features
    predictions = None
    try:
        # Check if we have the required columns
        missing_cols = [col for col in ALL_FEATURES if col not in df_results.columns]
        if not missing_cols:
            # Run predictions
            preds = pipeline.predict(df_results[ALL_FEATURES])
            predictions = preds.tolist()
            df_results['predicted_price'] = predictions
    except Exception:
        # If prediction fails, continue without predictions
        pass
    
    return {
        "results": df_results,
        "count": len(df),
        "predictions": predictions
    }


def create_synthetic_rows(filters: dict, preprocessor=None) -> list:
    """
    Create synthetic rows for prediction based on filters and defaults.
    Returns list of dicts with feature values.
    """
    import pandas as pd
    
    # Default values
    defaults = {}
    for f in NUMERIC_FEATURES:
        defaults[f] = 0.0
    for f in CATEGORICAL_FEATURES:
        defaults[f] = "Unknown"
    
    # Apply defaults from preprocessor if available
    if preprocessor is not None:
        try:
            transformers = getattr(preprocessor, "named_transformers_", {})
            num_pipeline = transformers.get("num")
            if num_pipeline is not None:
                num_imputer = num_pipeline.named_steps.get("imputer")
                if hasattr(num_imputer, "statistics_"):
                    for feature, value in zip(NUMERIC_FEATURES, num_imputer.statistics_):
                        defaults[feature] = float(value)
            
            cat_pipeline = transformers.get("cat")
            if cat_pipeline is not None:
                cat_imputer = cat_pipeline.named_steps.get("imputer")
                if hasattr(cat_imputer, "statistics_"):
                    for feature, value in zip(CATEGORICAL_FEATURES, cat_imputer.statistics_):
                        defaults[feature] = value
        except Exception:
            pass
    
    # Extract product types from filters
    product_types = filters.get("Tipo de producto", [])
    if isinstance(product_types, str):
        product_types = [product_types]
    if not product_types:
        product_types = ["Desktop", "Mini-PC"]  # Default if not specified
    
    # Create rows for each product type
    rows = []
    for product_type in product_types:
        row = defaults.copy()
        
        # Apply filters
        row["Tipo de producto"] = product_type
        
        # CPU filters
        if "cpu_clean" in filters:
            cpu_str = str(filters["cpu_clean"]).lower()
            if "core ultra 7" in cpu_str or "ultra 7" in cpu_str:
                row["cpu_brand"] = "Intel"
                row["cpu_family"] = "Core Ultra 7"
                row["cpu_bench_mark"] = defaults.get("cpu_bench_mark", 20000)
            elif "ryzen" in cpu_str:
                row["cpu_brand"] = "AMD"
                if "7" in cpu_str:
                    row["cpu_family"] = "Ryzen 7"
                    row["cpu_bench_mark"] = defaults.get("cpu_bench_mark", 18000)
        
        # GPU filters
        if "gpu_brand" in filters:
            gpu_brand = str(filters["gpu_brand"]).upper()
            if "NVIDIA" in gpu_brand:
                row["gpu_brand"] = "NVIDIA"
                row["gpu_series"] = filters.get("gpu_series", "GeForce")
                row["gpu_bench_mark"] = defaults.get("gpu_bench_mark", 15000)
            elif "AMD" in gpu_brand:
                row["gpu_brand"] = "AMD"
                row["gpu_series"] = filters.get("gpu_series", "Radeon")
        
        # RAM
        if "ram_gb" in filters:
            row["ram_gb"] = float(filters["ram_gb"])
        else:
            row["ram_gb"] = 16.0  # Default
        
        # Storage
        if "storage_total_gb" in filters:
            row["total_storage_gb"] = float(filters["storage_total_gb"])
            row["total_ssd_gb"] = float(filters["storage_total_gb"])
        else:
            row["total_storage_gb"] = 512.0
            row["total_ssd_gb"] = 512.0
        
        # Screen defaults (for desktops/mini-PC, these might be less relevant)
        row["screen_inches"] = 0.0 if product_type in ["Desktop", "Mini-PC"] else defaults.get("screen_inches", 15.0)
        row["screen_resolution_category"] = "Unknown"
        
        rows.append(row)
    
    return rows


def respond(user_msg: str, history: List[Dict[str, str]], dataset_df=None, pipeline=None, preprocessor=None) -> str:
    """Return an LLM answer or a graceful fallback."""
    client = _get_client()
    if client is None:
        return (
            f"I heard: '{user_msg}'. "
            "To enable smarter answers, set OPENAI_API_KEY in .env file "
            "or set OPENAI_API_KEY environment variable."
        )

    # Extract filters and generate predictions
    predictions_context = ""
    if dataset_df is not None and pipeline is not None:
        try:
            import json
            import pandas as pd
            
            # Use LLM to extract filters from user query
            filter_prompt = (
                f"Extract filter criteria from this user query: '{user_msg}'\n\n"
                "Return ONLY a JSON object with filter criteria. Use these keys:\n"
                "- 'Tipo de producto': array of product types like ['Desktop', 'Mini-PC', 'Portátil']\n"
                "- 'cpu_clean': CPU name string (e.g., 'core ultra 7', 'ryzen 7')\n"
                "- 'gpu_brand': GPU brand string like 'NVIDIA', 'AMD', 'Intel'\n"
                "- 'gpu_series': GPU series string (e.g., 'rtx 4070', 'geforce')\n"
                "- 'ram_gb': RAM in GB (number)\n"
                "- 'storage_total_gb': Storage in GB (number)\n\n"
                "Example: {\"Tipo de producto\": [\"Desktop\", \"Mini-PC\"], \"cpu_clean\": \"core ultra 7\", \"gpu_brand\": \"NVIDIA\"}\n"
                "Return ONLY valid JSON, no other text."
            )
            
            filter_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": filter_prompt}],
                temperature=0.1,
                max_tokens=200,
            )
            
            filters_str = filter_response.choices[0].message.content.strip()
            # Extract JSON from response
            if "{" in filters_str:
                filters_str = filters_str[filters_str.index("{"):filters_str.rindex("}")+1]
            filters = json.loads(filters_str)
            
            # Try to query dataset first
            query_result = query_dataset_and_predict(filters, dataset_df, pipeline, max_results=5)
            
            predictions_data = []
            if query_result["count"] > 0 and query_result["results"] is not None:
                # Use actual dataset results
                df_results = query_result["results"]
                for idx, row in df_results.iterrows():
                    pred_price = row.get('predicted_price')
                    if pred_price:
                        product_type = row.get('Tipo de producto', 'Unknown')
                        cpu = row.get('cpu_clean', 'Unknown')
                        gpu = row.get('gpu_clean', 'Unknown')
                        predictions_data.append({
                            'product_type': product_type,
                            'cpu': cpu,
                            'gpu': gpu,
                            'predicted_price': float(pred_price)
                        })
            
            # If no dataset matches, create synthetic predictions
            if not predictions_data and filters:
                synthetic_rows = create_synthetic_rows(filters, preprocessor)
                if synthetic_rows:
                    df_synthetic = pd.DataFrame(synthetic_rows)
                    try:
                        preds = pipeline.predict(df_synthetic[ALL_FEATURES])
                        for i, row in df_synthetic.iterrows():
                            predictions_data.append({
                                'product_type': row.get('Tipo de producto', 'Unknown'),
                                'cpu': row.get('cpu_family', 'Unknown'),
                                'gpu': f"{row.get('gpu_brand', 'Unknown')} {row.get('gpu_series', '')}".strip(),
                                'predicted_price': float(preds[i])
                            })
                    except Exception:
                        pass
            
            # Format predictions context
            if predictions_data:
                predictions_context = "\n\nPREDICTIONS FROM MODEL:\n"
                for pred in predictions_data:
                    predictions_context += f"- {pred['product_type']} with {pred['cpu']} CPU and {pred['gpu']} GPU: Predicted price {pred['predicted_price']:.0f} euros\n"
        except Exception as e:
            # If prediction fails, continue without it
            import sys
            print(f"Error in prediction flow: {e}", file=sys.stderr)
            pass

    try:
        # Build messages with predictions context
        user_msg_with_context = user_msg + predictions_context
        messages = _build_messages(user_msg_with_context, history)
        result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            max_tokens=400,
        )
        return result.choices[0].message.content.strip()
    except Exception as exc:
        # Provide user-friendly error messages
        error_type = exc.__class__.__name__
        error_msg = str(exc)
        
        if "insufficient_quota" in error_msg.lower() or "quota" in error_msg.lower():
            return (
                f"I heard: '{user_msg}'. "
                "⚠️ Your OpenAI account has insufficient credits. "
                "Please add billing credits at https://platform.openai.com/account/billing"
            )
        elif "invalid_api_key" in error_msg.lower() or "401" in error_msg:
            return (
                f"I heard: '{user_msg}'. "
                "⚠️ Invalid API key. Please check your OPENAI_API_KEY in .env file."
            )
        else:
            return (
                f"I heard: '{user_msg}'. "
                f"⚠️ API error ({error_type}). Please check your API key and billing status."
            )


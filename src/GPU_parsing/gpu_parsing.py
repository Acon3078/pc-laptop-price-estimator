import re
from typing import Dict, Optional

from CPU_parsing import strip_accents

# Regular Expression Patterns

GPU_NOISE_PATTERNS = [
    r"tarjeta grafica",
    r"graphics card",
    r"graphic card",
    r"gddr\d",
    r"integrada",
    r" dedicada",
    r"video",
    r",",
]

GPU_BRAND_PATTERNS = {
    "nvidia": r"nvidia|geforce|rtx|gtx|mx|tesla|quadro",
    "amd": r"amd|radeon|rx|vega",
    "intel": r"intel|iris|uhd|arc",
    "apple": r"apple|m\d",
}

GPU_SERIES_PATTERNS = [
    (re.compile(r"rtx\s*(\d{3,4})"), lambda m: ("rtx", m.group(1))),
    (re.compile(r"gtx\s*(\d{3,4})"), lambda m: ("gtx", m.group(1))),
    (re.compile(r"mx\s*(\d{3,4})"), lambda m: ("mx", m.group(1))),
    (re.compile(r"rx\s*(\d{3,4})"), lambda m: ("rx", m.group(1))),
    (re.compile(r"arc\s*(a\d{3})"), lambda m: ("arc", m.group(1))),
    (re.compile(r"iris\s*(xe|pro)"), lambda m: ("iris", m.group(1))),
    (re.compile(r"uhd\s*(\d{3})"), lambda m: ("uhd", m.group(1))),
    (re.compile(r"vega\s*(\d{1,2})"), lambda m: ("vega", m.group(1))),
    (re.compile(r"radeon\s*(\w+)"), lambda m: ("radeon", m.group(1))),
]

GPU_SUFFIX_NORMALIZATION = {
    "ti": "ti",
    "super": "super",
    "max q": "max-q",
    "max-q": "max-q",
    "maxq": "max-q",
    "mobile": "mobile",
    "laptop": "mobile",
    "xt": "xt",
    "x": "x",
    "pro": "pro",
}

GPU_MODEL_PATTERN = re.compile(r"\b(\d{3,4})(ti|super|xt|x|m)?\b")


def normalize_gpu_text(raw: Optional[str]) -> str:
    """
    Normalize GPU name text for parsing and matching.

    This function standardizes raw GPU text by removing accents, converting to lowercase,
    stripping noise phrases, and normalizing whitespace and symbols. The result is a
    clean, simplified string suitable for downstream model/series extraction and key matching.

    Args:
        raw (Optional[str]): The raw GPU string (could be None or non-string).

    Returns:
        str: A normalized version of the GPU string suitable for parsing.

    Error-handling:
        Returns an empty string if `raw` is None or not a string. Handles and ignores
        improper input types gracefully without raising exceptions.
    """
    if not isinstance(raw, str):
        return ""
    text = strip_accents(raw)
    text = text.lower()
    text = re.sub(r"\(.*?\)", " ", text)
    text = re.sub(r"[\-/]", " ", text)
    for pattern in GPU_NOISE_PATTERNS:
        text = re.sub(pattern, " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_gpu_brand(norm_text: str) -> Optional[str]:
    """
    Detect the GPU brand from a normalized GPU description string.

    Scans the preprocessed GPU string for brand-defining keywords/patterns and returns
    the corresponding canonical brand name.

    Args:
        norm_text (str): Normalized GPU string (typically output of `normalize_gpu_text`).

    Returns:
        Optional[str]: The detected brand ("nvidia", "amd", "intel", "apple"), or None if no brand matched.

    Error-handling:
        Returns None if no known brand is matched; handles empty strings gracefully.
    """
    for brand, pattern in GPU_BRAND_PATTERNS.items():
        if re.search(pattern, norm_text):
            return brand
    return None


def detect_gpu_series_and_model(norm_text: str) -> Dict[str, Optional[str]]:
    """
    Extract the GPU series and model number from normalized text.

    Attempts to pattern-match a known series (E.g: "rtx", "gtx", "mx", etc.) and model identifier.
    Falls back to extracting a number-only model if no series is detected.

    Args:
        norm_text (str): Normalized input string for a GPU.

    Returns:
        Dict[str, Optional[str]]: Dictionary with keys "series" and "model" (values as strings or None).

    Error-handling:
        If no suitable match is found, both fields are set to None. Handles empty or malformed input safely.
    """
    for pattern, formatter in GPU_SERIES_PATTERNS:
        match = pattern.search(norm_text)
        if match:
            series, model = formatter(match)
            # Clean any lingering words like 'geforce'
            series = series.strip()
            model = model.strip()
            return {"series": series, "model": model}

    model_match = GPU_MODEL_PATTERN.search(norm_text)
    if model_match:
        return {"series": None, "model": model_match.group(1)}

    return {"series": None, "model": None}


def detect_gpu_suffix(norm_text: str) -> Optional[str]:
    """
    Detect a qualifying suffix in the GPU string (E.g: "ti", "super", "max-q", etc.)

    Scans for common GPU suffixes and normalizes them using a lookup dictionary.

    Args:
        norm_text (str): Normalized GPU string.

    Returns:
        Optional[str]: Standardized suffix string (E.g: "ti", "super", "max-q"), or None if not found.

    Error-handling:
        Returns None if no recognized suffix is present; tolerant of malformed/empty input.
    """
    for raw_suffix, normalized in GPU_SUFFIX_NORMALIZATION.items():
        if re.search(rf"\b{re.escape(raw_suffix)}\b", norm_text):
            return normalized
    return None


def build_gpu_key(
    brand: Optional[str],
    series: Optional[str],
    model: Optional[str],
    suffix: Optional[str],
) -> Optional[str]:
    """
    Assemble a compact canonical GPU key from brand, series, model, and suffix.

    Constructs a normalized key for the GPU based on the detected brand, series, model number,
    and suffix. Omits any None fields, and appends the suffix only if it does not duplicate an existing
    word in the key (avoiding redundancy).

    Args:
        brand (Optional[str]): Standardized brand string (E.g: "nvidia").
        series (Optional[str]): Series string (E.g: "rtx", "mx", "radeon"), or None.
        model (Optional[str]): Model number string, or None.
        suffix (Optional[str]): Normalized suffix string (E.g: "ti"), or None.

    Returns:
        Optional[str]: Concatenated GPU canonical key, or None if no suitable components are present.

    Error-handling:
        Returns None if all components are falsy. Ignores and omits empty/None parts cleanly.
    """
    parts = [brand, series, model]
    key = " ".join([p for p in parts if p]).strip()
    if suffix and suffix not in key.split():
        key = f"{key} {suffix}".strip()
    return key or None


def parse_gpu_name(raw: Optional[str]) -> Dict[str, Optional[str]]:
    """
    Parse a raw GPU string into normalized components and a unique key.

    Combines all parsing subroutines to extract the cleaned GPU name, brand, series,
    model number, suffix, and normalized key. Returns a dict suitable for matching
    and lookup. If critical information is missing, sets fields to None and flags for review.

    Args:
        raw (Optional[str]): Raw GPU string to parse. Can be None.

    Returns:
        Dict[str, Optional[str]]: Dictionary containing keys:
          - \"gpu_name_clean\": normalized GPU string or None,
          - \"gpu_brand\": brand name or None,
          - \"gpu_series\": series code or None,
          - \"gpu_model_number\": model code or None,
          - \"gpu_suffix\": normalized suffix or None,
          - \"gpu_normalized_key\": compact canonical GPU key or None,
          - \"gpu_parse_status\": str, \"ok\" if parsed confidently, \"needs_review\" or \"empty\" if not.

    Error-handling:
        All failures gracefully return a dict with None fields and an appropriate status.
        No exceptions are raised due to bad or missing input; can be chained in DataFrame operations.
    """
    norm_text = normalize_gpu_text(raw)
    if not norm_text:
        return {
            "gpu_name_clean": None,
            "gpu_brand": None,
            "gpu_series": None,
            "gpu_model_number": None,
            "gpu_suffix": None,
            "gpu_normalized_key": None,
            "gpu_parse_status": "empty",
        }

    brand = detect_gpu_brand(norm_text)
    series_model = detect_gpu_series_and_model(norm_text)
    suffix = detect_gpu_suffix(norm_text)
    normalized_key = build_gpu_key(brand, series_model["series"], series_model["model"], suffix)

    return {
        "gpu_name_clean": norm_text,
        "gpu_brand": brand,
        "gpu_series": series_model["series"],
        "gpu_model_number": series_model["model"],
        "gpu_suffix": suffix,
        "gpu_normalized_key": normalized_key or norm_text,
        "gpu_parse_status": "ok" if normalized_key else "needs_review",
    }



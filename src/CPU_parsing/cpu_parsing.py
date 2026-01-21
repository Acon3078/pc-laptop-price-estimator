import re
import unicodedata
from typing import Dict, Optional

# Regular Expression Patterns for CPU Name Parsing
# Used to remove noise or irrelevant tokens from raw CPU name strings:
CPU_NOISE_PATTERNS = [
    r"procesador",
    r"processor",
    r"with.*graphics",
    r"grafica integrada",
    r"quad-?core",
    r"hexa-?core",
    r"octa-?core",
    r"dodeca-?core",
    r"\bup to\b",
    r"\bmax\.?\b",
]

# Regular Expression Patterns for CPU Brand Detection
CPU_BRAND_PATTERNS = {
    "intel": r"\bintel\b|core i[3579]|celeron|pentium|xeon|atom",
    "amd": r"\bamd\b|ryzen|athlon|threadripper|epyc",
    "apple": r"\bapple\b|m1|m2|m3|m4",
    "qualcomm": r"\bqualcomm\b|snapdragon",
}

# Regular Expression Patterns for Common Intel CPU Product Families
# Each tuple is (compiled_regex, function_to_format_match_result).
INTEL_FAMILY_PATTERNS = [
    (re.compile(r"core\s*i\s*([3579])"), lambda m: f"core i{m.group(1)}"),
    (re.compile(r"core\s*(duo|solo|m)"), lambda m: f"core {m.group(1)}"),
    (re.compile(r"xeon"), lambda _: "xeon"),
    (re.compile(r"celeron"), lambda _: "celeron"),
    (re.compile(r"pentium"), lambda _: "pentium"),
    (re.compile(r"atom"), lambda _: "atom"),
]

# Regular Expression Patterns for AMD CPU Families
AMD_FAMILY_PATTERNS = [
    (re.compile(r"ryzen\s*(\d)"), lambda m: f"ryzen {m.group(1)}"),
    (re.compile(r"threadripper"), lambda _: "ryzen threadripper"),
    (re.compile(r"athlon"), lambda _: "athlon"),
    (re.compile(r"epyc"), lambda _: "epyc"),
]

# Regular Expression Patterns for Apple CPU Families
APPLE_FAMILY_PATTERNS = [
    (re.compile(r"m(\d)\s*(pro|max|ultra)?"), lambda m: f"m{m.group(1)} {m.group(2)}".strip()),
]

# Regular Expression Patterns for Qualcomm CPU Families
QUALCOMM_FAMILY_PATTERNS = [
    (re.compile(r"snapdragon\s*(\d{3,4})"), lambda m: f"snapdragon {m.group(1)}"),
]

# Regular Expression Pattern for Model Code and Suffix Extraction
MODEL_PATTERN = re.compile(r"\b(\d{3,5})([a-z]{0,2})\b")

# List of Valid CPU Suffixes that Indicate Special Features
SUFFIX_CANDIDATES = {"k", "kf", "ks", "f", "g", "t", "p", "u", "h", "hx", "hk", "hs", "he", "x"}


def strip_accents(text: str) -> str:
    """
    Remove all accents/diacritics from a Unicode string.

    This function applies Unicode normalization ("NFKD") to decompose any accented or
    diacritic-laden characters to their base ASCII equivalents, then encodes the text to
    ASCII (ignoring non-ASCII characters, so all accents and diacritics are stripped),
    and finally decodes the result back to a standard string.

    Args:
        text (str): Input string which may contain unicode accented characters.

    Returns:
        str: The input string, with all accent marks and diacritics removed.

    Error-handling:
        Will raise a TypeError if `text` is not a string. If encoding or decoding fails
        (unexpected with string input), an exception will propagate.
    """
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def normalize_cpu_text(raw: Optional[str]) -> str:
    """
    Normalize a raw CPU name string for text pattern extraction.

    Converts raw CPU strings into a standardized form by:
    - Removing accents
    - Lowercasing
    - Removing trademark symbols, parentheses, noise/marketing words, and special characters
    - Collapsing multiple whitespaces
    - Returning ASCII-only, alphanumeric, space-separated text

    Args:
        raw (Optional[str]): The raw CPU name string to be normalized. May be None, NaN,
            or an invalid value.

    Returns:
        str: Normalized, noise-free CPU string. Returns an empty string if input is
        missing or not a string.

    Error-handling:
        If `raw` is not a string, the function will return an empty string, making this
        robust to null/missing fields.
    """
    if not isinstance(raw, str):
        return ""
    text = strip_accents(raw)
    text = text.lower()
    text = text.replace("®", " ").replace("™", " ")
    text = re.sub(r"\(.*?\)", " ", text)
    text = re.sub(r"[\-/]", " ", text)
    for pattern in CPU_NOISE_PATTERNS:
        text = re.sub(pattern, " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_brand(norm_text: str) -> Optional[str]:
    """
    Identify the CPU brand from a normalized CPU text string.

    Iterates through registered brand-specific regex patterns and returns the matching
    brand (such as "intel", "amd", "apple", "qualcomm", etc.), or None if no match is
    found.

    Args:
        norm_text (str): Normalized CPU name text (should be lowercased, accent-free).

    Returns:
        Optional[str]: Brand identifier as a string if detected, otherwise None.

    Error-handling:
        Will not raise if no brand matches; simply returns None.
    """
    for brand, pattern in CPU_BRAND_PATTERNS.items():
        if re.search(pattern, norm_text):
            return brand
    return None


def detect_family(norm_text: str, brand: Optional[str]) -> Optional[str]:
    """
    Extract the CPU product family from normalized CPU text, optionally using brand to
    refine matching.

    Uses brand-specific regex patterns (if a brand is provided or detected) to search for
    common CPU family names such as "core i7", "ryzen 5", etc. Returns the normalized
    family string if found, or None.

    Args:
        norm_text (str): Normalized CPU string.
        brand (Optional[str]): CPU brand name (e.g., "intel", "amd", "apple",
            "qualcomm", or None if undetected).

    Returns:
        Optional[str]: Detected CPU family string or None if no match.

    Error-handling:
        If the brand is not recognized, all brand family patterns are exhausted; result is
        None if no match.
    """
    if brand == "intel":
        family_patterns = INTEL_FAMILY_PATTERNS
    elif brand == "amd":
        family_patterns = AMD_FAMILY_PATTERNS
    elif brand == "apple":
        family_patterns = APPLE_FAMILY_PATTERNS
    elif brand == "qualcomm":
        family_patterns = QUALCOMM_FAMILY_PATTERNS
    else:
        family_patterns = (
            INTEL_FAMILY_PATTERNS
            + AMD_FAMILY_PATTERNS
            + APPLE_FAMILY_PATTERNS
            + QUALCOMM_FAMILY_PATTERNS
        )

    for pattern, formatter in family_patterns:
        match = pattern.search(norm_text)
        if match:
            return formatter(match)
    return None


def detect_model_and_suffix(norm_text: str) -> Dict[str, Optional[str]]:
    """
    Extract the numeric CPU model code and its optional suffix from normalized CPU text.

    This function searches for both common model number patterns (typically a 4- or
    5-digit number) and trailing 1-2 letter CPU suffixes indicating product variants.
    Extraction heuristics favor 4+ digit codes, but will fall back to the first match
    found.

    Args:
        norm_text (str): Normalized CPU name string (ASCII, alphanumeric, lowercased).

    Returns:
        Dict[str, Optional[str]]: A dictionary of the form {"model_code": str or None,
        "suffix": str or None}; both fields will be None if nothing could be detected.

    Error-handling:
        Never raises; if nothing found, both fields are None.
    """
    model_code = None
    suffix = None

    for match in MODEL_PATTERN.finditer(norm_text):
        candidate = match.group(1)
        letters = match.group(2)
        if len(candidate) >= 4:
            model_code = candidate + letters
            break
        if not model_code:
            model_code = candidate + letters

    if model_code:
        suffix_match = re.search(rf"{model_code}([a-z]{{1,2}})$", norm_text)
        if suffix_match:
            possible = suffix_match.group(1)
            if possible in SUFFIX_CANDIDATES:
                suffix = possible
        else:
            trailing_letters = re.findall(r"\b([a-z]{1,2})\b", norm_text)
            for candidate in trailing_letters[::-1]:
                if candidate in SUFFIX_CANDIDATES:
                    suffix = candidate
                    break

    return {"model_code": model_code, "suffix": suffix}


def build_cpu_key(
    brand: Optional[str],
    family: Optional[str],
    model_code: Optional[str],
    suffix: Optional[str],
) -> Optional[str]:
    """
    Construct a normalized CPU key by concatenating available feature components.

    Produces a whitespace-delimited CPU identity string using the brand, family, model
    code, and, if present, suffix. Intended for easy keying and matching. If all
    components are missing, returns None.

    Args:
        brand (Optional[str]): Brand name (e.g. "intel", "amd", ...), or None if
            undetected.
        family (Optional[str]): CPU family name or None.
        model_code (Optional[str]): Numeric model code, or None.
        suffix (Optional[str]): Product variant suffix, or None.

    Returns:
        Optional[str]: Assembled CPU key string if any field is present; otherwise None.

    Error-handling:
        No exception is raised for missing/empty strings; may return None.
    """
    parts = [brand, family, model_code]
    key = " ".join([p for p in parts if p]).strip()
    if suffix and suffix not in key.split():
        key = f"{key} {suffix}".strip()
    return key or None


def parse_cpu_name(raw: Optional[str]) -> Dict[str, Optional[str]]:
    """
    Parse a raw CPU name string to extract brand, family, model code, suffix, key, and
    parsing status.

    Combines the text normalization, brand/family/model extraction, and CPU key assembly
    processes into a single, robust pipeline for processing arbitrary free-text CPU name
    strings.

    Args:
        raw (Optional[str]): Raw (possibly messy) CPU name string.

    Returns:
        Dict[str, Optional[str]]: Dictionary with fields:
            - "cpu_name_clean": normalized text or None
            - "cpu_brand": detected brand or None
            - "cpu_family": detected family or None
            - "cpu_model_code": numeric model code or None
            - "cpu_suffix": known suffix or None
            - "cpu_normalized_key": unique normalized key, or fallback
            - "cpu_parse_status": flag string: "ok", "empty", or "needs_review"

    Error-handling:
        Always returns a dictionary; never raises on string/None/garbage input.
    """
    norm_text = normalize_cpu_text(raw)
    if not norm_text:
        return {
            "cpu_name_clean": None,
            "cpu_brand": None,
            "cpu_family": None,
            "cpu_model_code": None,
            "cpu_suffix": None,
            "cpu_normalized_key": None,
            "cpu_parse_status": "empty",
        }

    brand = detect_brand(norm_text)
    family = detect_family(norm_text, brand)
    model_info = detect_model_and_suffix(norm_text)
    model_code = model_info["model_code"]
    suffix = model_info["suffix"]
    normalized_key = build_cpu_key(brand, family, model_code, suffix)

    return {
        "cpu_name_clean": norm_text,
        "cpu_brand": brand,
        "cpu_family": family,
        "cpu_model_code": model_code,
        "cpu_suffix": suffix,
        "cpu_normalized_key": normalized_key or norm_text,
        "cpu_parse_status": "ok" if normalized_key else "needs_review",
    }



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
    """
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def normalize_cpu_text(raw: Optional[str]) -> str:
    """
    Normalize a raw CPU name string for text pattern extraction.
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
    """
    for brand, pattern in CPU_BRAND_PATTERNS.items():
        if re.search(pattern, norm_text):
            return brand
    return None


def detect_family(norm_text: str, brand: Optional[str]) -> Optional[str]:
    """
    Extract the CPU product family from normalized CPU text, optionally using brand to refine matching.
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
    """
    parts = [brand, family, model_code]
    key = " ".join([p for p in parts if p]).strip()
    if suffix and suffix not in key.split():
        key = f"{key} {suffix}".strip()
    return key or None


def parse_cpu_name(raw: Optional[str]) -> Dict[str, Optional[str]]:
    """
    Parse a raw CPU name string to extract brand, family, model code, suffix, key, and parsing status.
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



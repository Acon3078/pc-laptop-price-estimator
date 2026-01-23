import re
from typing import Optional

import pandas as pd

from CPU_parsing import strip_accents


def _safe_float(value: Optional[str]) -> Optional[float]:
    """
    Convert a string or numeric value to float, handling variations in decimal/thousand
    separators.

    This helper is designed for European-style numeric strings (e.g., "1.026,53") and
    mixed text such as "18,2 Wh" or "1,24 kg". It standardizes the representation and
    attempts a robust conversion to float.

    Args:
        value (Optional[str]): Raw value that may be a string, int, float, or NaN.

    Returns:
        Optional[float]: Parsed float value if conversion succeeds, otherwise None.

    Error-handling:
        - Returns None if the input is NaN or cannot be parsed as a float.
        - Silently ignores malformed inputs without raising exceptions.
    """
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = strip_accents(str(value)).strip().lower()
    if not text:
        return None

    # Keep only digits, comma, dot, minus sign
    text = re.sub(r"[^0-9,.-]", "", text)

    # European style handling
    if text.count(",") == 1 and text.count(".") == 0:
        # "18,2" -> "18.2"
        text = text.replace(",", ".")
    elif text.count(",") > 1 and "." in text:
        # "1.026,53" -> "1026.53"
        text = text.replace(".", "").replace(",", ".")
    else:
        # Treat commas as thousands separators
        text = text.replace(",", "")

    try:
        return float(text)
    except ValueError:
        return None


def parse_refresh_rate(raw: Optional[str]) -> Optional[float]:
    """
    Parse and convert a refresh rate value (Hz) from a string or numeric.

    This function delegates to `_safe_float` to handle European formats and mixed
    text (e.g., "144 Hz") and returns a numeric refresh rate in Hertz.

    Args:
        raw (Optional[str]): Raw refresh rate value.

    Returns:
        Optional[float]: Parsed refresh rate in Hz, or None if missing/invalid.

    Error-handling:
        Returns None when the input cannot be converted to a float, without raising.
    """
    return _safe_float(raw)


def categorize_resolution(height: Optional[float]) -> Optional[str]:
    """
    Categorize screen height in pixels into a resolution bucket.

    Maps vertical resolution (height in pixels) into coarse categories:
    - "UHD/4K" for height >= 2160
    - "QHD/2K" for 1440 <= height < 2160
    - "FHD" for 1080 <= height < 1440
    - "HD/Lower" for everything below 1080

    Args:
        height (Optional[float]): Vertical resolution in pixels.

    Returns:
        Optional[str]: Resolution category label, or None if height is missing.

    Error-handling:
        Returns None if height is None; does not raise on non-numeric inputs upstream
        as long as they are sanitized before calling this function.
    """
    if height is None:
        return None
    if height >= 2160:
        return "UHD/4K"
    if height >= 1440:
        return "QHD/2K"
    if height >= 1080:
        return "FHD"
    return "HD/Lower"


def parse_price_value(raw: Optional[str]) -> Optional[float]:
    """
    Parse 'Precio_Rango' values into a numeric price (in EUR).

    Handles single prices and ranges such as:
        - "1.026,53 € – 2.287,17 €"
        - "999,00 € – 999,90 €"
        - "847,99 €"

    Assumes European format:
        - '.' used as thousands separator
        - ',' used as decimal separator

    For ranges, returns the midpoint of [min_price, max_price]; for single values,
    returns the parsed value.

    Args:
        raw (Optional[str]): Raw price string or numeric value.

    Returns:
        Optional[float]: Parsed price in the same currency (typically EUR), or None
        if parsing fails.

    Error-handling:
        - Returns None if the input is NaN or no valid numeric component can be parsed.
        - Silently skips malformed segments in ranges; only segments that parse to a
          positive float are used.
    """
    if pd.isna(raw):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)

    text = str(raw)
    # Split on en dash or hyphen to separate min / max prices
    parts = re.split(r"[–-]", text)

    values = []
    for part in parts:
        s = strip_accents(str(part))
        # Keep only digits, dots, commas
        s = re.sub(r"[^0-9,\.]", "", s)
        if not s:
            continue

        # Normalize European style: '.' thousands, ',' decimals
        if "," in s and "." in s:
            # "1.026,53" -> "1026.53"
            s = s.replace(".", "").replace(",", ".")
        elif "," in s:
            # "999,00" -> "999.00"
            s = s.replace(",", ".")
        # else: digits and maybe '.', already fine

        try:
            v = float(s)
            if v > 0:
                values.append(v)
        except ValueError:
            continue

    if not values:
        return None

    return float(sum(values) / len(values))



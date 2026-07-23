"""Date parsing and formatting.

Dates are carried through the pipeline as YYYYMMDD integers.
"""

from __future__ import annotations

import re
from datetime import datetime

_FALLBACK = 0

_STRING_FORMATS = (
    "%d%b%Y",    # 01Jul2021
    "%d-%b-%Y",  # 01-Jul-2021
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%Y-%m",
    "%Y",
)


def parse_date_to_int(date_str) -> int:
    """Convert a date string in any of several GISAID formats to YYYYMMDD."""
    if not date_str or date_str == "Unknown":
        return _FALLBACK

    date_str = str(date_str).strip()

    if date_str.count("-") == 2:
        try:
            return int("".join(date_str.split("-")))
        except ValueError:
            pass

    for fmt in _STRING_FORMATS:
        try:
            return int(datetime.strptime(date_str, fmt).strftime("%Y%m%d"))
        except ValueError:
            continue

    # Last resort: pull a year out of the string and default to Jan 1.
    year_match = re.search(r"(19|20)\d{2}", date_str)
    if year_match:
        return int(f"{year_match.group()}0101")

    return _FALLBACK


def parse_date_int(d) -> datetime:
    """Convert a YYYYMMDD integer to a datetime, clamping out-of-range fields."""
    s = str(int(d)).zfill(8)
    year = max(int(s[:4]), 2000)
    month = min(max(int(s[4:6]), 1), 12)
    day = min(max(int(s[6:]), 1), 31)
    try:
        return datetime(year, month, day)
    except ValueError:
        return datetime(year, month, 1)


def int_dates_to_datetime(dates):
    return [parse_date_int(d) for d in dates]


def format_date_int(date_int, style: str = "MM-YYYY") -> str:
    s = str(int(date_int)).zfill(8)
    if style == "MM-YYYY":
        return f"{s[4:6]}-{s[:4]}"
    if style == "YYYY-MM-DD":
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    raise ValueError(f"Unknown date style: {style}")


def is_valid_date_in_range(date_int, start_date, end_date) -> bool:
    """Reject malformed dates as well as ones outside [start_date, end_date]."""
    try:
        s = str(int(date_int)).zfill(8)
        year, month, day = int(s[:4]), int(s[4:6]), int(s[6:8])
    except (ValueError, TypeError):
        return False

    if not 2000 <= year <= 2030:
        return False
    if not 1 <= month <= 12:
        return False
    if not 1 <= day <= 31:
        return False
    return start_date <= date_int <= end_date

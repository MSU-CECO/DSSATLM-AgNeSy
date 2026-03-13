"""Lightweight utility functions with no external dependencies."""

from datetime import datetime


def yrdoy_to_date(yrdoy) -> datetime:
    """Convert a DSSAT YRDOY integer (e.g. 2023127) to a datetime.

    Args:
        yrdoy: Integer or string in YRDOY format (year * 1000 + day-of-year).

    Returns:
        datetime object.

    Raises:
        ValueError: If yrdoy cannot be parsed.
    """
    yrdoy = int(yrdoy)
    year = yrdoy // 1000
    doy = yrdoy % 1000
    return datetime.strptime(f"{year} {doy}", "%Y %j")
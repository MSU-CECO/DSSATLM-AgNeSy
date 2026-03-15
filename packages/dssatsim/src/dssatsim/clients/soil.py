"""Soil API client.

Fetches the nearest soil profile from the soil-query API and returns
a SoilProfile object ready for use with DSSATTools v3.
"""

import tempfile
from pathlib import Path

import httpx
from DSSATTools.soil import SoilProfile

from dssatsim.config import SOIL_API_BASE_URL


class SoilAPIError(Exception):
    """Raised when the soil-query API returns an unexpected response."""


def fetch_soil_profile(lat: float, lon: float) -> SoilProfile:
    """Fetch the nearest soil profile as a DSSATTools SoilProfile object.

    Calls the soil-query API to get the nearest profile in DSSAT .SOL
    format, writes it to a temporary file, and loads it via
    SoilProfile.from_file().

    Args:
        lat: Latitude of the target location.
        lon: Longitude of the target location.

    Returns:
        A DSSATTools v3 SoilProfile instance.

    Raises:
        SoilAPIError: If the API returns a non-200 response.
    """
    url = f"{SOIL_API_BASE_URL}/soil"
    params = {"lat": lat, "lon": lon, "format": "sol"}

    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, params=params)

    if response.status_code != 200:
        raise SoilAPIError(
            f"Soil API returned {response.status_code}: {response.text}"
        )

    sol_text = response.text

    # Extract the profile name from the first non-comment line (e.g. "*IBMZ910214 ...")
    profile_name = None
    for line in sol_text.splitlines():
        line = line.strip()
        if line.startswith("*") and not line.startswith("*!"):
            profile_name = line[1:].split()[0]
            break

    if profile_name is None:
        raise SoilAPIError("Could not extract profile name from .SOL response.")

    # Write to a named temp file — must persist until SoilProfile.from_file() finishes
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".SOL",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write(sol_text)
        tmp_path = Path(tmp.name)

    try:
        soil_profile = SoilProfile.from_file(profile_name, str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)

    return soil_profile

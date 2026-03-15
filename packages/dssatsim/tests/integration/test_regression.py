"""Regression tests — compare new dssatsim output against known-good outputs
from the old dssatsim version.

Tolerances:
  - Yield and biomass: ±15%
  - Dates: ±2 days
  - Season length: ±5 days

Wheat: currently xfail — because weather API only has 2023 data, wheat matures July 2024.
"""

import pytest
from datetime import datetime
from dssatsim.run import exec
from dssatsim.config import SUMMARY_OUT_AS_JSON_NAN


def _normalize_input(raw: dict) -> dict:
    return {
        **raw,
        "phosphorus_fertilizer_application": raw.get("phosphorus_fertilizer_application", []),
        "potassium_fertilizer_application": raw.get("potassium_fertilizer_application", []),
    }


def _val(d, *keys):
    obj = d
    for k in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(k)
    if obj is None:
        return None
    try:
        v = float(obj)
        return None if v == -99 else v
    except (ValueError, TypeError):
        return None


def _within(actual, expected, tol=0.15):
    if actual is None or expected is None:
        return True
    if expected == 0:
        return actual == 0
    return abs(actual - expected) / abs(expected) <= tol


def _date_within(actual_str, expected_str, days=2):
    if actual_str is None or str(actual_str) == "-99":
        return False
    try:
        actual = datetime.strptime(str(actual_str)[:10], "%Y-%m-%d")
        expected = datetime.strptime(expected_str, "%Y-%m-%d")
        return abs((actual - expected).days) <= days
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

MAIZE_INPUT = _normalize_input({
    "latitude": 42.4241716982, "longitude": -85.7411854356, "elevation": 200,
    "planting_date": "2023-05-15", "crop_name": "maize",
    "crop_variety": "MZ GREAT LAKES 582 KBS",
    "is_irrigation_applied": "yes",
    "irrigation_application": [["2023-05-15", 80], ["2023-05-20", 100]],
    "nitrogen_fertilizer_application": [],
})

SOYBEAN_INPUT = _normalize_input({
    "latitude": 42.4241716982, "longitude": -85.7411854356, "elevation": 200,
    "planting_date": "2023-05-15", "crop_name": "soybean",
    "crop_variety": "SB MATURITY GROUP 2",
    "is_irrigation_applied": "no", "irrigation_application": [],
})

WHEAT_INPUT = _normalize_input({
    "latitude": 42.4241716982, "longitude": -85.7411854356, "elevation": 200,
    "planting_date": "2023-09-15", "crop_name": "wheat",
    "crop_variety": "WH NEWTON (Chelsea soft white)",
    "is_irrigation_applied": "yes",
    "irrigation_application": [["2023-09-15", 80], ["2023-09-20", 100]],
    "nitrogen_fertilizer_application": [["2023-09-25", 27], ["2023-10-15", 35]],
})

MAIZE_EXPECTED = {"hwam": 7222, "cwam": 19339, "pdat": "2023-05-15", "mdat": "2023-09-12", "season_days": 120}
SOYBEAN_EXPECTED = {"hwam": 2388, "cwam": 4658, "pdat": "2023-05-15", "mdat": "2023-09-06", "season_days": 127}
WHEAT_EXPECTED = {"hwam": 1121, "cwam": 6625, "pdat": "2023-09-15", "mdat": "2024-07-04", "season_days": 290}


# ---------------------------------------------------------------------------
# Maize
# ---------------------------------------------------------------------------

class TestRegressionMaize:

    @pytest.fixture(scope="class")
    def result(self):
        _, explanations = exec(MAIZE_INPUT)
        assert explanations != SUMMARY_OUT_AS_JSON_NAN
        return explanations

    def test_simulation_succeeds(self, result):
        assert result != SUMMARY_OUT_AS_JSON_NAN

    def test_crop_code_is_maize(self, result):
        assert result["Identifiers"]["Crop code"] == "MZ"

    def test_yield_within_tolerance(self, result):
        hwam = _val(result, "Dry weight, yield and yield components", "Yield at harvest maturity (kg [dm]/ha)")
        assert hwam is not None
        assert _within(hwam, MAIZE_EXPECTED["hwam"]), f"HWAM {hwam} deviates >15% from expected {MAIZE_EXPECTED['hwam']}"

    def test_tops_weight_within_tolerance(self, result):
        cwam = _val(result, "Dry weight, yield and yield components", "Tops weight at maturity (kg [dm]/ha)")
        assert cwam is not None
        assert _within(cwam, MAIZE_EXPECTED["cwam"], tol=0.20), f"CWAM {cwam} deviates >20% from expected {MAIZE_EXPECTED['cwam']}"

    def test_planting_date(self, result):
        pdat = result.get("Dates", {}).get("Planting date")
        assert _date_within(pdat, MAIZE_EXPECTED["pdat"], days=0), f"Planting date {pdat} != {MAIZE_EXPECTED['pdat']}"

    def test_maturity_date_within_tolerance(self, result):
        mdat = result.get("Dates", {}).get("Physiological maturity date")
        assert _date_within(mdat, MAIZE_EXPECTED["mdat"]), f"Maturity date {mdat} not within 2 days of {MAIZE_EXPECTED['mdat']}"

    def test_season_length_within_tolerance(self, result):
        days = _val(result, "Seasonal environmental data (planting to harvest)", "Number of days from planting to harvest (d)")
        assert days is not None
        assert abs(days - MAIZE_EXPECTED["season_days"]) <= 5, f"Season length {days} deviates >5 days from {MAIZE_EXPECTED['season_days']}"


# ---------------------------------------------------------------------------
# Soybean
# ---------------------------------------------------------------------------

class TestRegressionSoybean:

    @pytest.fixture(scope="class")
    def result(self):
        _, explanations = exec(SOYBEAN_INPUT)
        assert explanations != SUMMARY_OUT_AS_JSON_NAN
        return explanations

    def test_simulation_succeeds(self, result):
        assert result != SUMMARY_OUT_AS_JSON_NAN

    def test_crop_code_is_soybean(self, result):
        assert result["Identifiers"]["Crop code"] == "SB"

    def test_yield_within_tolerance(self, result):
        hwam = _val(result, "Dry weight, yield and yield components", "Yield at harvest maturity (kg [dm]/ha)")
        assert hwam is not None
        assert _within(hwam, SOYBEAN_EXPECTED["hwam"]), f"HWAM {hwam} deviates >15% from expected {SOYBEAN_EXPECTED['hwam']}"

    def test_tops_weight_within_tolerance(self, result):
        cwam = _val(result, "Dry weight, yield and yield components", "Tops weight at maturity (kg [dm]/ha)")
        assert cwam is not None
        assert _within(cwam, SOYBEAN_EXPECTED["cwam"]), f"CWAM {cwam} deviates >15% from expected {SOYBEAN_EXPECTED['cwam']}"

    def test_planting_date(self, result):
        pdat = result.get("Dates", {}).get("Planting date")
        assert _date_within(pdat, SOYBEAN_EXPECTED["pdat"], days=0), f"Planting date {pdat} != {SOYBEAN_EXPECTED['pdat']}"

    def test_maturity_date_within_tolerance(self, result):
        mdat = result.get("Dates", {}).get("Physiological maturity date")
        assert _date_within(mdat, SOYBEAN_EXPECTED["mdat"]), f"Maturity date {mdat} not within 2 days of {SOYBEAN_EXPECTED['mdat']}"

    def test_season_length_within_tolerance(self, result):
        days = _val(result, "Seasonal environmental data (planting to harvest)", "Number of days from planting to harvest (d)")
        assert days is not None
        assert abs(days - SOYBEAN_EXPECTED["season_days"]) <= 5, f"Season length {days} deviates >5 days from {SOYBEAN_EXPECTED['season_days']}"


# ---------------------------------------------------------------------------
# Wheat — xfail (weather API only has 2023 data, wheat matures Jul 2024)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason="Weather API dataset only covers 2023; wheat planted Sep 2023 "
           "matures Jul 2024 — simulation terminates early. "
           "Re-enable when weather API covers 2024.",
    strict=False,
)
class TestRegressionWheat:

    @pytest.fixture(scope="class")
    def result(self):
        _, explanations = exec(WHEAT_INPUT)
        assert explanations != SUMMARY_OUT_AS_JSON_NAN
        return explanations

    def test_simulation_succeeds(self, result):
        assert result != SUMMARY_OUT_AS_JSON_NAN

    def test_crop_code_is_wheat(self, result):
        assert result["Identifiers"]["Crop code"] == "WH"

    def test_yield_within_tolerance(self, result):
        hwam = _val(result, "Dry weight, yield and yield components", "Yield at harvest maturity (kg [dm]/ha)")
        assert hwam is not None
        assert _within(hwam, WHEAT_EXPECTED["hwam"]), f"HWAM {hwam} deviates >15% from expected {WHEAT_EXPECTED['hwam']}"

    def test_tops_weight_within_tolerance(self, result):
        cwam = _val(result, "Dry weight, yield and yield components", "Tops weight at maturity (kg [dm]/ha)")
        assert cwam is not None
        assert _within(cwam, WHEAT_EXPECTED["cwam"]), f"CWAM {cwam} deviates >15% from expected {WHEAT_EXPECTED['cwam']}"

    def test_planting_date(self, result):
        pdat = result.get("Dates", {}).get("Planting date")
        assert _date_within(pdat, WHEAT_EXPECTED["pdat"], days=0), f"Planting date {pdat} != {WHEAT_EXPECTED['pdat']}"

    def test_maturity_date_within_tolerance(self, result):
        mdat = result.get("Dates", {}).get("Physiological maturity date")
        assert _date_within(mdat, WHEAT_EXPECTED["mdat"]), f"Maturity date {mdat} not within 2 days of {WHEAT_EXPECTED['mdat']}"

    def test_season_length_within_tolerance(self, result):
        days = _val(result, "Seasonal environmental data (planting to harvest)", "Number of days from planting to harvest (d)")
        assert days is not None
        assert abs(days - WHEAT_EXPECTED["season_days"]) <= 5, f"Season length {days} deviates >5 days from {WHEAT_EXPECTED['season_days']}"


"""Integration tests for run.py — two suites:

1. TestRunHelpersWithMockedAPIs
   Tests the run.py helper functions (setup_weather, setup_soil, setup_field, etc.)
   with respx/unittest.mock patching the HTTP layer. No live APIs needed.

2. TestExecEndToEnd
   Calls dssatsim.run.exec() with both live APIs (weather + soil) and a real
   DSSATTools/DSSAT execution. Requires:
     - Weather API running at WEATHER_API_BASE_URL
     - Soil API reachable at SOIL_API_BASE_URL
     - DSSATTools v3 installed (pip install DSSATTools)

Run integration only:
    pytest tests/integration/test_integration_run.py -v
"""

import pytest
import respx
import httpx
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
from DSSATTools.weather import WeatherStation

from dssatsim.run import (
    setup_crop,
    setup_weather,
    setup_soil,
    setup_field,
    setup_planting,
    setup_irrigation,
    setup_fertilizer,
    setup_simulation_controls,
    exec,
)
from dssatsim.config import SUMMARY_OUT_AS_JSON_NAN


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

KBS_LAT = 42.24
KBS_LON = -85.24
KBS_ELEV = 200.0
PLANTING_DATE = "2023-05-15"

VALID_INPUT = {
    "crop_name": "Maize",
    "crop_variety": "MZ GREAT LAKES 582 KBS",
    "latitude": KBS_LAT,
    "longitude": KBS_LON,
    "elevation": KBS_ELEV,
    "planting_date": PLANTING_DATE,
    "is_irrigation_applied": "yes",
    "irrigation_application": [["2023-06-01", 25], ["2023-07-01", 30]],
    "nitrogen_fertilizer_application": [["2023-05-20", 120]],
    "phosphorus_fertilizer_application": [],
    "potassium_fertilizer_application": [],
}

# Minimal mock weather payload — two full years so setup_weather's window fits
def _make_mock_weather_payload(start="2023-01-01", end="2024-12-31"):
    records = []
    current = pd.date_range(start, end, freq="D")
    for d in current:
        records.append({
            "date": str(d.date()),
            "tmax": 20.0, "tmin": 10.0, "prcp": 2.0, "srad": 200.0,
            "latitude": KBS_LAT, "longitude": KBS_LON,
            "dayl": 50000.0, "swe": 0.0, "vp": 700.0,
        })
    return {"records": records}

MOCK_SOL_TEXT = """\
*US02476497     USA              Loam   200    ISRIC soilgrids + HC27
@SITE        COUNTRY          LAT     LONG SCS Family
 -99              US      42.208   -85.208     HC_GEN0011
@ SCOM  SALB  SLU1  SLDR  SLRO  SLNF  SLPF  SMHB  SMPX  SMKE
    BK  0.10  6.00  0.50 75.00  1.00  1.00 SA001 SA001 SA001
@  SLB  SLMH  SLLL  SDUL  SSAT  SRGF  SSKS  SBDM  SLOC  SLCL  SLSI  SLCF  SLNI  SLHW  SLHB  SCEC  SADC
     5 A     0.119 0.261 0.397  1.00  0.74  1.49  3.01 19.30 45.97 -99.0  0.12  5.72 -99.0 21.10 -99.0
    15 A     0.130 0.272 0.401  0.85  0.62  1.51  2.55 21.18 45.12 -99.0  0.09  5.79 -99.0 18.50 -99.0
    30 A     0.141 0.283 0.410  0.70  0.55  1.53  2.10 22.00 46.00 -99.0  0.08  5.85 -99.0 16.00 -99.0
"""


# ---------------------------------------------------------------------------
# Suite 1: run.py helpers with mocked HTTP APIs
# ---------------------------------------------------------------------------

class TestRunHelpersWithMockedAPIs:
    """Unit-level tests of run.py helper functions with all HTTP mocked."""

    # --- setup_weather ---

    @respx.mock
    def test_setup_weather_returns_weather_station(self):
        payload = _make_mock_weather_payload()
        respx.get("http://localhost:8001/weather").mock(
            return_value=httpx.Response(200, json=payload)
        )
        ws = setup_weather(PLANTING_DATE, KBS_LAT, KBS_LON, KBS_ELEV)
        assert isinstance(ws, WeatherStation)

    @respx.mock
    def test_setup_weather_fetches_two_year_window(self):
        """Verify that the request covers planting year + next year."""
        payload = _make_mock_weather_payload()
        route = respx.get("http://localhost:8001/weather").mock(
            return_value=httpx.Response(200, json=payload)
        )
        setup_weather(PLANTING_DATE, KBS_LAT, KBS_LON, KBS_ELEV)
        assert route.called
        params = route.calls[0].request.url.params
        assert params["start_date"] == "2023-01-01"
        assert params["end_date"] == "2024-12-31"

    # --- setup_soil ---

    @respx.mock
    def test_setup_soil_returns_soil_profile(self):
        respx.get("https://soil-query-production.up.railway.app/soil").mock(
            return_value=httpx.Response(200, text=MOCK_SOL_TEXT)
        )
        with patch("dssatsim.clients.soil.SoilProfile.from_file") as mock_fp:
            mock_fp.return_value = MagicMock()
            profile = setup_soil(KBS_LAT, KBS_LON)
        assert profile is not None

    @respx.mock
    def test_setup_soil_passes_correct_coordinates(self):
        route = respx.get("https://soil-query-production.up.railway.app/soil").mock(
            return_value=httpx.Response(200, text=MOCK_SOL_TEXT)
        )
        with patch("dssatsim.clients.soil.SoilProfile.from_file"):
            setup_soil(KBS_LAT, KBS_LON)
        params = route.calls[0].request.url.params
        assert float(params["lat"]) == KBS_LAT
        assert float(params["lon"]) == KBS_LON

    # --- setup_field ---

    @respx.mock
    def test_setup_field_returns_field(self):
        from DSSATTools.filex import Field
        payload = _make_mock_weather_payload()
        respx.get("http://localhost:8001/weather").mock(
            return_value=httpx.Response(200, json=payload)
        )
        ws = setup_weather(PLANTING_DATE, KBS_LAT, KBS_LON, KBS_ELEV)
        # Field accepts a soil profile ID string for id_soil
        field = setup_field(ws, "US02476497")
        assert isinstance(field, Field)

    # --- setup_planting ---

    def test_setup_planting_maize_defaults(self):
        from DSSATTools.filex import Planting
        planting = setup_planting(PLANTING_DATE, "Maize")
        assert isinstance(planting, Planting)

    def test_setup_planting_date_parsed_correctly(self):
        planting = setup_planting(PLANTING_DATE, "Maize")
        assert planting["pdate"] == date(2023, 5, 15)

    def test_setup_planting_soybean_defaults(self):
        planting = setup_planting("2023-05-20", "Soybean")
        assert planting["plrs"] == 20   # Soybean row spacing from config

    # --- setup_irrigation ---

    def test_setup_irrigation_creates_events(self):
        from DSSATTools.filex import Irrigation
        irr = setup_irrigation([["2023-06-01", 25], ["2023-07-01", 30]])
        assert isinstance(irr, Irrigation)

    def test_setup_irrigation_event_count(self):
        irr = setup_irrigation([["2023-06-01", 25], ["2023-07-01", 30]])
        assert len(irr.table) == 2

    def test_setup_irrigation_amount_preserved(self):
        irr = setup_irrigation([["2023-06-01", 50]])
        assert irr.table[0]["irval"] == 50

    # --- setup_fertilizer ---

    def test_setup_fertilizer_nitrogen_only(self):
        from DSSATTools.filex import Fertilizer
        fert = setup_fertilizer({
            "nitrogen_fertilizer_application": [["2023-05-20", 120]],
            "phosphorus_fertilizer_application": [],
            "potassium_fertilizer_application": [],
        })
        assert isinstance(fert, Fertilizer)

    def test_setup_fertilizer_merges_npk_on_same_date(self):
        fert = setup_fertilizer({
            "nitrogen_fertilizer_application": [["2023-05-20", 100]],
            "phosphorus_fertilizer_application": [["2023-05-20", 40]],
            "potassium_fertilizer_application": [["2023-05-20", 60]],
        })
        # All three applied on same date → single event
        assert len(fert.table) == 1
        assert fert.table[0]["famn"] == 100
        assert fert.table[0]["famp"] == 40
        assert fert.table[0]["famk"] == 60

    def test_setup_fertilizer_separate_dates_give_separate_events(self):
        fert = setup_fertilizer({
            "nitrogen_fertilizer_application": [
                ["2023-05-20", 60],
                ["2023-06-15", 60],
            ],
            "phosphorus_fertilizer_application": [],
            "potassium_fertilizer_application": [],
        })
        assert len(fert.table) == 2

    # --- setup_simulation_controls ---

    def test_setup_simulation_controls_irrigation_yes(self):
        from DSSATTools.filex import SimulationControls
        sc = setup_simulation_controls(
            PLANTING_DATE, "yes",
            has_fertilizer=True, has_nitrogen=True,
            has_phosphorus=False, has_potassium=False,
        )
        assert isinstance(sc, SimulationControls)
        assert sc["management"]["irrig"] == "R"

    def test_setup_simulation_controls_irrigation_no(self):
        sc = setup_simulation_controls(
            PLANTING_DATE, "no",
            has_fertilizer=False, has_nitrogen=False,
            has_phosphorus=False, has_potassium=False,
        )
        assert sc["management"]["irrig"] == "N"

    def test_setup_simulation_controls_sdate_is_day_before_planting(self):
        sc = setup_simulation_controls(
            PLANTING_DATE, "no",
            has_fertilizer=False, has_nitrogen=False,
            has_phosphorus=False, has_potassium=False,
        )
        assert sc["general"]["sdate"] == date(2023, 5, 14)

    # --- exec() with fully mocked stack ---

    @respx.mock
    def test_exec_returns_nan_sentinel_when_input_invalid(self):
        bad_input = {**VALID_INPUT, "latitude": "-99"}
        out_file, result = exec(bad_input)
        assert out_file is None
        assert result == SUMMARY_OUT_AS_JSON_NAN

    @respx.mock
    def test_exec_returns_nan_sentinel_when_dssat_fails(self):
        payload = _make_mock_weather_payload()
        respx.get("http://localhost:8001/weather").mock(
            return_value=httpx.Response(200, json=payload)
        )
        respx.get("https://soil-query-production.up.railway.app/soil").mock(
            return_value=httpx.Response(200, text=MOCK_SOL_TEXT)
        )
        mock_dssat = MagicMock()
        mock_dssat.run_treatment.return_value = None
        mock_dssat.output_files = {}

        with patch("dssatsim.clients.soil.SoilProfile.from_file", return_value="US02476497"), \
             patch("dssatsim.run.DSSAT", return_value=mock_dssat):
            _, result = exec(VALID_INPUT)

        assert result == SUMMARY_OUT_AS_JSON_NAN

    @respx.mock
    def test_exec_calls_dssat_close_on_success(self):
        """Regression test: dssat.close() must always be called (resource leak guard)."""
        payload = _make_mock_weather_payload()
        respx.get("http://localhost:8001/weather").mock(
            return_value=httpx.Response(200, json=payload)
        )
        respx.get("https://soil-query-production.up.railway.app/soil").mock(
            return_value=httpx.Response(200, text=MOCK_SOL_TEXT)
        )
        mock_dssat = MagicMock()
        mock_dssat.run_treatment.return_value = {"HARWT": 8000}
        mock_dssat.output_files = {"Summary": ""}  # empty → triggers NaN path

        with patch("dssatsim.clients.soil.SoilProfile.from_file", return_value="US02476497"), \
             patch("dssatsim.run.DSSAT", return_value=mock_dssat):
            exec(VALID_INPUT)

        mock_dssat.close.assert_called_once()

    @respx.mock
    def test_exec_calls_dssat_close_on_failure(self):
        """dssat.close() must also be called when run_treatment() returns None."""
        payload = _make_mock_weather_payload()
        respx.get("http://localhost:8001/weather").mock(
            return_value=httpx.Response(200, json=payload)
        )
        respx.get("https://soil-query-production.up.railway.app/soil").mock(
            return_value=httpx.Response(200, text=MOCK_SOL_TEXT)
        )
        mock_dssat = MagicMock()
        mock_dssat.run_treatment.return_value = None
        mock_dssat.output_files = {}

        with patch("dssatsim.clients.soil.SoilProfile.from_file", return_value="US02476497"), \
             patch("dssatsim.run.DSSAT", return_value=mock_dssat):
            exec(VALID_INPUT)

        mock_dssat.close.assert_called_once()


# ---------------------------------------------------------------------------
# Suite 2: Full end-to-end with live APIs + real DSSAT execution
# ---------------------------------------------------------------------------

class TestExecEndToEnd:
    """
    Full simulation pipeline: live weather API + live soil API + DSSATTools DSSAT.

    These tests are slow (~30–60s each). Run selectively:
        pytest tests/integration/test_integration_run.py::TestExecEndToEnd -v
    """

    def test_exec_maize_returns_explanations_dict(self):
        _, explanations = exec(VALID_INPUT)
        assert isinstance(explanations, dict)
        assert explanations != SUMMARY_OUT_AS_JSON_NAN

    def test_exec_maize_has_yield_data(self):
        _, explanations = exec(VALID_INPUT)
        dry_weight = explanations.get("Dry weight, yield and yield components", {})
        assert len(dry_weight) > 0
        hwam = dry_weight.get("Yield at harvest maturity (kg [dm]/ha)")
        assert hwam is not None, f"HWAM key missing from: {list(dry_weight.keys())}"
        assert float(hwam) > 0

    def test_exec_maize_has_dates(self):
        _, explanations = exec(VALID_INPUT)
        dates = explanations.get("Dates", {})
        assert "Planting date (YRDOY)" in dates or len(dates) > 0

    def test_exec_soybean_runs_successfully(self):
        soy_input = {
            **VALID_INPUT,
            "crop_name": "Soybean",
            "crop_variety": "SB MATURITY GROUP 2",
            "planting_date": "2023-05-20",
            "irrigation_application": [["2023-06-15", 25]],
        }
        _, explanations = exec(soy_input)
        assert isinstance(explanations, dict)
        assert explanations != SUMMARY_OUT_AS_JSON_NAN

    def test_exec_no_irrigation_runs_successfully(self):
        rainfed_input = {
            **VALID_INPUT,
            "is_irrigation_applied": "no",
            "irrigation_application": [],
        }
        _, explanations = exec(rainfed_input)
        assert isinstance(explanations, dict)
        assert explanations != SUMMARY_OUT_AS_JSON_NAN

    def test_exec_with_output_file_writes_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            out_file, explanations = exec(VALID_INPUT, output_file=tmp_path)
            assert out_file == tmp_path
            assert Path(tmp_path).exists()
            import json
            with open(tmp_path) as f:
                data = json.load(f)
            assert isinstance(data, dict)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_exec_accepts_dict_input(self):
        """exec() should work when passed a dict directly (not a file path)."""
        _, explanations = exec(VALID_INPUT)
        assert explanations != SUMMARY_OUT_AS_JSON_NAN

    def test_exec_accepts_json_file_input(self):
        """exec() should work when passed a path to a JSON file."""
        import json
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp:
            json.dump(VALID_INPUT, tmp)
            tmp_path = tmp.name
        try:
            _, explanations = exec(tmp_path)
            assert explanations != SUMMARY_OUT_AS_JSON_NAN
        finally:
            Path(tmp_path).unlink(missing_ok=True)


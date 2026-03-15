"""Core simulation runner.

Orchestrates weather/soil API clients, DSSATTools v3 objects, and DSSAT execution.
"""

import json
import os
from datetime import date, datetime, timedelta
from itertools import chain

import pandas as pd

from DSSATTools.crop import Maize, Soybean, Wheat
from DSSATTools.weather import WeatherStation
from DSSATTools.run import DSSAT
from DSSATTools.filex import (
    Field,
    Planting,
    Fertilizer,
    FertilizerEvent,
    Irrigation,
    IrrigationEvent,
    SimulationControls,
    SCGeneral,
    SCManagement,
    SCOptions,
)

from dssatsim.clients.weather import fetch_weather
from dssatsim.clients.soil import fetch_soil_profile
from dssatsim.config import (
    INSTI_CODE,
    SUMMARY_OUT_AS_JSON_NAN,
    MINIMUM_REQUIRED_FARMER_INPUTS,
    ASSUMPTIONS,
    CROP_NAMES_TO_CROP_VARIETIES,
    CROP_VARIETIES_TO_CULTIVAR_CODES,
    PLANTING_DEFAULTS,
)

# ---------------------------------------------------------------------------
# Crop class registry
# ---------------------------------------------------------------------------
CROP_CLASS_MAP = {
    "Maize": Maize,
    "Soybean": Soybean,
    "Wheat": Wheat,
}


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def is_simulation_possible(input_data: dict) -> bool:
    """Return False if any required input is missing or set to the NA sentinel."""
    for field in MINIMUM_REQUIRED_FARMER_INPUTS:
        val = input_data.get(field)
        if val is None:
            return False
        if isinstance(val, str) and val == "-99":
            return False
        if isinstance(val, int) and val == -99:
            return False
        if isinstance(val, list):
            is_irr = input_data.get("is_irrigation_applied", "")
            if is_irr.lower() == "yes" and -99 in list(chain.from_iterable(val)):
                return False
    return True


# ---------------------------------------------------------------------------
# Crop
# ---------------------------------------------------------------------------

def setup_crop(crop_name: str, crop_variety: str):
    """Return a configured DSSATTools v3 crop object.

    Args:
        crop_name: e.g. "Maize"
        crop_variety: e.g. "MZ GREAT LAKES 582 KBS"

    Returns:
        A DSSATTools crop instance with cultivar coefficients applied.
    """
    crop_name = crop_name.title()

    if crop_name not in CROP_NAMES_TO_CROP_VARIETIES:
        raise ValueError(f"Crop '{crop_name}' is not supported.")
    if crop_variety not in CROP_VARIETIES_TO_CULTIVAR_CODES:
        raise ValueError(f"Crop variety '{crop_variety}' is not supported.")

    props = CROP_VARIETIES_TO_CULTIVAR_CODES[crop_variety]
    cultivar_code = props["@var#"]

    crop_class = CROP_CLASS_MAP[crop_name]
    crop = crop_class(cultivar_code)

    for key, val in props.items():
        if key in ("@var#", "eco#"):
            continue
        crop[key] = val

    return crop


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

def setup_weather(
    planting_date: str,
    lat: float,
    lon: float,
    elev: float,
) -> WeatherStation:
    """Fetch weather from the Weather API and return a WeatherStation.

    Fetches a two-year window (planting year + next year) to support
    winter crops that cross the year boundary.

    Args:
        planting_date: ISO date string (YYYY-MM-DD).
        lat: Latitude.
        lon: Longitude.
        elev: Elevation in metres.

    Returns:
        A DSSATTools v3 WeatherStation instance.
    """
    year = int(planting_date.split("-")[0])
    start_date = f"{year}-01-01"
    end_date = f"{year + 1}-12-31"

    df = fetch_weather(lat=lat, lon=lon, start_date=start_date, end_date=end_date)

    weather_station = WeatherStation(
        insi=INSTI_CODE,
        lat=lat,
        long=lon,
        elev=elev,
        table=df,
    )

    return weather_station


# ---------------------------------------------------------------------------
# Soil
# ---------------------------------------------------------------------------

def setup_soil(lat: float, lon: float):
    """Fetch the nearest soil profile from the soil-query API.

    Args:
        lat: Latitude.
        lon: Longitude.

    Returns:
        A DSSATTools v3 SoilProfile instance.
    """
    return fetch_soil_profile(lat=lat, lon=lon)


# ---------------------------------------------------------------------------
# Irrigation
# ---------------------------------------------------------------------------

def setup_irrigation(irr_apps_list: list) -> Irrigation:
    """Build a DSSATTools v3 Irrigation object from the input schedule.

    Args:
        irr_apps_list: List of [date_str, amount_mm] pairs.

    Returns:
        A DSSATTools v3 Irrigation instance.
    """
    events = []
    for date_str, amount in irr_apps_list:
        events.append(
            IrrigationEvent(
                idate=datetime.strptime(date_str, "%Y-%m-%d").date(),
                irval=amount,
                irop=ASSUMPTIONS["irrigation_operation_code"],
            )
        )
    return Irrigation(table=events)


# ---------------------------------------------------------------------------
# Fertilizer
# ---------------------------------------------------------------------------

def _was_fertilizer_applied(fert_apps: dict) -> bool:
    return any(len(v) > 0 for v in fert_apps.values())


def setup_fertilizer(fert_apps: dict) -> Fertilizer:
    """Build a DSSATTools v3 Fertilizer object from N/P/K application lists.

    Args:
        fert_apps: Dict with keys:
            - nitrogen_fertilizer_application: [[date_str, amount], ...]
            - phosphorus_fertilizer_application: [[date_str, amount], ...]
            - potassium_fertilizer_application: [[date_str, amount], ...]

    Returns:
        A DSSATTools v3 Fertilizer instance.
    """
    records: dict[str, dict] = {}

    for date_str, famn in fert_apps.get("nitrogen_fertilizer_application", []):
        records.setdefault(date_str, {"famn": 0, "famp": 0, "famk": 0})
        records[date_str]["famn"] += famn

    for date_str, famp in fert_apps.get("phosphorus_fertilizer_application", []):
        records.setdefault(date_str, {"famn": 0, "famp": 0, "famk": 0})
        records[date_str]["famp"] += famp

    for date_str, famk in fert_apps.get("potassium_fertilizer_application", []):
        records.setdefault(date_str, {"famn": 0, "famp": 0, "famk": 0})
        records[date_str]["famk"] += famk

    events = []
    for date_str, amounts in sorted(records.items()):
        events.append(
            FertilizerEvent(
                fdate=datetime.strptime(date_str, "%Y-%m-%d").date(),
                fmcd=ASSUMPTIONS["fertilizer_material_code"],
                facd=ASSUMPTIONS["fertilizer_application_code"],
                fdep=ASSUMPTIONS["fertilizer_depth"],
                famn=amounts["famn"],
                famp=amounts["famp"],
                famk=amounts["famk"],
            )
        )

    return Fertilizer(table=events)


# ---------------------------------------------------------------------------
# Field
# ---------------------------------------------------------------------------

def setup_field(weather_station: WeatherStation, soil_profile) -> Field:
    """Build a DSSATTools v3 Field object.

    Args:
        weather_station: WeatherStation instance.
        soil_profile: SoilProfile instance.

    Returns:
        A DSSATTools v3 Field instance.
    """
    return Field(
        id_field=f"{INSTI_CODE}0001",
        wsta=weather_station,
        flob=0,
        fldd=0,
        flds=0,
        fldt="DR000",
        id_soil=soil_profile,
    )


# ---------------------------------------------------------------------------
# Planting
# ---------------------------------------------------------------------------

def setup_planting(planting_date: str, crop_name: str = "Maize") -> Planting:
    """Build a DSSATTools v3 Planting object.

    Args:
        planting_date: ISO date string (YYYY-MM-DD).
        crop_name: Crop name used to look up planting defaults from config.

    Returns:
        A DSSATTools v3 Planting instance.
    """
    pdate = datetime.strptime(planting_date, "%Y-%m-%d").date()
    defaults = PLANTING_DEFAULTS.get(crop_name.title(), PLANTING_DEFAULTS["Maize"])
    return Planting(
        pdate=pdate,
        ppop=defaults["ppop"],
        ppoe=defaults["ppoe"],
        plme=defaults["plme"],
        plds=defaults["plds"],
        plrs=defaults["plrs"],
        plrd=defaults["plrd"],
        pldp=defaults["pldp"],
    )


# ---------------------------------------------------------------------------
# Simulation controls
# ---------------------------------------------------------------------------

def setup_simulation_controls(
    planting_date: str,
    is_irrigation_applied: str,
    has_fertilizer: bool,
    has_nitrogen: bool,
    has_phosphorus: bool,
    has_potassium: bool,
) -> SimulationControls:
    """Build a DSSATTools v3 SimulationControls object.

    Args:
        planting_date: ISO date string (YYYY-MM-DD).
        is_irrigation_applied: "yes" or "no".
        has_fertilizer: Whether any fertilizer was applied.
        has_nitrogen: Whether nitrogen was applied.
        has_phosphorus: Whether phosphorus was applied.
        has_potassium: Whether potassium was applied.

    Returns:
        A DSSATTools v3 SimulationControls instance.
    """
    pdate = datetime.strptime(planting_date, "%Y-%m-%d").date()
    sdate = pdate - timedelta(days=1)

    irrig = "R" if is_irrigation_applied.lower() == "yes" else "N"
    ferti = "R" if has_fertilizer else "N"
    nitro = "Y" if has_nitrogen else "N"
    phosp = "Y" if has_phosphorus else "N"
    potas = "Y" if has_potassium else "N"

    return SimulationControls(
        general=SCGeneral(sdate=sdate),
        options=SCOptions(nitro=nitro, phosp=phosp, potas=potas),
        management=SCManagement(
            irrig=irrig,
            ferti=ferti,
            harvs=ASSUMPTIONS["harvest_management_option"],
        ),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def exec(input_file, output_file=None) -> tuple:
    """Run a DSSAT simulation from a JSON input file or dict.

    Args:
        input_file: Path to a JSON file, or a dict with input data.
        output_file: Optional path to write the Summary.OUT JSON output.

    Returns:
        Tuple of (output_file, explanations_dict).
        Returns (None, SUMMARY_OUT_AS_JSON_NAN) if simulation cannot run.
    """
    from dssatsim.outputs import explain_summary_out

    if isinstance(input_file, dict):
        input_data = input_file
    else:
        with open(os.path.abspath(input_file), "r", encoding="utf-8") as f:
            input_data = json.load(f)

    if not is_simulation_possible(input_data):
        return None, SUMMARY_OUT_AS_JSON_NAN

    # Fertilizer inputs
    fert_apps = {
        "nitrogen_fertilizer_application": input_data.get("nitrogen_fertilizer_application", []),
        "phosphorus_fertilizer_application": input_data.get("phosphorus_fertilizer_application", []),
        "potassium_fertilizer_application": input_data.get("potassium_fertilizer_application", []),
    }
    has_fertilizer = _was_fertilizer_applied(fert_apps)
    has_nitrogen = len(fert_apps["nitrogen_fertilizer_application"]) > 0
    has_phosphorus = len(fert_apps["phosphorus_fertilizer_application"]) > 0
    has_potassium = len(fert_apps["potassium_fertilizer_application"]) > 0

    # 1. Crop
    crop = setup_crop(input_data["crop_name"], input_data["crop_variety"])

    # 2. Weather
    weather_station = setup_weather(
        planting_date=input_data["planting_date"],
        lat=input_data["latitude"],
        lon=input_data["longitude"],
        elev=input_data["elevation"],
    )

    # 3. Soil
    soil_profile = setup_soil(input_data["latitude"], input_data["longitude"])

    # 4. Field
    field = setup_field(weather_station, soil_profile)

    # 5. Planting
    planting = setup_planting(input_data["planting_date"], input_data["crop_name"])

    # 6. Simulation controls
    simulation_controls = setup_simulation_controls(
        planting_date=input_data["planting_date"],
        is_irrigation_applied=input_data["is_irrigation_applied"],
        has_fertilizer=has_fertilizer,
        has_nitrogen=has_nitrogen,
        has_phosphorus=has_phosphorus,
        has_potassium=has_potassium,
    )

    # 7. Irrigation
    irrigation = None
    if input_data["is_irrigation_applied"].lower() == "yes":
        irrigation = setup_irrigation(input_data["irrigation_application"])

    # 8. Fertilizer
    fertilizer = None
    if has_fertilizer:
        fertilizer = setup_fertilizer(fert_apps)

    # 9. Run
    dssat = DSSAT()
    try:
        results = dssat.run_treatment(
            field=field,
            cultivar=crop,
            planting=planting,
            simulation_controls=simulation_controls,
            irrigation=irrigation,
            fertilizer=fertilizer,
        )

        # 10. Parse outputs
        if results is None:
            print(f"Simulation '{input_data.get('experiment_name', 'unknown')}' did not run successfully.")
            explanations = SUMMARY_OUT_AS_JSON_NAN
        else:
            summary_str = dssat.output_files.get("Summary")
            if not summary_str:
                print("Summary.OUT not found or empty in DSSAT output files.")
                explanations = SUMMARY_OUT_AS_JSON_NAN
            else:
                explanations, _ = explain_summary_out(summary_str, output_file)
                with open("log.json", "w") as fout:
                    json.dump(explanations, fout, indent=4)

    except Exception as e:
        print(f"Simulation '{input_data.get('experiment_name', 'unknown')}' raised an error: {e}")
        explanations = SUMMARY_OUT_AS_JSON_NAN
    finally:
        dssat.close()

    return output_file, explanations

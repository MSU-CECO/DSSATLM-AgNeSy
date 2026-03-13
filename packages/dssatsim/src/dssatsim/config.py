import os
from importlib.resources import files
from dotenv import load_dotenv

load_dotenv()

# API clients
WEATHER_API_BASE_URL = os.getenv("WEATHER_API_BASE_URL", "http://localhost:8001")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
SOIL_API_BASE_URL = os.getenv("SOIL_API_BASE_URL", "https://soil-query-production.up.railway.app")

# Static file paths
_static = files("dssatsim.static")
ALL_DSSAT_CDE_FILES = str(_static.joinpath("ALL_DSSAT_CDE_FILES.csv"))
DSSAT_CROP_COEFFS_METADATA = str(_static.joinpath("CUL_SPE_ECO_COEFFS_DEFINITIONS.csv"))
OFFICIAL_DSSAT_CROP_CODES = str(_static.joinpath("OFFICIAL_DSSAT_CROP_CODES.csv"))

# Runtime constants
INSTI_CODE = "AGXQ"
MISSING_NA_VALUE = "-99"
DSSAT_NA_VALUE = "-99"
SUMMARY_OUT_AS_JSON_NAN = {"simulation_results": "impossible"}

WTH_COLUMNS = ["@DATE", "SRAD", "TMAX", "TMIN", "RAIN", "DEWP", "WIND", "PAR", "EVAP", "RHUM"]

# Crop registry
CROP_NAMES_TO_CROP_VARIETIES = {
    "Maize": ["MZ GREAT LAKES 482", "MZ GREAT LAKES 582", "MZ GREAT LAKES 582 KBS"],
    "Soybean": ["SB MATURITY GROUP 2"],
    "Wheat": ["WH NEWTON (Chelsea soft white)"],
}

# Keys are lowercase to match DSSATTools v3 crop coefficient API (crop["p1"] = ...)
CROP_VARIETIES_TO_CULTIVAR_CODES = {
    # Maize
    "MZ GREAT LAKES 482": {
        "@var#": "IB0090",
        "eco#": "IB0001",
        "p1": 240.0,
        "p2": 0.7,
        "p5": 990.0,
        "g2": 907.0,
        "g3": 8.8,
        "phint": 38.9,
    },
    "MZ GREAT LAKES 582": {
        "@var#": "IB0089",
        "eco#": "IB0001",
        "p1": 200.0,
        "p2": 0.7,
        "p5": 750.0,
        "g2": 750.0,
        "g3": 8.6,
        "phint": 38.9,
    },
    "MZ GREAT LAKES 582 KBS": {
        "@var#": "IB0093",
        "eco#": "IB0001",
        "p1": 180.0,
        "p2": 0.7,
        "p5": 750.0,
        "g2": 750.0,
        "g3": 8.6,
        "phint": 38.9,
    },
    # Soybean
    "SB MATURITY GROUP 2": {
        "@var#": "990002",
        "eco#": "SB0201",
        "csdl": 13.59,
        "ppsen": 0.249,
        "em-fl": 17.4,
        "fl-sh": 6.0,
        "fl-sd": 13.5,
        "sd-pm": 32.4,
        "fl-lf": 26.0,
        "lfmax": 1.03,
        "slavr": 375.0,
        "sizlf": 180.0,
        "xfrt": 1.0,
        "wtpsd": 0.19,
        "sfdur": 23.0,
        "sdpdv": 2.2,
        "podur": 10.0,
        "thrsh": 77.0,
        "sdpro": 0.405,
        "sdlip": 0.205,
    },
    # Wheat
    "WH NEWTON (Chelsea soft white)": {
        "@var#": "IB0488",
        "exp#": "1,6",
        "eco#": "USWH01",
        "p1v": 48.45,
        "p1d": 73.5,
        "p5": 505.0,
        "g1": 35.42,
        "g2": 22.6,
        "g3": 0.78,
        "phint": 95.0,
    },
}

MINIMUM_REQUIRED_FARMER_INPUTS = [
    "crop_name", "crop_variety", "latitude", "longitude", "elevation",
    "planting_date", "is_irrigation_applied", "irrigation_application",
]

OUTPUT_CODE_TYPES = ["PlantGro", "Weather", "SoilWat"]

CDE_SUFIX_SEP = "__"

SUMMARY_OUT_CATEGORIES_COLS = {
    "IDENTIFIERS": ["RUNNO", "TRNO", "R#", "O#", "P#", "CR", "MODEL"],
    "EXPERIMENT AND TREATMENT": ["EXNAME", "TNAM"],
    "SITE INFORMATION": ["FNAM", "WSTA", "WYEAR", "SOIL_ID", "LAT", "LONG", "ELEV"],
    "DATES": ["SDAT", "PDAT", "EDAT", "ADAT", "MDAT", "HDAT", "HYEAR"],
    "DRY WEIGHT, YIELD AND YIELD COMPONENTS": ["DWAP", "CWAM", "HWAM", "HWAH", "BWAH", "PWAM", "HWUM", "H#AM", "H#UM", "HIAM", "LAIX"],
    "FRESH WEIGHT": ["FCWAM", "FHWAM", "HWAHF", "FBWAH", "FPWAM"],
    "WATER": ["IR#M", "IRCM", "PRCM", "ETCM", "EPCM", "ESCM", "ROCM", "DRCM", "SWXM"],
    "NITROGEN": ["NI#M", "NICM", "NFXM", "NUCM", "NLCM", "NIAM", "CNAM", "GNAM", "N2OEM"],
    "PHOSPHORUS": ["PI#M", "PICM", "PUPC", "SPAM"],
    "POTASSIUM": ["KI#M", "KICM", "KUPC", "SKAM"],
    "ORGANIC MATTER": ["RECM", "ONTAM", "ONAM", "OPTAM", "OPAM", "OCTAM", "OCAM", "CO2EM", "CH4EM"],
    "WATER PRODUCTIVITY": ["DMPPM", "DMPEM", "DMPTM", "DMPIM", "YPPM", "YPEM", "YPTM", "YPIM"],
    "NITROGEN PRODUCTIVITY": ["DPNAM", "DPNUM", "YPNAM", "YPNUM"],
    "SEASONAL ENVIRONMENTAL DATA (Planting to harvest)": ["NDCH", "TMAXA", "TMINA", "SRADA", "DAYLA", "CO2A", "PRCP", "ETCP", "ESCP", "EPCP"],
    "STATUS": ["CRST"],
}

ASSUMPTIONS = {
    "fertilizer_material_code": "IB001",
    "fertilizer_application_code": DSSAT_NA_VALUE,
    "fertilizer_depth": 10,
    "fertilizer_Ca": 0,
    "fertilizer_other_elements_applied": 0,
    "fertilizer_other_elements_code": DSSAT_NA_VALUE,
    "fertilizer_name": DSSAT_NA_VALUE,
    "avg_annual_soil_temperature": 9.2,
    "amplitude_soil_temperature": 13.0,
    "simulation_start": None,
    "emergence_date": None,
    "initial_swc": 1,
    "harvest_management_option": "M",
    "organic_matter_management_option": "G",
    "irrigation_operation_code": "IR001",
}

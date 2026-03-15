"""
packages/dssatlm/src/dssatlm/envs.py
Configuration, constants, and environment variable declarations.
"""

import os
import uuid

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEXTUAL_DB_DIR = os.path.join(ROOT_DIR, "textual_db")
TMP_DIR = "/tmp"

DEFINITIONS_BANK_FPATH = os.path.join(TEXTUAL_DB_DIR, "bank_of_definitions.txt")
QUESTIONS_BANK_FPATH = os.path.join(TEXTUAL_DB_DIR, "bank_of_questions.txt")
SAMPLE_DEFN_N_QUESTIONS_COVERED_FPATH = os.path.join(TEXTUAL_DB_DIR, "sample_dssat_questions.csv")

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

LLM_IDS_CONSIDERED = {
    "gpt-4o":         "openai/gpt-4o",
    "gpt-4o-mini":    "openai/gpt-4o-mini",
    "claude-sonnet":  "anthropic/claude-sonnet-4-5",
    "llama-3.3-70b":  "meta-llama/llama-3.3-70b-instruct",
    "dsr1-llama-70b": "deepseek/deepseek-r1-distill-llama-70b",
}

DEFAULT_PARSER_MODEL_ID = "gpt-4o"
DEFAULT_INTERPRETER_MODEL_ID = "gpt-4o"

DEFAULT_LLM_PARAMS = {
    "temperature": 0.6,
    "max_tokens": 4096,
    "top_p": 0.9,
}

# ---------------------------------------------------------------------------
# Required API keys
# ---------------------------------------------------------------------------
REQUIRED_API_KEYS = ["OPENROUTER_API_KEY", "WANDB_API_KEY"]

# ---------------------------------------------------------------------------
# WandB defaults
# ---------------------------------------------------------------------------
DEFAULT_WANDB_PROJECT_PARAMS = {
    "project": "dev-dssatlm-project",
    "job_type": "dev-dssatlm-QA-pipeline",
    "name": "run_for_user_" + str(uuid.uuid4()),
}

# ---------------------------------------------------------------------------
# Simulator output filtering
# ---------------------------------------------------------------------------
REQUIRED_DSSATSIM_OUTPUT_KEYS = {
    "Dates",
    "Dry weight, yield and yield components",
    "Nitrogen",
    "Nitrogen productivity",
    "Organic matter",
    "Phosphorus",
    "Potassium",
    "Seasonal environmental data (planting to harvest)",
    "Water",
    "Water productivity",
}

UNWANTED_SUB_KEYS_FROM_SIMULATOR_OUTPUT = {
    "Leaf area index, maximum",
    "By-product removed during harvest (kg [dm]/ha)",
    "Pod/Ear/Panicle weight at maturity (kg [dm]/ha)",
    "CH4EM",
    "Average daylength (hr/d), planting to harvest",
    "Simulation start date",
    "HYEAR",
    "Crop establishment start",
    "Crop establishment end",
    "Crop establishment duration",
    "Vegetative growth start",
    "Vegetative growth end",
    "Vegetative growth duration",
    "Yield formation start",
    "Yield formation end",
    "Yield formation duration",
    "Entire period start",
    "Entire period end",
    "Entire period duration",
}

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
MISSING_OR_NA_REPR = "-99"


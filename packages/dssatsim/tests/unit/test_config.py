"""Tests for config.py — sanity checks on crop registry and assumptions."""

import pytest
from dssatsim.config import (
    CROP_NAMES_TO_CROP_VARIETIES,
    CROP_VARIETIES_TO_CULTIVAR_CODES,
    PLANTING_DEFAULTS,
    ASSUMPTIONS,
    MINIMUM_REQUIRED_FARMER_INPUTS,
)


def test_all_varieties_have_cultivar_codes():
    for crop, varieties in CROP_NAMES_TO_CROP_VARIETIES.items():
        for variety in varieties:
            assert variety in CROP_VARIETIES_TO_CULTIVAR_CODES, (
                f"Variety '{variety}' for crop '{crop}' missing from CROP_VARIETIES_TO_CULTIVAR_CODES"
            )


def test_all_cultivar_codes_have_var_key():
    for variety, props in CROP_VARIETIES_TO_CULTIVAR_CODES.items():
        assert "@var#" in props, f"Missing '@var#' for variety '{variety}'"


def test_all_cultivar_keys_are_lowercase():
    for variety, props in CROP_VARIETIES_TO_CULTIVAR_CODES.items():
        for key in props:
            assert key == key.lower(), (
                f"Key '{key}' in variety '{variety}' is not lowercase — "
                "DSSATTools v3 requires lowercase coefficient keys"
            )


def test_planting_defaults_exist_for_all_crops():
    for crop in CROP_NAMES_TO_CROP_VARIETIES:
        assert crop in PLANTING_DEFAULTS, f"No planting defaults for crop '{crop}'"


def test_planting_defaults_have_required_keys():
    required = {"ppop", "ppoe", "plme", "plds", "plrs", "plrd", "pldp"}
    for crop, defaults in PLANTING_DEFAULTS.items():
        missing = required - set(defaults.keys())
        assert not missing, f"Planting defaults for '{crop}' missing keys: {missing}"


def test_minimum_required_inputs_not_empty():
    assert len(MINIMUM_REQUIRED_FARMER_INPUTS) > 0


def test_assumptions_has_required_keys():
    required = {
        "fertilizer_material_code", "fertilizer_depth",
        "irrigation_operation_code", "harvest_management_option",
    }
    for key in required:
        assert key in ASSUMPTIONS, f"Missing assumption: '{key}'"

        
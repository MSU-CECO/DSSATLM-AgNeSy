"""Tests for run.py helper functions."""

import pytest
from datetime import date
from dssatsim.run import is_simulation_possible, setup_crop


# ---------------------------------------------------------------------------
# is_simulation_possible
# ---------------------------------------------------------------------------

VALID_INPUT = {
    "crop_name": "Maize",
    "crop_variety": "MZ GREAT LAKES 582 KBS",
    "latitude": 42.24,
    "longitude": -85.24,
    "elevation": 200,
    "planting_date": "2023-05-15",
    "is_irrigation_applied": "yes",
    "irrigation_application": [["2023-05-15", 80], ["2023-05-20", 100]],
}


def test_simulation_possible_with_valid_input():
    assert is_simulation_possible(VALID_INPUT) is True


def test_simulation_not_possible_missing_field():
    data = {**VALID_INPUT}
    del data["planting_date"]
    assert is_simulation_possible(data) is False


def test_simulation_not_possible_na_string():
    data = {**VALID_INPUT, "latitude": "-99"}
    assert is_simulation_possible(data) is False


def test_simulation_not_possible_na_int():
    data = {**VALID_INPUT, "elevation": -99}
    assert is_simulation_possible(data) is False


def test_simulation_not_possible_irrigation_has_na():
    data = {**VALID_INPUT, "irrigation_application": [[-99, 80]]}
    assert is_simulation_possible(data) is False


def test_simulation_possible_no_irrigation():
    data = {**VALID_INPUT, "is_irrigation_applied": "no", "irrigation_application": []}
    assert is_simulation_possible(data) is True


# ---------------------------------------------------------------------------
# setup_crop
# ---------------------------------------------------------------------------

def test_setup_crop_maize():
    crop = setup_crop("Maize", "MZ GREAT LAKES 582 KBS")
    assert crop is not None


def test_setup_crop_case_insensitive():
    crop = setup_crop("maize", "MZ GREAT LAKES 582 KBS")
    assert crop is not None


def test_setup_crop_invalid_crop_name():
    with pytest.raises(ValueError, match="not supported"):
        setup_crop("Tomato", "MZ GREAT LAKES 582 KBS")


def test_setup_crop_invalid_variety():
    with pytest.raises(ValueError, match="not supported"):
        setup_crop("Maize", "UNKNOWN VARIETY")


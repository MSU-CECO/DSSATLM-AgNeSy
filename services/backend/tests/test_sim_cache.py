"""
Unit tests for the sim_cache module.

Covers:
  - compute_sim_hash determinism and sensitivity
  - store / get round-trip
  - get returns None on miss
  - store updates existing entry
  - FIFO eviction at MAX_SIZE
  - clear_cache / cache_size helpers
"""
import pytest
from backend import sim_cache
from backend.sim_cache import (
    MAX_SIZE,
    cache_size,
    clear_cache,
    compute_sim_hash,
    get,
    store,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BASE_INPUTS = dict(
    latitude=42.263,
    longitude=-85.648,
    crop="Maize",
    variety="MZ GREAT LAKES 582 KBS",
    planting_date="2023-05-01",
    irrigation_events=[],
    nitrogen_events=[],
    phosphorus_events=[],
    potassium_events=[],
)

FAKE_OUTPUTS = {"question_1": {"answer_for_farmer": "Yield is 9063 kg/ha."}}
FAKE_LOGS = {"dssatlm_simulator_response": {"Dates": {"Harvest date": "2023-09-11"}}}


@pytest.fixture(autouse=True)
def reset_sim_cache():
    """Flush the sim cache before and after every test."""
    clear_cache()
    yield
    clear_cache()


# ---------------------------------------------------------------------------
# compute_sim_hash
# ---------------------------------------------------------------------------

class TestComputeSimHash:
    def test_returns_string(self):
        h = compute_sim_hash(**BASE_INPUTS)
        assert isinstance(h, str)

    def test_returns_64_char_hex(self):
        h = compute_sim_hash(**BASE_INPUTS)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_same_inputs_same_hash(self):
        h1 = compute_sim_hash(**BASE_INPUTS)
        h2 = compute_sim_hash(**BASE_INPUTS)
        assert h1 == h2

    def test_different_latitude_different_hash(self):
        h1 = compute_sim_hash(**BASE_INPUTS)
        h2 = compute_sim_hash(**{**BASE_INPUTS, "latitude": 43.0})
        assert h1 != h2

    def test_different_longitude_different_hash(self):
        h1 = compute_sim_hash(**BASE_INPUTS)
        h2 = compute_sim_hash(**{**BASE_INPUTS, "longitude": -86.0})
        assert h1 != h2

    def test_different_crop_different_hash(self):
        h1 = compute_sim_hash(**BASE_INPUTS)
        h2 = compute_sim_hash(**{**BASE_INPUTS, "crop": "Soybean"})
        assert h1 != h2

    def test_different_variety_different_hash(self):
        h1 = compute_sim_hash(**BASE_INPUTS)
        h2 = compute_sim_hash(**{**BASE_INPUTS, "variety": "OTHER VARIETY"})
        assert h1 != h2

    def test_different_planting_date_different_hash(self):
        h1 = compute_sim_hash(**BASE_INPUTS)
        h2 = compute_sim_hash(**{**BASE_INPUTS, "planting_date": "2023-06-01"})
        assert h1 != h2

    def test_irrigation_events_order_does_not_affect_hash(self):
        """Events in different order must produce the same hash."""
        evs_a = [["2023-06-01", 25.0], ["2023-07-01", 30.0]]
        evs_b = [["2023-07-01", 30.0], ["2023-06-01", 25.0]]
        h1 = compute_sim_hash(**{**BASE_INPUTS, "irrigation_events": evs_a})
        h2 = compute_sim_hash(**{**BASE_INPUTS, "irrigation_events": evs_b})
        assert h1 == h2

    def test_different_irrigation_events_different_hash(self):
        evs_a = [["2023-06-01", 25.0]]
        evs_b = [["2023-06-01", 99.0]]
        h1 = compute_sim_hash(**{**BASE_INPUTS, "irrigation_events": evs_a})
        h2 = compute_sim_hash(**{**BASE_INPUTS, "irrigation_events": evs_b})
        assert h1 != h2

    def test_crop_name_is_case_insensitive(self):
        """'Maize' and 'maize' must produce the same hash."""
        h1 = compute_sim_hash(**{**BASE_INPUTS, "crop": "Maize"})
        h2 = compute_sim_hash(**{**BASE_INPUTS, "crop": "maize"})
        assert h1 == h2

    def test_variety_is_case_insensitive(self):
        h1 = compute_sim_hash(**{**BASE_INPUTS, "variety": "KBS 582"})
        h2 = compute_sim_hash(**{**BASE_INPUTS, "variety": "kbs 582"})
        assert h1 == h2


# ---------------------------------------------------------------------------
# store / get
# ---------------------------------------------------------------------------

class TestStoreGet:
    def test_store_and_get_round_trip(self):
        h = compute_sim_hash(**BASE_INPUTS)
        store(h, FAKE_OUTPUTS, FAKE_LOGS)
        entry = get(h)
        assert entry is not None
        assert entry["outputs"] == FAKE_OUTPUTS
        assert entry["logs"] == FAKE_LOGS

    def test_get_miss_returns_none(self):
        assert get("nonexistent-hash") is None

    def test_cache_size_increments_on_store(self):
        assert cache_size() == 0
        h = compute_sim_hash(**BASE_INPUTS)
        store(h, FAKE_OUTPUTS, FAKE_LOGS)
        assert cache_size() == 1

    def test_store_same_hash_updates_value(self):
        h = compute_sim_hash(**BASE_INPUTS)
        store(h, FAKE_OUTPUTS, FAKE_LOGS)
        new_outputs = {"question_1": {"answer_for_farmer": "Updated answer."}}
        store(h, new_outputs, FAKE_LOGS)
        assert cache_size() == 1
        assert get(h)["outputs"] == new_outputs

    def test_multiple_hashes_stored_independently(self):
        h1 = compute_sim_hash(**BASE_INPUTS)
        h2 = compute_sim_hash(**{**BASE_INPUTS, "crop": "Soybean"})
        store(h1, FAKE_OUTPUTS, FAKE_LOGS)
        store(h2, {"question_1": {"answer_for_farmer": "Soy answer."}}, FAKE_LOGS)
        assert cache_size() == 2
        assert get(h1)["outputs"] == FAKE_OUTPUTS
        assert get(h2)["outputs"]["question_1"]["answer_for_farmer"] == "Soy answer."


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------

class TestEviction:
    def test_eviction_at_max_size(self):
        hashes = []
        for i in range(MAX_SIZE + 2):
            inputs = {**BASE_INPUTS, "latitude": 40.0 + i * 0.01}
            h = compute_sim_hash(**inputs)
            store(h, FAKE_OUTPUTS, FAKE_LOGS)
            hashes.append(h)
        assert cache_size() == MAX_SIZE

    def test_oldest_entry_evicted_first(self):
        """First entry stored must be gone after MAX_SIZE + 1 inserts."""
        first_inputs = {**BASE_INPUTS, "latitude": 10.0}
        first_hash = compute_sim_hash(**first_inputs)
        store(first_hash, FAKE_OUTPUTS, FAKE_LOGS)

        for i in range(MAX_SIZE):
            inputs = {**BASE_INPUTS, "latitude": 20.0 + i * 0.01}
            h = compute_sim_hash(**inputs)
            store(h, FAKE_OUTPUTS, FAKE_LOGS)

        assert get(first_hash) is None


# ---------------------------------------------------------------------------
# clear_cache / cache_size
# ---------------------------------------------------------------------------

class TestCacheHelpers:
    def test_clear_cache_empties(self):
        h = compute_sim_hash(**BASE_INPUTS)
        store(h, FAKE_OUTPUTS, FAKE_LOGS)
        assert cache_size() == 1
        clear_cache()
        assert cache_size() == 0

    def test_get_after_clear_returns_none(self):
        h = compute_sim_hash(**BASE_INPUTS)
        store(h, FAKE_OUTPUTS, FAKE_LOGS)
        clear_cache()
        assert get(h) is None


"""
Unit tests for DSSATLMPipeline.

All external dependencies (dspy.LM, dssatsim, wandb) are mocked so these
tests run without API keys or DSSAT installed.
"""

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_PARSED = {
    "farm_name": "Test Farm",
    "crop_name": "Maize",
    "crop_variety": "MZ GREAT LAKES 582 KBS",
    "latitude": 42.263,
    "longitude": -85.648,
    "elevation": 288.0,
    "planting_date": "2023-05-01",
    "is_irrigation_applied": "no",
    "irrigation_application": [],
    "nitrogen_fertilizer_application": [],
    "phosphorus_fertilizer_application": [],
    "potassium_fertilizer_application": [],
    "question_statements": [
        "What would be my crop yield at maturity?",
        "When would I be able to harvest?",
    ],
}

FAKE_PARSED_WITH_FERTILIZER = {
    **FAKE_PARSED,
    "nitrogen_fertilizer_application": [["2023-05-15", 50.0], ["2023-06-01", 30.0]],
    "phosphorus_fertilizer_application": [["2023-05-15", 20.0]],
    "potassium_fertilizer_application": [["2023-05-15", 25.0]],
}

FAKE_SIM_OUTPUTS = {
    "Dates": {
        "Harvest date": "2023-10-15",
        "Anthesis date": "2023-07-20",
        "Physiological maturity date": "2023-09-30",
        "Number of days from planting to harvest (d)": 167,
    },
    "Dry weight, yield and yield components": {
        "Yield at harvest maturity (kg [dm]/ha)": 8500,
        "Harvested yield (kg [dm]/ha)": 8400,
        "Tops weight at maturity (kg [dm]/ha)": 18000,
        "Harvest index at maturity": 0.47,
        "Residue applied (kg/ha)": 9600,
    },
    "Nitrogen": {
        "N applications (no)": 0,
        "Inorganic N applied (kg [N]/ha)": 0,
        "N fixed during season (kg/ha)": 0,
        "N uptake during season (kg [N]/ha)": 145,
        "N leached during season (kg [N]/ha)": 8,
        "Inorganic N at maturity (kg [N]/ha)": 22,
        "N2OEM": 1.2,
    },
    "Nitrogen productivity": {
        "Dry matter-N fertilizer productivity (kg[DM]/kg[N fert])": -99,
        "Yield-N fertilizer productivity (kg[yield]/kg[N fert])": -99,
    },
    "Organic matter": {
        "CO2EM": 450,
    },
    "Phosphorus": {
        "Number of P applications (no)": 0,
        "Inorganic P applied (kg/ha)": 0,
        "Seasonal cumulative P uptake (kg[P]/ha)": 25,
        "Soil P at maturity (kg/ha)": 40,
    },
    "Potassium": {
        "Number of K applications (no)": 0,
        "Inorganic K applied (kg/ha)": 0,
        "Seasonal cumulative K uptake (kg[K]/ha)": 120,
        "Soil K at maturity (kg/ha)": 180,
    },
    "Seasonal environmental data (planting to harvest)": {
        "Total season precipitation (mm), planting to harvest": 380,
        "Total evapotranspiration, planting to harvest (mm)": 420,
    },
    "Water": {
        "Irrigation applications (no)": 0,
        "Season applied irrigation (includes losses) (mm)": 0,
        "Total season precipitation (mm), simulation - harvest": 400,
        "Total season evapotranspiration, simulation-harvest (mm)": 440,
        "Season surface runoff (mm)": 12,
        "Season water drainage (mm)": 71,
        "Extractable water at maturity (mm)": 85,
    },
    "Water productivity": {
        "Yield-ET productivity (kg[yield]/ha/mm[ET])": 20.0,
        "Yield-irrigation productivity (kg[yield]/ha/mm[irrig])": -99,
        "Dry matter-irrigation productivity (kg[DM]/ha/mm[irrig])": -99,
    },
}

FAKE_INTERPRETER_ANSWERS = {
    "question_1": {
        "question_statement": "What would be my crop yield at maturity?",
        "matched_question_found": "What would be my crop yield at maturity?",
        "retrieved_answer": "The Yield at harvest maturity (kg [dm]/ha) is 8500",
        "answer_for_farmer": "Your crop yield at maturity is 8500 kg/ha (7588 lbs/ac). ...",
    },
    "question_2": {
        "question_statement": "When would I be able to harvest?",
        "matched_question_found": "When would I be able to harvest?",
        "retrieved_answer": "The Harvest date is 2023-10-15",
        "answer_for_farmer": "You should be able to harvest around October 15, 2023. ...",
    },
}


def make_env_with_keys(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-openrouter-key")
    monkeypatch.setenv("WANDB_API_KEY", "fake-wandb-key")


# ---------------------------------------------------------------------------
# Helper: build a pipeline with all external calls mocked
# ---------------------------------------------------------------------------

def make_mocked_pipeline(monkeypatch, tmp_path):
    make_env_with_keys(monkeypatch)

    mock_lm = MagicMock()
    monkeypatch.setattr("dspy.LM", lambda **kwargs: mock_lm)

    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=None)
    mock_ctx.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr("dspy.context", lambda **kwargs: mock_ctx)

    monkeypatch.setattr("wandb.init", lambda **kwargs: MagicMock())

    monkeypatch.setattr(
        "dssatlm.pipeline.DSSATLMPipeline._load_text",
        staticmethod(lambda fpath: "MOCK BANK CONTENT"),
    )

    monkeypatch.setattr("dssatlm.pipeline.TMP_DIR", str(tmp_path))

    from dssatlm.pipeline import DSSATLMPipeline
    pipeline = DSSATLMPipeline(wandb_params=None)
    return pipeline


# ---------------------------------------------------------------------------
# Tests -- validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_openrouter_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("WANDB_API_KEY", "fake")
        from dssatlm.pipeline import DSSATLMPipeline
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            DSSATLMPipeline.__new__(DSSATLMPipeline)._validate_api_keys()

    def test_missing_wandb_key_raises(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        from dssatlm.pipeline import DSSATLMPipeline
        with pytest.raises(ValueError, match="WANDB_API_KEY"):
            DSSATLMPipeline.__new__(DSSATLMPipeline)._validate_api_keys()

    def test_invalid_model_id_raises(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
        monkeypatch.setenv("WANDB_API_KEY", "fake")
        from dssatlm.pipeline import DSSATLMPipeline
        p = DSSATLMPipeline.__new__(DSSATLMPipeline)
        with pytest.raises(ValueError, match="not valid"):
            p._validate_model_ids("not-a-real-model", "gpt-4o")

    def test_valid_model_ids_pass(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
        monkeypatch.setenv("WANDB_API_KEY", "fake")
        from dssatlm.pipeline import DSSATLMPipeline
        p = DSSATLMPipeline.__new__(DSSATLMPipeline)
        p._validate_model_ids("gpt-4o", "gpt-4o")

    def test_all_new_model_ids_are_valid(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
        monkeypatch.setenv("WANDB_API_KEY", "fake")
        from dssatlm.pipeline import DSSATLMPipeline
        p = DSSATLMPipeline.__new__(DSSATLMPipeline)
        for model_id in ["gpt-4o", "gpt-4o-mini", "claude-sonnet", "llama-3.3-70b", "dsr1-llama-70b"]:
            p._validate_model_ids(model_id, model_id)


# ---------------------------------------------------------------------------
# Tests -- parser step
# ---------------------------------------------------------------------------

class TestParserStep:
    def test_run_parser_returns_expected_keys(self, monkeypatch, tmp_path):
        pipeline = make_mocked_pipeline(monkeypatch, tmp_path)
        pipeline.parser = MagicMock(return_value=FAKE_PARSED)

        result = pipeline._run_parser("My farm is at 42.263, -85.648. I planted maize on May 1 2023. What is my yield?")

        assert result["crop_name"] == "Maize"
        assert result["latitude"] == 42.263
        assert "question_statements" in result
        assert len(result["question_statements"]) >= 1

    def test_run_parser_logs_parsed_response(self, monkeypatch, tmp_path):
        pipeline = make_mocked_pipeline(monkeypatch, tmp_path)
        pipeline.parser = MagicMock(return_value=FAKE_PARSED)

        pipeline._run_parser("some query")

        assert pipeline._logs["dssatlm_parser_response"] == FAKE_PARSED
        assert pipeline._logs["question_statements_parsed"] == FAKE_PARSED["question_statements"]

    def test_run_parser_raises_on_failure(self, monkeypatch, tmp_path):
        from dssatlm.pipeline import _PipelineStepError
        pipeline = make_mocked_pipeline(monkeypatch, tmp_path)
        pipeline.parser = MagicMock(side_effect=RuntimeError("LM call failed"))

        with pytest.raises(_PipelineStepError):
            pipeline._run_parser("any query")

    def test_run_parser_returns_all_fertilizer_fields(self, monkeypatch, tmp_path):
        pipeline = make_mocked_pipeline(monkeypatch, tmp_path)
        pipeline.parser = MagicMock(return_value=FAKE_PARSED_WITH_FERTILIZER)

        result = pipeline._run_parser("some query with fertilizer")

        assert "nitrogen_fertilizer_application" in result
        assert "phosphorus_fertilizer_application" in result
        assert "potassium_fertilizer_application" in result

    def test_run_parser_phosphorus_extracted(self, monkeypatch, tmp_path):
        pipeline = make_mocked_pipeline(monkeypatch, tmp_path)
        pipeline.parser = MagicMock(return_value=FAKE_PARSED_WITH_FERTILIZER)

        result = pipeline._run_parser("some query")

        assert result["phosphorus_fertilizer_application"] == [["2023-05-15", 20.0]]

    def test_run_parser_potassium_extracted(self, monkeypatch, tmp_path):
        pipeline = make_mocked_pipeline(monkeypatch, tmp_path)
        pipeline.parser = MagicMock(return_value=FAKE_PARSED_WITH_FERTILIZER)

        result = pipeline._run_parser("some query")

        assert result["potassium_fertilizer_application"] == [["2023-05-15", 25.0]]

    def test_run_parser_empty_pk_when_not_mentioned(self, monkeypatch, tmp_path):
        pipeline = make_mocked_pipeline(monkeypatch, tmp_path)
        pipeline.parser = MagicMock(return_value=FAKE_PARSED)

        result = pipeline._run_parser("query with no fertilizer")

        assert result["phosphorus_fertilizer_application"] == []
        assert result["potassium_fertilizer_application"] == []


# ---------------------------------------------------------------------------
# Tests -- simulator step
# ---------------------------------------------------------------------------

class TestSimulatorStep:
    def test_sim_not_possible_raises(self, monkeypatch, tmp_path):
        from dssatlm.pipeline import _PipelineStepError
        pipeline = make_mocked_pipeline(monkeypatch, tmp_path)

        with patch("dssatlm.pipeline.dssatsim_run") as mock_run:
            mock_run.is_simulation_possible.return_value = False
            with pytest.raises(_PipelineStepError):
                pipeline._run_simulator(FAKE_PARSED)

        assert pipeline._logs["simulation_is_possible"] is False

    def test_successful_sim_returns_filtered_outputs(self, monkeypatch, tmp_path):
        pipeline = make_mocked_pipeline(monkeypatch, tmp_path)

        with patch("dssatlm.pipeline.dssatsim_run") as mock_run:
            mock_run.is_simulation_possible.return_value = True
            mock_run.exec.return_value = (None, FAKE_SIM_OUTPUTS)
            result = pipeline._run_simulator(FAKE_PARSED)

        assert "Dates" in result
        assert "Dry weight, yield and yield components" in result
        assert pipeline._logs["simulation_is_successful"] is True

    def test_filter_removes_unwanted_subkeys(self, monkeypatch, tmp_path):
        pipeline = make_mocked_pipeline(monkeypatch, tmp_path)
        raw = {
            "Dates": {
                "Harvest date": "2023-10-15",
                "Simulation start date": "2023-01-01",
                "HYEAR": 2023,
            },
            **{k: {} for k in list(FAKE_SIM_OUTPUTS.keys())[1:]},
        }
        filtered = pipeline._filter_sim_outputs(raw)
        assert "Simulation start date" not in filtered.get("Dates", {})
        assert "HYEAR" not in filtered.get("Dates", {})
        assert "Harvest date" in filtered.get("Dates", {})


# ---------------------------------------------------------------------------
# Tests -- interpreter step
# ---------------------------------------------------------------------------

class TestInterpreterStep:
    def test_run_interpreter_returns_keyed_answers(self, monkeypatch, tmp_path):
        pipeline = make_mocked_pipeline(monkeypatch, tmp_path)
        pipeline.interpreter = MagicMock(return_value=FAKE_INTERPRETER_ANSWERS)

        result = pipeline._run_interpreter(
            question_statements=FAKE_PARSED["question_statements"],
            sim_outputs=FAKE_SIM_OUTPUTS,
        )

        assert "question_1" in result
        assert "question_2" in result
        assert "answer_for_farmer" in result["question_1"]

    def test_run_interpreter_raises_on_failure(self, monkeypatch, tmp_path):
        from dssatlm.pipeline import _PipelineStepError
        pipeline = make_mocked_pipeline(monkeypatch, tmp_path)
        pipeline.interpreter = MagicMock(side_effect=RuntimeError("LM failure"))

        with pytest.raises(_PipelineStepError):
            pipeline._run_interpreter(
                question_statements=["What is my yield?"],
                sim_outputs=FAKE_SIM_OUTPUTS,
            )


# ---------------------------------------------------------------------------
# Tests -- full answer_query (end-to-end mocked)
# ---------------------------------------------------------------------------

class TestAnswerQuery:
    def test_answer_query_returns_outputs_dict(self, monkeypatch, tmp_path):
        pipeline = make_mocked_pipeline(monkeypatch, tmp_path)
        pipeline.parser = MagicMock(return_value=FAKE_PARSED)
        pipeline.interpreter = MagicMock(return_value=FAKE_INTERPRETER_ANSWERS)

        with patch("dssatlm.pipeline.dssatsim_run") as mock_run:
            mock_run.is_simulation_possible.return_value = True
            mock_run.exec.return_value = (None, FAKE_SIM_OUTPUTS)
            outputs = pipeline.answer_query("My farm... What is my yield?")

        assert isinstance(outputs, dict)
        assert "question_1" in outputs

    def test_answer_query_sets_pipeline_ran_successfully_on_failure(self, monkeypatch, tmp_path):
        pipeline = make_mocked_pipeline(monkeypatch, tmp_path)
        pipeline.parser = MagicMock(side_effect=RuntimeError("Unexpected failure"))

        outputs = pipeline.answer_query("some query")

        assert pipeline._logs["pipeline_ran_successfully"] is False
        assert outputs == {}

    def test_answer_query_expert_answer_attached(self, monkeypatch, tmp_path):
        pipeline = make_mocked_pipeline(monkeypatch, tmp_path)
        pipeline.parser = MagicMock(return_value=FAKE_PARSED)
        pipeline.interpreter = MagicMock(return_value=FAKE_INTERPRETER_ANSWERS)

        with patch("dssatlm.pipeline.dssatsim_run") as mock_run:
            mock_run.is_simulation_possible.return_value = True
            mock_run.exec.return_value = (None, FAKE_SIM_OUTPUTS)
            outputs = pipeline.answer_query("query")

        for q_key in outputs:
            assert "expert_like_answer" in outputs[q_key]


# ---------------------------------------------------------------------------
# Tests -- logs
# ---------------------------------------------------------------------------

class TestLogs:
    def test_get_logs_returns_full_dict(self, monkeypatch, tmp_path):
        pipeline = make_mocked_pipeline(monkeypatch, tmp_path)
        logs = pipeline.get_logs()
        assert isinstance(logs, dict)
        assert "outputs" in logs
        assert "execution_errors" in logs

    def test_get_logs_subkey(self, monkeypatch, tmp_path):
        pipeline = make_mocked_pipeline(monkeypatch, tmp_path)
        outputs = pipeline.get_logs(subkey="outputs")
        assert isinstance(outputs, dict)

    def test_get_logs_unknown_subkey_returns_none(self, monkeypatch, tmp_path):
        pipeline = make_mocked_pipeline(monkeypatch, tmp_path)
        result = pipeline.get_logs(subkey="does_not_exist")
        assert result is None


# ---------------------------------------------------------------------------
# Tests -- _sim_was_successful
# ---------------------------------------------------------------------------

class TestSimWasSuccessful:
    def test_all_keys_present(self, monkeypatch, tmp_path):
        pipeline = make_mocked_pipeline(monkeypatch, tmp_path)
        assert pipeline._sim_was_successful(FAKE_SIM_OUTPUTS) is True

    def test_missing_key_returns_false(self, monkeypatch, tmp_path):
        pipeline = make_mocked_pipeline(monkeypatch, tmp_path)
        incomplete = {k: v for k, v in FAKE_SIM_OUTPUTS.items() if k != "Dates"}
        assert pipeline._sim_was_successful(incomplete) is False


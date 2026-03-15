"""
DSSATLMPipeline -- the top-level orchestrator.

Replaces the LangChain-based DSSATAnyLMPipeline with a DSPy + OpenRouter
equivalent. The three-step structure is still:

  Step 1 -- Parser:      free-text query -> structured dict (ParserModule)
  Step 2 -- Simulator:   structured dict -> DSSAT simulation outputs (dssatsim)
  Step 3 -- Interpreter: simulation outputs + questions -> farmer answers (InterpreterModule)

WandB logging is for experiment tracking.
"""

import json
import os
import uuid
from typing import Optional

import dspy
import pandas as pd
import wandb

from dssatlm.envs import (
    DEFINITIONS_BANK_FPATH,
    QUESTIONS_BANK_FPATH,
    SAMPLE_DEFN_N_QUESTIONS_COVERED_FPATH,
    LLM_IDS_CONSIDERED,
    DEFAULT_PARSER_MODEL_ID,
    DEFAULT_INTERPRETER_MODEL_ID,
    DEFAULT_LLM_PARAMS,
    OPENROUTER_BASE_URL,
    REQUIRED_API_KEYS,
    DEFAULT_WANDB_PROJECT_PARAMS,
    REQUIRED_DSSATSIM_OUTPUT_KEYS,
    UNWANTED_SUB_KEYS_FROM_SIMULATOR_OUTPUT,
    MISSING_OR_NA_REPR,
    TMP_DIR,
)
from dssatlm.modules import ParserModule, InterpreterModule
from dssatlm.utils import get_current_time, dict_to_json_file

# dssatsim is an optional runtime dependency; tests mock this at module level.
try:
    from dssatsim import run as dssatsim_run
except ImportError:  
    dssatsim_run = None  


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class DSSATLMPipeline:
    """
    Orchestrates the full DSSAT-LM pipeline:
      parser -> DSSAT simulator -> interpreter

    Args:
        parser_model_id:      Short key from LLM_IDS_CONSIDERED (default: "gpt-4o")
        interpreter_model_id: Short key from LLM_IDS_CONSIDERED (default: "gpt-4o")
        llm_params:           Optional overrides for temperature / max_tokens / top_p
        wandb_params:         WandB init kwargs. Pass None to disable WandB logging.
    """

    def __init__(
        self,
        parser_model_id: str = DEFAULT_PARSER_MODEL_ID,
        interpreter_model_id: str = DEFAULT_INTERPRETER_MODEL_ID,
        llm_params: Optional[dict] = None,
        wandb_params: Optional[dict] = DEFAULT_WANDB_PROJECT_PARAMS,
    ):
        self._validate_api_keys()
        self._validate_model_ids(parser_model_id, interpreter_model_id)

        self.parser_model_id = parser_model_id
        self.interpreter_model_id = interpreter_model_id
        self.llm_params = {**DEFAULT_LLM_PARAMS, **(llm_params or {})}

        # Configure DSPy LMs via OpenRouter
        self.parser_lm = self._make_lm(parser_model_id)
        self.interpreter_lm = self._make_lm(interpreter_model_id)

        # DSPy modules
        with dspy.context(lm=self.parser_lm):
            self.parser = ParserModule()

        with dspy.context(lm=self.interpreter_lm):
            self.interpreter = InterpreterModule()

        # Textual DB (loaded once, reused across calls)
        self._definitions_bank = self._load_text(DEFINITIONS_BANK_FPATH)
        self._questions_bank = self._load_text(QUESTIONS_BANK_FPATH)

        # WandB
        self._wandb_enabled = wandb_params is not None
        self._wandb_params = wandb_params or {}
        self._wandb_run = self._setup_wandb() if self._wandb_enabled else None

        # Pipeline state
        self._logs: dict = {}
        self._reset_logs()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer_query(self, farmer_input_query: str) -> dict:
        """
        Run the full pipeline for a farmer's free-text query.

        Returns:
            Dict keyed by 'question_1', 'question_2', ... Each value contains:
              - question_statement
              - matched_question_found
              - retrieved_answer
              - answer_for_farmer
              - expert_like_answer  (formulaic ground-truth from sample CSV)
        """
        self._reset_logs()

        try:
            # Step 1 -- Parse
            parsed = self._run_parser(farmer_input_query)

            # Step 2 -- Simulate
            sim_outputs = self._run_simulator(parsed)

            # Step 3 -- Interpret
            answers = self._run_interpreter(
                question_statements=parsed["question_statements"],
                sim_outputs=sim_outputs,
            )

            # Enrich with expert-like ground truth
            final_outputs = self._enrich_with_expert_answers(sim_outputs, answers)
            self._logs["outputs"] = final_outputs

        except _PipelineStepError as e:
            print(f"Pipeline stopped at: {e}")
            self._logs["pipeline_ran_successfully"] = False

        except Exception as e:
            self._logs["pipeline_ran_successfully"] = False
            self._logs["execution_errors"]["unexpected"] += (
                f"\nAt {get_current_time()}: Unexpected error: {e}"
            )
            print(f"Unexpected pipeline error: {e}")

        finally:
            self._save_logs()
            if self._wandb_enabled:
                self._close_wandb()

        return self._logs["outputs"]

    # ------------------------------------------------------------------
    # Step 1 -- Parser
    # ------------------------------------------------------------------

    def _run_parser(self, farmer_input_query: str) -> dict:
        try:
            with dspy.context(lm=self.parser_lm):
                parsed = self.parser(farmer_input_query=farmer_input_query)

            self._logs["dssatlm_parser_response"] = parsed
            self._logs["question_statements_parsed"] = parsed.get("question_statements", [])
            print("Step 1: Successfully parsed query to simulator structure.")
            return parsed

        except Exception as e:
            self._logs["execution_errors"]["step_1_parsing"] += (
                f"\nAt {get_current_time()}: {e}"
            )
            raise _PipelineStepError("Step 1 (parsing)") from e

    # ------------------------------------------------------------------
    # Step 2 -- Simulator
    # ------------------------------------------------------------------

    def _run_simulator(self, parsed: dict) -> dict:
        try:
            if not dssatsim_run.is_simulation_possible(parsed):
                msg = "Simulation not possible -- required inputs missing."
                self._logs["execution_errors"]["step_2_simulation"] += (
                    f"\nAt {get_current_time()}: {msg}"
                )
                self._logs["simulation_is_possible"] = False
                raise _PipelineStepError("Step 2 (simulation): " + msg)

            self._logs["simulation_is_possible"] = True

            _, raw_outputs = dssatsim_run.exec(input_file=parsed)
            filtered = self._filter_sim_outputs(raw_outputs)

            if not self._sim_was_successful(filtered):
                msg = "Simulation ran but required output keys are missing."
                self._logs["execution_errors"]["step_2_simulation"] += (
                    f"\nAt {get_current_time()}: {msg}"
                )
                self._logs["simulation_is_successful"] = False
                raise _PipelineStepError("Step 2 (simulation): " + msg)

            self._logs["simulation_is_successful"] = True
            self._logs["dssatlm_simulator_response"] = filtered
            print("Step 2: Successfully ran DSSAT simulation.")
            return filtered

        except _PipelineStepError:
            raise
        except Exception as e:
            self._logs["execution_errors"]["step_2_simulation"] += (
                f"\nAt {get_current_time()}: {e}"
            )
            raise _PipelineStepError("Step 2 (simulation)") from e

    def _filter_sim_outputs(self, raw: dict) -> dict:
        return {
            k: {sk: sv for sk, sv in v.items() if sk not in UNWANTED_SUB_KEYS_FROM_SIMULATOR_OUTPUT}
            for k, v in raw.items()
            if k in REQUIRED_DSSATSIM_OUTPUT_KEYS
        }

    def _sim_was_successful(self, outputs: dict) -> bool:
        return REQUIRED_DSSATSIM_OUTPUT_KEYS <= set(outputs.keys())

    # ------------------------------------------------------------------
    # Step 3 -- Interpreter
    # ------------------------------------------------------------------

    def _run_interpreter(self, question_statements: list, sim_outputs: dict) -> dict:
        try:
            with dspy.context(lm=self.interpreter_lm):
                answers = self.interpreter(
                    definitions_bank=self._definitions_bank,
                    questions_bank=self._questions_bank,
                    simulation_outputs_json=sim_outputs,
                    question_statements=question_statements,
                )

            self._logs["dssatlm_interpreter_response"] = answers
            print("Step 3: Successfully interpreted simulation results.")
            return answers

        except Exception as e:
            self._logs["execution_errors"]["step_3_interpreting"] += (
                f"\nAt {get_current_time()}: {e}"
            )
            raise _PipelineStepError("Step 3 (interpreting)") from e

    # ------------------------------------------------------------------
    # Expert-like ground truth enrichment
    # ------------------------------------------------------------------

    def _enrich_with_expert_answers(self, sim_outputs: dict, answers: dict) -> dict:
        """
        Attach a formulaic expert_like_answer to each question entry,
        derived from the sample CSV lookup table.
        """
        try:
            df = pd.read_csv(SAMPLE_DEFN_N_QUESTIONS_COVERED_FPATH)
        except Exception:
            df = pd.DataFrame()

        enriched = {}
        for q_key, q_data in answers.items():
            matched_q = q_data.get("matched_question_found", MISSING_OR_NA_REPR)
            expert_answer = self._form_expert_answer(df, sim_outputs, matched_q)
            enriched[q_key] = {**q_data, "expert_like_answer": expert_answer}

        return enriched

    def _form_expert_answer(self, df: pd.DataFrame, sim_outputs: dict, question_statement: str) -> str:
        if df.empty or question_statement not in df["QUESTIONS"].values:
            return MISSING_OR_NA_REPR

        row = df[df["QUESTIONS"] == question_statement].iloc[0]
        category_type = row["CATEGORY-TYPE"]
        category = row["CATEGORY"]
        category_definition = row["CATEGORY_DEFINITIONS"]

        try:
            value = sim_outputs[category_type][category]
        except KeyError:
            return MISSING_OR_NA_REPR

        return (
            f"The {category} is {value}. "
            f"Here is more definition: {category_definition}"
        )

    # ------------------------------------------------------------------
    # LM construction
    # ------------------------------------------------------------------

    def _make_lm(self, model_id: str) -> dspy.LM:
        full_model_id = LLM_IDS_CONSIDERED[model_id]
        return dspy.LM(
            model=f"openai/{full_model_id}",
            api_base=OPENROUTER_BASE_URL,
            api_key=os.environ["OPENROUTER_API_KEY"],
            temperature=self.llm_params["temperature"],
            max_tokens=self.llm_params["max_tokens"],
            top_p=self.llm_params["top_p"],
        )

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _reset_logs(self):
        self._logs = {
            "pipeline_ran_successfully": True,
            "simulation_is_possible": False,
            "simulation_is_successful": False,
            "question_statements_parsed": [],
            "dssatlm_parser_response": {},
            "dssatlm_simulator_response": {"simulation_results": "impossible"},
            "dssatlm_interpreter_response": {},
            "outputs": {},
            "execution_errors": {
                "step_1_parsing": "",
                "step_2_simulation": "",
                "step_3_interpreting": "",
                "unexpected": "",
            },
        }

    def _save_logs(self):
        fname = f"dssatlm_logs_{get_current_time()}".replace(" ", "_").replace(":", "-")
        fpath = os.path.join(TMP_DIR, f"{fname}.json")
        dict_to_json_file(self._logs, fpath)

        if self._wandb_enabled and self._wandb_run:
            try:
                artifact = wandb.Artifact(os.path.basename(fpath), type="logs")
                artifact.add_file(fpath)
                self._wandb_run.log_artifact(artifact)
            except Exception as e:
                print(f"WandB artifact upload failed: {e}")

        print(f"Logs saved to: {fpath}")

    def get_logs(self, subkey: Optional[str] = None):
        if subkey:
            return self._logs.get(subkey)
        return self._logs

    # ------------------------------------------------------------------
    # WandB helpers
    # ------------------------------------------------------------------

    def _setup_wandb(self):
        try:
            return wandb.init(**self._wandb_params, settings=wandb.Settings(start_method="thread"))
        except wandb.errors.AuthenticationError as e:
            raise ValueError("WANDB_API_KEY is invalid.") from e
        except Exception as e:
            raise ValueError(f"WandB init failed: {e}") from e

    def _close_wandb(self):
        if self._wandb_run:
            self._wandb_run.finish()

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_api_keys(self):
        missing = [k for k in REQUIRED_API_KEYS if not os.environ.get(k)]
        if missing:
            raise ValueError(
                f"Missing required environment variables: {missing}. "
                "Set them before instantiating DSSATLMPipeline."
            )

    def _validate_model_ids(self, parser_id: str, interpreter_id: str):
        valid = set(LLM_IDS_CONSIDERED.keys())
        for label, mid in [("parser_model_id", parser_id), ("interpreter_model_id", interpreter_id)]:
            if mid not in valid:
                raise ValueError(
                    f"{label}='{mid}' is not valid. Choose from: {sorted(valid)}"
                )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    @staticmethod
    def _load_text(fpath: str) -> str:
        with open(fpath, "r", encoding="utf-8") as f:
            return f.read()

    def __repr__(self):
        return (
            f"DSSATLMPipeline("
            f"parser={self.parser_model_id}, "
            f"interpreter={self.interpreter_model_id})"
        )


# ---------------------------------------------------------------------------
# Internal exception for controlled pipeline flow
# ---------------------------------------------------------------------------

class _PipelineStepError(Exception):
    """Raised internally to halt the pipeline at a specific step."""
    pass


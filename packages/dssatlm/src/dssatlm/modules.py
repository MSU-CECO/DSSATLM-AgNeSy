"""
DSPy Modules that wrap the Parser and Interpreter Signatures.

ParserModule      -- free-text farmer query -> structured dict
InterpreterModule -- simulation outputs + questions -> farmer-friendly answers

Both modules use dspy.Predict (deterministic structured output).
ChainOfThought is used for the Interpreter to improve reasoning quality
on the unit conversion + explanation task.
"""

import json
import dspy

from dssatlm.signatures import ParserSignature, InterpreterSignature


# ---------------------------------------------------------------------------
# Parser Module
# ---------------------------------------------------------------------------

class ParserModule(dspy.Module):
    """
    Converts a free-text farmer query into a structured dict suitable
    for passing to the DSSAT simulation runner.
    """

    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ParserSignature)

    def forward(self, farmer_input_query: str) -> dict:
        result = self.predict(farmer_input_query=farmer_input_query)

        return {
            "farm_name": result.farm_name,
            "crop_name": result.crop_name,
            "crop_variety": result.crop_variety,
            "latitude": result.latitude,
            "longitude": result.longitude,
            "elevation": result.elevation,
            "planting_date": result.planting_date,
            "is_irrigation_applied": result.is_irrigation_applied,
            "irrigation_application": result.irrigation_application,
            "nitrogen_fertilizer_application": result.nitrogen_fertilizer_application,
            "question_statements": result.question_statements,
        }


# ---------------------------------------------------------------------------
# Interpreter Module
# ---------------------------------------------------------------------------

class InterpreterModule(dspy.Module):
    """
    Takes DSSAT simulation outputs and a list of farmer questions and
    produces farmer-friendly answers with unit conversions and explanations.

    Uses ChainOfThought to encourage explicit step-by-step reasoning,
    which improves unit conversion accuracy and answer quality.
    """

    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(InterpreterSignature)

    def forward(
        self,
        definitions_bank: str,
        questions_bank: str,
        simulation_outputs_json: dict | str,
        question_statements: list[str],
    ) -> dict:
        """
        Args:
            definitions_bank: Contents of bank_of_definitions.txt
            questions_bank: Contents of bank_of_questions.txt
            simulation_outputs_json: Dict or JSON string of simulation results
            question_statements: List of question strings from the farmer

        Returns:
            Dict keyed by 'question_1', 'question_2', ... with answer dicts.
        """
        # Normalise simulation outputs to a JSON string for the prompt
        if isinstance(simulation_outputs_json, dict):
            sim_json_str = json.dumps(simulation_outputs_json, indent=2)
        else:
            sim_json_str = simulation_outputs_json

        # Format questions as a numbered list string
        questions_str = "\n".join(
            f"Question {i}: {q}" for i, q in enumerate(question_statements, 1)
        )

        result = self.predict(
            definitions_bank=definitions_bank,
            questions_bank=questions_bank,
            simulation_outputs_json=sim_json_str,
            farmer_question_statements=questions_str,
        )

        answers = result.answers

        # 'Defensive normalisation': just to ensure all expected question keys exist
        for i, q in enumerate(question_statements, 1):
            key = f"question_{i}"
            if key not in answers:
                answers[key] = {
                    "question_statement": q,
                    "matched_question_found": "Not found",
                    "retrieved_answer": "Not found",
                    "answer_for_farmer": (
                        "Sorry, your question could not be answered at this time."
                    ),
                }

        return answers


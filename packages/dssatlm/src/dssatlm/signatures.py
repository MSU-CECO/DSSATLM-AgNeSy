"""
DSPy Signatures replace the LangChain prompt templates (from v1).

Each Signature declares:
  - The task description (docstring) -> replaces the "YOUR ROLE / INSTRUCTIONS" block
  - Input fields  -> replaces template variables
  - Output fields -> replaces PydanticOutputParser schema

Two signatures are defined:
  1. ParserSignature      -- converts a free-text farmer query into structured JSON
  2. InterpreterSignature -- converts simulation outputs + farmer questions into answers
"""

import dspy


# ---------------------------------------------------------------------------
# Parser Signature
# ---------------------------------------------------------------------------

class ParserSignature(dspy.Signature):
    """
    You are a virtual QA assistant capable of explaining the characteristics
    of farming components and extracting the underlying question asked by farmers.

    Convert the farmer's free-text input into a structured JSON object.
    Use -99 (int) for missing numeric fields and "-99" (string) for missing
    string fields. Any "Not Applicable" or similar information must be set
    to the appropriate sentinel.

    Rules:
    - planting_date must be in "yyyy-mm-dd" format.
    - latitude and longitude must be decimal (e.g. 42.263, -85.648).
    - If irrigation is not mentioned, set is_irrigation_applied to "no".
    - If irrigation is mentioned but dates/amounts are missing, set
      is_irrigation_applied to "no" and irrigation_application to [].
    - If nitrogen fertilizer dates/amounts are not mentioned, set
      nitrogen_fertilizer_application to [].
    - Extract all distinct questions the farmer is asking and list them
      in question_statements.
    - If you do not understand the input or this task, return the following
      default structure exactly:
        farm_name: "-99", crop_name: "-99", crop_variety: "-99",
        latitude: -99, longitude: -99, elevation: -99,
        planting_date: "-99", is_irrigation_applied: "-99",
        irrigation_application: [], nitrogen_fertilizer_application: [],
        question_statements: []
    """

    farmer_input_query: str = dspy.InputField(
        desc="Free-text input from the farmer describing their farm context and question(s)."
    )

    farm_name: str = dspy.OutputField(
        desc='Name of the farm. Use "-99" if not mentioned.'
    )
    crop_name: str = dspy.OutputField(
        desc='Name of the crop (e.g. "Maize", "Soybean", "Wheat"). Use "-99" if not mentioned.'
    )
    crop_variety: str = dspy.OutputField(
        desc='Crop variety / cultivar name. Use "-99" if not mentioned.'
    )
    latitude: float = dspy.OutputField(
        desc="Decimal latitude of the farm. Use -99 if not mentioned."
    )
    longitude: float = dspy.OutputField(
        desc="Decimal longitude of the farm. Use -99 if not mentioned."
    )
    elevation: float = dspy.OutputField(
        desc="Elevation in metres. Use -99 if not mentioned."
    )
    planting_date: str = dspy.OutputField(
        desc='Planting date in "yyyy-mm-dd" format. Use "-99" if not mentioned.'
    )
    is_irrigation_applied: str = dspy.OutputField(
        desc='"yes" if irrigation was applied and complete date+amount data is present, otherwise "no".'
    )
    irrigation_application: list = dspy.OutputField(
        desc='List of [date_str, amount_mm] pairs, e.g. [["2023-06-01", 25.0]]. Empty list if none.'
    )
    nitrogen_fertilizer_application: list = dspy.OutputField(
        desc='List of [date_str, amount_kg_ha] pairs for nitrogen fertilizer. Empty list if none.'
    )
    phosphorus_fertilizer_application: list = dspy.OutputField(
        desc='List of [date_str, amount_kg_ha] pairs for phosphorus fertilizer. Empty list if none.'
    )
    potassium_fertilizer_application: list = dspy.OutputField(
        desc='List of [date_str, amount_kg_ha] pairs for potassium fertilizer. Empty list if none.'
    )
    question_statements: list = dspy.OutputField(
        desc="List of distinct question strings extracted from the farmer's input."
    )


# ---------------------------------------------------------------------------
# Interpreter Signature
# ---------------------------------------------------------------------------

class InterpreterSignature(dspy.Signature):
    """
    You are a virtual QA assistant capable of simplifying complex agricultural
    concepts for farmers.

    For EACH question Q_i in farmer_question_statements, produce a structured
    answer by doing the following three steps:

    Step 1 - matched_question_found:
        Match Q_i to the closest question from the questions_bank.

    Step 2 - retrieved_answer:
        Locate the key-value pair in simulation_outputs_json that corresponds
        to the matched question. Format it as: "The <key> is <value>".

    Step 3 - answer_for_farmer:
        Before generating the answer, convert any non-(-99) value from the
        international metric system to the US Imperial system:
          - mm -> inches: 1 mm = 0.3937 inches
            Example: "Season water drainage (mm)": 71
                  -> "Season water drainage (inches)": 27.95
          - kg/ha or any variant like kg[dm]/ha -> pounds per acre:
            1 kg[dm]/ha = 0.89 lbs[dm]/ac
        Always include BOTH the original metric value and the converted
        Imperial value in the answer.

        Then write a clear and detailed answer by combining the retrieved
        key-value pair with its corresponding definition from definitions_bank.
        Use simple, relatable terms with practical examples suited for
        American farmers. Indicate how they might use this information and
        where they should get additional help if necessary.

        The answer_for_farmer must end with this exact form:
        "<a paragraph applying the above instructions to answer the question>.
        This was the reasoning I used to answer your question:
        <a paragraph describing your chain of thought>."

    Special cases for answer_for_farmer:
    - If the extracted value is -99, inform the farmer that the simulator
      cannot provide an answer for that question at this time.
    - If simulation_outputs_json contains "simulation_results": "impossible",
      inform the farmer that the simulator cannot solve the problem due to
      missing or unrealistic inputs, and that they must provide at minimum:
      farm name, latitude, longitude, elevation, crop name, planting date,
      and irrigation details.
    - If you are unable to answer based on the inputs, state this clearly
      and stop.

    Return one entry per question, keyed as question_1, question_2, etc.
    Do not include any preamble, <think> blocks, or extra text outside
    the structured output.
    """

    definitions_bank: str = dspy.InputField(
        desc="Bank of definitions for each DSSAT simulation output variable."
    )
    questions_bank: str = dspy.InputField(
        desc="Bank of pre-written questions, one per DSSAT output variable."
    )
    simulation_outputs_json: str = dspy.InputField(
        desc="JSON string of DSSAT simulation results (the technical farming management outputs)."
    )
    farmer_question_statements: str = dspy.InputField(
        desc=(
            "Numbered list of the farmer's questions, formatted as:\n"
            "Question 1: <question text>\n"
            "Question 2: <question text>\n..."
        )
    )

    answers: dict = dspy.OutputField(
        desc=(
            "Dict keyed by 'question_1', 'question_2', ... (one key per farmer question). "
            "Each value is a dict with exactly these keys: "
            "question_statement, matched_question_found, retrieved_answer, answer_for_farmer."
        )
    )


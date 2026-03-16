"""
smoke_test.py
Live end-to-end test for DSSATLMPipeline.

Run from the repo root:
    uv run --package dssatlm python smoke_test.py

Requires:
    OPENROUTER_API_KEY and WANDB_API_KEY set in your environment.
"""

import os
import json
from dssatlm.pipeline import DSSATLMPipeline

# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

FARMER_QUERY = """
My farm is called KBS Farm. It is located at latitude 42.263 and longitude -85.648,
at an elevation of 288 meters. I planted Maize (variety: MZ GREAT LAKES 582 KBS)
on May 1st, 2023. I did not apply any irrigation or fertilizer.

I have two questions:
1. What would be my crop yield at maturity?
2. When would I be able to harvest?
"""

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("DSSATLM Live Smoke Test")
    print("=" * 60)

    # Verify keys are set before instantiating
    for key in ["OPENROUTER_API_KEY", "WANDB_API_KEY"]:
        val = os.environ.get(key, "")
        masked = val[:6] + "..." if len(val) > 6 else "(not set)"
        print(f"  {key}: {masked}")
    print()

    # Step 1 — instantiate
    print("Instantiating pipeline...")
    pipeline = DSSATLMPipeline(
        parser_model_id="gpt-4o",
        interpreter_model_id="gpt-4o",
    )
    print(f"  {pipeline}")
    print()

    # Step 2 — run
    print("Running answer_query()...")
    print("-" * 60)
    outputs = pipeline.answer_query(FARMER_QUERY)
    print("-" * 60)
    print()

    # Step 3 — print results
    if not outputs:
        print("Pipeline returned empty outputs — check logs above for errors.")
        logs = pipeline.get_logs()
        print("\nExecution errors:")
        for k, v in logs["execution_errors"].items():
            if v:
                print(f"  {k}: {v}")
    else:
        print(f"Got {len(outputs)} answer(s):\n")
        for q_key, q_data in outputs.items():
            print(f"  [{q_key}]")
            print(f"  Question   : {q_data.get('question_statement', 'N/A')}")
            print(f"  Matched Q  : {q_data.get('matched_question_found', 'N/A')}")
            print(f"  Retrieved  : {q_data.get('retrieved_answer', 'N/A')}")
            print(f"  Expert ans : {q_data.get('expert_like_answer', 'N/A')}")
            print()
            print(f"  Answer for farmer:")
            print(f"  {q_data.get('answer_for_farmer', 'N/A')}")
            print()
            print("-" * 60)

    # Step 4 — print full logs summary
    logs = pipeline.get_logs()
    print()
    print("Pipeline summary:")
    print(f"  pipeline_ran_successfully : {logs['pipeline_ran_successfully']}")
    print(f"  simulation_is_possible    : {logs['simulation_is_possible']}")
    print(f"  simulation_is_successful  : {logs['simulation_is_successful']}")
    print(f"  questions_parsed          : {logs['question_statements_parsed']}")



import json
from datetime import datetime, timezone
from pathlib import Path

from graph import build_graph
from state import AgentState

# -- File paths ----------------------------------------------------------------
BASE_DIR    = Path(__file__).parent
INPUT_FILE  = BASE_DIR / "inputs"  / "input.json"
OUTPUT_FILE = BASE_DIR / "outputs" / "output.json"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Output Writer
# =============================================================================

def write_output(results: list[dict]) -> None:
    """Serializes and writes the final results to outputs/output.json."""
    output = {
        "totalQuestions": len(results),
        "processedAt":    datetime.now(timezone.utc).isoformat(),
        "results":        results,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n Output saved to: {OUTPUT_FILE}")



# Entry Point


def main() -> None:
    # -- Load questions --------------------------------------------------------
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        questions: list[dict] = json.load(f)

    if not questions:
        raise ValueError("input.json is empty. Please add FAQ questions.")

    print(f"\n{'='*60}")
    print(f"  FAQ Reflection Agent  |  {len(questions)} questions")
    print(f"{'='*60}\n")

    # -- Build and run graph ---------------------------------------------------
    app = build_graph()

    first_q = questions[0]
    initial_state: AgentState = {
        "questions":        questions,
        "current_index":    0,
        "current_question": first_q["Question"],
        "faq_number":       first_q["faqQuestion"],
        "current_answer":   "",
        "iteration_count":  0,
        "is_valid":         False,
        "remarks":          "",
        "results":          [],
    }

    final_state = app.invoke(initial_state)

    # -- Write output ----------------------------------------------------------
    write_output(final_state["results"])

    # -- Summary ---------------------------------------------------------------
    answered  = sum(1 for r in final_state["results"] if r["status"] == "answered")
    no_answer = len(final_state["results"]) - answered

    print(f"\n{'-'*60}")
    print(f"  Summary: {answered} answered  |  {no_answer} no_answer")
    print(f"{'-'*60}\n")


if __name__ == "__main__":
    main()
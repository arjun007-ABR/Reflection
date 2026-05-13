

from langgraph.graph import END, StateGraph

from nodes.generation import generate_answer
from nodes.validator   import validate_answer
from state             import AgentState, FAQResult

MAX_RETRIES = 3



# Node 3 — Save Result

def save_result(state: AgentState) -> AgentState:
    
    if state["is_valid"]:
        status  = "answered"
        answer  = state["current_answer"]
        checked = True
        remarks = state["remarks"]
    else:
        status  = "no_answer"
        answer  = "-"
        checked = False
        remarks = "No valid answer after maximum retries."

    entry = FAQResult(
        faqQuestion    = state["faq_number"],
        Question       = state["current_question"],
        iterationCount = state["iteration_count"],
        status         = status,
        Answer         = answer,
        checkedAnswer  = checked,
        remarks        = remarks,
    )

    print(
        f"  [Save]   FAQ {state['faq_number']} → "
        f"status='{status}' after {state['iteration_count']} attempt(s).\n"
    )

    updated_results = state["results"] + [entry.model_dump()]
    next_index      = state["current_index"] + 1

    # Pre-load the next question into state (if any remain)
    questions = state["questions"]
    if next_index < len(questions):
        next_q      = questions[next_index]
        next_text   = next_q["Question"]
        next_faq_no = next_q["faqQuestion"]
    else:
        next_text   = ""
        next_faq_no = -1

    return {
        **state,
        "results":          updated_results,
        "current_index":    next_index,
        "current_question": next_text,
        "faq_number":       next_faq_no,
        "current_answer":   "",
        "iteration_count":  0,
        "is_valid":         False,
        "remarks":          "",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Conditional Routing Functions
# ═══════════════════════════════════════════════════════════════════════════════

def route_after_validation(state: AgentState) -> str:
    """
    Routes after validate_answer:
      - 'save_result'     → answer is valid
      - 'generate_answer' → invalid but retries still available
      - 'save_result'     → invalid and max retries exhausted
    """
    if state["is_valid"]:
        return "save_result"

    if state["iteration_count"] < MAX_RETRIES:
        print(
            f"  [Router] Invalid. Retrying "
            f"(attempt {state['iteration_count'] + 1}/{MAX_RETRIES})..."
        )
        return "generate_answer"

    print(f"  [Router] Max retries ({MAX_RETRIES}) reached. Saving as no_answer.")
    return "save_result"


def route_after_save(state: AgentState) -> str:
    """
    Routes after save_result:
      - 'generate_answer' → more questions remain
      - END               → all questions processed
    """
    if state["current_index"] < len(state["questions"]):
        return "generate_answer"
    return END


# ═══════════════════════════════════════════════════════════════════════════════
# Graph Builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph() -> StateGraph:
    """Constructs and compiles the reflection-agent StateGraph."""

    graph = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("validate_answer", validate_answer)
    graph.add_node("save_result",     save_result)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.set_entry_point("generate_answer")

    # ── Fixed edge ────────────────────────────────────────────────────────────
    graph.add_edge("generate_answer", "validate_answer")

    # ── Conditional edges ─────────────────────────────────────────────────────
    graph.add_conditional_edges(
        "validate_answer",
        route_after_validation,
        {
            "generate_answer": "generate_answer",  # retry loop
            "save_result":     "save_result",       # valid or exhausted
        },
    )

    graph.add_conditional_edges(
        "save_result",
        route_after_save,
        {
            "generate_answer": "generate_answer",  # next question
            END:               END,                 # all done
        },
    )

    return graph.compile()
from graph import build_graph

app = build_graph()

mermaid_text = app.get_graph().draw_mermaid_png()

with open("graph.png", "wb") as f:
    f.write(mermaid_text)

print("Mermaid diagram saved as graph.png")
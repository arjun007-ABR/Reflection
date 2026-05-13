
from llm import generator_llm
from state import AgentState, GeneratedAnswer


def generate_answer(state: AgentState) -> AgentState:

    question  = state["current_question"]
    iteration = state["iteration_count"] + 1

    print(f"  [Ollama] Generating answer (attempt {iteration}) for: '{question}'")

    prompt = (
        f"You are an expert technical writer.\n"
        f"Provide a clear, accurate, and concise answer (2-4 sentences) "
        f"to this FAQ question:\n\n"
        f"Question: {question}"
    )

    result: GeneratedAnswer = generator_llm.invoke(prompt)

    return {
        **state,
        "current_answer":  result.answer,
        "iteration_count": iteration,
    }
# """
# nodes/validator.py
# ==================
# Node 2 — Validate Answer (Groq / LLaMA)

# Calls the Groq LLM to assess whether the generated answer is
# accurate, relevant, and complete. Sets is_valid and remarks
# in state so the router can decide whether to retry or save.
# """

# from llm import validator_llm
# from state import AgentState, ValidationResult


# def validate_answer(state: AgentState) -> AgentState:
#     """
#     Validator node: prompts Groq / LLaMA to review the current answer.

#     Parameters
#     ----------
#     state : AgentState
#         Current graph state containing the question and the generated answer.

#     Returns
#     -------
#     AgentState
#         Updated state with 'is_valid' flag and 'remarks' from the validator.
#     """
#     question = state["current_question"]
#     answer   = state["current_answer"]

#     print(f"  [Groq]   Validating answer for: '{question}'")

#     prompt = (
#         f"You are a strict quality-assurance reviewer.\n"
#         f"if the answer that you are validating consist of  more than 10 words please reject it"
#         f"Evaluate whether the following answer correctly and completely "
#         f"addresses the FAQ question.\n\n"
#         f"Question : {question}\n"
#         f"Answer   : {answer}\n\n"
#         f"Return is_valid=true only if the answer is accurate, relevant, "
#         f"and at least 1 sentence long. "
#         f"Provide a brief remark explaining your decision."
#     )

#     result: ValidationResult = validator_llm.invoke(prompt)

#     print(f"  [Groq]   Valid={result.is_valid} | Remarks: {result.remarks}")

#     return {
#         **state,
#         "is_valid": result.is_valid,
#         "remarks":  result.remarks,
#     }













from llm import validator_llm
from state import AgentState, ValidationResult


def validate_answer(state: AgentState) -> AgentState:
   
    question = state["current_question"]
    answer   = state["current_answer"]

    print(f"  [Groq]   Validating answer for: '{question}'")

    prompt = (
        f"You are a factual answer reviewer. Your ONLY job is to check "
        f"whether the answer correctly addresses the question.\n\n"
        f"Question : {question}\n"
        f"Answer   : {answer}\n\n"
        f"Validation rules — apply ONLY these, nothing else:\n"
        f"  1. The answer must be factually correct.\n"
        f"  2. The answer must directly address what the question is asking.\n"
        f"  3. The answer must be at least 1 complete sentence.\n\n"
        f"DO NOT penalise for:\n"
        f"  - Length (short or long answers are both fine)\n"
        f"  - Writing style or tone\n"
        f"  - Word count\n"
        f"  - Extra context or explanation\n\n"
        f"Set is_valid=true if all 3 rules pass. "
        f"Set is_valid=false ONLY if the answer is factually wrong or does not "
        f"address the question at all.\n"
        f"Provide a single concise remark explaining your decision."
    )

    result: ValidationResult = validator_llm.invoke(prompt)

    print(f"  [Groq]   Valid={result.is_valid} | Remarks: {result.remarks}")

    return {
        **state,
        "is_valid": result.is_valid,
        "remarks":  result.remarks,
    }
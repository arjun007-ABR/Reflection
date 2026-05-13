import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from state import GeneratedAnswer, ValidationResult

# ── Load environment variables 
load_dotenv()

GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

if not GROQ_API_KEY:
    raise EnvironmentError("Missing GROQ_API_KEY. Please set it in .env")


# ── Node 1 — Generator (Ollama, local, no API key needed) ─────────────────────
_ollama_llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.3,
)
generator_llm = _ollama_llm.with_structured_output(GeneratedAnswer)


# ── Node 2 — Validator (Groq / LLaMA) ────────────────────────────────────────
_groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY,
)
validator_llm = _groq_llm.with_structured_output(ValidationResult)
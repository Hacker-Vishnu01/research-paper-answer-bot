"""
LLM configuration for the Research Paper Answer Bot.

Uses Ollama for free, local answer generation without requiring
an OpenAI API key.
"""

from langchain_ollama import ChatOllama


OLLAMA_MODEL = "llama3.2:3b"


def get_llm(
    model_name: str = OLLAMA_MODEL,
    temperature: float = 0.0,
) -> ChatOllama:
    """
    Create and return the local Ollama chat model.

    Args:
        model_name:
            Ollama model name.

        temperature:
            Generation temperature. 0.0 provides more deterministic
            answers, which is preferred for grounded academic QA.

    Returns:
        Configured ChatOllama instance.
    """

    if not model_name or not model_name.strip():
        raise ValueError("model_name cannot be empty.")

    if temperature < 0:
        raise ValueError("temperature cannot be negative.")

    return ChatOllama(
        model=model_name,
        temperature=temperature,
    )
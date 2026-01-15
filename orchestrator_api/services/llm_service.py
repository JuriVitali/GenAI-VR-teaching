import os
from langchain_ollama import ChatOllama

def get_llm_model(model: str, temperature: int):
    return ChatOllama(model=model, temperature=temperature)

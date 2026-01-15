import os
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

memory = InMemorySaver()

def get_llm_model(model: str, temperature: float):
    return ChatOllama(model=model, temperature=temperature)

def get_chat_agent(model_name: str, temperature: float, system_prompt: str):

    llm = get_llm_model(model=model_name, temperature=temperature)

    agent = create_agent(
        model=llm,
        tools=[],
        middleware=[
            SummarizationMiddleware(
                model=llm,          # Qua andrebbe usato un modello più leggero
                trigger=("tokens", 4000),
                keep=("messages", 20)
            )
        ],
        checkpointer=memory,
        state_modifier=system_prompt
    )

    return agent
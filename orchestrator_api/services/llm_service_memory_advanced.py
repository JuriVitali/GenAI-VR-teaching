import os
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage

def get_llm_model(model: str, temperature: float):
    return ChatOllama(model=model, temperature=temperature)

def get_chat_agent(model_name: str, temperature: float, system_prompt: str):
    """
    Creates a stateful graph with memory and auto-summarization.
    """
    # 1. Initialize Model
    llm = get_llm_model(model_name, temperature)
    
    # 2. Define the Logic
    def call_model(state: MessagesState):
        """Invoke the model with the current history."""
        # Ensure system prompt is always there if you want to enforce specific behavior
        messages = state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    def summarize_conversation(state: MessagesState):
        """Summarize the conversation if it gets too long."""
        summary_prompt = "Distill the above chat history into a single summary message. Include as many specific details as possible."
        
        # We get the history, excluding the very last latest user input to avoid summarizing the new query immediately
        conversation_history = state["messages"][:-1] 
        
        # Call LLM to generate summary
        summary_message = llm.invoke(
            conversation_history + [HumanMessage(content=summary_prompt)]
        )
        
        # Create a list of messages to delete (all except the last 2 usually, or all summarized ones)
        # Here we keep the summary and the very latest user message that triggered this run
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-1]]
        
        # Return the delete operations + the new summary (cast as SystemMessage for context)
        return {"messages": delete_messages + [SystemMessage(content=f"Summary of past conversation: {summary_message.content}")]}

    def should_summarize(state: MessagesState):
        """Check if we have more than 6 messages (3 turns)."""
        messages = state["messages"]
        if len(messages) > 6:
            return "summarize_conversation"
        return END

    # 3. Build the Graph
    workflow = StateGraph(MessagesState)
    
    # Add Nodes
    workflow.add_node("conversation", call_model)
    workflow.add_node("summarize_conversation", summarize_conversation)
    
    # Define Edges
    # Start -> Call Model
    workflow.add_edge(START, "conversation")
    
    # After Call Model -> Check if we need to summarize -> End or Summarize
    workflow.add_conditional_edges(
        "conversation",
        should_summarize,
    )
    
    # After Summarize -> End
    workflow.add_edge("summarize_conversation", END)

    # 4. Compile with Memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app
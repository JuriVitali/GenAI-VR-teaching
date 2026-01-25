from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Global in-memory storage: {session_id: [message1, message2, ...]}
chat_histories = {} 

def get_chat_history(session_id: str):
    return chat_histories.get(session_id, [])

def update_chat_history(session_id: str, human_query: str, ai_response: str):
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    
    # Add interactions
    chat_histories[session_id].append(HumanMessage(content=human_query))
    
    # DEEPSEEK TIP: Strip <think> tags before saving to memory to save context
    # Use regex to remove thinking blocks if present in the history
    import re
    clean_response = re.sub(r'<think>.*?</think>', '', ai_response, flags=re.DOTALL).strip()
    
    chat_histories[session_id].append(AIMessage(content=clean_response))
    
    # Optional: Limit history to last 20 turns (40 messages)
    if len(chat_histories[session_id]) > 40:
        chat_histories[session_id] = chat_histories[session_id][-40:]
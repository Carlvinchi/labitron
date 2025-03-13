####
## This the final work that was deployed to the cloud, a chatbot with memory and data persistance
####

import sys
import pickle
import streamlit as st
import ollama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from datetime import datetime
import uuid

#The chat history file
HISTORY_FILE = "chat_history.pkl"

#Method to save chat_history
def save_chat_history(history):
    with open(HISTORY_FILE, "wb") as f:
        pickle.dump(history, f)

#Method to load chat history
def load_chat_history():
    try:
        with open(HISTORY_FILE, "rb") as f:
            return pickle.load(f)
    
    except FileNotFoundError:
        return {}
    
#Method to clear chat history
def clear_chat_history():
    with open(HISTORY_FILE, "wb") as f:
        pickle.dump({}, f)

#Method to retrieve available models 
def get_models():
    models = ollama.list()

    if not models:
        print("No model available")
        sys.exit(1)

    models_list = []

    for model in models["models"]:
        
        models_list.append(model["model"])
    
    return models_list

#Method to initialize the LLM Agent with Memory
def initialize_chatbot():
    """Initialize the chatbot with LangGraph workflow"""
    ll = get_models()[0]
    model = ChatOllama(model=ll, temperature=0.3)
    workflow = StateGraph(state_schema=MessagesState)
     
    def call_model(state: MessagesState):
        system_prompt = (
            "You are a medical AI assistant who specializes in medical laboratory test analysis and report generation."
            "Your responses must be structured, concise, and medically accurate."
            "Compare test result value with the given test result range and flag result appropriately."
            "Follow standard medical and clinical reporting formats, and make objective interpretations."
            "Avoid making a diagnosis and always refer patients to specialist doctors for confirmation."
        )
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = model.invoke(messages)
        return {"messages": response.content}
    
    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

#Method that changes the LLM model based on user selection
def update_chatbot(llm_model):
    """Initialize the chatbot with LangGraph workflow"""
    
    model = ChatOllama(model=llm_model, temperature=0.3)
    workflow = StateGraph(state_schema=MessagesState)
    
    def call_model(state: MessagesState):
        system_prompt = (
            "You are a medical AI assistant who specializes in medical laboratory test analysis and report generation."
            "Your responses must be structured, concise, and medically accurate."
            "Flag values below the normal range as Low and values above the normal range as High."
            "Follow standard medical and clinical reporting formats, and make objective interpretations."
            "Avoid making a diagnosis and always refer patients to specialist doctors for confirmation."
        )
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = model.invoke(messages)
        return {"messages": response.content}
    
    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

#Method to initialize session variables
def initialize_session_state():
    """Initialize session state variables"""
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = get_models()[0]

    if "conversations" not in st.session_state:
        
        st.session_state.conversations = load_chat_history()

    if "current_chat_id" not in st.session_state:
        new_chat_id = create_new_chat()
        st.session_state.current_chat_id = new_chat_id
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = initialize_chatbot()

#Method to create new conversations
def create_new_chat():
    """Create a new chat and return its ID"""
    chat_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    st.session_state.conversations[chat_id] = {
        "messages": [],
        "timestamp": timestamp,
        "title": f"Chat {len(st.session_state.conversations) + 1}",
    }
    return chat_id

#Method for displaying chat history
def display_chat_history(chat_id):
    """Display the chat history for a specific chat"""
    if chat_id in st.session_state.conversations:
        
        for message in st.session_state.conversations[chat_id]["messages"]:
            
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.write(message.content)

            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.write(message.content)

            elif not isinstance(message, HumanMessage) and not isinstance(message, AIMessage):
                
                st.markdown(message, unsafe_allow_html=True)

#Method for updating the title of a chat
def update_chat_title(chat_id, first_message):
    """Update chat title based on first message"""
    if len(first_message) > 30:
        title = first_message[:30] + "..."
    else:
        title = first_message
    st.session_state.conversations[chat_id]["title"] = title

#Method to run the full app
def main():
    st.set_page_config(page_title="LABITRON", page_icon="🔬")
    st.header("MEDICAL LABORATORY ASSISTANT 🦜🔬🩺🔗")
    st.markdown("I am here to make your work easier in generating lab reports", unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for chat selection and new chat button
    with st.sidebar:
        st.selectbox("Select LLM: ", get_models(), key="selected_model")

        if st.button("Change Model"):
            
            st.session_state.chatbot = update_chatbot(st.session_state.selected_model)
            new_chat_id = create_new_chat()
            st.session_state.current_chat_id = new_chat_id
            st.rerun()

        st.divider()

        st.title("Conversations")
        
        # New Chat button
        if st.button("New Chat", key="new_chat", type="secondary"):
            new_chat_id = create_new_chat()
            st.session_state.current_chat_id = new_chat_id
            st.rerun()

        # Clear Chat button
        if st.button("Clear Conversations", key="clear", type="primary"):
            st.session_state.conversations = {}
            clear_chat_history()
            new_chat_id = create_new_chat()
            st.session_state.current_chat_id = new_chat_id
            
            st.rerun()
        
        st.divider()
        
        # Display all conversations
        for chat_id in reversed(st.session_state.conversations.keys()):
            chat_data = st.session_state.conversations[chat_id]
            if len(chat_data["messages"]) > 0:
                chat_title = chat_data["title"]
                timestamp = chat_data["timestamp"]
                
                # Create a unique key for each button
                button_key = f"chat_button_{chat_id}"
                
                # Style the current chat differently
                if chat_id == st.session_state.current_chat_id:
                    button_style = "primary"
                else:
                    button_style = "secondary"
                
                # Chat selection button with timestamp
                if st.button(
                    f"{chat_title}",
                    key=button_key,
                    type=button_style,
                    use_container_width=True
                ):
                    st.session_state.current_chat_id = chat_id
                    st.rerun()
    
    # Main chat interface
    current_chat = st.session_state.conversations[st.session_state.current_chat_id]
    st.title(current_chat["title"])
    
    # Display current chat history
    display_chat_history(st.session_state.current_chat_id)
    
    # Chat input
    if prompt := st.chat_input("Type your query..."):
        # Update chat title if this is the first message
        if not current_chat["messages"]:
            update_chat_title(st.session_state.current_chat_id, prompt)
        
        # Add user message
        user_message = HumanMessage(content=prompt)
        current_chat["messages"].append(user_message)
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get and display bot response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.invoke(
                    {"messages": [user_message]},
                    config={"configurable": {"thread_id": st.session_state.current_chat_id}}
                )
                
                ai_response = str(response["messages"][-1].content)
                message_placeholder.write(ai_response)
                
                # Add assistant response to history and save to local storage
                current_chat["messages"].append(AIMessage(content=ai_response))
                current_chat["messages"].append(f"response by {st.session_state.selected_model} model")
                save_chat_history(st.session_state.conversations)
        
        st.rerun()

if __name__ == "__main__":
    main()
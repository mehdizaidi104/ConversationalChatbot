import streamlit as st
import requests
import os

# Get the API URL from the environment variable we set in docker-compose
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

st.title("Linux Chatbot (API Version)")
st.write("Ask me anything about Linux commands!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if user_input := st.chat_input("How do I list files?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response_text = "Thinking..."
        message_placeholder = st.empty()
        message_placeholder.markdown(response_text + "...")
        
        try:
            # Call the backend API
            api_response = requests.post(API_URL, json={"text": user_input})
            api_response.raise_for_status() # Raise an error for bad responses (4xx, 5xx)
            
            response_text = api_response.json()["response_text"]
            message_placeholder.markdown(response_text)
            
        except requests.exceptions.RequestException as e:
            print(f"API Error: {e}")
            response_text = "Sorry, I'm having trouble connecting to my brain."
            message_placeholder.markdown(response_text)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})
import streamlit as st
import os
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from backend import app

# Page configuration
st.set_page_config(page_title="Global Career Advisor", layout="centered")
st.title("Global Career Guidance Engine")

# Initialize session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! Please upload your resume in the sidebar so I can analyze your background and map your global career opportunities."}]
if "resume_uploaded" not in st.session_state:
    st.session_state.resume_uploaded = False
    
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# Sidebar for file upload
with st.sidebar:
    st.header("Resume Upload")
    uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])
    
    if uploaded_file and not st.session_state.resume_uploaded:
        os.makedirs("data", exist_ok=True)
        file_path = os.path.join("data", uploaded_file.name).replace("\\", "/")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Initial trigger message
        initial_msg = f"Help me with my career based on my resume."
        inputs = {
            "messages": [HumanMessage(content=initial_msg)],
            "resume_path": file_path,
            "revision_count": 0
        }
        with st.spinner("Analyzing resume..."):
            for event in app.stream(inputs, config):
                pass
            
            # Grab the agent's response and update the UI
            state = app.get_state(config)
            last_msg = state.values["messages"][-1]
            
            # Save both the invisible prompt and the agent's reply to the chat history
            st.session_state.messages.append({"role": "user", "content": initial_msg})
            st.session_state.messages.append({"role": "assistant", "content": last_msg.content})
        
        st.session_state.resume_uploaded = True
        st.success("Resume processed!")
        
        # Force the UI to refresh so the chat appears <---
        st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input loop
if prompt := st.chat_input("Ask about your career..."):
    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            user_input = {"messages": [HumanMessage(content=prompt)]}
            final_content = ""
            for event in app.stream(user_input, config):
                # Update UI with the final state message
                state = app.get_state(config)
                last_msg = state.values["messages"][-1]
                if isinstance(last_msg, AIMessage):
                    final_content = last_msg.content
            
            st.markdown(final_content)
            st.session_state.messages.append({"role": "assistant", "content": final_content})
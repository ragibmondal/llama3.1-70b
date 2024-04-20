import os
import streamlit as st
from typing import Generator
from groq import Groq
from dotenv import load_dotenv
import yaml
import time
import pandas as pd

# Load environment variables from .env file
load_dotenv()

def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

st.set_page_config(page_icon="💬", layout="wide", page_title="Llama3 Chat App")

# Load configuration from config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load Groq API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY environment variable not found. Please set it in the .env file.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Define model details
model_options = list(config["models"].keys())
model_option = st.sidebar.selectbox("Select Model", model_options)
model_info = config["models"][model_option]

# Display model information
st.sidebar.header("Model Information")
st.sidebar.markdown(f"**Name:** {model_info['name']}")
st.sidebar.markdown(f"**Developer:** {model_info['developer']}")
st.sidebar.markdown(f"**Description:** {model_info['description']}")

max_tokens_range = model_info["tokens"]

# Adjust max_tokens slider
max_tokens = st.sidebar.slider(
    "Max Tokens:",
    min_value=512,
    max_value=max_tokens_range,
    value=min(config["default_max_tokens"], max_tokens_range),
    step=512,
    help=f"Adjust the maximum number of tokens (words) for the model's response. Max for selected model: {max_tokens_range}"
)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter your prompt here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar='👨‍💻'):
        st.markdown(prompt)

    # Fetch response from Groq API
    try:
        start_time = time.time()
        chat_completion = client.chat.completions.create(
            model=model_option,
            messages=[
                {
                    "role": m["role"],
                    "content": m["content"]
                }
                for m in st.session_state.messages
            ],
            max_tokens=max_tokens,
            stream=True
        )

        # Use the generator function with st.write_stream
        with st.chat_message("assistant", avatar="🤖"):
            chat_responses_generator = generate_chat_responses(chat_completion)
            full_response = st.write_stream(chat_responses_generator)

        end_time = time.time()
        response_time = end_time - start_time
        st.info(f"Response generated in {response_time:.2f} seconds.")
    except Exception as e:
        st.error(f"Error: {e}", icon="🚨")

    # Append the full response to session_state.messages
    if isinstance(full_response, str):
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})
    else:
        # Handle the case where full_response is not a string
        combined_response = "\n".join(str(item) for item in full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": combined_response})

# Add a clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []

# Add a download chat history button
if st.sidebar.button("Download Chat History"):
    chat_history = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages])
    st.download_button(
        label="Download Chat History",
        data=chat_history,
        file_name="chat_history.txt",
        mime="text/plain",
    )

# Add a prompt templates section
st.sidebar.header("Prompt Templates")
template_options = config["prompt_templates"].keys()
selected_template = st.sidebar.selectbox("Choose a prompt template", options=template_options)

if st.sidebar.button("Load Template"):
    prompt_template = config["prompt_templates"][selected_template]
    st.chat_input("", value=prompt_template, key="prompt_input")

# Add a feedback section
st.sidebar.header("Feedback")
if st.sidebar.button("Submit Feedback"):
    feedback = st.text_area("Enter your feedback here")
    if feedback:
        feedback_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "feedback": feedback
        }
        feedback_df = pd.DataFrame([feedback_data])
        feedback_df.to_csv("feedback.csv", mode="a", header=False, index=False)
        st.success("Thank you for your feedback!")

# Add a rate limiting feature
if "request_count" not in st.session_state:
    st.session_state.request_count = 0

if st.session_state.request_count >= config["rate_limit"]:
    st.warning(f"You have reached the maximum number of requests ({config['rate_limit']}) for the current session. Please wait or restart the app.")
else:
    st.session_state.request_count += 1

def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

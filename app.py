import os
import streamlit as st
from typing import Generator
from groq import Groq
from dotenv import load_dotenv
import yaml

# Load environment variables from .env file
load_dotenv()

def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

def load_config():
    """Load and validate the configuration from config.yaml"""
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        if "models" not in config:
            st.error("The 'models' section is missing from the config file.")
            st.stop()
        
        if "default_max_tokens" not in config:
            st.warning("'default_max_tokens' is not specified in the config. Using 1024 as default.")
            config["default_max_tokens"] = 1024
        
        if "prompt_templates" not in config:
            st.warning("No prompt templates found in the config.")
            config["prompt_templates"] = {}
        
        return config
    except FileNotFoundError:
        st.error("config.yaml file not found. Please ensure it exists in the same directory as this script.")
        st.stop()
    except yaml.YAMLError as e:
        st.error(f"Error reading config.yaml: {e}")
        st.stop()

# Set up the Streamlit page
st.set_page_config(page_icon="💬", layout="wide", page_title="llama-3.3-70b-versatile")

# Load configuration
config = load_config()

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
model_option = "llama-3.3-70b-versatile"
if model_option not in config["models"]:
    st.error(f"The model '{model_option}' is not defined in the config file.")
    st.stop()

model_info = config["models"][model_option]

# Display model information
st.sidebar.header("Model Information")
st.sidebar.markdown(f"**Name:** {model_info.get('name', 'N/A')}")
st.sidebar.markdown(f"**Developer:** {model_info.get('developer', 'N/A')}")
st.sidebar.markdown(f"**Description:** {model_info.get('description', 'N/A')}")

max_tokens_range = model_info.get("tokens", 1024)  # Default to 1024 if not specified

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
    except Exception as e:
        st.error(f"Error: {e}", icon="🚨")
    else:
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
template_options = list(config["prompt_templates"].keys())
if template_options:
    selected_template = st.sidebar.selectbox("Choose a prompt template", options=template_options)
    if st.sidebar.button("Load Template"):
        prompt_template = config["prompt_templates"][selected_template]
        st.session_state.messages.append({"role": "user", "content": prompt_template})
        st.experimental_rerun()
else:
    st.sidebar.info("No prompt templates available.")

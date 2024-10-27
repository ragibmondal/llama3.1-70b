import streamlit as st
import os
import base64
import openai
from typing import List, Optional
import tempfile
from audio_recorder_streamlit import audio_recorder
from dotenv import load_dotenv
import time
from datetime import datetime

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="AI Voice Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Discord-like UI
st.markdown("""
    <style>
        /* Global Styles */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        /* Main container */
        .main {
            background-color: #36393f;
            color: #dcddde;
            padding: 0 !important;
            max-width: 100% !important;
        }
        
        h1, h2, h3, h4, h5, h6, .markdown-text-container {
            color: #ffffff !important;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #2f3136;
        }
        
        .sidebar .sidebar-content {
            background-color: #2f3136;
        }
        
        /* Chat message containers */
        .message-container {
            background-color: #36393f;
            border-radius: 4px;
            padding: 1rem;
            margin: 0.5rem 0;
            border-bottom: 1px solid #42454a;
        }
        
        .user-message {
            background-color: #32353b;
        }
        
        .bot-message {
            background-color: #36393f;
        }
        
        /* Input area styling */
        .stTextArea textarea {
            background-color: #40444b;
            border: none;
            border-radius: 8px;
            color: #dcddde;
            font-size: 1rem;
            padding: 1rem;
        }
        
        .stTextArea textarea:focus {
            box-shadow: 0 0 0 2px #5865f2;
            border: none;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #5865f2;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: background-color 0.2s;
        }
        
        .stButton>button:hover {
            background-color: #4752c4;
        }
        
        /* Status indicators */
        .status-box {
            padding: 0.75rem;
            border-radius: 4px;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
            border: none;
        }
        
        .success-box {
            background-color: #3ba55c;
            color: white;
        }
        
        .error-box {
            background-color: #ed4245;
            color: white;
        }
        
        .info-box {
            background-color: #5865f2;
            color: white;
        }
        
        /* Audio recorder styling */
        .recorder-container {
            background-color: #40444b;
            border-radius: 4px;
            padding: 0.5rem;
            margin: 1rem 0;
        }
        
        /* Chat interface */
        .chat-interface {
            height: calc(100vh - 200px);
            overflow-y: auto;
            padding: 1rem;
            background-color: #36393f;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        
        /* Message timestamp */
        .message-timestamp {
            color: #72767d;
            font-size: 0.75rem;
            margin-bottom: 0.25rem;
        }
        
        /* User avatar */
        .user-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            margin-right: 1rem;
        }
        
        /* Input attachments */
        .input-attachments {
            background-color: #40444b;
            border-radius: 4px;
            padding: 0.5rem;
            margin-top: 0.5rem;
        }
        
        /* Custom checkbox */
        .stCheckbox {
            color: #dcddde;
        }
        
        /* Sidebar sections */
        .sidebar-section {
            background-color: #2f3136;
            padding: 1rem;
            border-radius: 4px;
            margin-bottom: 1rem;
        }
        
        /* Audio player */
        audio {
            width: 100%;
            margin: 0.5rem 0;
        }
        
        /* Progress bar */
        .stProgress > div > div {
            background-color: #5865f2;
        }
    </style>
""", unsafe_allow_html=True)

def initialize_lepton_client():
    api_token = os.getenv('LEPTON_API_TOKEN')
    if not api_token:
        raise ValueError("LEPTON_API_TOKEN not found in environment variables")
    return openai.OpenAI(
        base_url="https://llama3-1-405b.lepton.run/api/v1/",
        api_key=api_token
    )

def get_timestamp():
    return datetime.now().strftime("%H:%M")

def format_message(content, is_user=True, timestamp=None):
    if timestamp is None:
        timestamp = get_timestamp()
    
    avatar = "👤" if is_user else "🤖"
    bg_class = "user-message" if is_user else "bot-message"
    
    return f"""
    <div class="message-container {bg_class}">
        <div style="display: flex; align-items: start;">
            <div style="font-size: 1.5rem; margin-right: 0.5rem;">{avatar}</div>
            <div style="flex-grow: 1;">
                <div class="message-timestamp">{timestamp}</div>
                <div style="color: {'#ffffff' if is_user else '#dcddde'};">
                    {content}
                </div>
            </div>
        </div>
    </div>
    """

def render_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            st.markdown(
                format_message(
                    msg["content"],
                    msg["role"] == "user",
                    msg["timestamp"]
                ),
                unsafe_allow_html=True
            )

def process_audio_file(audio_file) -> str:
    audio_bytes = audio_file.read()
    return base64.b64encode(audio_bytes).decode()

def generate_response(client, 
                     prompt: str, 
                     audio_data: Optional[str] = None,
                     generate_audio: bool = False,
                     voice_preset: str = "jessica",
                     max_tokens: int = 128) -> tuple[str, List[str]]:
    
    messages = []
    if audio_data:
        messages.append({"role": "user", "content": [{"type": "audio", "data": audio_data}]})
    else:
        messages.append({"role": "user", "content": prompt})

    extra_body = {}
    if generate_audio:
        extra_body.update({
            "tts_audio_format": "mp3",
            "tts_audio_bitrate": 16,
            "require_audio": True,
            "tts_preset_id": voice_preset,
        })

    # Add progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    completion = client.chat.completions.create(
        model="llama3.1-405b",
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
        extra_body=extra_body if extra_body else None
    )

    full_response = ""
    audio_chunks = []
    
    for i, chunk in enumerate(completion):
        if not chunk.choices:
            continue
        
        content = chunk.choices[0].delta.content
        audio = getattr(chunk.choices[0], 'audio', [])
        
        if content:
            full_response += content
        if audio:
            audio_chunks.extend(audio)
            
        # Update progress
        progress = min(100, int((i + 1) / max_tokens * 100))
        progress_bar.progress(progress)
        status_text.text(f"Generating response... {progress}%")
    
    progress_bar.empty()
    status_text.empty()
    
    return full_response, audio_chunks

def save_audio(audio_chunks: List[str]) -> str:
    if not audio_chunks:
        return ""
    
    audio_data = b''.join([base64.b64decode(chunk) for chunk in audio_chunks])
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
        tmp_file.write(audio_data)
        return tmp_file.name

def render_sidebar():
    with st.sidebar:
        st.title("⚙️ Settings")
        
        with st.expander("🎤 Voice Settings", expanded=True):
            voice_preset = st.selectbox(
                "Voice Preset",
                ["jessica", "josh", "emma", "michael"],
                index=0
            )
        
        with st.expander("⚡ Model Settings", expanded=True):
            max_tokens = st.slider(
                "Response Length",
                min_value=50,
                max_value=500,
                value=128,
                step=10
            )
        
        with st.expander("ℹ️ About", expanded=True):
            st.markdown("""
            ### AI Voice Chat
            
            Talk naturally with AI using:
            - 🎤 Voice Input
            - ⌨️ Text Messages
            - 🔊 Voice Responses
            
            Made with Lepton AI
            """)
        
        return voice_preset, max_tokens

def main():
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    voice_preset, max_tokens = render_sidebar()
    
    # Main chat container
    st.markdown("""
        <div style="padding: 1rem; background-color: #36393f; border-bottom: 1px solid #42454a;">
            <h1 style="margin: 0; color: white;">🤖 AI Voice Chat</h1>
        </div>
    """, unsafe_allow_html=True)
    
    try:
        client = initialize_lepton_client()
    except Exception as e:
        st.markdown(f"""
            <div class="status-box error-box">
                ❌ Connection Error: {str(e)}<br>
                Check your LEPTON_API_TOKEN in .env file
            </div>
        """, unsafe_allow_html=True)
        return

    # Chat interface
    chat_container = st.container()
    with chat_container:
        render_chat_history()

    # Input section
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    input_container = st.container()
    
    with input_container:
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_area("Message", height=100, placeholder="Type a message...")
        
        with col2:
            st.markdown("<div style='height: 35px;'></div>", unsafe_allow_html=True)
            audio_record = audio_recorder(
                pause_threshold=2.0,
                sample_rate=44100,
                text="",
                recording_color="#5865f2",
                neutral_color="#72767d",
                icon_name="microphone",
                icon_size="2x"
            )
        
        col3, col4, col5 = st.columns([2, 2, 1])
        with col3:
            generate_audio = st.checkbox("Enable voice response", value=True)
        with col5:
            if st.button("Send", key="send_message"):
                if user_input:
                    # Add user message to chat
                    timestamp = get_timestamp()
                    st.session_state.messages.append({
                        "role": "user",
                        "content": user_input,
                        "timestamp": timestamp
                    })
                    
                    with st.spinner("🤖 AI is thinking..."):
                        response_text, audio_chunks = generate_response(
                            client, 
                            user_input,
                            generate_audio=generate_audio,
                            voice_preset=voice_preset,
                            max_tokens=max_tokens
                        )
                        
                        # Add AI response to chat
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_text,
                            "timestamp": get_timestamp()
                        })
                        
                        if generate_audio and audio_chunks:
                            audio_path = save_audio(audio_chunks)
                            if audio_path:
                                with open(audio_path, 'rb') as audio_file:
                                    st.audio(audio_file.read(), format='audio/mp3')
                                os.unlink(audio_path)
                    
                    # Clear input
                    st.experimental_rerun()
        
        if audio_record:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_record)
                
                # Add recording to chat
                timestamp = get_timestamp()
                st.session_state.messages.append({
                    "role": "user",
                    "content": "🎤 Voice message",
                    "timestamp": timestamp
                })
                
                with st.spinner("🎯 Processing voice..."):
                    with open(tmp_file.name, 'rb') as audio_file:
                        audio_data = process_audio_file(audio_file)
                    
                    response_text, audio_chunks = generate_response(
                        client,
                        "",
                        audio_data=audio_data,
                        generate_audio=generate_audio,
                        voice_preset=voice_preset,
                        max_tokens=max_tokens
                    )
                    
                    # Add AI response to chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "timestamp": get_timestamp()
                    })
                    
                    if generate_audio and audio_chunks:
                        audio_path = save_audio(audio_chunks)
                        if audio_path:
                            with open(audio_path, 'rb') as audio_file:
                                st.audio(audio_file.read(), format='audio/mp3')
                            os.unlink(audio_path)
                    
                    # Clear input
                    st.experimental_rerun()
                
                os.unlink(tmp_file.name)

    # Chat history section
    st.markdown("---")
    st.markdown("### 💬 Chat History")
    st.info("Chat history is not yet implemented. Coming soon!")

if __name__ == "__main__":
    main()

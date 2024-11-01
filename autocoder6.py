# AI Code Enhancer Streamlit App with Cloud Optimizations
import os
import ast
import streamlit as st
import streamlit_ace as st_ace
from streamlit_option_menu import option_menu
from langchain.memory import ConversationBufferMemory
import logging
import tempfile
from pathlib import Path
import time
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
try:
    client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")))
except Exception as e:
    st.error(f"Failed to initialize OpenAI client: {e}")
    st.stop()

# Streamlit Page Configuration
st.set_page_config(
    page_title="AI Code Enhancer",
    page_icon=":robot_face:",
    layout="wide",
)

@dataclass
class AppState:
    """Application state management"""
    code_blocks: Dict[str, str] = field(default_factory=dict)
    optimization_history: List[Dict] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=lambda: {
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 2000
    })
    session_start: datetime = field(default_factory=datetime.now)

    def is_session_expired(self, timeout_minutes: int = 30) -> bool:
        return datetime.now() - self.session_start > timedelta(minutes=timeout_minutes)

def initialize_session_state():
    """Initialize session state with defaults"""
    if 'app_state' not in st.session_state:
        st.session_state.app_state = AppState()

def optimize_code(code: str) -> Optional[str]:
    """Optimize code using OpenAI API"""
    try:
        response = client.chat.completions.create(
            model=st.session_state.app_state.settings["model"],
            messages=[
                {"role": "system", "content": "You are a Python optimization expert."},
                {"role": "user", "content": f"Optimize this code:\n{code}"}
            ],
            temperature=st.session_state.app_state.settings["temperature"],
            max_tokens=st.session_state.app_state.settings["max_tokens"]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Optimization error: {e}")
        logger.error(f"Optimization failed: {e}")
        return None

def render_code_editor():
    """Render code editor component"""
    return st_ace(
        value="# Enter your Python code here",
        language="python",
        theme="monokai",
        height=300,
        key="code_editor"
    )

def main():
    initialize_session_state()
    
    if st.session_state.app_state.is_session_expired():
        st.warning("Session expired. Please refresh the page.")
        st.stop()

    st.title("AI Code Enhancer")
    
    # Navigation
    selected = option_menu(
        "Main Menu",
        ["Editor", "Settings", "History"],
        icons=["code", "gear", "clock-history"],
        orientation="horizontal"
    )
    
    if selected == "Editor":
        code = render_code_editor()
        
        if st.button("Optimize"):
            with st.spinner("Optimizing code..."):
                if optimized := optimize_code(code):
                    st.session_state.app_state.optimization_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "original": code,
                        "optimized": optimized
                    })
                    st.success("Code optimized successfully!")
                    st.code(optimized, language="python")
                    
    elif selected == "Settings":
        st.subheader("Settings")
        st.session_state.app_state.settings["model"] = st.selectbox(
            "Model",
            ["gpt-4", "gpt-3.5-turbo"]
        )
        st.session_state.app_state.settings["temperature"] = st.slider(
            "Temperature",
            0.0, 1.0, 0.7
        )
        
    elif selected == "History":
        st.subheader("Optimization History")
        for entry in reversed(st.session_state.app_state.optimization_history):
            with st.expander(f"Optimization at {entry['timestamp']}"):
                st.code(entry["original"], language="python")
                st.code(entry["optimized"], language="python")

if __name__ == "__main__":
    main()

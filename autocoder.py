# Standard library imports
import ast
import io
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from difflib import unified_diff
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import streamlit as st
from openai import OpenAI
import plotly.graph_objects as go
from streamlit_ace import st_ace

# Initialize OpenAI client safely
try:
    client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")))
except Exception as e:
    st.error(f"Failed to initialize OpenAI client: {e}")
    st.stop()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Configure Streamlit only once
if 'page_config_set' not in st.session_state:
    st.set_page_config(page_title="AutocoderAI", layout="wide")
    st.session_state.page_config_set = True

@dataclass
class State:
    """Centralized state management with type validation"""
    script_sections: Dict[str, Any] = field(default_factory=dict)
    optimized_script: str = ""
    optimization_steps: List[str] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=lambda: {
        "openai_api_key": st.secrets.get("OPENAI_API_KEY", ""),
        "model": "gpt-3.5-turbo",
        "show_profiling": False,
        "max_retries": 3,
        "timeout": 30
    })
    error_log: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate state initialization"""
        if not isinstance(self.settings, dict):
            raise TypeError("Settings must be a dictionary")
        if not isinstance(self.optimization_steps, list):
            raise TypeError("Optimization steps must be a list")

# Initialize state
if "state" not in st.session_state:
    st.session_state.state = State()

def log_message(msg: str, level: str = "info") -> None:
    """Cloud-safe logging function"""
    getattr(st, level)(msg)
    st.session_state.state.logs.append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {level.upper()}: {msg}")

def validate_api_key(key: str) -> bool:
    """Validate OpenAI API key format"""
    if not key:
        return False
    return bool(re.match(r'^sk-[A-Za-z0-9]{48}$', key))

def optimize_code(code: str) -> Optional[str]:
    """Optimize code using OpenAI API"""
    try:
        response = client.chat.completions.create(
            model=st.session_state.state.settings["model"],
            messages=[
                {"role": "system", "content": "You are a Python optimization expert."},
                {"role": "user", "content": f"Optimize this code:\n{code}"}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        log_message(f"Optimization error: {e}", "error")
        return None

def main():
    st.title("AutocoderAI 🚀")
    
    # Input code
    code = st_ace(
        placeholder="Enter your Python code...",
        language="python",
        theme="monokai",
        height=300
    )
    
    if st.button("Optimize") and code:
        with st.spinner("Optimizing code..."):
            optimized = optimize_code(code)
            if optimized:
                st.session_state.state.optimized_script = optimized
                st.success("Code optimized successfully!")
                st.code(optimized, language="python")
            else:
                st.error("Failed to optimize code")

if __name__ == "__main__":
    main()

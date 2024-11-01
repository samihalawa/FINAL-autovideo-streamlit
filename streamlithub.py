import streamlit as st
import logging
from typing import Dict, Any, Optional
from openai import OpenAI
import os
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI Autocoder Hub", layout="wide")

# Initialize OpenAI client
try:
    client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")))
except Exception as e:
    st.error(f"Failed to initialize OpenAI client: {e}")
    st.stop()

def initialize_session_state():
    """Initialize session state with defaults"""
    defaults = {
        'code_storage': {},
        'history': [],
        'settings': {
            'model': 'gpt-4',
            'temperature': 0.7
        }
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def optimize_code(code: str) -> Optional[str]:
    """Optimize code using OpenAI"""
    try:
        response = client.chat.completions.create(
            model=st.session_state.settings['model'],
            messages=[
                {"role": "system", "content": "You are a Python optimization expert."},
                {"role": "user", "content": f"Optimize this code:\n{code}"}
            ],
            temperature=st.session_state.settings['temperature']
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Optimization failed: {e}")
        return None

def main():
    initialize_session_state()
    
    st.title("AI Autocoder Hub")
    
    tab1, tab2 = st.tabs(["Code Editor", "Settings"])
    
    with tab1:
        code = st.text_area("Enter your code:", height=300)
        if st.button("Optimize") and code:
            with st.spinner("Optimizing..."):
                if optimized := optimize_code(code):
                    st.session_state.history.append({
                        'timestamp': datetime.now().isoformat(),
                        'original': code,
                        'optimized': optimized
                    })
                    st.success("Code optimized!")
                    st.code(optimized, language="python")
    
    with tab2:
        st.session_state.settings['model'] = st.selectbox(
            "Model",
            ["gpt-4", "gpt-3.5-turbo"]
        )
        st.session_state.settings['temperature'] = st.slider(
            "Temperature",
            0.0, 1.0, 0.7
        )

if __name__ == "__main__":
    main()

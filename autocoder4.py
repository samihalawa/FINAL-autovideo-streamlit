# AutocoderAI - Enhanced Streamlit Application for Code Optimization

import os
import re
import ast
import json
import subprocess
import streamlit as st
from typing import List, Dict, Any, Optional, Union
import openai
from difflib import unified_diff
import networkx as nx
import matplotlib.pyplot as plt
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from streamlit_ace import st_ace
from streamlit_agraph import agraph, Node, Edge, Config
import plotly.graph_objects as go
import logging
import cProfile
import pstats
import io
from dotenv import load_dotenv
from streamlit_option_menu import option_menu
from aider import chat
from dataclasses import dataclass
from functools import cache, lru_cache
import asyncio
from typing import Tuple, Callable
import traceback
from contextlib import contextmanager
import psutil
from dataclasses import asdict

# Load environment variables
load_dotenv()

# Configuration
st.set_page_config(page_title="AutocoderAI", layout="wide")

# Constants
DEFAULT_OPENAI_MODEL = "gpt-4"
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MAX_RETRIES = 3
TIMEOUT_SECONDS = 30
SUPPORTED_MODELS = ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo-preview"]

# Type definitions
@dataclass
class Settings:
    openai_api_key: str
    model: str
    profiling_depth: int = 10
    show_profiling: bool = False
    timeout_seconds: int = TIMEOUT_SECONDS
    max_retries: int = MAX_RETRIES
    debug_mode: bool = False

@dataclass 
class AppState:
    script_sections: Dict[str, Any]
    optimized_script: str
    optimization_steps: List[str]
    current_step: int
    graph: nx.DiGraph
    optimization_history: List[str]
    settings: Settings
    profiling_results: Dict[str, str]
    error_log: List[str]
    logs: List[str]

# Initialize session state
if 'state' not in st.session_state:
    st.session_state.state = AppState(
        script_sections={},
        optimized_script="",
        optimization_steps=[],
        current_step=0,
        graph=nx.DiGraph(),
        optimization_history=[],
        settings=Settings(
            openai_api_key=DEFAULT_OPENAI_API_KEY,
            model=DEFAULT_OPENAI_MODEL
        ),
        profiling_results={},
        error_log=[],
        logs=[]
    )

state = st.session_state.state

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log(msg: str, level: str = "INFO") -> None:
    """Centralized logging function"""
    getattr(logging, level.lower())(msg)
    state.logs.append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {level}: {msg}")

# Validation functions
def validate_api_key(key: str) -> bool:
    """Validate OpenAI API key format"""
    return bool(re.match(r'^sk-[a-zA-Z0-9]{48}$', key))

def validate_input(script: str) -> bool:
    """Validate Python script input"""
    try:
        ast.parse(script)
        return True
    except SyntaxError:
        return False

# State management functions
@st.cache_data
def set_openai_creds(key: str, model: str) -> None:
    """Set OpenAI credentials with validation"""
    try:
        if not validate_api_key(key):
            raise ValueError("Invalid API key format")
        openai.api_key = key
        state.settings.openai_api_key = key
        state.settings.model = model
        openai.Model.list()
        log("OpenAI credentials set successfully.")
    except Exception as e:
        st.error(f"Error setting OpenAI credentials: {str(e)}")
        log(f"Error: {str(e)}", "ERROR")

# File operations
def save_final_code(code: str, filename: str = "optimized_script.py") -> None:
    """Save optimized code to file with error handling"""
    try:
        with open(filename, "w") as f:
            f.write(code)
        st.success(f"Saved optimized code to {filename}")
    except IOError as e:
        st.error(f"Error saving file: {str(e)}")
        log(f"File save error: {str(e)}", "ERROR")

# Performance profiling
@st.cache_resource
def profile_performance(script: str) -> str:
    """Profile code performance with error handling"""
    pr = cProfile.Profile()
    pr.enable()
    
    try:
        if not validate_input(script):
            raise SyntaxError("Invalid Python syntax")
        exec(script)
    except Exception as e:
        state.error_log.append(f"Error during profiling: {str(e)}")
        log(f"Profiling error: {str(e)}", "ERROR")
    finally:
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
    
    return s.getvalue()

# Script optimization
async def optimize_script(script_input: str, optimization_strategy: str) -> str:
    """Optimize script with progress tracking and error handling"""
    if not validate_input(script_input):
        raise ValueError("Invalid Python script")
        
    sections = extract_sections(script_input)
    state.script_sections = sections

    progress_bar = st.progress(0)
    status_text = st.empty()

    async with ThreadPoolExecutor() as executor:
        future_to_section = {
            executor.submit(optimize_section_async, name, content): name
            for name, content in sections.items() 
            if name != "package_installations"
        }
        
        total = len(future_to_section)
        for i, future in enumerate(as_completed(future_to_section)):
            section_name = future_to_section[future]
            try:
                _, optimized_content = await future.result()
                sections[section_name] = optimized_content
                progress = (i + 1) / total
                progress_bar.progress(progress)
                status_text.text(f"Optimizing {section_name} ({i+1}/{total})")
                
                if state.settings.show_profiling:
                    profile_result = profile_performance(optimized_content)
                    state.profiling_results[section_name] = profile_result
            except Exception as e:
                log(f"Error optimizing {section_name}: {str(e)}", "ERROR")

    return "\n\n".join(sections.values())

# UI Components
def display_profiling_results() -> None:
    """Display profiling results in expandable sections"""
    st.subheader("Performance Profiling Results")
    for section, result in state.profiling_results.items():
        with st.expander(f"Profiling for {section}"):
            st.text(result)

def display_error_log() -> None:
    """Display error log with proper formatting"""
    if state.error_log:
        st.subheader("Error Log")
        for error in state.error_log:
            st.error(error)

# Add resource management
@contextmanager
def manage_resources():
    """Context manager for application resources"""
    try:
        yield
    finally:
        # Cleanup resources
        if hasattr(st.session_state, 'state'):
            state = st.session_state.state
            state.profiling_results.clear()
            if hasattr(state, 'graph'):
                state.graph.clear()

# Update main function
def main() -> None:
    """Main application entry point with proper resource management"""
    with manage_resources(), error_boundary("main application"):
        st.title("AutocoderAI 🧑‍💻✨")
        
        # Initialize state if needed
        if 'state' not in st.session_state:
            initialize_state()
        
        # Sidebar navigation
        with st.sidebar:
            selected = option_menu(
                "Menu", 
                ["Optimize", "Settings", "Logs", "Debug"] if state.settings.debug_mode else ["Optimize", "Settings", "Logs"],
                icons=['magic', 'gear', 'journal-text', 'bug'],
                menu_icon="cast",
                default_index=0
            )
        
        # Route to appropriate view
        views = {
            "Optimize": display_optimization_interface,
            "Settings": display_settings,
            "Logs": display_logs,
            "Debug": display_debug_info if state.settings.debug_mode else lambda: None
        }
        
        if selected in views:
            views[selected]()

if __name__ == "__main__":
    main()

# Add missing imports and improve error handling
import asyncio
from typing import Tuple, Callable
import traceback
from contextlib import contextmanager

# Add error handling context manager
@contextmanager
def error_boundary(operation: str):
    """Context manager for consistent error handling"""
    try:
        yield
    except Exception as e:
        error_msg = f"Error during {operation}: {str(e)}\n{traceback.format_exc()}"
        log(error_msg, "ERROR")
        st.error(error_msg)
        state.error_log.append(error_msg)

# Add missing function definition
async def optimize_section_async(section_name: str, content: str) -> Tuple[str, str]:
    """Optimize a single section of code asynchronously with proper error handling"""
    async def _make_api_call():
        try:
            messages = [
                {"role": "system", "content": "You are a Python code optimization expert. Respond only with optimized code."},
                {"role": "user", "content": f"Optimize this code section:\n{content}"}
            ]
            
            async with asyncio.timeout(state.settings.timeout_seconds):
                response = await openai.ChatCompletion.acreate(
                    model=state.settings.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2000,
                    presence_penalty=0,
                    frequency_penalty=0
                )
                
                optimized_content = response.choices[0].message.content
                # Extract code from markdown if present
                if "```python" in optimized_content:
                    optimized_content = re.search(r"```python\n(.*?)\n```", optimized_content, re.DOTALL)
                    if optimized_content:
                        optimized_content = optimized_content.group(1)
                
                if validate_input(optimized_content):
                    return optimized_content
                raise ValueError("Generated code failed validation")
                
        except asyncio.TimeoutError:
            log(f"Timeout while optimizing section {section_name}", "WARNING")
            raise
        except Exception as e:
            log(f"API call failed: {str(e)}", "ERROR")
            raise

    for attempt in range(state.settings.max_retries):
        try:
            optimized_content = await _make_api_call()
            return section_name, optimized_content
        except Exception as e:
            if attempt == state.settings.max_retries - 1:
                log(f"All retries failed for section {section_name}: {str(e)}", "ERROR")
                return section_name, content
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    return section_name, content

# Add missing function
def extract_sections(script: str) -> Dict[str, str]:
    """Extract logical sections from the input script"""
    try:
        tree = ast.parse(script)
        sections = {}
        current_section = []
        current_section_name = "main"
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if current_section:
                    sections[current_section_name] = "\n".join(current_section)
                current_section = []
                current_section_name = node.name
                
            if hasattr(node, 'lineno'):
                lines = script.split('\n')[node.lineno-1:node.end_lineno]
                current_section.extend(lines)
                
        if current_section:
            sections[current_section_name] = "\n".join(current_section)
            
        return sections
    except Exception as e:
        log(f"Error extracting sections: {str(e)}", "ERROR")
        return {"main": script}

# Add missing UI functions
def display_settings():
    """Display and handle settings interface"""
    with st.form("settings_form"):
        api_key = st.text_input("OpenAI API Key", value=state.settings.openai_api_key, type="password")
        model = st.selectbox("Model", ["gpt-4", "gpt-3.5-turbo"], index=0)
        show_profiling = st.checkbox("Show Performance Profiling", value=state.settings.show_profiling)
        
        if st.form_submit_button("Save Settings"):
            with error_boundary("saving settings"):
                state.settings.show_profiling = show_profiling
                set_openai_creds(api_key, model)
                st.success("Settings saved successfully!")

def display_logs():
    """Display application logs"""
    st.subheader("Application Logs")
    for log_entry in state.logs:
        if "ERROR" in log_entry:
            st.error(log_entry)
        elif "WARNING" in log_entry:
            st.warning(log_entry)
        else:
            st.info(log_entry)

def display_optimization_interface():
    """Display main optimization interface"""
    with error_boundary("optimization interface"):
        script_input = st_ace(
            value="# Enter your Python code here",
            language="python",
            theme="monokai",
            height=300
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Optimize Code"):
                if not script_input.strip():
                    st.warning("Please enter some code to optimize")
                    return
                    
                with st.spinner("Optimizing code..."):
                    optimized_code = asyncio.run(optimize_script(script_input, "performance"))
                    state.optimized_script = optimized_code
                    
                    if state.settings.show_profiling:
                        display_profiling_results()
        
        with col2:
            if st.button("Save Optimized Code"):
                if state.optimized_script:
                    save_final_code(state.optimized_script)
                else:
                    st.warning("No optimized code to save")

# Add new validation and initialization functions
def initialize_state() -> None:
    """Initialize application state with defaults"""
    st.session_state.state = AppState(
        script_sections={},
        optimized_script="",
        optimization_steps=[],
        current_step=0,
        graph=nx.DiGraph(),
        optimization_history=[],
        settings=Settings(
            openai_api_key=DEFAULT_OPENAI_API_KEY,
            model=DEFAULT_OPENAI_MODEL
        ),
        profiling_results={},
        error_log=[],
        logs=[]
    )

def validate_environment() -> bool:
    """Validate required environment variables and dependencies"""
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        log(f"Missing required environment variables: {', '.join(missing_vars)}", "ERROR")
        return False
    
    try:
        import openai
        import streamlit
        import networkx
        return True
    except ImportError as e:
        log(f"Missing required dependency: {str(e)}", "ERROR")
        return False

def display_debug_info() -> None:
    """Display debug information when debug mode is enabled"""
    st.subheader("Debug Information")
    with st.expander("Application State"):
        st.json({
            "settings": asdict(state.settings),
            "error_count": len(state.error_log),
            "optimization_steps": len(state.optimization_steps),
            "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024  # MB
        })

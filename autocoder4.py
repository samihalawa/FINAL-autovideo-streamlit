# AutocoderAI - Production-Ready Streamlit Application for Code Optimization

import os
import re
import ast
import json
import subprocess
import streamlit as st
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
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
from dataclasses import dataclass, asdict
from functools import cache, lru_cache
import asyncio
import traceback
from contextlib import contextmanager
import psutil
import sys
from pathlib import Path
from openai import OpenAI
from langchain.memory import ConversationBufferMemory
from dataclasses import field
from datetime import datetime, timedelta

# Load environment variables from .env file if it exists
env_path = Path('.env')
if env_path.exists():
    load_dotenv(env_path)

# Configuration
st.set_page_config(
    page_title="AutocoderAI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DEFAULT_OPENAI_MODEL = "gpt-4"
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MAX_RETRIES = 3
TIMEOUT_SECONDS = 30
SUPPORTED_MODELS = ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo-preview"]
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_FILE = "autocoder.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Type definitions
@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    model: str
    profiling_depth: int = 10
    show_profiling: bool = False
    timeout_seconds: int = TIMEOUT_SECONDS
    max_retries: int = MAX_RETRIES
    debug_mode: bool = False
    log_level: str = "INFO"

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

    def clear(self):
        """Safely clear all state data"""
        self.script_sections.clear()
        self.optimized_script = ""
        self.optimization_steps.clear()
        self.current_step = 0
        self.graph.clear()
        self.optimization_history.clear()
        self.profiling_results.clear()
        self.error_log.clear()
        self.logs.clear()

def initialize_state() -> None:
    """Initialize application state with defaults"""
    if 'state' not in st.session_state:
        try:
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
        except Exception as e:
            logger.error(f"Failed to initialize state: {str(e)}")
            raise

initialize_state()
state = st.session_state.state

def log(msg: str, level: str = "INFO") -> None:
    """Centralized logging function with error handling"""
    try:
        log_level = getattr(logging, level.upper())
        logger.log(log_level, msg)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        state.logs.append(f"[{timestamp}] {level}: {msg}")
    except Exception as e:
        logger.error(f"Logging error: {str(e)}")

def validate_api_key(key: str) -> bool:
    """Validate OpenAI API key format"""
    if not key:
        return False
    return bool(re.match(r'^sk-[a-zA-Z0-9]{48}$', key))

def validate_input(script: str) -> bool:
    """Validate Python script input"""
    if not script or not isinstance(script, str):
        return False
    try:
        ast.parse(script)
        return True
    except (SyntaxError, ValueError, TypeError):
        return False

@st.cache_data(ttl=3600)
def set_openai_creds(key: str, model: str) -> None:
    """Set OpenAI credentials with validation and caching"""
    try:
        if not validate_api_key(key):
            raise ValueError("Invalid API key format")
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model. Must be one of: {', '.join(SUPPORTED_MODELS)}")
        
        openai.api_key = key
        state.settings = Settings(
            openai_api_key=key,
            model=model,
            profiling_depth=state.settings.profiling_depth,
            show_profiling=state.settings.show_profiling,
            timeout_seconds=state.settings.timeout_seconds,
            max_retries=state.settings.max_retries,
            debug_mode=state.settings.debug_mode
        )
        
        # Verify API key works
        openai.Model.list()
        log("OpenAI credentials set successfully")
    except Exception as e:
        error_msg = f"Error setting OpenAI credentials: {str(e)}"
        log(error_msg, "ERROR")
        raise ValueError(error_msg)

def save_final_code(code: str, filename: str = "optimized_script.py") -> None:
    """Save optimized code with Streamlit download button"""
    try:
        st.download_button(
            label="Download Optimized Code",
            data=code,
            file_name=filename,
            mime="text/plain"
        )
        log("Code ready for download")
    except Exception as e:
        error_msg = f"Error preparing download: {str(e)}"
        log(error_msg, "ERROR")
        raise

@st.cache_resource
def profile_performance(script: str) -> str:
    """Profile code performance with comprehensive error handling"""
    if not script or not isinstance(script, str):
        raise ValueError("Invalid script input")
        
    pr = cProfile.Profile()
    pr.enable()
    
    try:
        if not validate_input(script):
            raise SyntaxError("Invalid Python syntax")
            
        # Create isolated namespace for execution
        namespace = {}
        exec(script, namespace)
        
    except Exception as e:
        error_msg = f"Error during profiling: {str(e)}"
        state.error_log.append(error_msg)
        log(error_msg, "ERROR")
        raise
        
    finally:
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(state.settings.profiling_depth)
    
    return s.getvalue()

async def optimize_script(script_input: str, optimization_strategy: str = "performance") -> str:
    """Optimize script with progress tracking and comprehensive error handling"""
    if not validate_input(script_input):
        raise ValueError("Invalid Python script")
        
    sections = extract_sections(script_input)
    state.script_sections = sections

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        async with ThreadPoolExecutor() as executor:
            future_to_section = {
                executor.submit(
                    optimize_section_async, 
                    name, 
                    content
                ): name
                for name, content in sections.items()
                if name != "package_installations"
            }
            
            total = len(future_to_section)
            optimized_sections = {}
            
            for i, future in enumerate(as_completed(future_to_section)):
                section_name = future_to_section[future]
                try:
                    name, optimized_content = await future.result()
                    optimized_sections[name] = optimized_content
                    
                    # Update progress
                    progress = (i + 1) / total
                    progress_bar.progress(progress)
                    status_text.text(f"Optimizing {section_name} ({i+1}/{total})")
                    
                    # Profile if enabled
                    if state.settings.show_profiling:
                        profile_result = profile_performance(optimized_content)
                        state.profiling_results[section_name] = profile_result
                        
                except Exception as e:
                    error_msg = f"Error optimizing {section_name}: {str(e)}"
                    log(error_msg, "ERROR")
                    optimized_sections[section_name] = sections[section_name]  # Keep original on error
                    
        # Combine sections in correct order
        return "\n\n".join(
            optimized_sections.get(name, content)
            for name, content in sections.items()
        )
        
    except Exception as e:
        error_msg = f"Error in optimization process: {str(e)}"
        log(error_msg, "ERROR")
        raise
    finally:
        progress_bar.empty()
        status_text.empty()

def display_profiling_results() -> None:
    """Display profiling results in expandable sections with error handling"""
    try:
        if not state.profiling_results:
            st.info("No profiling results available")
            return
            
        st.subheader("Performance Profiling Results")
        for section, result in state.profiling_results.items():
            with st.expander(f"Profiling for {section}"):
                st.text(result)
    except Exception as e:
        error_msg = f"Error displaying profiling results: {str(e)}"
        log(error_msg, "ERROR")
        st.error(error_msg)

def display_error_log() -> None:
    """Display error log with proper formatting and filtering"""
    try:
        if not state.error_log:
            return
            
        st.subheader("Error Log")
        for error in state.error_log:
            if isinstance(error, str):
                st.error(error)
            else:
                st.error(str(error))
    except Exception as e:
        log(f"Error displaying error log: {str(e)}", "ERROR")

@contextmanager
def manage_resources():
    """Context manager for application resources with comprehensive cleanup"""
    try:
        yield
    except Exception as e:
        log(f"Error in resource management: {str(e)}", "ERROR")
        raise
    finally:
        try:
            if hasattr(st.session_state, 'state'):
                state = st.session_state.state
                state.clear()
        except Exception as e:
            log(f"Error during cleanup: {str(e)}", "ERROR")

@contextmanager
def error_boundary(operation: str):
    """Context manager for consistent error handling"""
    try:
        yield
    except Exception as e:
        error_msg = f"Error during {operation}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        st.error(error_msg)
        state.error_log.append(error_msg)

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
            logger.warning(f"Timeout while optimizing section {section_name}")
            raise
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            raise

    for attempt in range(state.settings.max_retries):
        try:
            optimized_content = await _make_api_call()
            return section_name, optimized_content
        except Exception as e:
            if attempt == state.settings.max_retries - 1:
                logger.error(f"All retries failed for section {section_name}: {str(e)}")
                return section_name, content
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    return section_name, content

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
        logger.error(f"Error extracting sections: {str(e)}")
        return {"main": script}

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
                st.success("Settings applied successfully!")

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

def main():
    """Main application entry point"""
    if not validate_environment():
        st.error("Environment validation failed. Please check logs.")
        return

    initialize_state()
    
    with st.sidebar:
        selected = st.selectbox(
            "Navigation",
            ["Optimize", "Settings", "Logs", "Debug"] if state.settings.debug_mode else ["Optimize", "Settings", "Logs"]
        )
    
    if selected == "Optimize":
        display_optimization_interface()
    elif selected == "Settings":
        display_settings()
    elif selected == "Logs":
        display_logs()
    elif selected == "Debug" and state.settings.debug_mode:
        display_debug_info()

def validate_environment() -> bool:
    """Validate required environment variables and dependencies"""
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    try:
        import openai
        import streamlit
        import networkx
        return True
    except ImportError as e:
        logger.error(f"Missing required dependency: {str(e)}")
        return False

if __name__ == "__main__":
    main()

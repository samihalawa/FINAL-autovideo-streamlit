import os
import re
import ast
import json
import subprocess
import streamlit as st
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
import openai
from difflib import unified_diff
import networkx as nx
import matplotlib.pyplot as plt
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from streamlit_ace import st_ace
from streamlit_agraph import agraph, Node, Edge, Config
from tqdm import tqdm
from langchain.memory import ConversationBufferMemory
from dataclasses import dataclass, field
from functools import lru_cache
import logging
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autocoder.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ------------------- Configuration -------------------

st.set_page_config(page_title="AutocoderAI", layout="wide")

# ------------------- State Management -------------------

@dataclass
class State:
    """Centralized state management with type validation"""
    script_sections: Dict[str, Any] = field(default_factory=dict)
    optimized_script: str = ""
    optimization_steps: List[str] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=lambda: {
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "model": "gpt-3.5-turbo",
        "coder_initialized": False,
        "show_profiling": False,
        "profiling_depth": 10,
        "max_retries": 3,
        "timeout": 30
    })
    profiling_results: Dict[str, Any] = field(default_factory=dict)
    error_log: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    memory: ConversationBufferMemory = field(default_factory=lambda: ConversationBufferMemory(memory_key="chat_history"))

    def __post_init__(self):
        """Validate state initialization"""
        if not isinstance(self.settings, dict):
            raise TypeError("Settings must be a dictionary")
        if not isinstance(self.optimization_steps, list):
            raise TypeError("Optimization steps must be a list")
        # Validate required settings
        required_settings = ["openai_api_key", "model", "max_retries", "timeout"]
        for setting in required_settings:
            if setting not in self.settings:
                raise ValueError(f"Missing required setting: {setting}")

# Initialize state with error handling
try:
    if "state" not in st.session_state:
        st.session_state.state = State()
    state = st.session_state.state
except Exception as e:
    logger.error(f"Failed to initialize state: {str(e)}")
    st.error("Failed to initialize application state. Please refresh the page.")
    sys.exit(1)

# ------------------- Error Handling -------------------

class AutocoderError(Exception):
    """Base exception class for Autocoder errors"""
    pass

class ValidationError(AutocoderError):
    """Raised when input validation fails"""
    pass

class OptimizationError(AutocoderError):
    """Raised when code optimization fails"""
    pass

class APIError(AutocoderError):
    """Raised when API calls fail"""
    pass

def handle_error(func: Callable) -> Callable:
    """Error handling decorator with user feedback and logging"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AutocoderError as e:
            error_msg = f"Operation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            st.error(error_msg)
            state.error_log.append(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            st.error(error_msg)
            state.error_log.append(error_msg)
        return None
    return wrapper

# ------------------- Core Functions -------------------

def validate_script(script: str) -> bool:
    """Validate Python script syntax"""
    try:
        ast.parse(script)
        return True
    except SyntaxError:
        return False

def validate_api_key(api_key: str) -> bool:
    """Validate OpenAI API key format and connectivity"""
    if not api_key or len(api_key) < 20:
        return False
    try:
        openai.api_key = api_key
        # Make a minimal API call to test connectivity
        openai.Model.list()
        return True
    except Exception as e:
        logger.error(f"API key validation failed: {str(e)}")
        return False

@lru_cache(maxsize=100)
def extract_sections(script: str) -> Dict[str, Any]:
    """Parse and extract code sections with caching and validation"""
    if not script.strip():
        raise ValidationError("Empty script provided")
    
    try:
        tree = ast.parse(script)
    except SyntaxError as e:
        raise ValidationError(f"Syntax Error: {e}")
    
    sections = {
        "imports": [],
        "settings": "",
        "function_definitions": {},
        "class_definitions": {},
        "global_code": []
    }
    
    for node in ast.walk(tree):
        try:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                sections["imports"].append(ast.unparse(node))
            elif isinstance(node, ast.FunctionDef):
                sections["function_definitions"][node.name] = ast.unparse(node)
            elif isinstance(node, ast.ClassDef):
                sections["class_definitions"][node.name] = ast.unparse(node)
            elif isinstance(node, ast.Assign):
                sections["settings"] += ast.unparse(node) + "\n"
            elif isinstance(node, ast.Expr):
                sections["global_code"].append(ast.unparse(node))
        except Exception as e:
            logger.warning(f"Failed to process node: {str(e)}")
            continue
    
    return sections

async def optimize_section_async(section_name: str, section_content: str) -> Tuple[str, str]:
    """Asynchronously optimize code section with validation and retry logic"""
    if not section_content.strip():
        return section_name, section_content
        
    if not validate_script(section_content):
        logger.warning(f"Invalid script in section {section_name}")
        return section_name, section_content
        
    retries = 0
    while retries < state.settings["max_retries"]:
        try:
            prompt = f"""Optimize this Python code section for production use:
            ```python
            {section_content}
            ```
            Requirements:
            - Improve performance
            - Enhance readability
            - Ensure PEP 8 compliance
            - Add proper error handling
            - Include type hints
            - Add docstrings"""
            
            response = await openai.ChatCompletion.acreate(
                model=state.settings["model"],
                messages=[
                    {"role": "system", "content": "You are a Python optimization expert focused on production-ready code."},
                    {"role": "user", "content": prompt}
                ],
                timeout=state.settings["timeout"]
            )
            
            optimized = response.choices[0].message.content.strip()
            # Extract code from markdown if present
            optimized = re.search(r'```python\n(.*?)\n```', optimized, re.DOTALL)
            if optimized:
                optimized = optimized.group(1)
            
            if not validate_script(optimized):
                logger.warning(f"Optimization result for {section_name} was invalid")
                retries += 1
                continue
                
            return section_name, optimized
            
        except Exception as e:
            logger.error(f"Optimization attempt {retries + 1} failed for {section_name}: {str(e)}")
            retries += 1
            if retries == state.settings["max_retries"]:
                raise OptimizationError(f"Failed to optimize {section_name} after {retries} attempts")
            time.sleep(1)  # Backoff before retry

@handle_error
def optimize_script(script_input: str, optimization_strategy: str) -> str:
    """Optimize full script with progress tracking and validation"""
    if not script_input.strip():
        raise ValidationError("Empty script provided")
        
    if not state.settings.get("openai_api_key"):
        raise ValidationError("OpenAI API key not configured")
        
    sections = extract_sections(script_input)
    state.script_sections = sections

    progress = st.progress(0)
    status = st.empty()

    optimized_sections = {}
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(optimize_section_async, name, content): name
            for name, content in sections.items() 
            if name != "package_installations"
        }
        
        total = len(futures)
        completed = 0
        
        for future in as_completed(futures):
            section_name = futures[future]
            try:
                _, optimized_content = future.result()
                optimized_sections[section_name] = optimized_content
                completed += 1
                progress.progress(completed / total)
                status.text(f"Optimizing {section_name} ({completed}/{total})")
                logger.info(f"Successfully optimized section: {section_name}")
            except Exception as e:
                logger.error(f"Failed to optimize section {section_name}: {str(e)}")
                optimized_sections[section_name] = sections[section_name]  # Keep original on failure

    # Assemble final script maintaining proper order
    final_script = "\n\n".join([
        "\n".join(optimized_sections.get("imports", [])),
        optimized_sections.get("settings", ""),
        "\n\n".join(optimized_sections.get("class_definitions", {}).values()),
        "\n\n".join(optimized_sections.get("function_definitions", {}).values()),
        "\n".join(optimized_sections.get("global_code", []))
    ])

    return final_script

# ------------------- UI Components -------------------

def render_code_editor(key: str = "main_editor") -> str:
    """Standardized code editor component with error handling"""
    try:
        return st_ace(
            placeholder="Paste your Python script here...",
            language="python",
            theme="monokai",
            keybinding="vscode",
            font_size=14,
            min_lines=20,
            key=key,
            auto_update=True
        )
    except Exception as e:
        logger.error(f"Failed to render code editor: {str(e)}")
        st.error("Failed to load code editor. Please refresh the page.")
        return ""

def render_metrics(script_input: str):
    """Display unified metrics component with error handling"""
    try:
        col1, col2, col3 = st.columns(3)
        
        input_lines = len(script_input.splitlines())
        output_lines = len(state.optimized_script.splitlines()) if state.optimized_script else 0
        
        col1.metric(
            "Lines of Code", 
            input_lines,
            output_lines - input_lines if output_lines else None
        )
        
        col2.metric(
            "Functions", 
            len(state.script_sections.get("function_definitions", {}))
        )
        
        col3.metric(
            "Imports", 
            len(state.script_sections.get("imports", []))
        )
    except Exception as e:
        logger.error(f"Failed to render metrics: {str(e)}")
        st.warning("Unable to display metrics")

def render_optimization_chat():
    """Interactive optimization suggestions component with error handling"""
    try:
        st.subheader("Interactive Optimization Suggestions")
        user_input = st.chat_input("Ask for optimization suggestions")
        
        if user_input:
            with st.chat_message("user"):
                st.write(user_input)
            with st.chat_message("assistant"):
                response = generate_optimization_suggestion(user_input)
                if response:
                    st.write(response)
                else:
                    st.error("Failed to generate suggestion")
    except Exception as e:
        logger.error(f"Chat component failed: {str(e)}")
        st.error("Chat component is currently unavailable")

def generate_optimization_suggestion(query: str) -> Optional[str]:
    """Generate optimization suggestions with error handling"""
    try:
        response = openai.ChatCompletion.create(
            model=state.settings["model"],
            messages=[
                {"role": "system", "content": "You are a Python optimization expert."},
                {"role": "user", "content": query}
            ],
            timeout=state.settings["timeout"]
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Failed to generate suggestion: {str(e)}")
        return None

# ------------------- Main Application -------------------

def main():
    """Main application with comprehensive error handling"""
    try:
        st.title("AutocoderAI 🧑‍💻✨")
        st.markdown("**Automated Python Script Manager and Optimizer using OpenAI's API**")

        with st.sidebar:
            optimization_strategy = st.selectbox(
                "Select Optimization Strategy",
                ["Basic", "Advanced", "Experimental"]
            )

            st.subheader("OpenAI Settings")
            api_key = st.text_input("OpenAI API Key", type="password")
            model = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4"])
            
            if api_key:
                if validate_api_key(api_key):
                    state.settings.update({
                        "openai_api_key": api_key,
                        "model": model
                    })
                    st.success("API key validated successfully")
                else:
                    st.error("Invalid API Key")

        script_input = render_code_editor()

        if st.button("🚀 Optimize Script", disabled=not state.settings["openai_api_key"]):
            with st.spinner("Optimizing script..."):
                state.optimized_script = optimize_script(script_input, optimization_strategy)

            if state.optimized_script:
                st.success("Optimization complete!")
                st.download_button(
                    label="Download Optimized Script",
                    data=state.optimized_script,
                    file_name="optimized_script.py",
                    mime="text/plain"
                )

        if state.optimized_script:
            st.subheader("Optimized Script Sections")
            for section, content in state.script_sections.items():
                if section != "package_installations":
                    try:
                        state.script_sections[section] = st.text_area(
                            f"Edit {section}", 
                            content, 
                            key=f"editor_{section}"
                        )
                    except Exception as e:
                        logger.error(f"Failed to render section editor: {str(e)}")
                        st.error(f"Failed to load editor for {section}")

            col1, col2 = st.columns(2)
            if col1.button("Save Changes"):
                try:
                    state.optimized_script = assemble_script()
                    st.success("Changes saved successfully")
                except Exception as e:
                    logger.error(f"Failed to save changes: {str(e)}")
                    st.error("Failed to save changes")

            if col2.button("Export Final Code"):
                try:
                    save_path = Path("optimized_script.py")
                    save_path.write_text(state.optimized_script)
                    st.success(f"Code exported to {save_path.absolute()}")
                except Exception as e:
                    logger.error(f"Failed to export code: {str(e)}")
                    st.error("Failed to export code")

            render_metrics(script_input)
            render_optimization_chat()

            st.subheader("Advanced Options")
            enable_type_hints = st.toggle("Enable Type Hints")
            enable_async = st.toggle("Enable Async Optimization")
            
            if enable_type_hints or enable_async:
                try:
                    state.optimized_script = apply_advanced_optimizations(
                        state.optimized_script, 
                        enable_type_hints, 
                        enable_async
                    )
                    st.success("Advanced optimizations applied")
                except Exception as e:
                    logger.error(f"Advanced optimization failed: {str(e)}")
                    st.error("Failed to apply advanced optimizations")

            # Display final optimized script
            st.subheader("Final Optimized Script")
            st.code(state.optimized_script, language="python")

    except Exception as e:
        logger.critical(f"Application crashed: {str(e)}", exc_info=True)
        st.error("Application encountered a critical error. Please refresh the page.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Application failed to start: {str(e)}", exc_info=True)
        st.error("Failed to start application. Please check the logs and try again.")

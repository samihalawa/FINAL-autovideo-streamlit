import os
import re
import ast
import json
import subprocess
import streamlit as st
from typing import List, Dict, Any, Optional, Union, Callable
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
        "profiling_depth": 10
    })
    profiling_results: Dict[str, Any] = field(default_factory=dict)
    error_log: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    memory: ConversationBufferMemory = field(default_factory=ConversationBufferMemory)

    def __post_init__(self):
        """Validate state initialization"""
        if not isinstance(self.settings, dict):
            raise TypeError("Settings must be a dictionary")
        if not isinstance(self.optimization_steps, list):
            raise TypeError("Optimization steps must be a list")

if "state" not in st.session_state:
    st.session_state.state = State()

state = st.session_state.state

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

def handle_error(func: Callable) -> Callable:
    """Error handling decorator with user feedback"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AutocoderError as e:
            st.error(f"Operation failed: {str(e)}")
            log_error(str(e))
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            log_error(f"Unexpected: {str(e)}")
        return None
    return wrapper

# ------------------- Core Functions -------------------

@lru_cache(maxsize=100)
def extract_sections(script: str) -> Dict[str, Any]:
    """Parse and extract code sections with caching"""
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
    
    return sections

async def optimize_section_async(section_name: str, section_content: str) -> tuple[str, str]:
    """Asynchronously optimize code section with validation"""
    if not section_content.strip():
        return section_name, section_content
        
    if not validate_script(section_content):
        return section_name, section_content
        
    try:
        prompt = f"""Optimize this Python code section:
        ```python
        {section_content}
        ```
        Ensure: performance, readability, PEP 8 compliance"""
        
        response = await openai.ChatCompletion.acreate(
            model=state.settings["model"],
            messages=[
                {"role": "system", "content": "You are a Python optimization expert."},
                {"role": "user", "content": prompt}
            ]
        )
        
        optimized = response.choices[0].message.content.strip()
        if not validate_script(optimized):
            st.warning(f"Optimization result for {section_name} was invalid, keeping original")
            return section_name, section_content
            
        return section_name, optimized
    except Exception as e:
        raise OptimizationError(f"Error optimizing {section_name}: {str(e)}")

@handle_error
def optimize_script(script_input: str, optimization_strategy: str) -> str:
    """Optimize full script with progress tracking"""
    sections = extract_sections(script_input)
    state.script_sections = sections

    progress = st.progress(0)
    status = st.empty()

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(optimize_section_async, name, content): name
            for name, content in sections.items() 
            if name != "package_installations"
        }
        
        total = len(futures)
        for i, future in enumerate(as_completed(futures)):
            section_name = futures[future]
            try:
                _, optimized_content = future.result()
                sections[section_name] = optimized_content
                progress.progress((i + 1) / total)
                status.text(f"Optimizing {section_name} ({i+1}/{total})")
            except Exception as e:
                raise OptimizationError(f"Error in section {section_name}: {str(e)}")

    return "\n\n".join(sections.values())

# ------------------- UI Components -------------------

def render_code_editor(key: str = "main_editor") -> str:
    """Standardized code editor component"""
    return st_ace(
        placeholder="Paste your Python script here...",
        language="python",
        theme="monokai",
        keybinding="vscode",
        font_size=14,
        min_lines=20,
        key=key
    )

def render_metrics(script_input: str):
    """Display unified metrics component"""
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Lines of Code", 
        len(script_input.splitlines()),
        len(state.optimized_script.splitlines()) - len(script_input.splitlines())
    )
    col2.metric(
        "Functions", 
        len(state.script_sections.get("function_definitions", {}))
    )
    col3.metric(
        "Imports", 
        len(state.script_sections.get("imports", []))
    )

def render_optimization_chat():
    """Interactive optimization suggestions component"""
    st.subheader("Interactive Optimization Suggestions")
    user_input = st.chat_input("Ask for optimization suggestions")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            st.write(generate_optimization_suggestion(user_input))

# ------------------- Main Application -------------------

def main():
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
        
        if api_key and validate_api_key(api_key):
            state.settings.update({
                "openai_api_key": api_key,
                "model": model
            })
        elif api_key:
            st.error("Invalid API Key")

    script_input = render_code_editor()

    if st.button("🚀 Optimize Massive Script"):
        with st.spinner("Optimizing script..."):
            state.optimized_script = optimize_script(script_input, optimization_strategy)

        st.success("Optimization complete!")
        st.download_button(
            label="Download Optimized Script",
            data=state.optimized_script,
            file_name="optimized_massive_script.py",
            mime="text/plain"
        )

    if state.optimized_script:
        st.subheader("Optimized Script Sections")
        for section, content in state.script_sections.items():
            if section != "package_installations":
                state.script_sections[section] = st.text_area(
                    f"Edit {section}", 
                    content, 
                    key=f"editor_{section}"
                )

        col1, col2 = st.columns(2)
        if col1.button("Save Changes"):
            state.optimized_script = assemble_script()
            save_final_code(state.optimized_script)

        if col2.button("Save Final Code"):
            save_final_code(state.optimized_script)

        render_metrics(script_input)
        render_optimization_chat()

        st.subheader("Advanced Options")
        enable_type_hints = st.toggle("Enable Type Hints")
        enable_async = st.toggle("Enable Async Optimization")
        if enable_type_hints or enable_async:
            state.optimized_script = apply_advanced_optimizations(
                state.optimized_script, 
                enable_type_hints, 
                enable_async
            )

        display_optimized_script()
        display_function_graph()

if __name__ == "__main__":
    main()

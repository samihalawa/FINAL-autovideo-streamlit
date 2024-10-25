# AutocoderAI - Enhanced Streamlit Application for Code Optimization

import os
import re
import ast
import json
import subprocess
import streamlit as st
from typing import List, Dict, Any
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

# Load environment variables
load_dotenv()

# Configuration
st.set_page_config(page_title="AutocoderAI", layout="wide")
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Initialize session state
if 'state' not in st.session_state:
    st.session_state.state = {
        "script_sections": {},
        "optimized_script": "",
        "optimization_steps": [],
        "current_step": 0,
        "graph": nx.DiGraph(),
        "optimization_history": [],
        "settings": {
            "openai_api_key": DEFAULT_OPENAI_API_KEY,
            "model": DEFAULT_OPENAI_MODEL,
        },
        "profiling_results": {},
        "error_log": [],
        "logs": [],
        "show_profiling": False
    }

state = st.session_state.state

# Logging function
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log(msg: str, level: str = "INFO"):
    getattr(logging, level.lower())(msg)
    state["logs"].append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {level}: {msg}")

# Validate API key
def validate_api_key(key: str) -> bool:
    return bool(re.match(r'^sk-[a-zA-Z0-9]{48}$', key))

# Set OpenAI credentials
def set_openai_creds(key: str, model: str):
    try:
        openai.api_key = key
        state["settings"]["openai_api_key"] = key
        state["settings"]["model"] = model
        openai.Model.list()
        log("OpenAI credentials set successfully.")
    except Exception as e:
        st.error(f"Error setting OpenAI credentials: {str(e)}")
        log(f"Error: {str(e)}")

# Add a function to save the final code
def save_final_code(code: str, filename: str = "optimized_script.py"):
    with open(filename, "w") as f:
        f.write(code)
    st.success(f"Saved optimized code to {filename}")

# Add this function near the other utility functions
def profile_performance(script: str):
    pr = cProfile.Profile()
    pr.enable()
    
    try:
        exec(script)
    except Exception as e:
        state["error_log"].append(f"Error during profiling: {str(e)}")
    
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    return s.getvalue()

def optimize_script(script_input: str, optimization_strategy: str) -> str:
    sections = extract_sections(script_input)
    state["script_sections"] = sections

    progress_bar = st.progress(0)
    status_text = st.empty()

    with ThreadPoolExecutor() as executor:
        future_to_section = {
            executor.submit(optimize_section_async, name, content): name
            for name, content in sections.items() if name != "package_installations"
        }
        
        total = len(future_to_section)
        for i, future in enumerate(as_completed(future_to_section)):
            section_name = future_to_section[future]
            try:
                _, optimized_content = future.result()
                sections[section_name] = optimized_content
                progress = (i + 1) / total
                progress_bar.progress(progress)
                status_text.text(f"Optimizing {section_name} ({i+1}/{total})")
                
                # Profile the optimized section
                if state["show_profiling"]:
                    profile_result = profile_performance(optimized_content)
                    state["profiling_results"][section_name] = profile_result
            except Exception as e:
                log(f"Error optimizing {section_name}: {str(e)}", "ERROR")

    return "\n\n".join(sections.values())

def display_profiling_results():
    st.subheader("Performance Profiling Results")
    for section, result in state["profiling_results"].items():
        with st.expander(f"Profiling for {section}"):
            st.text(result)

def display_error_log():
    if state["error_log"]:
        st.subheader("Error Log")
        for error in state["error_log"]:
            st.error(error)

def main():
    st.title("AutocoderAI 🧑‍💻✨")
    
    # Sidebar
    with st.sidebar:
        selected = option_menu("Menu", ["Optimize", "Settings", "Logs"], 
            icons=['magic', 'gear', 'journal-text'], menu_icon="cast", default_index=0)
        
        if selected == "Settings":
            st.subheader("OpenAI Settings")
            state["settings"]["openai_api_key"] = st.text_input("OpenAI API Key", value=state["settings"]["openai_api_key"], type="password")
            state["settings"]["model"] = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"], index=2)
        
        elif selected == "Logs":
            st.subheader("Logs")
            for log in state["logs"]:
                st.text(log)

    if selected == "Optimize":
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("Input Code")
            script_input = st_ace(
                placeholder="Paste your Python script here...",
                language="python",
                theme="monokai",
                keybinding="vscode",
                font_size=14,
                min_lines=20,
                key="ace_editor"
            )

            if st.button("🚀 Optimize with Aider"):
                with st.spinner("Optimizing script..."):
                    optimized_script = optimize_with_aider(script_input)
                    
                    animate_optimization(script_input, optimized_script)
                    
                    state["optimized_script"] = optimized_script

                if st.button("Save Optimized Code"):
                    save_final_code(state["optimized_script"])

        with col2:
            st.subheader("Optimization Process")
            for step in state["optimization_steps"]:
                st.info(step)

            if state["optimized_script"]:
                st.subheader("Optimized Script")
                st.code(state["optimized_script"], language="python")

                if st.button("Save Optimized Code"):
                    save_final_code(state["optimized_script"])

            if state["profiling_results"]:
                st.subheader("Performance Profiling")
                for section, result in state["profiling_results"].items():
                    with st.expander(f"Profiling for {section}"):
                        st.text(result)

            if state["error_log"]:
                st.subheader("Error Log")
                for error in state["error_log"]:
                    st.error(error)

        if state["optimized_script"]:
            st.subheader("Function Dependency Graph")
            graph = generate_script_map(state["script_sections"].get("function_definitions", {}))
            fig = go.Figure(data=[go.Sankey(
                node = dict(
                  pad = 15,
                  thickness = 20,
                  line = dict(color = "black", width = 0.5),
                  label = list(graph.nodes()),
                  color = "blue"
                ),
                link = dict(
                  source = [list(graph.nodes()).index(edge[0]) for edge in graph.edges()],
                  target = [list(graph.nodes()).index(edge[1]) for edge in graph.edges()],
                  value = [1] * len(graph.edges())
              ))])
            fig.update_layout(title_text="Function Dependencies", font_size=10)
            st.plotly_chart(fig, use_container_width=True)

# Add missing functions
def extract_sections(script_content: str) -> Dict[str, Any]:
    try:
        tree = ast.parse(script_content)
    except SyntaxError as e:
        st.error(f"Syntax Error in script: {e}")
        return {}
    
    sections = {
        "package_installations": [],
        "imports": [],
        "settings": "",
        "function_definitions": {}
    }

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            sections["imports"].append(ast.unparse(node))
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            sections["settings"] += ast.unparse(node) + "\n"
        elif isinstance(node, ast.FunctionDef):
            sections["function_definitions"][node.name] = ast.unparse(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Attribute) and node.value.func.attr in ["system", "check_call", "run"]:
                for arg in node.value.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        match = re.findall(r'pip install ([\w\-\.\@]+)', arg.value)
                        if match:
                            sections["package_installations"].extend(match)
    return sections

def add_log(message: str):
    state["optimization_steps"].append(message)

def generate_script_map(function_definitions: Dict[str, str]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for func_name in function_definitions.keys():
        graph.add_node(func_name)
    
    for func_name, func_code in function_definitions.items():
        calls = re.findall(r'\b{}\b'.format('|'.join(function_definitions.keys())), func_code)
        for call in calls:
            if call != func_name:
                graph.add_edge(func_name, call)
    return graph

def optimize_section_async(section_name: str, section_content: str):
    if not section_content.strip():
        return section_name, section_content
        
    try:
        prompt = f"""Optimize this Python code section:
        ```python
        {section_content}
        ```
        Focus on: performance, readability, PEP 8 compliance"""
        
        response = openai.ChatCompletion.create(
            model=state["settings"]["model"],
            messages=[
                {"role": "system", "content": "You are a Python optimization expert."},
                {"role": "user", "content": prompt}
            ]
        )
        
        optimized = response.choices[0].message.content.strip()
        return section_name, optimized
    except Exception as e:
        log(f"Error in section {section_name}: {str(e)}", "ERROR")
        return section_name, section_content

def optimize_with_aider(script_input: str) -> str:
    try:
        aider_chat = chat.Chat(io=None, coder=None)  # Initialize Aider chat
        optimized_script = aider_chat.send_message(f"Optimize this Python code:\n\n{script_input}")
        return optimized_script.content
    except Exception as e:
        st.error(f"Error during Aider optimization: {e}")
        return script_input

def animate_optimization(original_script: str, optimized_script: str):
    placeholder = st.empty()
    lines_original = original_script.split('\n')
    lines_optimized = optimized_script.split('\n')
    max_lines = max(len(lines_original), len(lines_optimized))
    
    for i in range(max_lines):
        current_original = '\n'.join(lines_original[:i+1])
        current_optimized = '\n'.join(lines_optimized[:i+1])
        
        col1, col2 = placeholder.columns(2)
        with col1:
            st.code(current_original, language='python')
        with col2:
            st.code(current_optimized, language='python')
        
        time.sleep(0.1)  # Adjust speed of animation

def display_optimized_script():
    """Displays the optimized script with metrics and visualization."""
    if not state["optimized_script"]:
        return
        
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original vs Optimized")
        st.code(state["optimized_script"], language="python")
        
    with col2:
        st.subheader("Optimization Metrics")
        metrics = {
            "Lines": len(state["optimized_script"].splitlines()),
            "Functions": len(state["script_sections"].get("function_definitions", {})),
            "Imports": len(state["script_sections"].get("imports", []))
        }
        for label, value in metrics.items():
            st.metric(label, value)

def profile_section(section_name: str, code: str):
    """Profile a section of code."""
    if not state["show_profiling"]:
        return
        
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        exec(code, {})
    except Exception as e:
        log(f"Profiling error in {section_name}: {str(e)}", "ERROR")
    
    profiler.disable()
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    stats.print_stats(state["settings"].get("profiling_depth", 10))
    state["profiling_results"][section_name] = s.getvalue()

def display_optimization_results():
    """Display optimization results with metrics and visualizations."""
    if not state["optimized_script"]:
        return
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Code Comparison")
        tabs = st.tabs(["Original", "Optimized"])
        with tabs[0]:
            st.code(state["script_sections"]["original"], language="python")
        with tabs[1]:
            st.code(state["optimized_script"], language="python")
            
    with col2:
        st.subheader("Optimization Metrics")
        metrics = calculate_metrics()
        for label, value in metrics.items():
            st.metric(label, value)
        
        if state["show_profiling"]:
            st.subheader("Performance Profiling")
            for section, result in state["profiling_results"].items():
                with st.expander(f"Profile: {section}"):
                    st.text(result)

if __name__ == "__main__":
    main()

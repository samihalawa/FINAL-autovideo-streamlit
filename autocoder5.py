# AutocoderAI - Enhanced Streamlit Application for Code Optimization

import os
import sys
import ast
import re
import streamlit as st
import openai
import networkx as nx
from typing import Dict, List, Any
from difflib import unified_diff
import time
from streamlit_ace import st_ace
from streamlit_agraph import agraph, Node, Edge, Config
from aider.io import InputOutput
from aider import models, chat
from aider.coders import Coder
import subprocess
import concurrent.futures
import plotly.graph_objects as go
from openai import OpenAI
import json

# Configuration
st.set_page_config(page_title="AutocoderAI", layout="wide")
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Initialize session state
def init_session_state():
    defaults = {
        'script_sections': {}, 'optimized_sections': {}, 'final_script': "",
        'progress': 0, 'logs': [], 'function_list': [], 'current_function_index': 0,
        'openai_api_key': DEFAULT_OPENAI_API_KEY, 'openai_model': DEFAULT_OPENAI_MODEL,
        'openai_endpoint': "https://api.openai.com/v1", 'script_map': None,
        'optimization_status': {}, 'error_log': [], 'chat_history': [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# Logging function
def log(msg: str):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    st.session_state['logs'].append(f"[{timestamp}] {msg}")

# Extract script sections using AST
def extract_sections(script: str) -> Dict[str, Any]:
    try:
        tree = ast.parse(script)
    except SyntaxError as e:
        st.error(f"Syntax Error: {e}")
        log(f"Syntax Error: {e}")
        return {}
    
    sections = {
        "package_installations": [], "imports": [], "settings": [],
        "function_definitions": {}, "class_definitions": {}, "global_code": []
    }
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            sections["imports"].append(ast.get_source_segment(script, node))
        elif isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
            sections["settings"].append(ast.get_source_segment(script, node))
        elif isinstance(node, ast.FunctionDef):
            sections["function_definitions"][node.name] = ast.get_source_segment(script, node)
            st.session_state['function_list'].append(node.name)
        elif isinstance(node, ast.ClassDef):
            sections["class_definitions"][node.name] = ast.get_source_segment(script, node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Attribute) and node.value.func.attr in ["system", "check_call", "run"]:
                for arg in node.value.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        match = re.findall(r'pip install ([\w\-\.\@]+)', arg.value)
                        if match:
                            sections["package_installations"].extend(match)
        elif isinstance(node, (ast.Expr, ast.Assign)):
            sections["global_code"].append(ast.get_source_segment(script, node))
    
    return sections

# Generate requirements
def gen_requirements(pkgs: List[str]) -> str:
    reqs = []
    for pkg in sorted(set(pkgs)):
        try:
            ver = subprocess.check_output([sys.executable, '-m', 'pip', 'show', pkg]).decode().split('\n')[1].split(': ')[1]
            reqs.append(f"{pkg}=={ver}")
        except subprocess.CalledProcessError:
            reqs.append(pkg)
    return "\n".join(f"pip install {req}" for req in reqs)

# Validate API key
def validate_api_key(key: str) -> bool:
    return bool(re.match(r'^sk-[a-zA-Z0-9]{48}$', key))

# Set OpenAI credentials
def set_openai_creds(key: str, model: str, endpoint: str):
    try:
        openai.api_key, openai.api_base = key, endpoint
        st.session_state.update({'openai_api_key': key, 'openai_model': model, 'openai_endpoint': endpoint})
        openai.Model.list()
        log("OpenAI credentials set successfully.")
    except Exception as e:
        st.error(f"Error setting OpenAI credentials: {str(e)}")
        log(f"Error: {str(e)}")

# Analyze and optimize code section using Aider
def analyze_and_optimize(content: str, name: str, coder: Coder) -> str:
    prompt = (
        f"Optimize the following `{name}` section:\n\n```python\n{content}\n```\n\n"
        "Enhance for: time/space complexity, readability, PEP 8 compliance, modern Python features, error handling, type hints, efficient imports, optimal data structures, and caching."
    )
    
    try:
        optimized_content = coder.edit(content, prompt)
        return optimized_content
    except Exception as e:
        st.error(f"Aider Optimization error: {str(e)}")
        log(f"Aider Optimization Error: {str(e)}")
        return content

# Optimize all sections with Aider integration
def optimize_sections(coder: Coder):
    progress_bar = st.progress(0)
    total_steps = len(st.session_state['script_sections']['function_definitions']) + \
                  len(st.session_state['script_sections']['class_definitions']) + 3
    current_step = 0

    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_section = {
                executor.submit(analyze_and_optimize, content, section, coder): section
                for section, content in [
                    ('imports', '\n'.join(st.session_state['script_sections']['imports'])),
                    ('settings', '\n'.join(st.session_state['script_sections']['settings'])),
                    *[(f"function {func}", func_content) for func, func_content in st.session_state['script_sections']['function_definitions'].items()],
                    *[(f"class {cls}", cls_content) for cls, cls_content in st.session_state['script_sections']['class_definitions'].items()]
                ]
            }
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_section)):
                section = future_to_section[future]
                try:
                    optimized = future.result()
                    section_type = section.split()[0] + ('s' if section in ['imports', 'settings'] else '')
                    if section_type not in st.session_state['optimized_sections']:
                        st.session_state['optimized_sections'][section_type] = {}
                    if section in ['imports', 'settings']:
                        st.session_state['optimized_sections'][section_type] = optimized.splitlines()
                    else:
                        st.session_state['optimized_sections'][section_type][section.split()[1]] = optimized
                    current_step += 1
                    progress_bar.progress(current_step / total_steps)
                    st.session_state['optimization_status'][section] = "Completed"
                    log(f"{section.capitalize()} optimized.")
                    if i == len(future_to_section) - 1:
                        st.info(f"Completed optimization of all sections.")
                        break
                except Exception as e:
                    st.error(f"Error optimizing {section}: {str(e)}")
                    log(f"Error: {str(e)}")
    except Exception as e:
        st.error(f"Optimization error: {str(e)}")
        log(f"Error: {str(e)}")
    finally:
        progress_bar.empty()
        log("Optimization process finished.")

# Assemble optimized script
def assemble_script() -> str:
    parts = []
    if st.session_state['script_sections']['package_installations']:
        parts.append("# Package Installations")
        parts.append(gen_requirements(st.session_state['script_sections']['package_installations']))
    parts.append("# Imports")
    parts.extend(st.session_state['optimized_sections']['imports'])
    parts.append("\n# Settings")
    parts.extend(st.session_state['optimized_sections']['settings'])
    for class_name, class_content in st.session_state['optimized_sections'].get('class_definitions', {}).items():
        parts.append(f"\n# Class: {class_name}")
        parts.append(class_content)
    for func_name, func_content in st.session_state['optimized_sections'].get('function_definitions', {}).items():
        parts.append(f"\n# Function: {func_name}")
        parts.append(func_content)
    if 'global_code' in st.session_state['script_sections']:
        parts.append("\n# Global Code")
        parts.extend(st.session_state['script_sections']['global_code'])
    return "\n\n".join(parts)

# Display logs
def display_logs():
    st.header("📋 Optimization Logs")
    for log_entry in st.session_state['logs']:
        if "Error" in log_entry:
            st.error(log_entry)
        elif "Optimizing" in log_entry or "optimized" in log_entry:
            st.success(log_entry)
        else:
            st.info(log_entry)

# Generate script dependency map
def gen_script_map() -> nx.DiGraph:
    G = nx.DiGraph()
    for func_name in st.session_state['function_list']:
        G.add_node(func_name)
        func_content = st.session_state['optimized_sections']['function_definitions'].get(func_name, "")
        for other_func in st.session_state['function_list']:
            if other_func != func_name and re.search(rf'\b{re.escape(other_func)}\b', func_content):
                G.add_edge(func_name, other_func)
    return G

# Plot interactive graph
def plot_graph(G: nx.DiGraph):
    nodes = [Node(id=n, label=n, size=1000) for n in G.nodes()]
    edges = [Edge(source=e[0], target=e[1]) for e in G.edges()]
    config = Config(width=800, height=600, directed=True, physics=True, hierarchical=False)
    return agraph(nodes=nodes, edges=edges, config=config)

# Highlight code changes
def highlight_changes(orig: str, opt: str) -> str:
    diff = unified_diff(orig.splitlines(), opt.splitlines(), lineterm='')
    return "<br>".join(
        f"<span style='background-color: {'#d4fcdc' if l.startswith('+ ') else '#fcdcdc' if l.startswith('- ') else 'transparent'}'>{l}</span>"
        for l in diff
    )

# Display function changes
def display_func_changes():
    st.header("🔍 Code Changes")
    for item_type in ['function_definitions', 'class_definitions']:
        if item_type in st.session_state['optimized_sections']:
            for name, opt_content in st.session_state['optimized_sections'][item_type].items():
                with st.expander(f"{item_type.split('_')[0].capitalize()}: {name}", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Original")
                        st.code(st.session_state['script_sections'][item_type][name], language='python')
                    with col2:
                        st.subheader("Optimized")
                        st.code(opt_content, language='python')
                    st.markdown("### Diff")
                    st.markdown(
                        highlight_changes(
                            st.session_state['script_sections'][item_type][name],
                            opt_content
                        ),
                        unsafe_allow_html=True
                    )

# Display final script
def display_final_script():
    st.header("📄 Optimized Script")
    st.code(st.session_state['final_script'], language='python')
    st.download_button(
        "💾 Download Optimized Script",
        st.session_state['final_script'],
        "optimized_script.py",
        "text/plain"
    )

# Add a function to save the final code
def save_final_code(code: str, filename: str = "optimized_script.py"):
    with open(filename, "w") as f:
        f.write(code)
    st.success(f"Saved optimized code to {filename}")

# Main interface
def main_interface(coder: Coder):
    st.title("AutocoderAI 🧑‍💻✨")
    st.markdown("**Advanced Python Script Optimizer using OpenAI's API and Aider**")
    
    tab1, tab2, tab3 = st.tabs(["Script Input", "Optimization", "Results"])
    
    with tab1:
        script_input = st_ace(
            placeholder="Paste your Python script here...",
            language="python",
            theme="monokai",
            keybinding="vscode",
            font_size=14,
            min_lines=20,
            key="ace_editor"
        )
    
    with tab2:
        if st.button("🚀 Optimize Script", key="optimize_button"):
            if not script_input.strip():
                st.error("Please paste a Python script before proceeding.")
                return
            
            if not validate_api_key(st.session_state['openai_api_key']):
                st.error("Please enter a valid OpenAI API key in the sidebar.")
                return
            
            with st.spinner("Analyzing and optimizing script..."):
                progress_bar = st.progress(0)
                st.session_state['script_sections'] = extract_sections(script_input)
                progress_bar.progress(25)
                optimize_sections(coder)
                progress_bar.progress(75)
                st.session_state['final_script'] = assemble_script()
                st.session_state['script_map'] = gen_script_map()
                progress_bar.progress(100)
            
            st.success("Script optimization completed!")
    
    with tab3:
        if st.session_state['final_script']:
            st.subheader("📊 Optimized Script Sections")
            for section, content in st.session_state['optimized_sections'].items():
                with st.expander(f"{section.capitalize()}"):
                    if isinstance(content, dict):
                        for subsection, subcontent in content.items():
                            st.session_state['optimized_sections'][section][subsection] = st.text_area(f"Edit {subsection}", subcontent, key=f"editor_{section}_{subsection}")
                    else:
                        st.session_state['optimized_sections'][section] = st.text_area(f"Edit {section}", "\n".join(content) if isinstance(content, list) else content, key=f"editor_{section}")

            if st.button("Save Changes"):
                st.session_state['final_script'] = assemble_script()
                save_final_code(st.session_state['final_script'])

            if st.button("Save Final Code"):
                save_final_code(st.session_state['final_script'])

# Sidebar settings
def sidebar_settings(coder: Coder):
    st.sidebar.header("⚙️ Settings")
    
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        value=st.session_state['openai_api_key'],
        type="password",
        help="Enter your OpenAI API key."
    )
    model = st.sidebar.selectbox(
        "OpenAI Model",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4-0613", "gpt-4-32k"],
        index=2,
        help="Select the OpenAI model to use."
    )
    endpoint = st.sidebar.text_input(
        "OpenAI Endpoint",
        value=st.session_state['openai_endpoint'],
        help="Enter the OpenAI API endpoint."
    )
    
    if st.sidebar.button("Apply Settings"):
        if validate_api_key(api_key):
            set_openai_creds(api_key, model, endpoint)
        else:
            st.sidebar.error("Invalid API Key")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 Optimization Status")
    if st.session_state['optimization_status']:
        for section, status in st.session_state['optimization_status'].items():
            st.sidebar.text(f"{section}: {status}")
    else:
        st.sidebar.info("No optimization tasks yet.")

# Initialize Aider components
def initialize_aider() -> Coder:
    io = InputOutput()
    model = models.Model.create("gpt-4")
    coder = Coder.create(main_model=model)
    aider_chat = chat.Chat(io=io, coder=coder)
    return coder

# Run app
def run_app():
    try:
        init_session_state()
        coder = initialize_aider()
        sidebar_settings(coder)
        main_interface(coder)
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        log(f"Unexpected Error: {str(e)}")

# Main execution
if __name__ == "__main__":
    run_app()

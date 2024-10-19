import os
import re
import ast
import json
import subprocess
import streamlit as st
from typing import List, Dict, Any
import openai
from aider_chat import Aider
import autogen
from langchain.text_splitter import CharacterTextSplitter
from difflib import unified_diff
import networkx as nx
import matplotlib.pyplot as plt
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from streamlit_ace import st_ace
from streamlit_agraph import agraph, Node, Edge, Config

# ------------------- Configuration -------------------

st.set_page_config(page_title="AutocoderAI", layout="wide")

# ------------------- Helper Functions -------------------

def extract_sections(script_content: str) -> Dict[str, Any]:
    """
    Parses the Python script and divides it into sections:
    - Package Installations
    - Imports
    - Settings
    - Function Definitions
    """
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

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            sections["imports"].append(ast.unparse(node))
        elif isinstance(node, ast.Assign):
            sections["settings"] += ast.unparse(node) + "\n"
        elif isinstance(node, ast.FunctionDef):
            sections["function_definitions"][node.name] = ast.unparse(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Attribute):
                if node.value.func.attr in ["system", "check_call", "run"]:
                    for arg in node.value.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            match = re.findall(r'pip install ([\w\-\.\@]+)', arg.value)
                            if match:
                                sections["package_installations"].extend(match)
    return sections

def generate_requirements(packages: List[str]) -> str:
    """
    Generates a requirements string with one-liner pip install commands.
    """
    unique_packages = sorted(set(packages))
    pip_commands = [f"pip install {pkg}" for pkg in unique_packages]
    return "\n".join(pip_commands)

def analyze_and_optimize(section_content: str, section_name: str, coder: Coder) -> str:
    """
    Uses Aider to analyze and optimize a given section of the code.
    """
    prompt = f"""
    You are an expert Python developer. Analyze the following `{section_name}` section of a Python script and optimize it. Ensure that:

    - No functions or essential components are removed.
    - Function names are not hallucinated or changed.
    - Logic is optimized for efficiency and clarity.
    - All inputs and outputs remain consistent.
    - If using ORMs like SQLAlchemy, schema integrity is maintained.
    - Improve error handling and add input validation where necessary.
    - Enhance code readability and add comments for complex logic.

    Here is the `{section_name}` section:

    ```python
    {section_content}
    ```

    Provide the optimized `{section_name}` section.
    """

    try:
        optimized_section = coder.edit(section_content, prompt)
        return optimized_section
    except Exception as e:
        st.error(f"Error during Aider optimization: {e}")
        return section_content  # Return original if error occurs

def supervise_changes(original: str, optimized: str) -> bool:
    """
    Compares the original and optimized code to ensure no unintended changes were made.
    Returns True if changes are acceptable, False otherwise.
    """
    original_lines = original.splitlines()
    optimized_lines = optimized.splitlines()
    diff = list(unified_diff(original_lines, optimized_lines, lineterm=''))
    removed = [line for line in diff if line.startswith('- ') and not line.startswith('---')]

    if removed:
        st.warning("Some lines were removed during optimization. Please review the changes.")
        st.write("**Removed Lines:**")
        st.write("\n".join(removed))
        return False
    return True

def install_packages(packages: List[str]):
    """
    Installs packages listed in the packages list.
    """
    for pkg in packages:
        command = f"pip install {pkg}"
        try:
            subprocess.run(command, shell=True, check=True)
            st.success(f"✅ Successfully installed: {pkg}")
        except subprocess.CalledProcessError as e:
            st.error(f"❌ Failed to install package: {pkg}\nError: {e}")

def initialize_session_state():
    """
    Initializes necessary session states for Streamlit.
    """
    if 'script_sections' not in st.session_state:
        st.session_state.script_sections = {}
    if 'optimized_script' not in st.session_state:
        st.session_state.optimized_script = ""
    if 'optimization_steps' not in st.session_state:
        st.session_state.optimization_steps = []
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'graph' not in st.session_state:
        st.session_state.graph = nx.DiGraph()
    if 'optimization_history' not in st.session_state:
        st.session_state.optimization_history = []

def add_log(message: str):
    """
    Adds a log message to the optimization steps.
    """
    st.session_state.optimization_steps.append(message)

def generate_script_map(function_definitions: Dict[str, str]) -> nx.DiGraph:
    """
    Generates a graph representing the relationships between functions.
    """
    graph = nx.DiGraph()
    for func_name in function_definitions.keys():
        graph.add_node(func_name)
    
    # Simple heuristic: if a function calls another, add an edge
    for func_name, func_code in function_definitions.items():
        calls = re.findall(r'\b{}\b'.format('|'.join(function_definitions.keys())), func_code)
        for call in calls:
            if call != func_name:
                graph.add_edge(func_name, call)
    return graph

def plot_graph(graph: nx.DiGraph):
    """
    Plots the graph representing the script structure.
    """
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph, k=0.5, iterations=20)
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10, arrowsize=20)
    st.pyplot(plt)

def get_diff_html(original: str, optimized: str) -> str:
    """
    Generates HTML to display diffs between original and optimized code.
    """
    diff = list(unified_diff(original.splitlines(), optimized.splitlines(), lineterm=''))
    diff_html = ""
    for line in diff:
        if line.startswith('+ ') and not line.startswith('+++'):
            diff_html += f"<span style='background-color: #d4fcdc'>{line}</span><br>"
        elif line.startswith('- ') and not line.startswith('---'):
            diff_html += f"<span style='background-color: #fcdcdc'>{line}</span><br>"
        else:
            diff_html += f"{line}<br>"
    return diff_html

def visualize_changes(original: str, optimized: str):
    """
    Visualizes changes between original and optimized code.
    """
    diff_html = get_diff_html(original, optimized)
    st.markdown("### 🛠️ Changes")
    st.markdown(diff_html, unsafe_allow_html=True)

# ------------------- New Helper Functions -------------------

def validate_api_key(api_key: str) -> bool:
    """Validate the OpenAI API key."""
    if not api_key:
        return False
    try:
        openai.api_key = api_key
        openai.Model.list()
        return True
    except:
        return False

def optimize_section_async(section_name: str, section_content: str, coder: Coder):
    """Asynchronously optimize a section of the script."""
    optimized_content = analyze_and_optimize(section_content, section_name, coder)
    return section_name, optimized_content

def optimize_script(script_input: str, optimization_strategy: str, coder: Coder) -> str:
    sections = extract_sections(script_input)
    st.session_state.script_sections = sections

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_section = {
            executor.submit(analyze_and_optimize, content, name, coder): name
            for name, content in sections.items() if name != "package_installations"
        }
        
        for future in concurrent.futures.as_completed(future_to_section):
            section_name = future_to_section[future]
            try:
                optimized_content = future.result()
                sections[section_name] = optimized_content
                add_log(f"Optimized {section_name}")
            except Exception as e:
                add_log(f"Error optimizing {section_name}: {str(e)}")

    # ... (rest of the function remains the same)

# ------------------- Streamlit UI -------------------

def main():
    st.title("AutocoderAI 🧑‍💻✨")
    st.markdown("**Automated Python Script Manager and Optimizer using OpenAI's API**")

    initialize_session_state()

    optimization_strategy = st.sidebar.selectbox(
        "Select Optimization Strategy",
        ["Basic", "Advanced", "Experimental"]
    )

    st.sidebar.subheader("OpenAI Settings")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    model = st.sidebar.selectbox("Model", ["gpt-3.5-turbo", "gpt-4"])
    
    if api_key:
        if validate_api_key(api_key):
            st.secrets["OPENAI_API_KEY"] = api_key
            st.session_state['openai_model'] = model
        else:
            st.sidebar.error("Invalid API Key")

    script_input = st_ace(
        placeholder="Paste your Python script here...",
        language="python",
        theme="monokai",
        keybinding="vscode",
        font_size=14,
        min_lines=20,
        key="ace_editor"
    )

    coder = initialize_aider()  # Add this line to initialize Aider

    if st.button("🚀 Optimize Script"):
        with st.spinner("Optimizing script..."):
            optimized_script = optimize_script(script_input, optimization_strategy, coder)
            st.session_state.optimized_script = optimized_script

    if st.session_state.optimized_script:
        st.subheader("Optimized Script Sections")
        for section, content in st.session_state.script_sections.items():
            if section != "package_installations":
                st.session_state.script_sections[section] = st.text_area(f"Edit {section}", content, key=f"editor_{section}")

        if st.button("Save Changes"):
            st.session_state.optimized_script = assemble_script()
            save_final_code(st.session_state.optimized_script)

        if st.button("Save Final Code"):
            save_final_code(st.session_state.optimized_script)

        col1, col2, col3 = st.columns(3)
        col1.metric("Lines of Code", len(script_input.splitlines()), 
                    len(st.session_state.optimized_script.splitlines()) - len(script_input.splitlines()))
        col2.metric("Functions", len(st.session_state.script_sections.get("function_definitions", {})))
        col3.metric("Imports", len(st.session_state.script_sections.get("imports", [])))

        st.subheader("Interactive Optimization Suggestions")
        user_input = st.chat_input("Ask for optimization suggestions")
        if user_input:
            with st.chat_message("user"):
                st.write(user_input)
            with st.chat_message("assistant"):
                st.write(generate_optimization_suggestion(user_input))

        st.subheader("Advanced Options")
        enable_type_hints = st.toggle("Enable Type Hints")
        enable_async = st.toggle("Enable Async Optimization")
        if enable_type_hints or enable_async:
            st.session_state.optimized_script = apply_advanced_optimizations(
                st.session_state.optimized_script, enable_type_hints, enable_async
            )

        display_optimized_script()

# ... (rest of the existing code)

if __name__ == "__main__":
    main()

# Add this function to initialize Aider
def initialize_aider() -> Coder:
    io = InputOutput()
    model = models.Model.create("gpt-4")
    coder = Coder.create(main_model=model)
    return coder

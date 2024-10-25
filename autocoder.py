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
from tqdm import tqdm
from langchain.memory import ConversationBufferMemory

# ------------------- Configuration -------------------

st.set_page_config(page_title="AutocoderAI", layout="wide")

# ------------------- State Management -------------------

class State:
    def __init__(self):
        self.script_sections = {}
        self.optimized_script = ""
        self.optimization_steps = []
        self.settings = {
            "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
            "model": "gpt-3.5-turbo",
            "coder_initialized": False,
            "show_profiling": False,
            "profiling_depth": 10
        }
        self.profiling_results = {}
        self.error_log = []
        self.logs = []
        self.memory = ConversationBufferMemory()

state = State()

# ------------------- Helper Functions -------------------

def extract_sections(script: str) -> Dict[str, Any]:
    try:
        tree = ast.parse(script)
    except SyntaxError as e:
        log_error(f"Syntax Error: {e}")
        return {}
    
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

def generate_requirements(packages: List[str]) -> str:
    """
    Generates a requirements string with one-liner pip install commands.
    """
    unique_packages = sorted(set(packages))
    pip_commands = [f"pip install {pkg}" for pkg in unique_packages]
    return "\n".join(pip_commands)

def analyze_and_optimize(section_content: str, section_name: str) -> str:
    """
    Uses OpenAI API to analyze and optimize a given section of the code.
    """
    prompt = f"""
    Analyze and optimize the following `{section_name}` section:
    ```python
    {section_content}
    ```
    Ensure efficiency, clarity, and best practices are followed.
    """

    try:
        response = openai.ChatCompletion.create(
            model=state.settings["model"],
            messages=[
                {"role": "system", "content": "You are a Python code optimization assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        optimized_section = response.choices[0].message.content
        return optimized_section
    except Exception as e:
        st.error(f"Error during OpenAI optimization: {e}")
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

def add_log(message: str):
    """
    Adds a log message to the optimization steps.
    """
    state.optimization_steps.append(message)

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

def validate_api_key(api_key: str) -> bool:
    """Validate the OpenAI API key."""
    if not api_key:
        return False
    try:
        openai.api_key = api_key
        openai.Model.list()
        return True
    except Exception as e:
        st.error(f"API Key validation failed: {str(e)}")
        return False

def validate_script(script: str) -> bool:
    """Validates script before optimization."""
    try:
        ast.parse(script)
        return True
    except SyntaxError as e:
        st.error(f"Syntax error in script: Line {e.lineno}, {e.msg}")
        return False
    except Exception as e:
        st.error(f"Error validating script: {str(e)}")
        return False

def optimize_section_async(section_name: str, section_content: str):
    """Optimizes code section with proper error handling."""
    if not section_content.strip():
        return section_name, section_content
        
    try:
        if not validate_script(section_content):
            return section_name, section_content
            
        prompt = f"""Optimize this Python code section:
        ```python
        {section_content}
        ```
        Ensure: performance, readability, PEP 8 compliance"""
        
        response = openai.ChatCompletion.create(
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
        st.error(f"Error optimizing {section_name}: {str(e)}")
        return section_name, section_content

def optimize_script(script_input: str, optimization_strategy: str) -> str:
    sections = extract_sections(script_input)
    state.script_sections = sections

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
            except Exception as e:
                st.error(f"Error optimizing {section_name}: {str(e)}")

    return "\n\n".join(sections.values())

def generate_optimization_suggestion(user_input: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model=state.settings["model"],
            messages=[
                {"role": "system", "content": "You are a Python code optimization assistant."},
                {"role": "user", "content": f"Suggest improvements for this code:\n{user_input}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating suggestion: {e}")
        return "Unable to generate suggestion at this time."

def apply_advanced_optimizations(script: str, enable_type_hints: bool, enable_async: bool) -> str:
    # Implement advanced optimizations here
    # This is a placeholder and should be expanded based on specific requirements
    return script

def assemble_script() -> str:
    """Assembles the final optimized script from sections."""
    parts = []
    
    # Add package installations if present
    if state.script_sections.get("package_installations"):
        parts.append("# Package Installations")
        parts.extend(state.script_sections["package_installations"])
    
    # Add imports
    parts.append("\n# Imports")
    parts.extend(state.script_sections.get("imports", []))
    
    # Add settings
    parts.append("\n# Settings")
    parts.append(state.script_sections.get("settings", ""))
    
    # Add function definitions
    for name, content in state.script_sections.get("function_definitions", {}).items():
        parts.append(f"\n# Function: {name}")
        parts.append(content)
    
    return "\n".join(filter(None, parts))

def save_final_code(code: str, filename: str = "optimized_script.py"):
    with open(filename, "w") as f:
        f.write(code)
    st.success(f"Saved optimized code to {filename}")

def display_optimized_script():
    st.subheader("Optimized Script")
    st.code(state.optimized_script, language="python")

def display_function_graph():
    """Displays interactive function dependency graph."""
    if not state.script_sections.get("function_definitions"):
        return
        
    graph = generate_script_map(state.script_sections["function_definitions"])
    nodes = [Node(id=n, label=n, size=1000) for n in graph.nodes()]
    edges = [Edge(source=e[0], target=e[1]) for e in graph.edges()]
    config = Config(width=800, height=400, directed=True, physics=True)
    
    st.subheader("Function Dependencies")
    agraph(nodes=nodes, edges=edges, config=config)

def log_event(message: str, state: State):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    state.optimization_steps.append(f"[{timestamp}] {message}")

def log_error(message: str):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    state.error_log.append(f"[{timestamp}] {message}")

# ------------------- Streamlit UI -------------------

def main():
    st.title("AutocoderAI 🧑‍💻✨")
    st.markdown("**Automated Python Script Manager and Optimizer using OpenAI's API**")

    optimization_strategy = st.sidebar.selectbox(
        "Select Optimization Strategy",
        ["Basic", "Advanced", "Experimental"]
    )

    st.sidebar.subheader("OpenAI Settings")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    model = st.sidebar.selectbox("Model", ["gpt-3.5-turbo", "gpt-4"])
    
    if api_key:
        if validate_api_key(api_key):
            state.settings["openai_api_key"] = api_key
            state.settings["model"] = model
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

    if st.button("🚀 Optimize Massive Script"):
        with st.spinner("Optimizing script..."):
            optimized_script = optimize_script(script_input, optimization_strategy)
            state.optimized_script = optimized_script

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
                state.script_sections[section] = st.text_area(f"Edit {section}", content, key=f"editor_{section}")

        if st.button("Save Changes"):
            state.optimized_script = assemble_script()
            save_final_code(state.optimized_script)

        if st.button("Save Final Code"):
            save_final_code(state.optimized_script)

        col1, col2, col3 = st.columns(3)
        col1.metric("Lines of Code", len(script_input.splitlines()), 
                    len(state.optimized_script.splitlines()) - len(script_input.splitlines()))
        col2.metric("Functions", len(state.script_sections.get("function_definitions", {})))
        col3.metric("Imports", len(state.script_sections.get("imports", [])))

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
            state.optimized_script = apply_advanced_optimizations(
                state.optimized_script, enable_type_hints, enable_async
            )

        display_optimized_script()

if __name__ == "__main__":
    main()

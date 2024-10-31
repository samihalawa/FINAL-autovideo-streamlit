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
from autocoder import State, optimize_script, validate_api_key, save_final_code, generate_optimization_suggestion
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Callable
from pathlib import Path
import tempfile
from contextlib import contextmanager
from functools import wraps
import pickle

# Configuration
st.set_page_config(page_title="AutocoderAI", layout="wide")
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize session state
def init_session_state():
    """Initialize session state with proper error handling"""
    try:
        defaults = {
            'script_sections': {}, 
            'optimized_sections': {}, 
            'final_script': "",
            'progress': 0, 
            'logs': [], 
            'function_list': [], 
            'current_function_index': 0,
            'openai_api_key': DEFAULT_OPENAI_API_KEY,
            'openai_model': DEFAULT_OPENAI_MODEL,
            'openai_endpoint': "https://api.openai.com/v1",
            'script_map': None,
            'optimization_status': {},
            'error_log': [],
            'chat_history': [],
            'state': State()
        }
        
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v
                
        # Validate critical settings
        if not st.session_state['openai_api_key']:
            logger.warning("OpenAI API key not set")
            
    except Exception as e:
        logger.error(f"Session state initialization error: {str(e)}")
        raise

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

# Add error handling decorator
def error_handler(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}\n{traceback.format_exc()}")
            st.error(f"An error occurred in {func.__name__}. Check logs for details.")
            return None
    return wrapper

# Add context manager for temporary files
@contextmanager
def temp_file_manager():
    temp_files = []
    try:
        yield temp_files
    finally:
        for file in temp_files:
            try:
                Path(file).unlink(missing_ok=True)
            except Exception as e:
                logger.error(f"Failed to delete temp file {file}: {e}")

# Add authentication class
class Authentication:
    def __init__(self):
        self.max_attempts = 3
        self.attempt_count = 0
        
    def validate_credentials(self, username: str, password: str) -> bool:
        # In production, replace with secure authentication
        return username == "admin" and password == "admin"
    
    @error_handler
    def authenticate(self) -> Tuple[bool, Optional[str]]:
        if 'authentication_status' not in st.session_state:
            st.session_state['authentication_status'] = False

        if not st.session_state['authentication_status']:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Login"):
                if self.validate_credentials(username, password):
                    st.session_state['authentication_status'] = True
                    st.session_state['username'] = username
                    return True, username
                else:
                    self.attempt_count += 1
                    if self.attempt_count >= self.max_attempts:
                        st.error("Maximum login attempts exceeded. Please try again later.")
                        time.sleep(30)  # Add delay after max attempts
                    else:
                        st.error("Invalid credentials")
                    return False, None
            return False, None
        
        return True, st.session_state.get('username')

# Add session management class
class SessionManager:
    def __init__(self, state: State):
        self.state = state
        self.session_dir = Path("sessions")
        self.session_dir.mkdir(exist_ok=True)
    
    @error_handler
    def save_session(self, username: str):
        session_data = {
            k: getattr(self.state, k) 
            for k in dir(self.state) 
            if not k.startswith('__') and not callable(getattr(self.state, k))
        }
        session_file = self.session_dir / f"{username}_session.pkl"
        with open(session_file, "wb") as f:
            pickle.dump(session_data, f)
        st.success("Session saved successfully.")
    
    @error_handler
    def load_session(self, username: str):
        session_file = self.session_dir / f"{username}_session.pkl"
        if session_file.exists():
            with open(session_file, "rb") as f:
                session_data = pickle.load(f)
                for k, v in session_data.items():
                    setattr(self.state, k, v)
            st.success("Session loaded successfully.")
        else:
            st.warning("No saved session found.")

# Add code optimization class
class CodeOptimizer:
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    @error_handler
    def optimize_code(self, code: str, optimization_type: str = "standard") -> Optional[str]:
        if not code.strip():
            raise ValueError("Empty code provided")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a Python code optimization expert."},
                {"role": "user", "content": f"Optimize this Python code ({optimization_type} optimization):\n\n{code}"}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        return response.choices[0].message.content

# Update main interface
def main_interface(coder: Coder):
    tabs = st.tabs(["Code Input", "Optimization", "Results"])
    
    with tabs[0]:
        with st.form("code_input_form"):
            script_input = render_code_editor()
            submit_button = st.form_submit_button("Submit Code")
            
            if submit_button and script_input.strip():
                st.session_state['script_sections'] = extract_sections(script_input)
                st.success("Code submitted successfully!")

    with tabs[1]:
        if st.button("🚀 Optimize Script") and 'script_sections' in st.session_state:
            with st.spinner("Optimizing..."):
                try:
                    optimizer = CodeOptimizer(
                        st.session_state['openai_api_key'],
                        st.session_state['openai_model']
                    )
                    optimized = optimizer.optimize_code(script_input)
                    if optimized:
                        st.session_state['final_script'] = optimized
                        st.success("Optimization complete!")
                except Exception as e:
                    st.error(f"Optimization failed: {str(e)}")
                    logger.error(f"Optimization error: {str(e)}\n{traceback.format_exc()}")

    with tabs[2]:
        if st.session_state.get('final_script'):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original vs Optimized")
                st.code(st.session_state['final_script'], language="python")
            with col2:
                if st.session_state.get('script_sections'):
                    display_dependency_graph(generate_script_map(
                        st.session_state['script_sections'].get('function_definitions', {})
                    ))

# Update run_app function
def run_app():
    """Main application runner with comprehensive error handling"""
    try:
        auth = Authentication()
        auth_status, username = auth.authenticate()
        if not auth_status:
            return

        st.title("AutocoderAI - Code Optimization")
        
        init_session_state()
        
        if not validate_api_key(st.session_state['openai_api_key']):
            st.warning("Please enter a valid OpenAI API key in settings")
            return
            
        session_manager = SessionManager(st.session_state['state'])
        
        with st.sidebar:
            if st.button("Save Session"):
                session_manager.save_session(username)
            if st.button("Load Session"):
                session_manager.load_session(username)
            
        coder = initialize_aider()
        sidebar_settings(coder)
        main_interface(coder)
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}\n{traceback.format_exc()}")
        st.error("An unexpected error occurred. Please check the logs for details.")
        
        with st.expander("Technical Details"):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    run_app()

def render_code_editor():
    """Render the code editor with advanced features."""
    editor_options = {
        "theme": st.selectbox("Editor Theme", ["monokai", "github", "twilight"]),
        "keybinding": st.selectbox("Keybinding", ["vscode", "vim", "emacs"]),
        "font_size": st.slider("Font Size", 12, 24, 14),
        "min_lines": 20,
        "max_lines": 40,
        "wrap": True,
        "auto_update": True
    }
    
    return st_ace(
        placeholder="Enter your Python code here...",
        language="python",
        **editor_options,
        key="code_editor"
    )

def render_optimization_controls():
    """Render optimization controls and options."""
    st.sidebar.subheader("Optimization Settings")
    
    optimization_options = {
        "show_profiling": st.sidebar.checkbox("Show Profiling Results", False),
        "profiling_depth": st.sidebar.slider("Profiling Depth", 5, 20, 10),
        "parallel_optimization": st.sidebar.checkbox("Parallel Optimization", True),
        "type_hints": st.sidebar.checkbox("Add Type Hints", False),
        "async_code": st.sidebar.checkbox("Use Async/Await", False)
    }
    
    if st.sidebar.button("Apply Settings"):
        state.settings.update(optimization_options)
        st.success("Settings updated!")

def optimize_script(script: str, optimization_type: str) -> str:
    try:
        if not script.strip():
            raise ValueError("Empty script provided")
            
        # Initialize OpenAI client with error handling
        client = OpenAI(api_key=st.session_state.get('openai_api_key'))
        if not client:
            raise ValueError("Failed to initialize OpenAI client")
            
        # Perform optimization
        response = client.chat.completions.create(
            model=st.session_state.get('openai_model', DEFAULT_OPENAI_MODEL),
            messages=[
                {"role": "system", "content": "You are a Python code optimization expert."},
                {"role": "user", "content": f"Optimize this Python script:\n\n{script}"}
            ]
        )
        
        optimized_code = response.choices[0].message.content
        return optimized_code
        
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}\n{traceback.format_exc()}")
        raise

def generate_script_map(functions: Dict[str, str]) -> nx.DiGraph:
    """Generate dependency graph from function definitions"""
    try:
        G = nx.DiGraph()
        for func_name, func_content in functions.items():
            G.add_node(func_name)
            # Parse function content to find dependencies
            tree = ast.parse(func_content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    called_func = node.func.id
                    if called_func in functions:
                        G.add_edge(func_name, called_func)
        return G
    except Exception as e:
        logger.error(f"Error generating script map: {str(e)}")
        return nx.DiGraph()

def display_dependency_graph(G: nx.DiGraph):
    """Display dependency graph with error handling"""
    try:
        if not G or G.number_of_nodes() == 0:
            st.warning("No dependencies to display")
            return
            
        # Create Plotly figure
        pos = nx.spring_layout(G)
        edge_trace = go.Scatter(
            x=[], y=[], line=dict(width=0.5, color='#888'),
            hoverinfo='none', mode='lines')
            
        node_trace = go.Scatter(
            x=[], y=[], text=[], mode='markers+text',
            hoverinfo='text', marker=dict(size=20))
            
        # Add edges
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += (x0, x1, None)
            edge_trace['y'] += (y0, y1, None)
            
        # Add nodes
        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += (x,)
            node_trace['y'] += (y,)
            node_trace['text'] += (node,)
            
        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                         showlegend=False,
                         hovermode='closest',
                         margin=dict(b=0,l=0,r=0,t=0),
                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                     ))
                     
        st.plotly_chart(fig)
        
    except Exception as e:
        logger.error(f"Graph visualization error: {str(e)}")
        st.error("Failed to display dependency graph")

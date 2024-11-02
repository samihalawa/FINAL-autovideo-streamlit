# AutocoderAI - Enhanced Streamlit Application for Code Optimization

import os
import sys
import ast
import re
import streamlit as st
import openai
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Callable
from difflib import unified_diff
import time
import logging
import traceback
from streamlit_ace import st_ace
import plotly.graph_objects as go
from openai import OpenAI
import json
from dataclasses import dataclass
from pathlib import Path
import tempfile
from contextlib import contextmanager
from functools import wraps
import pickle

# Configuration
st.set_page_config(page_title="AutocoderAI", layout="wide")
DEFAULT_OPENAI_MODEL = "gpt-4"  # Using standard GPT-4 model
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
            'chat_history': []
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
        "imports": [], 
        "settings": [],
        "function_definitions": {}, 
        "class_definitions": {}, 
        "global_code": []
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
        elif isinstance(node, (ast.Expr, ast.Assign)):
            sections["global_code"].append(ast.get_source_segment(script, node))
    
    return sections

# Validate API key
def validate_api_key(key: str) -> bool:
    return bool(re.match(r'^sk-[a-zA-Z0-9]{48}$', key))

# Set OpenAI credentials
def set_openai_creds(key: str, model: str, endpoint: str):
    try:
        client = OpenAI(api_key=key, base_url=endpoint)
        # Test API connection
        client.models.list()
        st.session_state.update({
            'openai_api_key': key,
            'openai_model': model, 
            'openai_endpoint': endpoint
        })
        log("OpenAI credentials set successfully.")
    except Exception as e:
        st.error(f"Error setting OpenAI credentials: {str(e)}")
        log(f"Error: {str(e)}")

# Code optimization class
class CodeOptimizer:
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
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
        # In production, implement secure authentication
        raise NotImplementedError("Implement secure authentication mechanism")
    
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
                        time.sleep(30)
                    else:
                        st.error("Invalid credentials")
                    return False, None
            return False, None
        
        return True, st.session_state.get('username')

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

def generate_script_map(functions: Dict[str, str]) -> nx.DiGraph:
    """Generate dependency graph from function definitions"""
    try:
        G = nx.DiGraph()
        for func_name, func_content in functions.items():
            G.add_node(func_name)
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
            
        pos = nx.spring_layout(G)
        edge_trace = go.Scatter(
            x=[], y=[], line=dict(width=0.5, color='#888'),
            hoverinfo='none', mode='lines')
            
        node_trace = go.Scatter(
            x=[], y=[], text=[], mode='markers+text',
            hoverinfo='text', marker=dict(size=20))
            
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += (x0, x1, None)
            edge_trace['y'] += (y0, y1, None)
            
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

@contextmanager
def safe_file_operation(filepath: str):
    """Safe file operations with cleanup"""
    temp_file = None
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        yield temp_file
    finally:
        if temp_file:
            try:
                os.unlink(temp_file.name)
            except OSError:
                pass

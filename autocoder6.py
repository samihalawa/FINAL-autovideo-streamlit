#!pip install aider langchain streamlit streamlit-ace streamlit-modal streamlit-sortables
# AI Code Enhancer Streamlit App
# ==========================
# Imports and Configuration
# ==========================

import os
import difflib
import hashlib
import datetime
import ast
import streamlit as st
import streamlit_ace as st_ace
import streamlit_modal
import streamlit_sortables
from streamlit_option_menu import option_menu
from aider.coders import Coder
from aider.main import main as cli_main
from langchain.memory import ConversationBufferMemory
import cProfile
import pstats
import io
import pickle
import logging
import git
import tempfile
from pathlib import Path
import subprocess
import streamlit_authenticator as stauth
from pyvis.network import Network
from autocoder import State, optimize_script, validate_api_key, save_final_code, generate_optimization_suggestion
from aider import chat
import time
import plotly.graph_objects as go
from streamlit_agraph import agraph, Node, Edge, Config
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
import sys
from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# Streamlit Page Configuration
st.set_page_config(
    page_title="AI Code Enhancer",
    page_icon=":robot_face:",
    layout="wide",
)

# Configure logging
logging.basicConfig(
    filename='autocoder6.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ==========================
# State Management
# ==========================
@dataclass
class State:
    settings: dict = field(default_factory=lambda: {
        "model": "gpt-3.5-turbo",
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "theme": "Light",
        "session_timeout": 30,
        "coder_initialized": False,
        "show_profiling": False
    })
    code_blocks: dict = field(default_factory=dict)
    block_versions: dict = field(default_factory=dict)
    processing_status: dict = field(default_factory=dict)
    custom_prompts: dict = field(default_factory=dict)
    ordered_blocks: list = field(default_factory=list)
    retry_counts: dict = field(default_factory=dict)
    global_prompt: str = ""
    undo_stack: dict = field(default_factory=dict)
    redo_stack: dict = field(default_factory=dict)
    tags: dict = field(default_factory=dict)
    agent_outputs: dict = field(default_factory=dict)
    logs: list = field(default_factory=list)
    session_start: datetime = field(default_factory=datetime.now)
    coder: Optional[Coder] = None
    profiling_results: dict = field(default_factory=dict)
    unused_imports: dict = field(default_factory=dict)
    memory: ConversationBufferMemory = field(default_factory=ConversationBufferMemory)
    profiling_results_depth: int = 10
    show_logs: bool = True

    def is_session_expired(self) -> bool:
        """Check if the current session has expired"""
        timeout = timedelta(minutes=self.settings["session_timeout"])
        return datetime.now() - self.session_start > timeout

@st.cache_resource
def get_state():
    return State()

state = get_state()

# ==========================
# Helper Functions
# ==========================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def log_event(event, state):
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event": event
    }
    state.logs.append(log_entry)
    if state.show_logs:
        logging.info(f"Event: {event}")

def log_error(error_message: str):
    logging.error(error_message)
    st.error(f"Error: {error_message}")

def safe_execute(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log_error(str(e))
    return wrapper

def segment_code(code):
    """
    Segments the input code into blocks using the AST module.
    Handles imports, classes, functions, and global code.
    """
    segments = {}
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        st.error(f"Syntax error in code: {e}")
        return {}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            block_name = "Imports"
            if block_name not in segments:
                segments[block_name] = []
            segments[block_name].append(ast.get_source_segment(code, node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            block_name = f"def {node.name}"
            segments[block_name] = ast.get_source_segment(code, node)
        elif isinstance(node, ast.ClassDef):
            block_name = f"class {node.name}"
            segments[block_name] = ast.get_source_segment(code, node)
        else:
            # For any other top-level code
            block_name = "Global"
            if block_name not in segments:
                segments[block_name] = []
            segments[block_name].append(ast.get_source_segment(code, node))

    # Combine segments that are lists
    for key in segments:
        if isinstance(segments[key], list):
            segments[key] = '\n'.join(segments[key])
    return segments

def show_diff(block_name):
    """
    Shows the differences between the last two versions of a code block.
    """
    versions = state.block_versions.get(block_name, [])
    if len(versions) < 2:
        st.info("No changes to show.")
    else:
        old_version = versions[-2]
        new_version = versions[-1]
        diff = difflib.unified_diff(
            old_version.splitlines(),
            new_version.splitlines(),
            lineterm='',
            fromfile='Original',
            tofile='Edited',
        )
        with streamlit_modal.modal(key=f"modal_diff_{block_name}", title=f"Changes in {block_name}"):
            st.code('\n'.join(diff), language='diff')

def undo_block(block_name):
    """
    Undoes the last change to a code block.
    """
    versions = state.block_versions.get(block_name, [])
    if len(versions) > 1:
        if block_name not in state.redo_stack:
            state.redo_stack[block_name] = []
        state.redo_stack[block_name].append(versions.pop())
        state.code_blocks[block_name] = versions[-1]
        state.processing_status[block_name] = "Undone"
        st.success(f"Undo successful for {block_name}.")
        log_event(f"Undid changes for {block_name}", state)
    else:
        st.warning("Nothing to undo.")

def redo_block(block_name):
    """
    Redoes the last undone change to a code block.
    """
    if block_name in state.redo_stack and state.redo_stack[block_name]:
        version = state.redo_stack[block_name].pop()
        state.block_versions[block_name].append(version)
        state.code_blocks[block_name] = version
        state.processing_status[block_name] = "Redone"
        st.success(f"Redo successful for {block_name}.")
        log_event(f"Redid changes for {block_name}", state)
    else:
        st.warning("Nothing to redo.")

def generate_prompt(block_name):
    """
    Generates a prompt for the AI agent to process a code block.
    """
    custom_prompt = state.custom_prompts.get(block_name, "")
    if custom_prompt:
        return custom_prompt
    else:
        return state.global_prompt

def initialize_coder():
    """
    Initializes the Coder instance from the 'aider' package.
    """
    if state.settings["openai_api_key"]:
        os.environ['OPENAI_API_KEY'] = state.settings["openai_api_key"]
    else:
        st.error("Please provide your OpenAI API Key in the Settings.")
        st.stop()
    # Ensure the OpenAI API key is set
    try:
        state.coder = cli_main(return_coder=True)
        if not isinstance(state.coder, Coder):
            st.error("Failed to initialize the Coder.")
            st.stop()
        else:
            state.settings["coder_initialized"] = True
    except Exception as e:
        st.error(f"Failed to initialize the Coder: {e}")
        st.stop()

# ==========================
# Code Segmentation and Management
# ==========================

SUPPORTED_LANGUAGES = ["python", "javascript", "java", "c++", "ruby"]

def segment_code_multilang(code: str, language: str) -> dict:
    segments = {}
    if language == "python":
        segments = segment_code(code)
    elif language in ["javascript", "java", "c++", "ruby"]:
        # Placeholder for language-specific segmentation
        segments["Global"] = code
    else:
        segments["Global"] = code
    return segments

# ==========================
# Version Control Integration
# ==========================

def initialize_git_repo(repo_path: str):
    if not Path(os.path.join(repo_path, '.git')).exists():
        repo = git.Repo.init(repo_path)
        st.success("Initialized empty Git repository.")
    else:
        repo = git.Repo(repo_path)
    return repo

def commit_changes(repo: git.Repo, message: str):
    repo.git.add(all=True)
    repo.index.commit(message)
    st.success(f"Changes committed: {message}")

def view_commit_history(repo: git.Repo):
    commits = list(repo.iter_commits('master'))
    for commit in commits:
        st.write(f"**Commit:** {commit.hexsha[:7]}")
        st.write(f"**Author:** {commit.author.name}")
        st.write(f"**Date:** {datetime.datetime.fromtimestamp(commit.committed_date).strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"**Message:** {commit.message}")
        st.write("---")

# ==========================
# Code Formatting
# ==========================

def format_code_with_black(code: str) -> str:
    try:
        process = subprocess.run(
            ['black', '-', '--quiet'],
            input=code.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        formatted_code = process.stdout.decode()
        return formatted_code
    except subprocess.CalledProcessError as e:
        st.error(f"Black formatting failed: {e.stderr.decode()}")
        return code

# ==========================
# Agent Classes
# ==========================

class SupervisorAgent:
    def __init__(self, planner_agent, editor_agent, verifier_agent, profiler_agent, dependency_checker_agent):
        self.planner_agent = planner_agent
        self.editor_agent = editor_agent
        self.verifier_agent = verifier_agent
        self.profiler_agent = profiler_agent
        self.dependency_checker_agent = dependency_checker_agent
        self.memory = state.memory

    @safe_execute
    def execute_workflow(self, block_name, block_content):
        try:
            plan = self.planner_agent.analyze_blocks({block_name: block_content}).get(block_name, None)
            if plan == 'edit':
                edited_block, agent_output = self.editor_agent.edit_block(block_name, block_content, generate_prompt(block_name))
                state.agent_outputs[block_name] = agent_output
                self.memory.add(f"Edited {block_name}: {agent_output}")

                profiling_output = self.profiler_agent.profile_block(block_name, edited_block)
                state.profiling_results[f"{block_name}_profile_after"] = profiling_output

                unused_imports = self.dependency_checker_agent.check_dependencies({block_name: edited_block})
                if unused_imports:
                    st.warning(f"Unused imports detected in {block_name}: {unused_imports}")

                if self.verifier_agent.verify_block(block_name, edited_block):
                    state.block_versions[block_name].append(edited_block)
                    state.code_blocks[block_name] = edited_block
                    state.processing_status[block_name] = "Completed"
                    log_event(f"Processed block {block_name} successfully", state)
                else:
                    st.error(f"Verification failed for {block_name}.")
            else:
                st.warning(f"No action needed for {block_name}")
        except Exception as e:
            st.error(f"Error processing block {block_name}: {e}")
            log_event(f"Error processing block {block_name}: {e}", state)

class PlannerAgent:
    def analyze_blocks(self, code_blocks):
        """
        Analyzes code blocks and decides on actions to take.
        """
        plan = {}
        for block_name, block_content in code_blocks.items():
            plan[block_name] = 'edit'  # Placeholder logic
        return plan

class EditorAgent:
    @safe_execute
    def edit_block(self, block_name, block_content, prompt):
        """
        Edits a code block using the AI agent.
        """
        if not state.settings.get("coder_initialized"):
            initialize_coder()
        try:
            edited_block = state.coder.run(prompt + "\n\n" + block_content)
            agent_output = f"Edited {block_name} using AI agent with prompt: {prompt}"
            return edited_block, agent_output
        except Exception as e:
            raise Exception(f"Editor Agent failed: {str(e)}")

class VerifierAgent:
    def verify_block(self, block_name, block_content):
        """
        Verifies the syntax of a code block.
        """
        try:
            compile(block_content, '<string>', 'exec')
            return True
        except SyntaxError as e:
            st.error(f"Syntax error in {block_name}: {e}")
            return False

class ProfilerAgent:
    def profile_block(self, block_name, block_content):
        """
        Profiles the execution of a code block.
        """
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            exec(block_content, {})
        except Exception as e:
            st.warning(f"Profiling of {block_name} failed: {e}")
        profiler.disable()
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(10)
        return s.getvalue()

class DependencyCheckerAgent:
    def check_dependencies(self, code_blocks: Dict[str, str]) -> Dict[str, List[str]]:
        """Analyzes code blocks for unused imports and dependencies."""
        unused_imports = {}
        for block_name, content in code_blocks.items():
            try:
                tree = ast.parse(content)
                imports = set()
                used_names = set()
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        for name in node.names:
                            imports.add(name.name)
                    elif isinstance(node, ast.Name):
                        used_names.add(node.id)
                
                unused = imports - used_names
                if unused:
                    unused_imports[block_name] = list(unused)
                    
            except Exception as e:
                log_error(f"Dependency check failed for {block_name}: {str(e)}")
                
        return unused_imports

@st.cache_resource
def get_agents():
    planner_agent = PlannerAgent()
    editor_agent = EditorAgent()
    verifier_agent = VerifierAgent()
    profiler_agent = ProfilerAgent()
    dependency_checker_agent = DependencyCheckerAgent()
    supervisor_agent = SupervisorAgent(planner_agent, editor_agent, verifier_agent, profiler_agent, dependency_checker_agent)
    return supervisor_agent

supervisor_agent = get_agents()

# ==========================
# GUI Components
# ==========================

def authenticate_user():
    """Handle user authentication."""
    if 'authentication_status' not in st.session_state:
        st.session_state['authentication_status'] = False

    if not st.session_state['authentication_status']:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            # Simple authentication for demo
            if username == "admin" and password == "admin":
                st.session_state['authentication_status'] = True
                st.session_state['username'] = username
                return True, username
            else:
                st.error("Invalid credentials")
                return False, None
        return False, None
    
    return True, st.session_state.get('username')

def render_settings():
    st.title("Settings")
    st.write("Configure application settings.")
    
    # Existing settings inputs...
    
    st.header("Advanced Settings")
    
    # AI Prompt Customization
    st.subheader("AI Prompt Customization")
    state.global_prompt = st.text_area(
        "Global AI Prompt",
        value=state.global_prompt,
        help="This prompt will be used as the default for all code block enhancements."
    )
    
    custom_prompts = state.custom_prompts.copy()
    for block_name in state.ordered_blocks:
        prompt = st.text_area(
            f"Custom Prompt for {block_name}",
            value=custom_prompts.get(block_name, ""),
            key=f"prompt_{block_name}",
            help=f"Customize the AI prompt for the {block_name} block."
        )
        if prompt:
            custom_prompts[block_name] = prompt
        else:
            custom_prompts.pop(block_name, None)
    state.custom_prompts = custom_prompts
    
    # Profiling Depth
    st.subheader("Profiling Configuration")
    profiling_depth = st.slider(
        "Profiling Depth",
        min_value=1,
        max_value=20,
        value=state.profiling_results_depth,
        help="Number of profiling lines to display."
    )
    state.profiling_results_depth = profiling_depth
    
    # User Preferences
    st.subheader("User Preferences")
    show_logs = st.checkbox("Show Logs", value=state.show_logs)
    state.show_logs = show_logs
    
    if st.button("Apply Settings"):
        # ... (update settings)
        initialize_coder()
        st.success("Settings applied successfully.")
        log_event("Applied new settings", state)

    st.header("Session Management")
    username = st.text_input("Username for Session")
    if st.button("Save Session"):
        if username:
            save_session(state, username)
        else:
            st.warning("Please enter a username to save the session.")
    
    if st.button("Load Session"):
        if username:
            load_session(state, username)
        else:
            st.warning("Please enter a username to load the session.")

def render_input_code():
    st.title("Input Code")
    language = st.selectbox("Select Programming Language", SUPPORTED_LANGUAGES, index=SUPPORTED_LANGUAGES.index("python"))
    code_input = st_ace.st_ace(
        language=language,
        theme='monokai' if state.settings["theme"] == "Dark" else 'github',
        placeholder="Paste your code here...",
        height=400,
        key="code_input",
    )
    if st.button("Segment Code"):
        if code_input.strip():
            segments = segment_code_multilang(code_input, language)
            if segments:
                state.code_blocks = segments
                state.ordered_blocks = list(segments.keys())
                for block_name in segments:
                    state.processing_status[block_name] = "Pending"
                    state.retry_counts[block_name] = 0
                    state.block_versions[block_name] = [segments[block_name]]
                st.success("Code segmented successfully.")
                log_event("Segmented input code", state)
            else:
                st.warning("No valid code segments found.")
        else:
            st.warning("Please input code to segment.")

def render_code_blocks():
    st.title("Code Blocks")
    
    if not state.code_blocks:
        st.info("No code blocks found. Please input code and segment it first.")
        return
    
    # Synchronize with shared state (for collaboration)
    sync_code_blocks(shared_state)
    
    # Reorder blocks
    st.subheader("Reorder Code Blocks")
    ordered_blocks = streamlit_sortables.sort_items(
        items=state.ordered_blocks,
        direction='vertical',
        key='reorder_blocks',
    )
    state.ordered_blocks = ordered_blocks
    update_shared_state()
    
    for block_name in state.ordered_blocks:
        block_content = state.code_blocks[block_name]
        with st.expander(f"{block_name}", expanded=False):
            new_content = st_ace.st_ace(
                value=block_content,
                language='python',
                theme='monokai' if state.settings["theme"] == "Dark" else 'github',
                height=300,
                key=f"ace_{block_name}"
            )
            if new_content != block_content:
                state.block_versions[block_name].append(new_content)
                state.code_blocks[block_name] = new_content
                state.processing_status[block_name] = "Modified"
                st.success(f"Changes saved for {block_name}")
                log_event(f"Modified {block_name}", state)
                update_shared_state()

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button(f"Undo {block_name}", key=f"undo_{block_name}"):
                    undo_block(block_name)
            with col2:
                if st.button(f"Redo {block_name}", key=f"redo_{block_name}"):
                    redo_block(block_name)
            with col3:
                if st.button(f"Show Changes {block_name}", key=f"show_changes_{block_name}"):
                    show_diff(block_name)
            with col4:
                if st.button(f"Format {block_name}", key=f"format_{block_name}"):
                    formatted = format_code_with_black(new_content)
                    state.block_versions[block_name].append(formatted)
                    state.code_blocks[block_name] = formatted
                    st.success(f"Formatted {block_name} successfully.")
                    log_event(f"Formatted {block_name}", state)
                    update_shared_state()
            
            if st.button(f"Execute {block_name}"):
                safe_execute(execute_code)(block_name, new_content)
                output = execute_code(block_name, new_content)
                st.text_area(f"Output for {block_name}", value=output, height=200)

def render_graph_view():
    st.title("Graph Visualization")
    net = Network(height='600px', width='100%', directed=True, notebook=False)
    
    colors = {"def ": "lightblue", "class ": "orange", "Imports": "green"}
    for block_name in state.ordered_blocks:
        color = next((v for k, v in colors.items() if block_name.startswith(k)), "gray")
        net.add_node(block_name, label=block_name, color=color)
    
    net.add_edges([(state.ordered_blocks[i], state.ordered_blocks[i + 1]) for i in range(len(state.ordered_blocks) - 1)])
    
    net.set_options("""
    var options = {
      "nodes": {"shape": "dot", "size": 16},
      "edges": {
        "arrows": {"to": {"enabled": true}},
        "color": {"color": "#848484", "highlight": "#848484", "inherit": false, "opacity": 0.8},
        "smooth": {"enabled": false}
      },
      "interaction": {"hover": true, "zoomView": true, "dragNodes": true, "multiselect": true},
      "physics": {
        "enabled": true,
        "barnesHut": {"gravitationalConstant": -8000, "centralGravity": 0.3, "springLength": 95}
      }
    }
    """)
    
    graph_path = os.path.join(tempfile.gettempdir(), "enhanced_graph.html")
    net.show(graph_path)
    with open(graph_path, "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=600, scrolling=True)

def render_help_page():
    st.title("Help & Documentation")
    
    sections = {
        "Getting Started": """
        Welcome to the AI Code Enhancer! Follow these steps:
        1. **Settings:** Configure API keys, select AI model, adjust preferences.
        2. **Input Code:** Paste Python code or select another supported language.
        3. **Segment Code:** Divide code into manageable blocks for targeted enhancements.
        4. **Code Blocks:** Edit, undo, redo, and view changes for each block.
        5. **Graph Visualization:** Visualize codebase structure and dependencies.
        6. **Export:** Download enhanced code and reports for external use.
        """,
        "Feature Guides": {
            "AI Code Enhancement": "Leverages advanced AI models to analyze and optimize your code. Customize prompts for specific improvements.",
            "Version Control": "Integrated Git version control for tracking changes, committing updates, and viewing codebase history within the app."
        },
        "Troubleshooting": """
        - **Authentication Issues:** Ensure correct username and password.
        - **API Errors:** Verify valid OpenAI API key with necessary permissions.
        - **Formatting Failures:** Check for syntax errors before formatting.
        - **Execution Errors:** Review error messages during code execution to identify issues.
        """,
        "Contact and Support": "For further assistance, contact our support team at [support@example.com](mailto:support@example.com)."
    }
    
    for title, content in sections.items():
        st.header(title)
        if isinstance(content, dict):
            for subtitle, subcontent in content.items():
                st.subheader(subtitle)
                st.write(subcontent)
        else:
            st.write(content)

def save_session(state: State, username: str):
    session_data = {k: getattr(state, k) for k in dir(state) if not k.startswith('__') and not callable(getattr(state, k))}
    with open(f"{username}_session.pkl", "wb") as f:
        pickle.dump(session_data, f)
    st.success("Session saved successfully.")

def load_session(state: State, username: str):
    session_file = f"{username}_session.pkl"
    if Path(session_file).exists():
        with open(session_file, "rb") as f:
            session_data = pickle.load(f)
            for k, v in session_data.items():
                setattr(state, k, v)
        st.success("Session loaded successfully.")

shared_state = {"code_blocks": {}, "block_versions": {}, "processing_status": {}, "ordered_blocks": []}

def sync_code_blocks(shared_state: dict):
    for k, v in shared_state.items():
        setattr(state, k, v)

def update_shared_state():
    for k in shared_state.keys():
        shared_state[k] = getattr(state, k)

def save_session_state():
    """Save current session state."""
    if 'username' in st.session_state:
        session_data = {
            'script_sections': st.session_state.get('script_sections', {}),
            'optimized_sections': st.session_state.get('optimized_sections', {}),
            'final_script': st.session_state.get('final_script', ''),
            'optimization_status': st.session_state.get('optimization_status', {})
        }
        
        with open(f"{st.session_state['username']}_session.pkl", 'wb') as f:
            pickle.dump(session_data, f)

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

def optimize_section(self, name: str, content: str) -> str:
    """Optimize a single section with error handling."""
    try:
        if not content.strip():
            return content
                
        prompt = self.generate_prompt(name, content)
        response = openai.ChatCompletion.create(
            model=self.state.settings["model"],
            messages=[
                {"role": "system", "content": "You are a Python optimization expert."},
                {"role": "user", "content": prompt}
            ]
        )
        
        optimized = response.choices[0].message.content.strip()
        if not self.validate_section(optimized):
            return content
                
        return optimized
    except Exception as e:
        self.log_error(f"optimization_{name}", e)
        return content

class OptimizationWorkflow:
    def __init__(self, state: State):
        self.state = state
        self.history = []
        
    def execute(self, script: str) -> str:
        """Execute the complete optimization workflow."""
        try:
            # Extract and validate
            sections = extract_sections(script)
            if not sections:
                raise ValueError("No valid code sections found")
            
            # Optimize sections
            optimized_sections = {}
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(self.optimize_section, name, content): name
                    for name, content in sections.items()
                }
                
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        optimized = future.result()
                        optimized_sections[name] = optimized
                        self.log_success(name)
                    except Exception as e:
                        self.log_error(name, e)
                        optimized_sections[name] = sections[name]
            
            # Profile if enabled
            if self.state.settings["show_profiling"]:
                self.profile_sections(optimized_sections)
            
            # Assemble final script
            return self.assemble_script(optimized_sections)
            
        except Exception as e:
            self.log_error("workflow", e)
            return script
    
    def optimize_section(self, name: str, content: str) -> str:
        """Optimize a single section with error handling."""
        try:
            if not content.strip():
                return content
                
            prompt = self.generate_prompt(name, content)
            response = openai.ChatCompletion.create(
                model=self.state.settings["model"],
                messages=[
                    {"role": "system", "content": "You are a Python optimization expert."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            optimized = response.choices[0].message.content.strip()
            if not self.validate_section(optimized):
                return content
                
            return optimized
        except Exception as e:
            self.log_error(f"optimization_{name}", e)
            return content

    def validate_section(self, code: str) -> bool:
        """Validate that the optimized code is syntactically correct"""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def log_success(self, name: str) -> None:
        """Log successful optimization"""
        log_event(f"Successfully optimized section: {name}", self.state)

    def log_error(self, name: str, error: Exception) -> None:
        """Log optimization error"""
        error_msg = f"Error optimizing {name}: {str(error)}"
        log_event(error_msg, self.state)
        log_error(error_msg)

    def generate_prompt(self, name: str, content: str) -> str:
        """Generate optimization prompt for a code section"""
        return f"""Please optimize the following Python code section '{name}':
        
{content}

Focus on:
1. Performance improvements
2. Code readability
3. Best practices
4. Memory efficiency

Return only the optimized code without explanations."""

    def profile_sections(self, sections: Dict[str, str]) -> None:
        """Profile optimized code sections"""
        for name, content in sections.items():
            try:
                profiler = ProfilerAgent()
                result = profiler.profile_block(name, content)
                self.state.profiling_results[name] = result
            except Exception as e:
                self.log_error(name, e)

    def assemble_script(self, sections: Dict[str, str]) -> str:
        """Assemble optimized sections into final script"""
        parts = []
        if sections.get("imports"):
            parts.extend(sections["imports"])
        if sections.get("settings"):
            parts.append(sections["settings"])
        if sections.get("class_definitions"):
            parts.extend(sections["class_definitions"].values())
        if sections.get("function_definitions"):
            parts.extend(sections["function_definitions"].values())
        if sections.get("global_code"):
            parts.extend(sections["global_code"])
        
        return "\n\n".join(parts)

def execute_code(block_name: str, code: str) -> str:
    """
    Safely executes a code block and returns the output.
    """
    output = io.StringIO()
    try:
        # Redirect stdout to capture output
        sys.stdout = output
        # Create a restricted globals dictionary
        restricted_globals = {
            '__builtins__': {
                name: __builtins__[name] 
                for name in ['print', 'len', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple']
            }
        }
        exec(code, restricted_globals)
        return output.getvalue()
    except Exception as e:
        return f"Error executing code: {str(e)}"
    finally:
        sys.stdout = sys.__stdout__

def check_session_expiry():
    """Check and handle session expiration"""
    if state.is_session_expired():
        st.warning("Session expired. Please log in again.")
        st.session_state['authentication_status'] = False
        return False
    return True

def main():
    auth_status, username = authenticate_user()
    if not auth_status:
        st.stop()

    if not check_session_expiry():
        st.stop()

    # Reset session timer on activity
    state.session_start = datetime.now()

    menu_items = ["Settings", "Input Code", "Code Blocks", "Graph Visualization", "Help"]
    icons = ['gear', 'input-cursor-text', 'code-slash', 'share', 'question-circle']
    selected = option_menu("Main Menu", menu_items, icons=icons, menu_icon="menu-button", default_index=0, orientation="horizontal")

    render_functions = {
        "Settings": render_settings,
        "Input Code": render_input_code,
        "Code Blocks": render_code_blocks,
        "Graph Visualization": render_graph_view,
        "Help": render_help_page
    }
    
    try:
        render_functions[selected]()
    except Exception as e:
        log_error(f"Error rendering {selected}: {str(e)}")
        st.error("An error occurred. Please try again or contact support.")

if __name__ == "__main__":
    main()

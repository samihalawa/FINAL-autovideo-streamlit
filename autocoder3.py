# AutocoderAI - Advanced Streamlit Application for Code Optimization

from typing import Dict, List, Any, Optional, Union, Callable, TypeVar, cast
import os
import ast
import re
import streamlit as st
import openai
import networkx as nx
from difflib import unified_diff
import time
import threading
from streamlit_ace import st_ace
from streamlit_agraph import agraph, Node, Edge, Config
from aider.io import InputOutput
from aider import models, chat
from aider.coders import Coder
from aider.handlers import ConversationHandler
from aider.repomap import RepoMap
from openai import OpenAI
import subprocess
import sys
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
import concurrent.futures
from streamlit_folium import folium_static
import folium
import plotly.graph_objects as go
import json
import functools
import logging
from pathlib import Path
import traceback
from dataclasses import dataclass, field, asdict
from typing_extensions import TypedDict
from contextlib import contextmanager

# Configure logging with proper error handling
try:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('autocoder.log', mode='a', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
except Exception as e:
    print(f"Failed to configure logging: {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)

# Type definitions
T = TypeVar('T')
SectionType = Dict[str, Union[List[str], Dict[str, str]]]

# Constants with validation
DEFAULT_OPENAI_MODEL = "gpt-4"
DEFAULT_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MAX_RETRIES = 3
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 200

# Validate environment
if not DEFAULT_OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY environment variable not set")

@dataclass(frozen=True)
class AppState:
    """Immutable application state management with type hints"""
    script_sections: Dict[str, Any] = field(default_factory=dict)
    optimized_sections: Dict[str, Any] = field(default_factory=dict)
    final_script: str = ""
    progress: float = 0.0
    logs: List[str] = field(default_factory=list)
    function_list: List[str] = field(default_factory=list)
    current_function_index: int = 0
    openai_api_key: str = DEFAULT_OPENAI_API_KEY
    openai_model: str = DEFAULT_OPENAI_MODEL
    openai_endpoint: str = "https://api.openai.com/v1"
    script_map: Optional[nx.DiGraph] = None
    optimization_status: Dict[str, str] = field(default_factory=dict)
    error_log: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate state after initialization"""
        if not isinstance(self.progress, float) or not 0 <= self.progress <= 1:
            raise ValueError("Progress must be a float between 0 and 1")

class ThreadSafeStateManager:
    """Thread-safe state management with proper locking"""
    
    def __init__(self):
        self._lock = threading.RLock()
        
    @contextmanager
    def state_lock(self):
        """Context manager for thread-safe state access"""
        with self._lock:
            yield
    
    def init_state(self) -> None:
        """Initialize session state with default values"""
        with self.state_lock():
            if not hasattr(st.session_state, "_initialized"):
                default_state = AppState()
                for field_name, field_value in asdict(default_state).items():
                    if field_name not in st.session_state:
                        st.session_state[field_name] = field_value
                st.session_state._initialized = True

    def update_state(self, key: str, value: Any) -> None:
        """Thread-safe state update with validation"""
        with self.state_lock():
            if key not in asdict(AppState()):
                raise KeyError(f"Invalid state key: {key}")
            st.session_state[key] = value
            logger.debug(f"Updated state: {key}={value}")

state_manager = ThreadSafeStateManager()

def log_error(e: Exception, context: str = "") -> None:
    """Centralized error logging with context"""
    error_msg = f"{context}: {str(e)}\n{traceback.format_exc()}"
    logger.error(error_msg)
    with state_manager.state_lock():
        st.session_state.error_log.append(error_msg)
    st.error(error_msg)

def safe_ast_parse(script: str) -> Optional[ast.AST]:
    """Safely parse Python code with detailed error handling"""
    try:
        return ast.parse(script)
    except SyntaxError as e:
        log_error(e, f"Syntax error at line {e.lineno}, column {e.offset}: {e.text}")
        return None
    except Exception as e:
        log_error(e, "Failed to parse script")
        return None

def extract_sections(script: str) -> Dict[str, Any]:
    """Extract code sections with comprehensive error handling"""
    if not isinstance(script, str):
        raise TypeError("Script must be a string")
        
    tree = safe_ast_parse(script)
    if tree is None:
        return {}
        
    sections: SectionType = {
        "package_installations": [],
        "imports": [],
        "settings": [],
        "function_definitions": {},
        "class_definitions": {},
        "global_code": []
    }
    
    try:
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                sections["imports"].append(cast(str, ast.get_source_segment(script, node)))
            elif isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
                sections["settings"].append(cast(str, ast.get_source_segment(script, node)))
            elif isinstance(node, ast.FunctionDef):
                func_code = cast(str, ast.get_source_segment(script, node))
                sections["function_definitions"][node.name] = func_code
                with state_manager.state_lock():
                    if node.name not in st.session_state.function_list:
                        st.session_state.function_list.append(node.name)
            elif isinstance(node, ast.ClassDef):
                sections["class_definitions"][node.name] = cast(str, ast.get_source_segment(script, node))
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Attribute) and node.value.func.attr in ["system", "check_call", "run"]:
                    for arg in node.value.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            match = re.findall(r'pip install ([\w\-\.\@]+)', arg.value)
                            if match:
                                sections["package_installations"].extend(match)
            elif isinstance(node, (ast.Expr, ast.Assign)):
                sections["global_code"].append(cast(str, ast.get_source_segment(script, node)))
                
    except Exception as e:
        log_error(e, "Error extracting sections")
        return {}
        
    return sections

def validate_api_key(key: str) -> bool:
    """Validate OpenAI API key format with strict checking"""
    if not isinstance(key, str):
        return False
    return bool(re.match(r'^sk-[A-Za-z0-9]{48}$', key.strip()))

@dataclass(frozen=True)
class OptimizationResult:
    """Immutable optimization result with validation"""
    content: str
    success: bool
    error: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.content, str):
            raise TypeError("Content must be a string")
        if not isinstance(self.success, bool):
            raise TypeError("Success must be a boolean")
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("Error must be a string or None")

def analyze_and_optimize(content: str, name: str, coder: Coder) -> OptimizationResult:
    """Analyze and optimize code with comprehensive error handling"""
    if not content or not name or not coder:
        return OptimizationResult(
            content=content,
            success=False,
            error="Invalid input parameters"
        )
    
    try:
        prompt = (
            f"Optimize the following `{name}` section:\n\n```python\n{content}\n```\n\n"
            "Consider:\n"
            "1. Time/space complexity optimization\n"
            "2. Code readability and maintainability\n"
            "3. PEP 8 compliance\n"
            "4. Modern Python features (3.7+)\n"
            "5. Comprehensive error handling\n"
            "6. Type hints and documentation\n"
            "7. Import optimization\n"
            "8. Efficient data structures\n"
            "9. Proper caching mechanisms\n"
            "10. Thread safety considerations"
        )
        
        optimized_content = coder.edit(content, prompt)
        if not optimized_content:
            raise ValueError("Optimization returned empty content")
            
        # Validate optimized code
        if not safe_ast_parse(optimized_content):
            raise ValueError("Optimized code contains syntax errors")
            
        return OptimizationResult(content=optimized_content, success=True)
        
    except Exception as e:
        error_msg = f"Optimization error in {name}: {str(e)}"
        log_error(e, error_msg)
        return OptimizationResult(content=content, success=False, error=error_msg)

def optimize_sections(coder: Coder) -> None:
    """Optimize code sections with robust error handling and progress tracking"""
    if not isinstance(coder, Coder):
        log_error(TypeError("Invalid coder instance"), "Optimization initialization failed")
        return

    progress_bar = st.progress(0)
    total_sections = sum(
        len(sections) for sections in [
            st.session_state.script_sections.get('function_definitions', {}),
            st.session_state.script_sections.get('class_definitions', {}),
        ]
    ) + 2  # +2 for imports and settings

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(os.cpu_count() or 1, 4)) as executor:
            future_to_section = {}
            
            # Prepare optimization tasks
            sections_to_optimize = [
                ('imports', '\n'.join(st.session_state.script_sections.get('imports', []))),
                ('settings', '\n'.join(st.session_state.script_sections.get('settings', []))),
            ]
            
            sections_to_optimize.extend(
                (f"function {func}", func_content)
                for func, func_content in st.session_state.script_sections.get('function_definitions', {}).items()
            )
            
            sections_to_optimize.extend(
                (f"class {cls}", cls_content)
                for cls, cls_content in st.session_state.script_sections.get('class_definitions', {}).items()
            )
            
            # Submit tasks
            for section_name, content in sections_to_optimize:
                if content.strip():  # Only optimize non-empty sections
                    future = executor.submit(analyze_and_optimize, content, section_name, coder)
                    future_to_section[future] = section_name
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_section):
                section_name = future_to_section[future]
                try:
                    result = future.result()
                    if result.success:
                        with state_manager.state_lock():
                            section_type = section_name.split()[0] + ('s' if section_name in ['imports', 'settings'] else '')
                            if section_type not in st.session_state.optimized_sections:
                                st.session_state.optimized_sections[section_type] = {}
                            
                            if section_name in ['imports', 'settings']:
                                st.session_state.optimized_sections[section_type] = result.content.splitlines()
                            else:
                                st.session_state.optimized_sections[section_type][section_name.split()[1]] = result.content
                            
                            st.session_state.optimization_status[section_name] = "Completed"
                            
                    else:
                        with state_manager.state_lock():
                            st.session_state.optimization_status[section_name] = "Failed"
                        logger.error(f"Failed to optimize {section_name}: {result.error}")
                        
                    completed += 1
                    progress_bar.progress(completed / total_sections)
                    
                except Exception as e:
                    log_error(e, f"Error processing {section_name}")
                    with state_manager.state_lock():
                        st.session_state.optimization_status[section_name] = "Failed"
                    
    except Exception as e:
        log_error(e, "Error in optimization process")
        return
    finally:
        progress_bar.empty()
        
    # Combine optimized sections into final script
    try:
        final_script = []
        
        # Add imports
        if 'imports' in st.session_state.optimized_sections:
            final_script.extend(st.session_state.optimized_sections['imports'])
        
        # Add settings
        if 'settings' in st.session_state.optimized_sections:
            final_script.extend([''] + st.session_state.optimized_sections['settings'])
        
        # Add optimized functions
        if 'functions' in st.session_state.optimized_sections:
            for func_content in st.session_state.optimized_sections['functions'].values():
                final_script.extend(['', func_content])
        
        # Add optimized classes
        if 'classes' in st.session_state.optimized_sections:
            for class_content in st.session_state.optimized_sections['classes'].values():
                final_script.extend(['', class_content])
        
        # Add global code
        if 'global_code' in st.session_state.script_sections:
            final_script.extend([''] + st.session_state.script_sections['global_code'])
        
        with state_manager.state_lock():
            st.session_state.final_script = '\n'.join(final_script).strip()
        
        # Validate final script
        if not safe_ast_parse(st.session_state.final_script):
            raise ValueError("Generated script contains syntax errors")
            
        logger.info("Successfully combined optimized sections into final script")
        
    except Exception as e:
        log_error(e, "Error combining optimized sections")
        with state_manager.state_lock():
            st.session_state.final_script = ""
        st.error("Failed to combine optimized code sections. Please check the error log and try again.")

# Replace file operations with session state
def save_state():
    with state_manager.state_lock():
        st.session_state['app_state'] = asdict(AppState())

# Add safe imports
try:
    from aider.coders import Coder
except ImportError:
    st.warning("aider-chat not installed")
    Coder = None

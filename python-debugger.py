import streamlit as st
import sys
import traceback
from io import StringIO
from typing import Dict, Any, Optional
from streamlit_ace import st_ace
import logging
from contextlib import contextmanager
import ast
import time
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class SafeExecutionError(Exception):
    """Custom exception for safe execution errors"""
    pass

def create_safe_globals() -> Dict[str, Any]:
    """Create restricted globals dictionary"""
    return {
        '__builtins__': {
            name: __builtins__[name]
            for name in ['print', 'len', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple']
        }
    }

# Configure Streamlit page
st.set_page_config(
    page_title="Python Debugger",
    layout="wide",
    initial_sidebar_state="expanded"
)

@contextmanager
def stdout_redirect():
    """Safely redirect stdout and restore it after"""
    old_stdout = sys.stdout
    stdout = StringIO()
    sys.stdout = stdout
    try:
        yield stdout
    finally:
        sys.stdout = old_stdout

def safe_exec(code: str, globals_dict: Optional[Dict[str, Any]] = None, locals_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Safely execute code and return local variables"""
    if globals_dict is None:
        globals_dict = {}
    if locals_dict is None:
        locals_dict = {}
    
    try:
        # Validate code before execution
        ast.parse(code)
        
        # Create restricted globals
        safe_globals = create_safe_globals()
        globals_dict.update(safe_globals)
        
        exec(code, globals_dict, locals_dict)
        return locals_dict
    except Exception as e:
        logger.error(f"Code execution error: {str(e)}")
        raise SafeExecutionError(f"Code execution error: {str(e)}")

def format_variable(name: str, value: Any) -> str:
    """Format variable for display with type safety"""
    try:
        # Handle potentially problematic types
        if isinstance(value, (int, float, str, bool, list, dict, set, tuple)):
            return f"**{name}**: {type(value).__name__} = {repr(value)}"
        else:
            return f"**{name}**: {type(value).__name__} = <Complex Object>"
    except Exception as e:
        logger.warning(f"Error formatting variable {name}: {str(e)}")
        return f"**{name}**: Error displaying value"

def initialize_session_state():
    """Initialize session state variables"""
    if "code_history" not in st.session_state:
        st.session_state.code_history = []
    if "last_code" not in st.session_state:
        st.session_state.last_code = ""
    if "execution_count" not in st.session_state:
        st.session_state.execution_count = 0

def save_execution_history(code: str):
    """Save code execution history with timestamp"""
    if 'execution_history' not in st.session_state:
        st.session_state.execution_history = []
    st.session_state.execution_history.append({
        'code': code,
        'timestamp': datetime.now().isoformat()
    })

def main():
    """Main application entry point with comprehensive error handling"""
    try:
        initialize_session_state()
        st.title("Python Debugger 🐞")

        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader("Input Code")
            code = st_ace(
                value=st.session_state.get("last_code", ""),
                placeholder="Enter Python code to debug...",
                language="python",
                theme="monokai",
                keybinding="vscode",
                font_size=14,
                min_lines=20,
                key="debug_editor",
                wrap=True,
                auto_update=True
            )
            
            if code:
                st.session_state.last_code = code

        with col2:
            st.subheader("Execution Output")
            if st.button("🚀 Run Code", key="run_button", help="Execute the code"):
                with stdout_redirect() as output:
                    try:
                        locals_dict = safe_exec(code)
                        st.success("Code executed successfully!")
                        if output.getvalue():
                            st.code(output.getvalue(), language="python")
                        save_execution_history(code)
                    except SafeExecutionError as e:
                        st.error(f"Safe execution error: {str(e)}")
                        st.code(traceback.format_exc(), language="python")
                        logger.error(f"Execution error: {str(e)}")

            st.subheader("Variable Inspector")
            if st.button("🔍 Inspect Variables", key="inspect_button", help="View all variables"):
                try:
                    locals_dict = safe_exec(code)
                    if not locals_dict:
                        st.info("No variables to display")
                    else:
                        with st.expander("Variables", expanded=True):
                            for var, value in locals_dict.items():
                                if not var.startswith("__"):
                                    st.markdown(format_variable(var, value))
                except Exception as e:
                    st.error(f"Error during variable inspection: {str(e)}")
                    logger.error(f"Variable inspection error: {str(e)}")

            if st.session_state.code_history:
                with st.expander("Code History"):
                    for entry in reversed(st.session_state.code_history[-5:]):
                        st.text(f"Run {entry['execution_number']} at {entry['timestamp']}:")
                        st.code(entry['code'], language="python")

    except Exception as e:
        logger.critical(f"Critical application error: {str(e)}")
        st.error("An unexpected error occurred. Please try again or contact support.")
        if st.checkbox("Show Error Details"):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()

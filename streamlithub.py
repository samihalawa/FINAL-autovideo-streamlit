# Add at the top with other imports
import pkg_resources
from pkg_resources import parse_requirements
import streamlit as st
import importlib
import os
import logging
import time
from github import Github
from dotenv import load_dotenv
from streamlit_ace import st_ace
from streamlit_option_menu import option_menu
import sys
from contextlib import contextmanager
import io
from datetime import datetime
import psutil
import gradio as gr
import subprocess
import shutil
import tempfile

st.set_page_config(page_title="AI Autocoder Hub", layout="wide")

load_dotenv()

# Enhanced logging with file output
log_file = "streamlithub.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Memory monitoring
def check_memory_usage():
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB
    if mem_usage > 1000:  # 1GB threshold
        logger.warning(f"High memory usage detected: {mem_usage:.2f} MB")
        return False
    return True

# Enhanced error capture
@contextmanager
def capture_streamlit_error():
    stdout = sys.stdout
    stderr = sys.stderr
    string_io = io.StringIO()
    sys.stdout = string_io
    sys.stderr = string_io
    try:
        yield string_io
    finally:
        sys.stdout = stdout
        sys.stderr = stderr
        
# Add timeout for module loading
@st.cache_resource(ttl=3600)  # Cache for 1 hour
def load_module(module_name, timeout=30):
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Copy module to temp dir to avoid conflicts
            if os.path.exists(f"{module_name}.py"):
                shutil.copy(f"{module_name}.py", f"{tmp_dir}/{module_name}.py")
            
            sys.path.insert(0, tmp_dir)
            
            if module_name in sys.modules:
                del sys.modules[module_name]
                
            # Use timeout to prevent hanging
            with st.spinner(f"Loading {module_name}..."):
                result = subprocess.run(
                    [sys.executable, "-c", f"import {module_name}"],
                    timeout=timeout,
                    capture_output=True
                )
                
            if result.returncode != 0:
                raise ImportError(f"Failed to import {module_name}")
                
            return importlib.import_module(module_name)
            
    except Exception as e:
        logger.error(f"Error loading module {module_name}: {str(e)}")
        return None
    finally:
        if tmp_dir in sys.path:
            sys.path.remove(tmp_dir)

# Support for multiple app types
def detect_app_type(file_path):
    with open(file_path) as f:
        content = f.read()
    if "gradio" in content:
        return "gradio"
    elif "streamlit" in content:
        return "streamlit" 
    return "unknown"

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_apps_from_directory():
    apps = {}
    for file in os.listdir():
        if file.endswith('.py') and file != 'streamlithub.py':
            try:
                app_name = os.path.splitext(file)[0].replace('_', ' ').title()
                app_type = detect_app_type(file)
                apps[app_name] = {
                    'path': file,
                    'type': app_type
                }
            except Exception as e:
                logger.error(f"Error loading {file}: {str(e)}")
    return apps

def run_app_safely(module, app_name, app_type):
    st.markdown(f"### Running: {app_name} ({app_type})")
    
    if st.button("🏠 Back to Hub"):
        st.session_state.current_app = None
        st.experimental_rerun()
        return

    try:
        # Memory check before running
        if not check_memory_usage():
            st.warning("High memory usage detected. Consider restarting the app.")
            
        with capture_streamlit_error() as captured:
            if app_type == "gradio":
                if hasattr(module, 'interface'):
                    gr.Interface.load(module.interface).launch()
                else:
                    st.error("No Gradio interface found")
            else:  # streamlit
                if hasattr(module, 'main'):
                    module.main()
                else:
                    st.error(f"No main() function found in {app_name}")
        
        error_output = captured.getvalue()
        if error_output:
            with st.expander("Show App Errors/Logs"):
                st.code(error_output)
                
    except Exception as e:
        st.error(f"Error running {app_name}: {str(e)}")
        with st.expander("Show Error Details"):
            st.exception(e)
        logger.error(f"App crash: {app_name} - {str(e)}")

# Rest of the code remains the same...

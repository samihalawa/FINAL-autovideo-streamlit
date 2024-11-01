import streamlit as st
from autocoder import State, optimize_script, validate_api_key, save_final_code, generate_optimization_suggestion, generate_script_map
import plotly.graph_objects as go
from aider import chat
import time
from tqdm import tqdm
from typing import Dict, Any, List, Optional
import ast
import re
import logging
import sys
import traceback
from contextlib import contextmanager
from streamlit.runtime.scriptrunner import add_script_run_ctx
import psutil
import requests
import threading
from queue import Queue
import signal
import gc
import json
from pathlib import Path
import tempfile

# Set up logging with file handler for error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use temp directory for logs in Streamlit Cloud
temp_dir = tempfile.gettempdir()
log_path = Path(temp_dir) / 'app_errors.log'
fh = logging.FileHandler(log_path)
fh.setLevel(logging.ERROR)
logger.addHandler(fh)

# Global settings
API_TIMEOUT = 20  # Reduced timeout for cloud environment
MAX_RETRIES = 3
MAX_SCRIPT_SIZE = 500_000  # 500KB limit for cloud
MAX_MEMORY_PCT = 85  # Memory threshold
MAX_CPU_PCT = 90  # CPU threshold

class StreamlitAppError(Exception):
    """Custom exception for Streamlit app errors"""
    pass

class ResourceExhaustedError(StreamlitAppError):
    """Raised when system resources are critically low"""
    pass

class CloudEnvironmentError(StreamlitAppError):
    """Raised for cloud-specific issues"""
    pass

@contextmanager 
def timeout_handler(seconds=API_TIMEOUT):
    """Context manager to handle timeouts with cloud considerations"""
    def signal_handler(signum, frame):
        raise TimeoutError("Operation timed out in cloud environment")
    
    # Only use SIGALRM on Unix systems
    if sys.platform != 'win32':
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
    try:
        yield
    finally:
        if sys.platform != 'win32':
            signal.alarm(0)

@contextmanager
def error_handling():
    """Enhanced error handling for cloud deployment"""
    try:
        yield
    except Exception as e:
        st.error(f"❌ Error in cloud environment: {str(e)}")
        logger.error(f"Cloud exception: {traceback.format_exc()}")
        # Clear memory and session state
        gc.collect()
        cleanup_cloud_resources()

def cleanup_cloud_resources():
    """Cloud-specific resource cleanup"""
    try:
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Clear temp files
        temp_files = Path(tempfile.gettempdir()).glob('streamlit_*')
        for f in temp_files:
            try:
                f.unlink()
            except:
                pass
                
        gc.collect()
        
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

def check_system_resources() -> Dict[str, float]:
    """Cloud-optimized resource monitoring"""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.5)
        memory = psutil.virtual_memory()
        
        if cpu_percent > MAX_CPU_PCT or memory.percent > MAX_MEMORY_PCT:
            cleanup_cloud_resources()
            raise ResourceExhaustedError("Cloud resources critically low")
            
        return {
            "cpu_usage": cpu_percent,
            "memory_usage": memory.percent,
            "memory_available": memory.available / (1024 * 1024 * 1024)  # GB
        }
    except Exception as e:
        logger.error(f"Resource check error: {e}")
        return {"cpu_usage": 0, "memory_usage": 0, "memory_available": 0}

def validate_script_input(script: str) -> List[str]:
    """Enhanced validation for cloud deployment"""
    issues = []
    if not script or not script.strip():
        issues.append("Script input is empty")
        return issues
        
    # Check script size
    if len(script) > MAX_SCRIPT_SIZE:
        issues.append(f"Script exceeds cloud size limit of {MAX_SCRIPT_SIZE/1000}KB")
        
    try:
        tree = ast.parse(script)
        
        # Enhanced cloud compatibility checks
        for node in ast.walk(tree):
            if isinstance(node, ast.While) and isinstance(node.test, ast.Constant) and node.test.value == True:
                issues.append("❌ Infinite loop detected - not suitable for cloud")
                
            if isinstance(node, ast.Call):
                func_name = ''
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                    
                # Check for problematic operations
                if func_name in ['sleep', 'wait', 'system', 'popen', 'exec']:
                    issues.append(f"❌ Unsafe operation detected: {func_name}")
                    
            # Check for file operations
            if isinstance(node, (ast.Open, ast.Write)):
                issues.append("⚠️ File operations may not persist in cloud")
                
    except SyntaxError as e:
        issues.append(f"Syntax error: {str(e)}")
    except Exception as e:
        issues.append(f"Validation error: {str(e)}")
        
    return issues

def simulate_cloud_workflows():
    """Simulate various cloud deployment scenarios"""
    issues = []
    
    # Simulate concurrent users
    try:
        for _ in range(3):
            threading.Thread(target=check_system_resources).start()
        issues.append("⚠️ Threading may cause issues in cloud environment")
    except:
        pass
        
    # Check for session state persistence
    if len(st.session_state) > 100:
        issues.append("❌ Too many session state variables")
        
    # Simulate memory pressure
    try:
        large_data = "x" * 1000000
        st.session_state.temp = large_data
        issues.append("⚠️ Large session state data detected")
    except:
        pass
        
    # Check for rate limiting
    if 'api_calls' not in st.session_state:
        st.session_state.api_calls = []
    
    current_time = time.time()
    st.session_state.api_calls = [t for t in st.session_state.api_calls if current_time - t < 60]
    
    if len(st.session_state.api_calls) > 50:
        issues.append("❌ API rate limit may be exceeded")
        
    return issues

def save_app_state(data: dict):
    st.session_state['app_data'] = data

def main():
    """Cloud-optimized main application"""
    try:
        st.set_page_config(page_title="AutocoderAI Cloud", layout="wide")
        st.title("AutocoderAI Cloud Deployment")
        
        # Initialize cloud monitoring
        if 'cloud_errors' not in st.session_state:
            st.session_state.cloud_errors = []
            
        # Simulate cloud workflows
        cloud_issues = simulate_cloud_workflows()
        if cloud_issues:
            st.warning("Cloud Deployment Issues Detected:")
            for issue in cloud_issues:
                st.write(issue)
                
        # Resource monitoring
        resources = check_system_resources()
        if resources["memory_usage"] > MAX_MEMORY_PCT:
            st.error("⚠️ High memory usage in cloud environment")
            cleanup_cloud_resources()
            
        # Rest of the main function implementation...
        # (Previous main function code continues here)
        
    except Exception as e:
        logger.error(f"Cloud runtime error: {traceback.format_exc()}")
        st.error(f"Cloud deployment error: {str(e)}")
        cleanup_cloud_resources()

if __name__ == "__main__":
    main()

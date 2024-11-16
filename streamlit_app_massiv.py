import streamlit as st
import plotly.graph_objects as go
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
import time
import openai

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

def validate_api_key(api_key: str) -> bool:
    """Validate OpenAI API key with rate limiting"""
    try:
        # Track API validation attempts
        if 'api_validations' not in st.session_state:
            st.session_state.api_validations = []
        
        current_time = time.time()
        # Clean old validation attempts
        st.session_state.api_validations = [t for t in st.session_state.api_validations 
                                          if current_time - t < 60]
        
        # Check rate limit (max 10 validations per minute)
        if len(st.session_state.api_validations) >= 10:
            logger.warning("API key validation rate limit exceeded")
            return False
            
        st.session_state.api_validations.append(current_time)
        
        # Existing validation code
        openai.api_key = api_key
        with timeout_handler(seconds=5):
            response = openai.Completion.create(
                model="gpt-3.5-turbo-instruct",
                prompt="test",
                max_tokens=1
            )
            if 'error' in response:
                return False
        return True
    except Exception as e:
        logger.error(f"API key validation error: {str(e)}")
        return False

def optimize_script(script: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
    """Optimize code using OpenAI API with resource management"""
    try:
        # Check resources before heavy operation
        resources = check_system_resources()
        if resources['memory_usage'] > MAX_MEMORY_PCT * 0.9:  # 90% of threshold
            cleanup_cloud_resources()
            
        # Existing optimization code
        prompt = f"""Optimize this Python code for better performance and readability:
        {script}
        Return only the optimized code without explanations."""
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a Python optimization expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        optimized_code = response.choices[0].message.content.strip()
        
        # Store optimization history with memory management
        if 'optimization_history' not in st.session_state:
            st.session_state.optimization_history = []
        
        # Limit history size to prevent memory issues
        if len(st.session_state.optimization_history) > 100:
            st.session_state.optimization_history = st.session_state.optimization_history[-100:]
            
        original_lines = len(script.split('\n'))
        optimized_lines = len(optimized_code.split('\n'))
        optimization_score = (original_lines - optimized_lines) / original_lines * 100
        st.session_state.optimization_history.append(optimization_score)
        
        return optimized_code
        
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}")
        raise StreamlitAppError(f"Failed to optimize code: {str(e)}")

def generate_optimization_suggestion(script: str) -> str:
    """Generate optimization suggestions for the code"""
    try:
        prompt = f"""Analyze this Python code and suggest optimization improvements:

{script}

Focus on:
1. Performance bottlenecks
2. Code readability
3. Best practices
4. Memory usage

Provide specific suggestions."""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a Python optimization expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Suggestion generation error: {str(e)}")
        return "Failed to generate suggestions"

def save_final_code(code: str, filename: str = "optimized_code.py"):
    """Save optimized code to a file in temp directory"""
    try:
        temp_path = Path(tempfile.gettempdir()) / filename
        with open(temp_path, 'w') as f:
            f.write(code)
        return str(temp_path)
    except Exception as e:
        logger.error(f"Save error: {str(e)}")
        raise StreamlitAppError(f"Failed to save code: {str(e)}")

def main():
    """Self-sufficient main application"""
    try:
        st.set_page_config(page_title="Code Optimizer", layout="wide")
        st.title("Python Code Optimizer")
        
        # Initialize state
        if 'api_calls' not in st.session_state:
            st.session_state.api_calls = []
        
        # Resource monitoring
        resources = check_system_resources()
        
        # API Key input
        api_key = st.text_input("Enter OpenAI API Key", type="password")
        if api_key:
            if not validate_api_key(api_key):
                st.error("Invalid OpenAI API key")
                return
            
        # Settings sidebar
        st.sidebar.header("Optimization Settings")
        max_tokens = st.sidebar.slider("Max Tokens", 100, 2000, 1000)
        temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
        
        # Main input
        script_input = st.text_area("Enter Python Code to Optimize", height=300)
        
        if script_input and api_key and st.button("Optimize Code"):
            with st.spinner("Optimizing code..."):
                # Validate input
                issues = validate_script_input(script_input)
                if issues:
                    for issue in issues:
                        st.warning(issue)
                else:
                    try:
                        # Track API call
                        st.session_state.api_calls.append(time.time())
                        
                        # Get optimization suggestions
                        suggestions = generate_optimization_suggestion(script_input)
                        
                        # Optimize code
                        optimized_code = optimize_script(
                            script_input,
                            max_tokens=max_tokens,
                            temperature=temperature
                        )
                        
                        # Display results
                        st.success("Code optimized successfully!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Optimization Suggestions")
                            st.write(suggestions)
                            
                        with col2:
                            st.subheader("Optimized Code")
                            st.code(optimized_code, language='python')
                            
                        # Save button
                        if st.button("Save Optimized Code"):
                            save_path = save_final_code(optimized_code)
                            st.success(f"Code saved to: {save_path}")
                            
                    except Exception as e:
                        st.error(f"Optimization failed: {str(e)}")
                        logger.error(traceback.format_exc())
        
        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("System Status")
            st.metric("CPU Usage", f"{resources['cpu_usage']}%")
            st.metric("Memory Usage", f"{resources['memory_usage']}%")
            
        with col2:
            st.subheader("API Usage")
            recent_calls = len([t for t in st.session_state.api_calls 
                              if time.time() - t < 3600])
            st.metric("API Calls (last hour)", recent_calls)
        
        # Show optimization history
        if 'optimization_history' in st.session_state:
            st.subheader("Optimization History")
            fig = go.Figure(data=[
                go.Scatter(x=list(range(len(st.session_state.optimization_history))),
                          y=st.session_state.optimization_history,
                          mode='lines+markers')
            ])
            fig.update_layout(
                xaxis_title="Iteration",
                yaxis_title="Optimization Score (%)"
            )
            st.plotly_chart(fig)
        
        # Footer
        st.markdown("---")
        st.markdown("Code Optimizer © 2024")
        
    except Exception as e:
        logger.error(f"Application error: {traceback.format_exc()}")
        st.error(f"Application error: {str(e)}")
        cleanup_cloud_resources()

if __name__ == "__main__":
    main()

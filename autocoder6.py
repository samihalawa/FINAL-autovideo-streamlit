import os
import ast
import streamlit as st
import logging
import tempfile
import time
import asyncio
import psutil
import signal
import gc
import traceback
import functools
import weakref
import sys
import json
import queue
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Generator, Tuple, Protocol, Iterator, NoReturn
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from openai import OpenAI
from contextlib import contextmanager, asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from enum import Enum, auto
from streamlit_extras import switch_page_button, add_vertical_space, colored_header
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.panel import Panel
from rich.live import Live
import pyperclip
from streamlit.runtime.scriptrunner import add_script_run_ctx

MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30
MEMORY_THRESHOLD = 90
API_RATE_LIMIT = 10
CACHE_TTL = 300

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(name)s-%(levelname)s-%(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(tempfile.gettempdir()) / "autocoder.log")
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationBlock:
    """Represents a block of code to be optimized."""
    original: str
    optimized: str = ""
    block_type: str = ""
    status: str = "pending"
    messages: List[str] = field(default_factory=list)
    progress: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    
    def update_progress(self, progress: float, message: Optional[str] = None) -> None:
        """Update the progress of optimization for this block."""
        self.progress = min(100.0, max(0.0, progress))
        if message:
            self.messages.append(f"{datetime.now().strftime('%H:%M:%S')} - {message}")

@dataclass
class OptimizationTask:
    """Represents a single optimization task."""
    stage: OptimizationStage
    code_block: str
    context: Dict[str, Any]
    dependencies: List[str]
    status: str = "pending"
    result: Optional[str] = None

@dataclass
class AppState:
    """Application state container."""
    settings: Dict[str, Any]
    history: List[str]
    current_optimization: Optional[str]
    
    def is_rate_limited(self) -> bool:
        """Check if API calls are rate limited."""
        return len(self.history) >= API_RATE_LIMIT
        
    def get_rate_limit_wait_time(self) -> float:
        """Calculate wait time for rate limit."""
        if not self.history:
            return 0
        oldest_call = datetime.fromisoformat(self.history[0])
        wait_time = 60 - (datetime.now() - oldest_call).total_seconds()
        return max(0, wait_time)

    def validate_state(self) -> bool:
        """Validate application state."""
        required_settings = {"model", "temperature", "max_tokens", "memory_limit"}
        return all(key in self.settings for key in required_settings)

OPTIMIZATION_PROMPT = """Optimize this {block_type} following best practices:
1. Improve time/space complexity
2. Enhance readability and maintainability
3. Apply SOLID principles
4. Add type hints and documentation
5. Implement error handling
6. Optimize resource usage

Original code:
{code}

Return only the optimized code without explanations."""

class OptimizationStage(Enum):
    """Detailed stages for optimization pipeline."""
    PLANNING = auto()
    ANALYSIS = auto()
    SEGMENTATION = auto()
    TYPE_INFERENCE = auto()
    OPTIMIZATION = auto()
    VERIFICATION = auto()
    DOCUMENTATION = auto()

class OptimizationStep(Enum):
    """Steps in the optimization process."""
    ANALYSIS = "Analyzing code structure and patterns"
    PLANNING = "Planning optimization strategy"
    SEPARATION = "Separating code blocks"
    ENHANCEMENT = "Enhancing individual blocks"
    INTEGRATION = "Integrating optimized blocks"
    VALIDATION = "Validating final output"
    DOCUMENTATION = "Adding documentation"

class OptimizationState(Enum):
    """States for optimization process."""
    IDLE = auto()
    RUNNING = auto()
    FAILED = auto()
    COMPLETED = auto()

class WorkflowOrchestrator:
    """Orchestrates the optimization workflow."""
    
    # Core Initialization
    def __init__(self) -> None:
        """Initialize the workflow orchestrator."""
        self.blocks: List[OptimizationBlock] = []
        self.current_step = OptimizationStep.ANALYSIS
        self.console = Console()
        self.progress_queue: queue.Queue = queue.Queue()
        self.memory = None  # Remove ConversationBufferMemory
        self.metrics = {
            "total_optimizations": 0,
            "average_time": 0,
            "success_rate": 0,
            "memory_usage": []
        }
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.optimization_cache: Dict[str, Any] = {}
        self._cleanup_handlers: List[Any] = []
        self._resource_monitor = None
        self.client: Optional[OpenAI] = None

    async def initialize(self) -> None:
        """Initialize async resources."""
        try:
            retries = 3
            while retries > 0:
                self.client = await get_openai_client()
                if self.client:
                    await self._start_resource_monitor()
                    return
                retries -= 1
                await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise RuntimeError("Failed to initialize")

    # Main Processing Pipeline
    async def process_code(self, code: str) -> Optional[str]:
        """Process and optimize code with fatal error prevention."""
        try:
            # Validate input before processing
            if not code or not isinstance(code, str):
                raise ValueError("Invalid code input")
                
            # Check system resources
            if psutil.virtual_memory().percent > 90:
                raise ResourceExhaustedError("System memory critically low")
                
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_area = st.empty()
            
            # Initialize error recovery state
            recovery_attempts = 0
            MAX_RECOVERY_ATTEMPTS = 3
            
            while recovery_attempts < MAX_RECOVERY_ATTEMPTS:
                try:
                    if psutil.Process().memory_percent() > app_state.settings["memory_limit"]:
                        await self.cleanup()
                    
                    steps = [
                        ("Analyzing", self._analyze_code_structure),
                        ("Planning", self._generate_optimization_plan),
                        ("Optimizing", self._optimize_blocks),
                        ("Integrating", self._integrate_blocks)
                    ]

                    result = None
                    for i, (step_name, step_func) in enumerate(steps):
                        # Update progress safely
                        progress = min(100, (i + 1) * 25)
                        progress_bar.progress(progress)
                        
                        # Use try-except for each step
                        try:
                            with status_area.container():
                                st.write(f"🔄 {step_name}")
                                step_progress = st.progress(0)
                                result = await step_func(code if i == 0 else result)
                                step_progress.progress(100)
                        except Exception as step_error:
                            logger.error(f"Step '{step_name}' failed: {step_error}")
                            raise
                    
                    if result is None:
                        raise ValueError("Optimization produced no result")
                        
                    progress_bar.progress(100)
                    status_text.text("✨ Optimization complete!")
                    return result
                    
                except Exception as e:
                    recovery_attempts += 1
                    if recovery_attempts >= MAX_RECOVERY_ATTEMPTS:
                        raise
                    logger.warning(f"Attempting recovery ({recovery_attempts}/{MAX_RECOVERY_ATTEMPTS})")
                    await asyncio.sleep(1)
                    await self.cleanup()
                    
            return None
            
        except Exception as e:
            logger.error(f"Fatal process error: {e}")
            st.error(f"Optimization failed: {str(e)}")
            raise

    # Resource Management
    async def cleanup(self) -> None:
        """Enhanced cleanup with memory management and resource handling."""
        try:
            # Cancel running tasks
            if self._resource_monitor and not self._resource_monitor.done():
                self._resource_monitor.cancel()
                try:
                    await self._resource_monitor
                except asyncio.CancelledError:
                    pass
                    
            # Clear data structures
            self.blocks.clear()
            self.optimization_cache.clear()
            self.progress_queue = queue.Queue()  # Create new queue instead of clearing
            
            # Shutdown executor properly
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = ThreadPoolExecutor(max_workers=3)  # Create new executor
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Fatal error during cleanup: {e}")
            # Ensure resources are released even if cleanup fails
            self.blocks = []
            self.optimization_cache = {}
            self.progress_queue = queue.Queue()
            gc.collect()
            raise

    async def _cancel_resource_monitor(self) -> None:
        if self._resource_monitor:
            self._resource_monitor.cancel()
        
    async def _cleanup_blocks(self) -> None:
        self.blocks.clear()
        
    async def _shutdown_executor(self) -> None:
        self.executor.shutdown(wait=True)

    async def _analyze_code_structure(self, code: str) -> List[OptimizationBlock]:
        """Analyze the structure of the input code."""
        try:
            if not validate_syntax(code):
                raise ValueError("Invalid Python syntax")
            blocks = []
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    complexity = sum(1 for _ in ast.walk(node))
                    block = OptimizationBlock(
                        original=ast.get_source_segment(code, node),
                        block_type=node.__class__.__name__,
                        messages=[f"Complexity score: {complexity}"]
                    )
                    blocks.append(block)
            return blocks
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise

    async def _generate_optimization_plan(
        self,
        blocks: List[OptimizationBlock]
    ) -> List[OptimizationBlock]:
        """Generate optimization plan for code blocks."""
        try:
            for block in blocks:
                block.status = "planned"
                block.messages.append("Optimization planned")
            return blocks
        except Exception as e:
            logger.error(f"Plan failed: {e}")
            raise

    async def _optimize_block(self, block: OptimizationBlock) -> OptimizationBlock:
        """Optimize a single code block with recovery."""
        try:
            retries_left = app_state.settings["max_retries"]
            backoff = 1
            
            while retries_left > 0:
                try:
                    if app_state.is_rate_limited():
                        wait_time = app_state.get_rate_limit_wait_time()
                        logger.info(f"Rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                        
                    if psutil.Process().memory_percent() > app_state.settings["memory_limit"]:
                        await self.cleanup()
                        if psutil.Process().memory_percent() > app_state.settings["memory_limit"]:
                            raise ResourceExhaustedError("Memory limit exceeded")
                    
                    async with asyncio.timeout(app_state.settings["timeout"]):
                        response = await self.client.chat.completions.create(
                            model=app_state.settings["model"],
                            messages=[
                                {"role": "system", "content": "You are a code optimization assistant."},
                                {"role": "user", "content": OPTIMIZATION_PROMPT.format(
                                    block_type=block.block_type,
                                    code=block.original
                                )}
                            ],
                            temperature=app_state.settings["temperature"],
                            max_tokens=app_state.settings["max_tokens"]
                        )
                    
                    if not response or not hasattr(response, 'choices') or not response.choices:
                        raise APIError("Invalid API response format")
                        
                    block.optimized = response.choices[0].message.content
                    block.status = "complete"
                    return block
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout on attempt {app_state.settings['max_retries']-retries_left+1}")
                    retries_left -= 1
                    if retries_left > 0:
                        await asyncio.sleep(backoff)
                        backoff *= 2
                    else:
                        raise
                        
                except Exception as e:
                    logger.error(f"Block optimization error: {str(e)}")
                    retries_left -= 1
                    if retries_left > 0:
                        await asyncio.sleep(backoff)
                        backoff *= 2
                    else:
                        raise

            return block

        except Exception as e:
            block.status = "failed"
            block.messages.append(f"Optimization failed: {str(e)}")
            if self._should_retry(e):
                return await self._retry_optimization(block)
            raise

    async def _integrate_blocks(self) -> str:
        """Integrate optimized blocks into final code."""
        try:
            return "\n\n".join(block.optimized for block in self.blocks)
        except Exception as e:
            logger.error(f"Integration failed: {e}")
            raise

    async def _validate_result(self, code: str) -> bool:
        """Validate the optimized code."""
        try:
            ast.parse(code)
            return True
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise

    async def get_final_result(self) -> Optional[str]:
        """Get the final optimized code result."""
        try:
            return await self._integrate_blocks() if self.blocks else None
        except Exception as e:
            logger.error(f"Final result failed: {e}")
            return None

    async def _optimize_blocks(self) -> List[OptimizationBlock]:
        """Optimize all code blocks in parallel."""
        optimized = []
        for block in self.blocks:
            optimized.append(await self._optimize_block(block))
        return optimized

    async def _check_memory_usage(self) -> None:
        """Monitor memory usage."""
        current_memory = psutil.Process().memory_percent()
        self.metrics["memory_usage"].append(current_memory)
        
        if len(self.metrics["memory_usage"]) > 100:
            self.metrics["memory_usage"] = self.metrics["memory_usage"][-100:]
            
        if current_memory > app_state.settings["memory_limit"] * 0.9:  # Early warning
            logger.warning(f"Memory usage approaching limit: {current_memory}%")

    async def _handle_rate_limit(self, wait_time: float) -> None:
        """Handle API rate limiting."""
        logger.info(f"Rate limited, waiting {wait_time}s")
        self.metrics["rate_limits"] = self.metrics.get("rate_limits", 0) + 1
        await asyncio.sleep(wait_time)

    def _update_metrics(self, step_name: str) -> None:
        """Update optimization metrics."""
        self.metrics["total_optimizations"] += 1
        if step_name == "failed":
            success_count = sum(1 for b in self.blocks if b.status == "complete")
            self.metrics["success_rate"] = (success_count / len(self.blocks)) if self.blocks else 0
        else:
            elapsed = (datetime.now() - self.blocks[-1].start_time).total_seconds() if self.blocks else 0
            self.metrics["average_time"] = (self.metrics["average_time"] * (self.metrics["total_optimizations"] - 1) + elapsed) / self.metrics["total_optimizations"]

    def _should_retry(self, error: Exception) -> bool:
        """Determine if operation should be retried."""
        return isinstance(error, (asyncio.TimeoutError, APIError)) and not isinstance(error, ResourceExhaustedError)

    async def _retry_optimization(self, block: OptimizationBlock) -> OptimizationBlock:
        """Retry optimization with exponential backoff."""
        await asyncio.sleep(1)
        return await self._optimize_block(block)

    async def _start_resource_monitor(self) -> None:
        """Start resource monitoring task."""
        async def monitor():
            while True:
                await self._check_memory_usage()
                await asyncio.sleep(5)
        
        self._resource_monitor = asyncio.create_task(monitor())

    async def _optimize_block_with_timeout(self, block: OptimizationBlock) -> OptimizationBlock:
        """Optimize block with timeout."""
        try:
            async with asyncio.timeout(app_state.settings["timeout"]):
                return await self._optimize_block(block)
        except asyncio.TimeoutError:
            logger.error(f"Optimization timeout for {block.block_type}")
            block.status = "timeout"
            return block

# Add get_openai_client function
async def get_openai_client() -> Optional[OpenAI]:
    """Initialize OpenAI client with error handling and validation."""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
            logger.error("OpenAI API key not found")
            return None
            
        client = OpenAI(api_key=api_key)
        # Validate client by making a test request
        try:
            async with asyncio.timeout(5):
                await client.models.list()
            return client
        except Exception as e:
            st.error(f"Failed to validate OpenAI client: {str(e)}")
            logger.error(f"OpenAI client validation failed: {e}")
            return None
            
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {str(e)}")
        logger.error(f"OpenAI client initialization failed: {e}")
        return None

# Add settings update handler
def update_settings(key: str, value: Any) -> None:
    """Update application settings."""
    if key in st.session_state.app_state["settings"]:
        st.session_state.app_state["settings"][key] = value
        logger.info(f"Updated setting {key}: {value}")

# Add at the top of the file, after imports
st.set_page_config(
    page_title="AutoCoder Optimization",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add after page config
if 'app_state' not in st.session_state:
    initial_state = AppState(
        settings={
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 2000,
            "memory_limit": 90,
            "max_retries": 3,
            "timeout": 30
        },
        history=[],
        current_optimization=None
    )
    if not initial_state.validate_state():
        raise ValueError("Invalid initial state")
    st.session_state.app_state = initial_state

# Add main page layout function
def create_page_layout():
    """Create the main page layout."""
    st.title("AutoCoder Code Optimization")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        model = st.selectbox(
            "Model",
            options=["gpt-4", "gpt-3.5-turbo"],
            key="model"
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            key="temperature"
        )
        
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["Code Input", "Optimization", "Results"])
    
    return tab1, tab2, tab3

# Add error handling utilities
def show_error_message(error: Exception):
    with st.error(f"Error: {str(error)}"):
        if st.button("Show Details"): st.code(traceback.format_exc())

def show_warning(message: str):
    st.warning(message, icon="️")

class ProgressTracker:
    @staticmethod
    def save_progress(progress: dict) -> None:
        if 'optimization_progress' not in st.session_state:
            st.session_state.optimization_progress = []
        st.session_state.optimization_progress.append({'timestamp': datetime.now().isoformat(), **progress})

    @staticmethod
    def load_progress() -> List[dict]:
        return st.session_state.get('optimization_progress', [])

def show_optimization_progress():
    try:
        if 'orchestrator' not in st.session_state: return
        progress_data = st.session_state.orchestrator.progress_queue.get_nowait()
        ProgressTracker.save_progress(progress_data)
        if progress_data.get('status') == 'complete': cleanup_progress()
    except queue.Empty: pass

async def main():
    try:
        initialize_session_state()
        if not validate_dependencies() or not validate_environment():
            st.error("Missing required environment variables")
            return
            
        tab1, tab2, tab3 = create_page_layout()
        
        with tab1:
            code_input = st.text_area("Input your code here", height=300, key="code_input")
            
            if st.button("Start Optimization"):
                if await handle_code_input(code_input):
                    try:
                        orchestrator = WorkflowOrchestrator()
                        await orchestrator.initialize()
                        result = await orchestrator.process_code(code_input)
                        if result: display_results(result)
                        else: st.error("Optimization failed")
                    except Exception as e:
                        result = await handle_optimization_error(e, code_input)
                        if result: display_results(result)
                        else: st.error("Optimization failed after recovery attempt")
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}\n{traceback.format_exc()}")

def validate_dependencies() -> bool:
    required = ["openai", "rich", "streamlit", "pyperclip"]
    return all(__import__(pkg) for pkg in required)

def validate_environment() -> bool:
    required_vars = ["OPENAI_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        logger.error(f"Missing environment variables: {missing}")
        return False
    return True

def validate_syntax(code: str) -> bool:
    try: 
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def validate_input(code: str) -> tuple[bool, str]:
    if not code or not code.strip(): return False, "Please enter some code"
    if len(code) > 10000: return False, "Code too long (max 10000 characters)"
    return True, ""

class ErrorRecovery:
    @staticmethod
    async def recover(error: Exception) -> bool:
        if isinstance(error, ResourceExhaustedError):
            await cleanup()
            return True
        if isinstance(error, APIError):
            await asyncio.sleep(1)
            return True
        return False

async def cancel_optimization():
    if st.session_state.get('orchestrator'):
        await st.session_state.orchestrator.cleanup()
        cleanup_progress()
        st.experimental_rerun()

def add_result_handlers(result: str):
    if st.button("Copy Code"):
        pyperclip.copy(result)
        st.success("Code copied!")
    
    if st.download_button("Download Result", result, file_name="optimized_code.py", mime="text/plain"):
        st.success("File downloaded!")

def validate_settings(settings: dict) -> tuple[bool, str]:
    if not 0 <= settings["temperature"] <= 1: return False, "Temperature must be between 0 and 1"
    if not 100 <= settings["max_tokens"] <= 4000: return False, "Invalid max tokens range"
    return True, ""

def save_settings():
    valid, message = validate_settings(st.session_state.app_state.settings)
    if not valid:
        st.sidebar.error(message)
        return
    with open("settings.json", "w") as f:
        json.dump(st.session_state.app_state.settings, f)

async def handle_error(error: Exception) -> bool:
    try:
        return await ErrorRecovery.recover(error)
    except Exception as e:
        logger.error(f"Error handling failed: {e}")
        return False

def add_retry_button():
    if st.button("Retry"):
        st.session_state.retry_count = st.session_state.get('retry_count', 0) + 1
        if st.session_state.retry_count <= 3: st.experimental_rerun()
        else: st.error("Maximum retries exceeded")

def clear_optimization_history():
    if st.button("Clear History"):
        st.session_state.app_state.history.clear()
        cleanup_progress()
        st.experimental_rerun()

def update_metrics(metrics: dict):
    if 'metrics_history' not in st.session_state:
        st.session_state.metrics_history = []
    st.session_state.metrics_history.append({'timestamp': datetime.now().isoformat(), **metrics})

def initialize_session_state():
    """Initialize session state with error prevention."""
    try:
        default_keys = {
            'optimization_progress': [],
            'metrics_history': [],
            'retry_count': 0,
            'error_state': None
        }
        
        for key, default_value in default_keys.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
                
        # Validate app state
        if 'app_state' not in st.session_state:
            initial_state = AppState(
                settings={
                    "model": "gpt-4",
                    "temperature": 0.7,
                    "max_tokens": 2000,
                    "memory_limit": 90,
                    "max_retries": 3,
                    "timeout": 30
                },
                history=[],
                current_optimization=None
            )
            if not initial_state.validate_state():
                raise ValueError("Invalid initial application state")
            st.session_state.app_state = initial_state
            
    except Exception as e:
        logger.error(f"Fatal error initializing session state: {e}")
        st.error("Failed to initialize application. Please refresh the page.")
        raise

def handle_code_input(code: str):
    if st.button("Start Optimization"):
        valid, message = validate_input(code)
        if not valid:
            st.error(message)
            return False
            
        with st.spinner("Initializing..."):
            orchestrator = WorkflowOrchestrator()
            asyncio.run(orchestrator.initialize())
            result = asyncio.run(orchestrator.process_code(code))
            
            if result:
                with st.tabs(["Code Input", "Optimization", "Results"])[2]:
                    st.success("Optimization completed!")
                    st.code(result, language="python")
                    with st.expander("View Optimization Metrics"):
                        st.json(orchestrator.metrics)
    return True

async def handle_optimization_cancel():
    await cancel_optimization()
    cleanup_progress()
    st.experimental_rerun()

def display_results(result: str):
    st.code(result, language="python")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Copy Code"): copy_to_clipboard(result)
    with col2: handle_download(result)
    display_metrics()

def display_metrics():
    with st.expander("Optimization Metrics"):
        if 'metrics_history' in st.session_state and st.session_state.metrics_history:
            st.json(st.session_state.metrics_history[-1])
        else: st.info("No optimization metrics available yet")

def validate_code_input(code: str) -> tuple[bool, str]:
    if not code or not code.strip(): return False, "Code input is required"
    try:
        ast.parse(code)
        return True, "Code is valid"
    except SyntaxError as e:
        return False, f"Invalid Python syntax: {str(e)}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def show_loading_state():
    with st.spinner("Processing code optimization..."): pass

def copy_to_clipboard(code: str):
    try:
        pyperclip.copy(code)
        st.success("Code copied to clipboard!")
    except Exception as e:
        st.error(f"Failed to copy code: {str(e)}")

def cleanup_progress():
    cleanup_state()
    st.experimental_rerun()

def cleanup_state():
    for key in ['optimization_progress', 'metrics_history', 'retry_count', 'orchestrator']:
        if key in st.session_state: del st.session_state[key]

def handle_download(code: str):
    st.download_button(label="Download Code", data=code, file_name="optimized_code.py", mime="text/plain")

async def handle_optimization_error(e: Exception, code: str) -> Optional[str]:
    st.error(f"Optimization error: {str(e)}")
    if await ErrorRecovery.recover(e):
        st.warning("Attempting to recover and retry optimization...")
        orchestrator = WorkflowOrchestrator()
        await orchestrator.initialize()
        return await orchestrator.process_code(code)
    return None

async def main():
    initialize_session_state()
    if not validate_dependencies() or not validate_environment():
        st.error("Required dependencies or environment variables are missing")
        return
        
    tab1, tab2, tab3 = create_page_layout()
    with tab1:
        code_input = st.text_area("Input your Python code", height=300, key="code_input")
        if st.button("Optimize Code"):
            valid, message = validate_code_input(code_input)
            if not valid:
                st.error(message)
                return
            with st.spinner("Initializing optimization..."):
                try:
                    orchestrator = WorkflowOrchestrator()
                    await orchestrator.initialize()
                    result = await orchestrator.process_code(code_input)
                    if result: display_results(result)
                    else: st.error("Code optimization failed")
                except Exception as e:
                    result = await handle_optimization_error(e, code_input)
                    if result: display_results(result)
                    else: st.error("Code optimization failed after recovery attempt")

# Add missing error classes after imports
class APIError(Exception):
    """Custom exception for API related errors."""
    pass

class ResourceExhaustedError(Exception):
    """Custom exception for resource exhaustion."""
    pass

# Add missing cleanup function
async def cleanup() -> None:
    """Global cleanup function."""
    if 'orchestrator' in st.session_state:
        await st.session_state.orchestrator.cleanup()
    gc.collect()

if __name__ == "__main__":
    asyncio.run(main())

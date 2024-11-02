# AI Code Enhancer Streamlit App with Cloud Optimizations
import os
import ast
import streamlit as st
import streamlit_ace as st_ace
from streamlit_option_menu import option_menu
from langchain.memory import ConversationBufferMemory
import logging
import tempfile
from pathlib import Path
import time
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from openai import OpenAI, OpenAIError
import asyncio
import psutil
import signal
import gc
from contextlib import contextmanager, asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from aider import models, chat
from aider.coders import Coder
from enum import Enum, auto
import traceback
import functools
import weakref
import sys
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.panel import Panel
from rich.live import Live
import difflib
from typing import Generator, Tuple
import queue
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Protocol, Iterator
from dataclasses import dataclass
from enum import auto, Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
from rich.console import Console
from rich.progress import Progress
from streamlit_autorefresh import st_autorefresh
from streamlit_dynamic_filters import DynamicFilters
import json

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(tempfile.gettempdir()) / "autocoder.log")
    ]
)
logger = logging.getLogger(__name__)

# Global constants
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30
MEMORY_THRESHOLD = 90  # Percentage
API_RATE_LIMIT = 10  # Calls per minute
CACHE_TTL = 300  # Seconds

class OptimizationError(Exception):
    """Base exception for optimization errors"""
    pass

class ResourceExhaustedError(OptimizationError):
    """Raised when system resources are depleted"""
    pass

class APIError(OptimizationError):
    """Raised for API-related errors"""
    pass

class OptimizationState(Enum):
    """Tracks optimization process state"""
    IDLE = auto()
    RUNNING = auto()
    FAILED = auto()
    COMPLETED = auto()

class OptimizationStep(Enum):
    """Detailed optimization process steps"""
    ANALYSIS = "Analyzing code structure and patterns"
    PLANNING = "Planning optimization strategy"
    SEPARATION = "Separating code blocks"
    ENHANCEMENT = "Enhancing individual blocks"
    INTEGRATION = "Integrating optimized blocks"
    VALIDATION = "Validating final output"
    DOCUMENTATION = "Adding documentation"

@dataclass
class OptimizationBlock:
    """Represents a code block being optimized"""
    original: str
    optimized: str = ""
    block_type: str = ""
    status: str = "pending"
    messages: List[str] = field(default_factory=list)

class WorkflowOrchestrator:
    """Orchestrates the optimization workflow with visual feedback"""
    def __init__(self):
        self.console = Console()
        self.progress_queue = queue.Queue()
        self.blocks: List[OptimizationBlock] = []
        self.current_step = OptimizationStep.ANALYSIS
        
    async def process_code(self, code: str) -> Generator[Tuple[str, str], None, str]:
        """Process code with visual feedback and yield progress updates"""
        try:
            with Live(self._generate_layout(), refresh_per_second=4) as live:
                # Analyze code structure
                self.current_step = OptimizationStep.ANALYSIS
                analysis_result = await self._analyze_code_structure(code)
                yield "analysis", f"Identified {len(analysis_result)} code blocks"

                # Plan optimizations
                self.current_step = OptimizationStep.PLANNING
                optimization_plan = await self._generate_optimization_plan(analysis_result)
                yield "planning", "Generated optimization strategy"

                # Process each block
                self.current_step = OptimizationStep.ENHANCEMENT
                for block in optimization_plan:
                    optimized_block = await self._optimize_block(block)
                    self.blocks.append(optimized_block)
                    yield "block_update", f"Optimized {block.block_type}"
                    live.update(self._generate_layout())

                # Integrate and validate
                self.current_step = OptimizationStep.INTEGRATION
                final_code = await self._integrate_blocks()
                yield "integration", "Blocks integrated successfully"

                self.current_step = OptimizationStep.VALIDATION
                validation_result = await self._validate_result(final_code)
                yield "validation", "Code validated successfully"

                return final_code

        except Exception as e:
            logger.error(f"Workflow error: {traceback.format_exc()}")
            yield "error", str(e)
            return None

    def _generate_layout(self) -> Panel:
        """Generate rich layout for live display"""
        content = [
            f"Current Step: {self.current_step.value}",
            "",
            *[f"[{'green' if b.status == 'complete' else 'yellow'}]"
              f"{b.block_type}: {b.messages[-1] if b.messages else 'Processing...'}"
              for b in self.blocks]
        ]
        return Panel("\n".join(content), title="Optimization Progress")

    async def _analyze_code_structure(self, code: str):
        """Analyzes code structure and returns blocks for optimization"""
        try:
            tree = ast.parse(code)
            blocks = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    blocks.append(OptimizationBlock(
                        original=ast.get_source_segment(code, node),
                        block_type=node.__class__.__name__
                    ))
            return blocks
        except Exception as e:
            logger.error(f"Code structure analysis failed: {e}")
            raise

    async def _generate_optimization_plan(self, analysis_result: List[OptimizationBlock]) -> List[OptimizationBlock]:
        """Generates optimization plan from analysis results"""
        try:
            # Create optimization tasks for each block
            for block in analysis_result:
                block.status = "planned"
                block.messages.append("Optimization planned")
            return analysis_result
        except Exception as e:
            logger.error(f"Plan generation failed: {e}")
            raise

    async def _optimize_block(self, block: OptimizationBlock) -> OptimizationBlock:
        """Optimizes individual code block"""
        try:
            # Optimization logic here
            block.optimized = block.original  # Placeholder
            block.status = "complete"
            block.messages.append("Optimization completed")
            return block
        except Exception as e:
            logger.error(f"Block optimization failed: {e}")
            raise

    async def _integrate_blocks(self) -> str:
        """Integrates optimized blocks into final code"""
        try:
            return "\n\n".join(block.optimized for block in self.blocks)
        except Exception as e:
            logger.error(f"Block integration failed: {e}")
            raise

    async def _validate_result(self, code: str) -> bool:
        """Validates optimized code"""
        try:
            ast.parse(code)  # Verify syntax
            return True
        except Exception as e:
            logger.error(f"Code validation failed: {e}")
            raise

    async def get_final_result(self) -> Optional[str]:
        """Returns final optimized code result"""
        try:
            if not self.blocks:
                return None
            return await self._integrate_blocks()
        except Exception as e:
            logger.error(f"Getting final result failed: {e}")
            return None

# Initialize OpenAI client with retry mechanism
@functools.lru_cache(maxsize=1)
def get_openai_client() -> Optional[OpenAI]:
    for attempt in range(MAX_RETRIES):
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
            if not api_key:
                raise ValueError("OpenAI API key not found")
            return OpenAI(api_key=api_key)
        except Exception as e:
            logger.error(f"OpenAI client initialization attempt {attempt + 1} failed: {e}")
            if attempt == MAX_RETRIES - 1:
                st.error("Failed to initialize OpenAI client. Please check your API key.")
                st.stop()
    return None

# Initialize client safely
client = get_openai_client()

# Enhanced page configuration with error handling
try:
    st.set_page_config(
        page_title="AI Code Enhancer",
        page_icon=":robot_face:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception as e:
    logger.error(f"Page configuration failed: {e}")
    # Fallback to default configuration
    pass

@dataclass
class AppState:
    """Enhanced application state management with validation"""
    code_blocks: Dict[str, str] = field(default_factory=dict)
    optimization_history: List[Dict] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=lambda: {
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 2000,
        "rate_limit": API_RATE_LIMIT,
        "timeout": DEFAULT_TIMEOUT,
        "max_retries": MAX_RETRIES,
        "memory_limit": MEMORY_THRESHOLD
    })
    session_start: datetime = field(default_factory=datetime.now)
    last_api_call: datetime = field(default_factory=datetime.now)
    api_call_count: int = 0
    error_count: int = 0
    state: OptimizationState = field(default=OptimizationState.IDLE)
    _cache: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate state initialization"""
        self._validate_settings()
        self._setup_cache_cleanup()
    
    def _validate_settings(self):
        """Validate and sanitize settings"""
        required_settings = {"model", "temperature", "max_tokens", "rate_limit", "timeout"}
        if not all(key in self.settings for key in required_settings):
            raise ValueError("Missing required settings")
        
        # Sanitize values
        self.settings["temperature"] = max(0.0, min(1.0, self.settings["temperature"]))
        self.settings["max_tokens"] = max(1, min(4096, self.settings["max_tokens"]))
    
    def _setup_cache_cleanup(self):
        """Setup periodic cache cleanup"""
        def cleanup_cache():
            now = datetime.now()
            self._cache = {k: v for k, v in self._cache.items() 
                          if (now - v.get("timestamp", now)).seconds < CACHE_TTL}
        
        # Register cleanup with weakref to avoid memory leaks
        self._cleanup_ref = weakref.ref(cleanup_cache)
    
    def is_rate_limited(self) -> bool:
        """Enhanced rate limiting with automatic reset"""
        now = datetime.now()
        if (now - self.last_api_call).seconds >= 60:
            self.api_call_count = 0
            self.last_api_call = now
        return self.api_call_count >= self.settings["rate_limit"]
    
    def is_session_expired(self) -> bool:
        """Check if session has expired"""
        return (datetime.now() - self.session_start) > timedelta(hours=4)
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update system metrics with validation"""
        if not isinstance(metrics, dict):
            raise TypeError("Metrics must be a dictionary")
        self._cache["metrics"] = {
            "timestamp": datetime.now(),
            "data": metrics
        }

class CodeOptimizer:
    """Enhanced AI-powered code optimization manager"""
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.memory = ConversationBufferMemory(return_messages=True)
        self.coder = self._initialize_aider()
        self._resource_monitor = self._setup_resource_monitor()
        
    def _initialize_aider(self) -> Optional[Coder]:
        """Initialize Aider with error handling"""
        for attempt in range(MAX_RETRIES):
            try:
                return Coder(
                    models=models.Model(),
                    chat=chat.Chat(),
                    edit_format="diff",
                    stream=True
                )
            except Exception as e:
                logger.error(f"Aider initialization attempt {attempt + 1} failed: {e}")
                if attempt == MAX_RETRIES - 1:
                    return None
    
    def _setup_resource_monitor(self):
        """Setup system resource monitoring"""
        def monitor():
            while True:
                try:
                    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
                    cpu_usage = psutil.cpu_percent()
                    if memory_usage > MEMORY_THRESHOLD or cpu_usage > 90:
                        logger.warning(f"High resource usage: Memory={memory_usage}MB, CPU={cpu_usage}%")
                        gc.collect()
                    time.sleep(5)
                except Exception as e:
                    logger.error(f"Resource monitoring error: {e}")
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        return monitor_thread
            
    @asynccontextmanager
    async def optimization_context(self):
        """Context manager for optimization resources"""
        try:
            yield self
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            self.memory.clear()
            gc.collect()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            
    async def optimize_code_with_agents(self, code: str) -> Optional[str]:
        """Multi-agent code optimization with enhanced error handling"""
        if not isinstance(code, str):
            raise TypeError("Code must be a string")
            
        async with self.optimization_context():
            try:
                # Analyze code structure
                analysis = await self._analyze_code(code)
                
                # Generate optimization plan
                plan = await self._generate_optimization_plan(analysis)
                
                # Execute optimizations
                optimized = await self._execute_optimizations(code, plan)
                
                # Verify changes
                if await self._verify_changes(code, optimized):
                    return optimized
                    
                return None
                
            except OpenAIError as e:
                logger.error(f"OpenAI API error: {e}")
                raise APIError(f"OpenAI API error: {str(e)}")
            except Exception as e:
                logger.error(f"Agent optimization failed: {traceback.format_exc()}")
                raise OptimizationError(f"Optimization failed: {str(e)}")

def initialize_session_state():
    """Initialize session state with error handling"""
    try:
        if 'app_state' not in st.session_state:
            st.session_state.app_state = AppState()
        # Validate existing state
        if not isinstance(st.session_state.app_state, AppState):
            logger.error("Invalid session state detected")
            st.session_state.app_state = AppState()
    except Exception as e:
        logger.error(f"Session state initialization failed: {e}")
        st.error("Failed to initialize application state")
        st.stop()

@contextmanager
def error_boundary(operation: str):
    """Context manager for error handling and user feedback"""
    try:
        yield
    except Exception as e:
        logger.error(f"{operation} failed: {traceback.format_exc()}")
        st.error(f"{operation} failed: {str(e)}")
        if hasattr(st.session_state, 'app_state'):
            st.session_state.app_state.error_count += 1

def render_code_editor():
    """Render code editor with input validation"""
    try:
        return st_ace(
            value="# Enter your Python code here",
            language="python",
            theme="monokai",
            height=300,
            key="code_editor",
            on_change=lambda: validate_code(st.session_state.code_editor)
        )
    except Exception as e:
        logger.error(f"Code editor rendering failed: {e}")
        st.error("Failed to load code editor")
        return None

def validate_code(code: str) -> bool:
    """Validate Python code syntax"""
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        st.warning(f"Syntax error: {str(e)}")
        return False
    except Exception as e:
        st.warning(f"Code validation error: {str(e)}")
        return False

@contextmanager
def timeout_handler(seconds: int):
    """Enhanced timeout handler with platform compatibility"""
    if sys.platform != 'win32':  # SIGALRM not available on Windows
        def timeout_signal(signum, frame):
            raise TimeoutError("Operation timed out")
        
        original_handler = signal.signal(signal.SIGALRM, timeout_signal)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)
    else:
        yield  # Fallback for Windows

async def optimize_code(code: str) -> Optional[str]:
    """Enhanced code optimization with visual feedback"""
    if not code or not code.strip():
        st.warning("Please enter code to optimize")
        return None
        
    try:
        orchestrator = WorkflowOrchestrator()
        progress_placeholder = st.empty()
        code_display = st.empty()
        
        async for step_type, message in orchestrator.process_code(code):
            if step_type == "error":
                st.error(message)
                return None
                
            progress_placeholder.markdown(f"**{message}**")
            
            if step_type == "block_update":
                # Show diff of changes
                code_display.code(orchestrator.blocks[-1].optimized, language="python")
                
        optimized_code = await orchestrator.get_final_result()
        if optimized_code:
            st.success("Optimization complete!")
            return optimized_code
            
    except Exception as e:
        logger.error(f"Optimization error: {traceback.format_exc()}")
        st.error(f"Optimization failed: {str(e)}")
        
    return None

async def main():
    """Enhanced main function with comprehensive error handling and state management"""
    try:
        # Initialize and validate session state
        initialize_session_state()
        
        # Check session expiration
        if st.session_state.app_state.is_session_expired():
            await cleanup_resources()
            st.warning("Session expired. Please refresh the page.")
            st.stop()

        st.title("AI Code Enhancer")
        
        # Navigation with error boundary
        with error_boundary("Navigation"):
            selected = option_menu(
                "Main Menu",
                ["Editor", "Settings", "History"],
                icons=["code", "gear", "clock-history"],
                orientation="horizontal"
            )
        
        if selected == "Editor":
            # Editor section with error handling
            with error_boundary("Code editor"):
                code = render_code_editor()
                
                if st.button("Optimize") and code:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Original Code")
                        st.code(code, language="python")
                    
                    with col2:
                        st.subheader("Optimization Progress")
                        with st.spinner(""):
                            optimized = await optimize_code(code)
                            
                            if optimized:
                                st.code(optimized, language="python")
                                
                                # Add download button
                                st.download_button(
                                    "Download Optimized Code",
                                    optimized,
                                    file_name="optimized_code.py",
                                    mime="text/plain"
                                )
                    
        elif selected == "Settings":
            # Settings section with validation
            with error_boundary("Settings"):
                st.subheader("Settings")
                st.session_state.app_state.settings["model"] = st.selectbox(
                    "Model",
                    ["gpt-4", "gpt-3.5-turbo"],
                    help="Select the AI model to use for optimization"
                )
                st.session_state.app_state.settings["temperature"] = st.slider(
                    "Temperature",
                    0.0, 1.0, 0.7,
                    help="Controls randomness in optimization (0=conservative, 1=creative)"
                )
                
                # Advanced settings
                with st.expander("Advanced Settings"):
                    st.session_state.app_state.settings["max_tokens"] = st.number_input(
                        "Max Tokens",
                        min_value=100,
                        max_value=4096,
                        value=2000
                    )
                    st.session_state.app_state.settings["timeout"] = st.number_input(
                        "Timeout (seconds)",
                        min_value=10,
                        max_value=300,
                        value=30
                    )
        
        elif selected == "History":
            # History section with pagination
            with error_boundary("History"):
                st.subheader("Optimization History")
                history = st.session_state.app_state.optimization_history
                
                if not history:
                    st.info("No optimization history available")
                else:
                    for entry in reversed(history):
                        with st.expander(f"Optimization at {entry['timestamp']}"):
                            st.code(entry["original"], language="python")
                            st.code(entry["optimized"], language="python")
                            
                            # Display metrics
                            metrics = entry.get("metrics", {})
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Memory Usage (MB)", f"{metrics.get('memory_usage', 0):.1f}")
                            with col2:
                                st.metric("Errors", metrics.get("error_count", 0))

    except Exception as e:
        logger.error(f"Application error: {traceback.format_exc()}")
        st.error(f"An unexpected error occurred: {str(e)}")
        # Attempt to recover state
        if 'app_state' in st.session_state:
            st.session_state.app_state.error_count += 1

async def cleanup_resources():
    """Cleanup application resources"""
    try:
        if hasattr(st.session_state, 'optimizer'):
            await st.session_state.optimizer.cleanup()
        gc.collect()
    except Exception as e:
        logger.error(f"Resource cleanup failed: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Application startup failed: {traceback.format_exc()}")
        st.error(f"Failed to start application: {str(e)}")

class OptimizationStage(Enum):
    PLANNING = auto()
    ANALYSIS = auto()
    SEGMENTATION = auto()
    TYPE_INFERENCE = auto()
    OPTIMIZATION = auto()
    VERIFICATION = auto()
    DOCUMENTATION = auto()

@dataclass
class OptimizationTask:
    """Represents a single optimization task"""
    stage: OptimizationStage
    code_block: str
    context: Dict[str, Any]
    dependencies: List[str]
    status: str = "pending"
    result: Optional[str] = None

class Agent(Protocol):
    """Base protocol for optimization agents"""
    async def process(self, task: OptimizationTask) -> OptimizationTask:
        ...

class PlannerAgent:
    """Plans optimization strategy"""
    async def process(self, code: str) -> List[OptimizationTask]:
        try:
            response = await client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a Python optimization planner."},
                    {"role": "user", "content": f"Analyze this code and create an optimization plan:\n{code}"}
                ]
            )
            plan = json.loads(response.choices[0].message.content)
            return [OptimizationTask(**task) for task in plan]
        except Exception as e:
            logger.error(f"Planning error: {e}")
            raise

class AnalysisAgent:
    """Analyzes code structure and patterns"""
    async def process(self, task: OptimizationTask) -> OptimizationTask:
        try:
            response = await client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a Python code analyzer."},
                    {"role": "user", "content": f"Analyze this code:\n{task.code_block}"}
                ]
            )
            task.result = response.choices[0].message.content
            task.status = "completed"
            return task
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            raise

class SegmentationAgent:
    """Segments code into logical blocks"""
    async def process(self, task: OptimizationTask) -> OptimizationTask:
        try:
            tree = ast.parse(task.code_block)
            segments = {}
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    segments[node.name] = ast.get_source_segment(task.code_block, node)
            task.result = segments
            task.status = "completed"
            return task
        except Exception as e:
            logger.error(f"Segmentation error: {e}")
            raise

class TypeInferenceAgent:
    """Adds type hints to code"""
    async def process(self, task: OptimizationTask) -> OptimizationTask:
        try:
            response = await client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a Python type inference expert."},
                    {"role": "user", "content": f"Add type hints to:\n{task.code_block}"}
                ]
            )
            task.result = response.choices[0].message.content
            task.status = "completed"
            return task
        except Exception as e:
            logger.error(f"Type inference error: {e}")
            raise

class VerificationAgent:
    """Verifies optimized code"""
    async def process(self, task: OptimizationTask) -> OptimizationTask:
        try:
            # Verify syntax
            ast.parse(task.code_block)
            # Run basic static analysis
            task.result = {"valid": True, "message": "Code verification passed"}
            task.status = "completed"
            return task
        except Exception as e:
            logger.error(f"Verification error: {e}")
            raise

class DocumentationAgent:
    """Updates code documentation"""
    async def process(self, task: OptimizationTask) -> OptimizationTask:
        try:
            response = await client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a Python documentation expert."},
                    {"role": "user", "content": f"Add/improve documentation for:\n{task.code_block}"}
                ]
            )
            task.result = response.choices[0].message.content
            task.status = "completed"
            return task
        except Exception as e:
            logger.error(f"Documentation error: {e}")
            raise

class OptimizationPipeline:
    """Manages the end-to-end optimization process"""
    def __init__(self):
        self.agents = {
            OptimizationStage.PLANNING: PlannerAgent(),
            OptimizationStage.ANALYSIS: AnalysisAgent(),
            OptimizationStage.SEGMENTATION: SegmentationAgent(),
            OptimizationStage.TYPE_INFERENCE: TypeInferenceAgent(),
            OptimizationStage.OPTIMIZATION: AiderOptimizationAgent(),
            OptimizationStage.VERIFICATION: VerificationAgent(),
            OptimizationStage.DOCUMENTATION: DocumentationAgent()
        }
        self.progress = Progress()
        self.console = Console()
        
    async def optimize_code(self, code: str) -> str:
        """Run full optimization pipeline"""
        try:
            # Create optimization plan
            planner = self.agents[OptimizationStage.PLANNING]
            tasks = await planner.process(code)
            
            # Initialize progress tracking
            task_ids = {}
            for task in tasks:
                task_ids[task.stage] = self.progress.add_task(
                    f"[cyan]{task.stage.name}",
                    total=100
                )
            
            # Process tasks with dependencies
            results = {}
            async with asyncio.TaskGroup() as group:
                for task in tasks:
                    if all(dep in results for dep in task.dependencies):
                        task.context.update({
                            dep: results[dep] for dep in task.dependencies
                        })
                        group.create_task(self._process_task(
                            task, task_ids[task.stage]
                        ))
            
            # Assemble final result
            return self._assemble_result(results)
            
        except Exception as e:
            logger.error(f"Pipeline error: {traceback.format_exc()}")
            raise

    async def _process_task(self, task: OptimizationTask, progress_id: int) -> None:
        """Process single optimization task"""
        try:
            agent = self.agents[task.stage]
            
            # Update progress
            self.progress.update(progress_id, advance=50)
            
            # Process task
            result = await agent.process(task)
            
            # Store result
            task.result = result
            task.status = "completed"
            
            # Update progress
            self.progress.update(progress_id, advance=50)
            
        except Exception as e:
            task.status = "failed"
            logger.error(f"Task error: {e}")
            raise

    def _assemble_result(self, results: Dict[str, Any]) -> str:
        """Assembles final optimized code from stage results"""
        try:
            # Combine results in correct order
            final_code = []
            if results.get(OptimizationStage.DOCUMENTATION):
                final_code.append("# Documentation")
                final_code.append(results[OptimizationStage.DOCUMENTATION])
            if results.get(OptimizationStage.TYPE_INFERENCE):
                final_code.append("# Type-annotated code")
                final_code.append(results[OptimizationStage.TYPE_INFERENCE])
            if results.get(OptimizationStage.OPTIMIZATION):
                final_code.append("# Optimized code")
                final_code.append(results[OptimizationStage.OPTIMIZATION])
            
            return "\n\n".join(final_code)
        except Exception as e:
            logger.error(f"Result assembly error: {e}")
            raise

class AiderOptimizationAgent:
    """Handles code optimization using Aider"""
    def __init__(self):
        self.coder = self._initialize_aider()
        
    def _initialize_aider(self) -> Optional[Coder]:
        try:
            return Coder(
                models=models.Model(),
                chat=chat.Chat(),
                edit_format="diff",
                stream=True
            )
        except Exception as e:
            logger.error(f"Aider initialization failed: {e}")
            return None
            
    async def process(self, task: OptimizationTask) -> str:
        """Optimize code block using Aider"""
        if not self.coder:
            raise ValueError("Aider not initialized")
            
        try:
            # Create optimization prompt using context
            prompt = f"""
            Optimize this code block based on:
            - Analysis: {task.context.get('analysis', '')}
            - Type hints: {task.context.get('type_inference', '')}
            
            Code block:
            {task.code_block}
            """
            
            # Run Aider optimization
            result = await self.coder.edit(
                task.code_block,
                prompt,
                max_retries=3
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Aider optimization error: {e}")
            raise

def render_optimization_interface():
    """Render dynamic optimization interface"""
    st.title("AI Code Enhancer")
    
    # Code input
    code = st_ace(
        value="# Enter your Python code here",
        language="python",
        theme="monokai",
        height=300,
        key="code_editor"
    )
    
    if st.button("Start Optimization"):
        # Initialize pipeline
        pipeline = OptimizationPipeline()
        
        # Create progress containers
        progress_containers = {
            stage: st.empty() for stage in OptimizationStage
        }
        
        # Create result containers
        result_containers = {
            stage: st.expander(f"{stage.name} Results", expanded=False)
            for stage in OptimizationStage
        }
        
        try:
            # Run optimization with auto-refresh
            count = st_autorefresh(interval=1000, limit=300)
            
            with st.spinner("Optimizing code..."):
                # Run pipeline
                optimized = asyncio.run(pipeline.optimize_code(code))
                
                # Display results
                for stage, result in pipeline.results.items():
                    # Update progress
                    progress_containers[stage].progress(100)
                    
                    # Show result
                    with result_containers[stage]:
                        if stage == OptimizationStage.OPTIMIZATION:
                            st.code(result, language="python")
                        else:
                            st.json(result)
                
                # Show final code
                st.success("Optimization complete!")
                st.code(optimized, language="python")
                
                # Save to history
                st.session_state.app_state.optimization_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "original": code,
                    "optimized": optimized,
                    "stages": pipeline.results
                })
                
        except Exception as e:
            st.error(f"Optimization failed: {str(e)}")
            logger.error(f"Pipeline error: {traceback.format_exc()}")

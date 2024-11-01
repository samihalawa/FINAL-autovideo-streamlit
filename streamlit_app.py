# Add at the top with other imports
import streamlit as st
import importlib
import os
import logging
import time
from typing import Optional, Dict, List, Any, Union, Tuple
from github import Github, GithubException
from dotenv import load_dotenv
from streamlit_ace import st_ace
from streamlit_option_menu import option_menu
import sys
from contextlib import contextmanager
import io
from datetime import datetime
import base64
import uuid
from importlib.metadata import distribution
import functools
from cryptography.fernet import Fernet
from filelock import FileLock
import glob
import re
from pathlib import Path
from types import ModuleType
from typing import Callable
import shutil
import psutil
import signal
from typing import Generator
import json
import random
import tempfile

# Add proper logger initialization
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI Autocoder Hub", layout="wide")

load_dotenv()

# Setup logging with proper error handling
def setup_logging() -> None:
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ] if is_streamlit_cloud() else [
                logging.FileHandler('app.log'),
                logging.StreamHandler()
            ]
        )
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        sys.exit(1)

@contextmanager
def capture_streamlit_error() -> Any:
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

@st.cache_resource
def load_module(module_name: str) -> Optional[Any]:
    try:
        # Get module code
        module_path = f"{module_name}.py"
        with open(module_path, 'r') as f:
            code = f.read()
            
        # Check imports first
        is_safe, missing_packages = safe_import_requirements(code)
        if not is_safe:
            st.error(f"Missing required packages: {', '.join(missing_packages)}")
            return None
            
        # Safe module loading
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            logger.error(f"Module {module_name} not found")
            return None
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
        
    except Exception as e:
        logger.error(f"Module load error: {e}")
        return None

# Centralized session state initialization with type hints
def initialize_session_state() -> None:
    """Enhanced session state initialization with recovery"""
    try:
        # Initialize defaults if not already set
        defaults = {
            'settings': {
                'auto_save': True,
                'show_previews': True,
                'dark_mode': False,
                'github': {
                    'username': os.getenv('GITHUB_USERNAME', ''),
                    'repo': os.getenv('GITHUB_REPO', ''),
                    'token': os.getenv('GITHUB_TOKEN', '')
                }
            },
            'current_app': None,
            'app_code_storage': {},
            'version_history': [],
            'error_count': 0,
            'last_error': None,
            'resource_usage': {},
            'active_apps': set(),
            'pending_operations': []
        }
        
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v
                    
    except Exception as e:
        logger.error(f"Session initialization error: {e}")
        st.error("Failed to initialize session state")

# Consolidated state management utilities with proper typing
class StateManager:
    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """Safe session state access with nested key support"""
        try:
            keys = key.split('.')
            value = st.session_state
            for k in keys:
                value = value.get(k, {})
            return value if value != {} else default
        except Exception:
            return default
    
    @staticmethod
    def set(key: str, value: Any) -> None:
        """Safe session state update with nested key support"""
        try:
            keys = key.split('.')
            if len(keys) == 1:
                st.session_state[key] = value
            else:
                current = st.session_state
                for k in keys[:-1]:
                    current = current.setdefault(k, {})
                current[keys[-1]] = value
        except Exception as e:
            logger.error(f"State update error: {e}")
    
    @staticmethod
    def delete(key: str) -> None:
        """Safe session state cleanup"""
        if key in st.session_state:
            del st.session_state[key]

# Unified error handling decorator with proper typing
def handle_errors(func: Any) -> Any:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Optional[Any]:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_id = str(uuid.uuid4())
            logger.error(f"Error ID {error_id}: {str(e)}")
            st.error(f"""
                An error occurred (ID: {error_id}). 
                Please try again or contact support if the issue persists.
                
                Details: {str(e)}
            """)
            return None
    return wrapper

# Consolidated file operations with proper error handling
class FileManager:
    @staticmethod
    @handle_errors
    def save_app_code(file_name: str, content: str) -> bool:
        """Universal save function with validation and error handling"""
        try:
            if not isinstance(content, str):
                raise ValueError("Content must be a string")
            
            if not file_name.endswith('.py'):
                file_name = f"{file_name}.py"
                
            # Add atomic file writing with backup
            backup_file = f"{file_name}.bak"
            lock_file = f"{file_name}.lock"
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_name) if os.path.dirname(file_name) else '.', exist_ok=True)
            
            with FileLock(lock_file):
                try:
                    # Create backup of existing file
                    if os.path.exists(file_name):
                        shutil.copy2(file_name, backup_file)
                    
                    # Write new content
                    with open(file_name, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    # Store in session state
                    if 'app_code_storage' not in st.session_state:
                        st.session_state.app_code_storage = {}
                    st.session_state.app_code_storage[file_name] = content
                    
                    # Remove backup if successful
                    if os.path.exists(backup_file):
                        os.remove(backup_file)
                        
                    return True
                    
                except Exception as e:
                    logger.error(f"File write error: {e}")
                    # Restore from backup
                    if os.path.exists(backup_file):
                        shutil.copy2(backup_file, file_name)
                    return False
                finally:
                    # Cleanup lock file
                    if os.path.exists(lock_file):
                        try:
                            os.remove(lock_file)
                        except OSError:
                            pass
                            
        except Exception as e:
            logger.error(f"File operation error: {e}")
            return False
    
    @staticmethod
    @handle_errors
    def load_app_code(file_name: str) -> Optional[str]:
        try:
            with open(file_name, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"File not found: {file_name}")
            return None
        except Exception as e:
            logger.error(f"Error loading file {file_name}: {e}")
            return None

# Unified GitHub operations with proper error handling
class GitHubManager:
    def __init__(self) -> None:
        self.config: Dict[str, str] = StateManager.get('settings', {}).get('github', {})
        self.client: Optional[Github] = None
        if self.config.get('token'):
            try:
                self.client = Github(self.config['token'])
                # Test connection
                self.client.get_user().login
            except GithubException as e:
                logger.error(f"GitHub initialization error: {e}")
                self.client = None
            except Exception as e:
                logger.error(f"Unexpected GitHub error: {e}")
                self.client = None

    @handle_errors
    def sync_files(self, files: List[str], commit_msg: str) -> bool:
        if not self.client:
            raise ValueError("GitHub client not initialized")
            
        try:
            repo = self.client.get_repo(f"{self.config['username']}/{self.config['repo']}")
            
            for file in files:
                content = FileManager.load_app_code(file)
                if content is None:
                    logger.warning(f"Skipping {file}: File not found or empty")
                    continue
                    
                try:
                    contents = repo.get_contents(file)
                    repo.update_file(contents.path, commit_msg, content, contents.sha)
                except GithubException as e:
                    if e.status == 404:  # File doesn't exist
                        repo.create_file(file, commit_msg, content)
                    else:
                        raise
            return True
        except Exception as e:
            logger.error(f"GitHub sync error: {e}")
            raise

def is_streamlit_cloud() -> bool:
    """Check if running on Streamlit Cloud"""
    return os.getenv('STREAMLIT_RUNTIME_ENV') == 'cloud'

# Add memory monitoring and cleanup
def monitor_resource_usage() -> Dict[str, float]:
    process = psutil.Process()
    return {
        'memory': process.memory_info().rss / 1024 / 1024,  # MB
        'cpu': process.cpu_percent()
    }

@contextmanager
def timeout(seconds: int):
    """Custom timeout context manager compatible with all Python versions."""
    def signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    # Only use SIGALRM on Unix systems
    if hasattr(signal, 'SIGALRM'):
        # Register a function to raise a TimeoutError on the signal
        signal.signal(signal.SIGALRM, signal_handler)
        # Schedule the signal to be sent after ``time``
        signal.alarm(seconds)

        try:
            yield
        finally:
            # Disable the alarm
            signal.alarm(0)
    else:
        # On Windows, just execute without timeout
        yield

def run_app_safely(module: Any, app_name: str) -> None:
    try:
        if not hasattr(module, 'main'):
            st.error(f"App {app_name} has no main() function")
            cleanup_failed_app()
            return
            
        # Enhanced memory tracking
        initial_resources = monitor_resource_usage()
        
        with st.spinner("Running app..."):
            try:
                with timeout(30):  # Use our custom timeout
                    module.main()
            except TimeoutError:
                raise TimeoutError("App execution timed out")
            finally:
                # Always cleanup resources
                cleanup_app_state()
            
    except TimeoutError:
        st.error(f"App {app_name} timed out")
        cleanup_failed_app()
    except MemoryError as e:
        st.error(f"Memory limit exceeded: {str(e)}")
        cleanup_failed_app()
    except Exception as e:
        st.error(f"App crashed: {str(e)}")
        cleanup_failed_app()

def create_new_app() -> None:
    """Home page with app creation functionality"""
    st.title("AI Autocoder Hub")
    st.markdown("### Create New App")
    
    # App creation form
    with st.form("new_app_form"):
        app_name = st.text_input("App Name").strip()
        
        template_options: Dict[str, str] = {
            "Basic": """import streamlit as st\n\ndef main():\n    st.title("{{APP_NAME}}")\n    st.write("Welcome to {{APP_NAME}}!")\n\nif __name__ == "__main__":\n    main()""",
            "Data Analysis": """import streamlit as st\nimport pandas as pd\nimport plotly.express as px\n\ndef main():\n    st.title("{{APP_NAME}}")\n    st.write("Upload your data to begin analysis")\n    \n    uploaded_file = st.file_uploader("Choose a CSV file")\n    if uploaded_file:\n        df = pd.read_csv(uploaded_file)\n        st.dataframe(df)\n\nif __name__ == "__main__":\n    main()""",
            "Machine Learning": """import streamlit as st\nimport pandas as pd\nfrom sklearn.model_selection import train_test_split\n\ndef main():\n    st.title("{{APP_NAME}}")\n    st.write("Upload your dataset to train the model")\n\nif __name__ == "__main__":\n    main()"""
        }
        
        template_type = st.selectbox("Choose Template", list(template_options.keys()))
        submitted = st.form_submit_button("Create App")
        
        if submitted and app_name:
            try:
                file_name = f"{app_name.lower().replace(' ', '_')}.py"
                if file_name in st.session_state.app_code_storage:
                    st.error("An app with this name already exists!")
                    return
                
                # Generate app code from template
                template_code = template_options[template_type].replace("{{APP_NAME}}", app_name)
                if FileManager.save_app_code(file_name, template_code):
                    st.success(f"Created {file_name} successfully!")
                    time.sleep(1)
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"Error creating app: {str(e)}")

def manage_dependencies() -> None:
    st.subheader("Dependency Management")
    
    # Define core dependencies that should always be present
    core_dependencies: List[str] = [
        "streamlit",
        "streamlit-ace",
        "streamlit-option-menu",
        "PyGithub",
        "python-dotenv",
        "importlib-metadata"
    ]
    
    try:
        with open('requirements.txt', 'r', encoding='utf-8') as f:
            current_reqs = f.read()
    except FileNotFoundError:
        current_reqs = "\n".join(core_dependencies)
        
    col1, col2 = st.columns([2,1])
    
    with col1:
        new_reqs = st_ace(
            value=current_reqs,
            language="text",
            theme="monokai",
            height=300
        )
        
    with col2:
        st.markdown("### Actions")
        if st.button("Save Changes"):
            # Ensure core dependencies are included
            req_lines = set(new_reqs.split('\n'))
            for dep in core_dependencies:
                if not any(line.startswith(dep) for line in req_lines):
                    req_lines.add(dep)
            
            # Write updated requirements
            try:
                with open('requirements.txt', 'w', encoding='utf-8') as f:
                    f.write('\n'.join(sorted(req_lines)))
                st.success("Requirements updated!")
                
                # Install requirements
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
                st.success("Dependencies installed successfully!")
            except Exception as e:
                st.error(f"Error installing dependencies: {str(e)}")

@handle_errors
def github_integration() -> None:
    st.subheader("GitHub Integration")
    
    if 'settings' not in st.session_state:
        initialize_session_state()
        
    github_settings = st.session_state.settings.get('github', {})
    if not all([github_settings.get('token'), github_settings.get('username'), github_settings.get('repo')]):
        st.warning("Please configure GitHub settings first in the Settings page")
        if st.button("Go to Settings"):
            st.session_state.selected_page = "Settings"
            st.experimental_rerun()
        return
        
    try:
        g = Github(github_settings['token'])
        repo = g.get_repo(f"{github_settings['username']}/{github_settings['repo']}")
        
        # Get files from session state only when on cloud
        available_files = set(st.session_state.get('app_code_storage', {}).keys())
        if not is_streamlit_cloud():
            available_files.update([f for f in os.listdir() if f.endswith('.py')])
        
        if not available_files:
            st.info("No files available to sync")
            return
            
        files_to_sync = st.multiselect(
            "Select Files to Sync",
            sorted(list(available_files))
        )
        
        commit_msg = st.text_input("Commit Message", "Update from Streamlit Hub")
        
        if st.button("Sync with GitHub"):
            if not files_to_sync:
                st.warning("Please select files to sync")
                return
                
            with st.spinner("Syncing with GitHub..."):
                try:
                    gh_manager = GitHubManager()
                    if gh_manager.sync_files(files_to_sync, commit_msg):
                        st.success("Successfully synced all files to GitHub!")
                    else:
                        st.error("Sync failed")
                except Exception as e:
                    st.error(f"Error during sync: {str(e)}")
                    
    except Exception as e:
        st.error(f"GitHub Error: {str(e)}")

def update_requirements(template: str) -> None:
    template_requirements: Dict[str, List[str]] = {
        "Basic": ["streamlit"],
        "Data Analysis": ["streamlit", "pandas", "plotly", "numpy"],
        "Machine Learning": ["streamlit", "pandas", "scikit-learn", "numpy"]
    }
    
    try:
        current_reqs = set()
        if os.path.exists('requirements.txt'):
            with open('requirements.txt', 'r', encoding='utf-8') as f:
                current_reqs = set(line.strip() for line in f.readlines())
        
        # Add core dependencies
        core_deps = {
            "streamlit",
            "streamlit-ace",
            "streamlit-option-menu",
            "PyGithub",
            "python-dotenv",
            "importlib-metadata"
        }
        
        # Combine all required dependencies
        new_reqs = current_reqs.union(template_requirements.get(template, [])).union(core_deps)
        
        with open('requirements.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted(new_reqs)))
            
        # Install new requirements
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            
    except Exception as e:
        st.error(f"Error updating requirements: {str(e)}")

def get_app_metadata(file_path: str) -> Dict[str, Union[str, int, bool]]:
    try:
        stats = os.stat(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return {
            "size": f"{stats.st_size/1024:.1f} KB",
            "modified": datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M'),
            "lines": sum(1 for _ in open(file_path, encoding='utf-8')),
            "has_main": "main()" in content
        }
    except Exception as e:
        logger.error(f"Error getting metadata: {str(e)}")
        return {}

def show_breadcrumbs(selected: str, current_app: Optional[str] = None) -> None:
    crumbs = ["🏠 Home"]
    if selected != " Home":
        crumbs.append(selected)
    if current_app:
        crumbs.append(f"📱 {current_app}")
    st.markdown(" > ".join(crumbs))

def show_app_preview(app_path: str, with_edit: bool = False) -> None:
    try:
        # First try session state
        content = st.session_state.get('app_code_storage', {}).get(app_path)
        
        # If not in session state and not on cloud, try file
        if not content and not is_streamlit_cloud():
            try:
                with open(app_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Cache in session state
                    st.session_state.setdefault('app_code_storage', {})[app_path] = content
            except Exception as e:
                logger.warning(f"File read failed: {e}")
                content = ""
        
        if not content:
            st.warning("No content available")
            return
            
        if with_edit:
            col1, col2 = st.columns([3, 1])
            with col1:
                edited_code = st_ace(
                    value=content,
                    language="python", 
                    theme="monokai",
                    height=400,
                    key=f"editor_{app_path}"
                )
            with col2:
                if st.button("Save Changes"):
                    if FileManager.save_app_code(app_path, edited_code):
                        st.success("Changes saved!")
                        time.sleep(1)
                        st.experimental_rerun()
        else:
            st.code(content, language="python")
    except Exception as e:
        st.error(f"Error in preview: {str(e)}")

def search_apps(apps: Dict[str, str], query: str) -> Dict[str, str]:
    if not query:
        return apps
    results = {}
    for name, path in apps.items():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().lower()
            if query.lower() in name.lower() or query.lower() in content:
                results[name] = path
        except Exception as e:
            logger.error(f"Error searching {name}: {str(e)}")
    return results

def clone_app(original_path: str, new_name: str) -> Tuple[bool, str]:
    try:
        new_path = f"{new_name.lower().replace(' ', '_')}.py"
        if os.path.exists(new_path):
            return False, "App already exists"
        with open(original_path, 'r', encoding='utf-8') as src, open(new_path, 'w', encoding='utf-8') as dst:
            dst.write(src.read())
        return True, f"Created {new_path}"
    except Exception as e:
        return False, str(e)

def validate_requirements(requirements_text: str) -> Tuple[bool, List[str]]:
    try:
        requirements = [r.strip() for r in requirements_text.split('\n') if r.strip()]
        invalid = []
        for req in requirements:
            try:
                package_name = req.split('==')[0].split('>=')[0].split('<=')[0].strip()
                __import__(package_name)
            except ImportError:
                invalid.append(req)
        return not invalid, invalid
    except Exception as e:
        return False, [str(e)]

def show_diff(original: str, modified: str) -> None:
    st.code("\n".join(
        f"- {line}" if line in original.splitlines() and line not in modified.splitlines()
        else f"+ {line}" if line in modified.splitlines() and line not in original.splitlines()
        else f"  {line}"
        for line in set(original.splitlines() + modified.splitlines())
    ))

def git_workflow(repo: Any, branch: str = 'main') -> None:
    branches = [b.name for b in repo.get_branches()]
    selected_branch = st.selectbox("Select Branch", branches, index=branches.index('main') if 'main' in branches else 0)
    
    if st.button("Pull Latest Changes"):
        try:
            contents = repo.get_contents("")
            for content in contents:
                if content.path.endswith('.py'):
                    with open(content.path, 'w', encoding='utf-8') as f:
                        f.write(content.decoded_content.decode())
            st.success("Successfully pulled latest changes")
        except Exception as e:
            st.error(f"Error pulling changes: {str(e)}")

    with st.expander("Commit History"):
        commits = repo.get_commits()
        for commit in list(commits)[:5]:
            st.markdown(f"**{commit.commit.message}**")
            st.markdown(f"Author: {commit.commit.author.name}")
            st.markdown(f"Date: {commit.commit.author.date}")
            st.markdown("---")

def handle_template_customization() -> None:
    st.subheader("Template Customization")
    template = st.session_state.get('current_template', '')
    if not template:
        return
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Edit Template")
        modified_template = st_ace(
            value=template,
            language="python",
            theme="monokai",
            height=300
        )
        st.session_state.current_template = modified_template
    
    with col2:
        st.markdown("### Preview")
        with st.expander("Template Preview", expanded=True):
            st.code(modified_template, language="python")

def version_control() -> Optional[str]:
    if 'version_history' not in st.session_state:
        st.session_state.version_history = []
    if 'current_code' not in st.session_state:
        st.session_state.current_code = ''
    
    current_version = len(st.session_state.version_history)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Undo") and current_version > 0:
            st.session_state.version_history.pop()
            return st.session_state.version_history[-1] if st.session_state.version_history else ''
    
    with col2:
        if st.button("Save Version"):
            st.session_state.version_history.append(st.session_state.current_code)
    return None

def save_settings(settings: Dict[str, Any]) -> None:
    for key, value in settings.items():
        st.session_state[key] = value
        if key in ['GITHUB_USERNAME', 'GITHUB_REPO', 'GITHUB_TOKEN']:
            os.environ[key] = str(value)

def settings_page() -> None:
    st.title("Settings")
    general_tab, github_tab, advanced_tab = st.tabs(["General", "GitHub", "Advanced"])
    
    with general_tab:
        st.markdown("### General Settings")
        settings = st.session_state.settings
        settings['auto_save'] = st.toggle("Auto Save", settings.get('auto_save', True))
        settings['show_previews'] = st.toggle("Show Code Previews", settings.get('show_previews', True))
        settings['dark_mode'] = st.toggle("Dark Mode", settings.get('dark_mode', False))
        
    with github_tab:
        st.markdown("### GitHub Integration")
        github_settings = settings.get('github', {})
        github_settings['username'] = st.text_input("GitHub Username", github_settings.get('username', ''))
        github_settings['repo'] = st.text_input("GitHub Repository", github_settings.get('repo', ''))
        github_settings['token'] = st.text_input("GitHub Token", github_settings.get('token', ''), type="password")
        
        if st.button("Test Connection"):
            test_github_connection(github_settings)
    
    with advanced_tab:
        st.markdown("### Advanced Settings")
        if st.button("Clear All Data"):
            if st.checkbox("I understand this will delete all apps and settings"):
                clear_all_data()
                st.success("All data cleared! Restarting app...")
                time.sleep(2)
                st.experimental_rerun()

def get_github_config() -> Dict[str, str]:
    return {
        'username': st.session_state.get('GITHUB_USERNAME', os.getenv('GITHUB_USERNAME', '')),
        'repo': st.session_state.get('GITHUB_REPO', os.getenv('GITHUB_REPO', '')),
        'token': st.session_state.get('GITHUB_TOKEN', os.getenv('GITHUB_TOKEN', ''))
    }
def get_app_url(app_path: str) -> str:
    config = get_github_config()
    return f"https://share.streamlit.io/{config['username']}/{config['repo']}/main/{app_path}"

def ensure_dependencies() -> bool:
    required = ["PyGithub", "streamlit-ace", "streamlit-option-menu", "python-dotenv"]
    try:
        for pkg in required:
            try:
                __import__(pkg.replace('-', '_').lower())
            except ImportError:
                st.error(f"Missing required package: {pkg}")
                return False
        return True
    except Exception as e:
        st.error(f"Dependency error: {e}")
        return False

def get_session_state(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, default)

def validate_app_data(data: Dict[str, Any]) -> Dict[str, Any]:
    required = {'name': str, 'content': str, 'metadata': dict}
    if not all(isinstance(data.get(k), t) for k,t in required.items()):
        raise TypeError("Invalid app data types")
    return data

def api_call_wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to handle API call errors and retries"""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"API call failed after {max_retries} attempts: {e}")
                    raise
                time.sleep(retry_delay * (attempt + 1))
        return None
    return wrapper

def validate_callback(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to validate callback functions"""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            if not callable(func):
                raise ValueError("Invalid callback function")
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Callback validation failed: {e}")
            return None
    return wrapper

def cleanup_app_state() -> None:
    """Safe cleanup of app state and temporary files"""
    try:
        # Add atomic state cleanup
        lock_file = "state.lock"
        try:
            with FileLock(lock_file):
                # Clear temp session state
                keys_to_remove = [k for k in st.session_state.keys() 
                                if k.startswith('temp_') or k in ['gh_token', 'current_code']]
                for k in keys_to_remove:
                    st.session_state.pop(k, None)
        finally:
            if os.path.exists(lock_file):
                try:
                    os.remove(lock_file)
                except OSError:
                    pass
        
        # Remove temp files safely
        temp_files = glob.glob("*.tmp")
        for f in temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except OSError as e:
                logger.warning(f"Failed to remove temp file {f}: {e}")
                
    except Exception as e:
        logger.error(f"Critical state cleanup error: {e}")
        # Only clear session state on critical failure
        try:
            st.session_state.clear()
            initialize_session_state()
        except Exception as cleanup_error:
            logger.critical(f"Failed to reset session state: {cleanup_error}")

def safe_import(module: str) -> Optional[ModuleType]:
    try:
        return __import__(module)
    except ImportError as e:
        logger.error(f"Import error {module}: {e}")
        return None

def transform_app_data(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise TypeError("Invalid app data")
        
    required = ['name', 'content', 'metadata']
    if not all(k in data for k in required):
        raise ValueError(f"Missing fields: {required}")
        
    return {
        'name': str(data['name']),
        'content': str(data['content']), 
        'metadata': data.get('metadata', {})
    }

def initialize_session_state() -> None:
    defaults = {
        'current_app': None,
        'app_code_storage': {},
        'version_history': [],
        'current_code': '',
        'settings': {
            'auto_save': True,
            'show_previews': True,
            'dark_mode': False,
            'github': {
                'username': os.getenv('GITHUB_USERNAME', ''),
                'repo': os.getenv('GITHUB_REPO', ''),
                'token': os.getenv('GITHUB_TOKEN', '')
            }
        },
        'current_template': '',
        '_is_running': True,
        'last_edited': None,
        'search_query': '',
        'selected_files': []
    }
    
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def validate_github_response(resp: Dict[str, Any]) -> Dict[str, Any]:
    if not resp:
        raise ValueError("Empty GitHub response")
    if 'message' in resp:
        raise ValueError(f"GitHub error: {resp['message']}")
    return resp

def save_app_code(name: str, content: str) -> bool:
    try:
        st.session_state.setdefault('app_code_storage', {})[name] = content
        if not is_streamlit_cloud():
            with open(name, 'w', encoding='utf-8') as f:
                f.write(content)
        return True
    except Exception as e:
        logger.error(f"Save error: {e}")
        return False

def recover_session() -> None:
    """Attempt to recover corrupted session state"""
    try:
        # Check for backup state
        backup_file = "session_backup.json"
        if os.path.exists(backup_file):
            with open(backup_file, 'r') as f:
                backup_state = json.load(f)
                st.session_state.update(backup_state)
                st.success("Session recovered from backup")
                return
                
        # If no backup, reinitialize
        initialize_session_state()
        st.warning("Session state was reset")
        
    except Exception as e:
        logger.error(f"Session recovery failed: {e}")
        st.error("Could not recover session state")
        initialize_session_state()

def main() -> None:
    try:
        # Add session recovery
        if '_recovered' not in st.session_state:
            recover_session()
            st.session_state['_recovered'] = True
            
        if not initialize_app():
            handle_app_error(Exception("Initialization failed"), "main")
            return

        # Add progress tracking
        progress_update = show_operation_progress("Loading application", 3)
        
        with st.sidebar:
            progress_update(1, "Loading navigation")
            selected = option_menu(
                "AI Autocoder Hub",
                ["🏠 Home", "📱 Apps", "⚙️ Settings", "🔄 Sync", "📦 Dependencies"],
                icons=['house', 'app', 'gear', 'cloud-upload', 'box'],
                menu_icon="cast",
                default_index=0
            )

        progress_update(2, "Loading current app")
        if current_app := st.session_state.get('current_app'):
            try:
                if module := load_module(current_app):
                    with st.spinner("Running app..."):
                        run_app_safely(module, current_app)
                    return
            except Exception as e:
                handle_app_error(e, current_app)
                return

        progress_update(3, f"Loading {selected}")
        pages = {
            "🏠 Home": create_new_app,
            "📱 Apps": show_apps_page, 
            "⚙️ Settings": settings_page,
            "🔄 Sync": github_integration,
            "📦 Dependencies": manage_dependencies
        }

        if selected in pages:
            try:
                with st.spinner(f"Loading {selected}..."):
                    pages[selected]()
            except Exception as e:
                handle_app_error(e, selected)
                cleanup_app_state()

    except Exception as e:
        handle_app_error(e, "main")
        cleanup_app_state()
        raise

def show_apps_page() -> None:
    """Display and manage apps with proper error handling"""
    try:
        # Prevent concurrent operations
        if '_loading' in st.session_state:
            st.warning("Another operation is in progress...")
            return
            
        st.session_state['_loading'] = True
        try:
            st.title("Manage Apps")
            
            # Get apps from session state and filesystem
            apps = set(st.session_state.get('app_code_storage', {}))
            if not is_streamlit_cloud():
                apps.update(f for f in os.listdir() 
                          if f.endswith('.py') and f != 'streamlit_app.py')
            
            apps = sorted(apps)
            if not apps:
                st.info("No apps found. Create a new app to get started!")
                return
                
            # Search functionality
            query = st.text_input("Search Apps", st.session_state.get('search_query', ''))
            if query:
                apps = [a for a in apps if query.lower() in a.lower()]
                
            # Display apps
            for app in apps:
                with st.expander(app):
                    try:
                        c1, c2, c3 = st.columns([3,1,1])
                        with c1:
                            show_app_preview(app)
                        with c2:
                            if st.button("Edit", key=f"edit_{app}"):
                                st.session_state.update({
                                    'current_app': app,
                                    'editing': True
                                })
                                st.experimental_rerun()
                        with c3:
                            if st.button("Delete", key=f"delete_{app}"):
                                if st.session_state.get('current_app') == app:
                                    del st.session_state['current_app']
                                st.session_state.get('app_code_storage', {}).pop(app, None)
                                if not is_streamlit_cloud() and os.path.exists(app):
                                    os.remove(app)
                                st.success(f"Deleted {app}")
                                time.sleep(1)
                                st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error displaying app {app}: {str(e)}")
                        
        finally:
            st.session_state.pop('_loading', None)
            
    except Exception as e:
        logger.error(f"Apps page error: {e}")
        st.error("Failed to load apps page")
        cleanup_app_state()

def cleanup_failed_app() -> None:
    try:
        st.session_state.pop('current_app', None)
        for f in glob.glob("*.tmp"):
            try: os.remove(f)
            except OSError: pass
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

def test_github_connection(settings: Dict[str, str]) -> bool:
    try:
        if not all(settings.get(k) for k in ['token','username','repo']):
            st.error("Missing GitHub settings")
            return False
            
        g = Github(settings['token'])
        repo = g.get_repo(f"{settings['username']}/{settings['repo']}")
        repo.get_contents("")
        st.success("GitHub connected!")
        return True
        
    except Exception as e:
        logger.error(f"GitHub error: {e}")
        st.error(f"GitHub connection failed: {e}")
        return False

def clear_all_data() -> bool:
    try:
        st.session_state.clear()
        for f in glob.glob("*.tmp"):
            try: os.remove(f)
            except OSError: pass
        initialize_session_state()
        return True
    except Exception as e:
        logger.error(f"Clear error: {e}")
        return False

def initialize_app() -> bool:
    """Initialize app with proper error handling and state management"""
    try:
        # Setup logging
        setup_logging()
        
        # Initialize session state
        initialize_session_state()
        
        # Create necessary directories if not on cloud
        if not is_streamlit_cloud():
            os.makedirs('apps', exist_ok=True)
            os.makedirs('templates', exist_ok=True)
            os.makedirs('logs', exist_ok=True)
        
        # Monitor initial resource usage
        st.session_state['resource_usage'] = monitor_resource_usage()
        
        return True
        
    except Exception as e:
        logger.error(f"App initialization error: {e}")
        st.error("Failed to initialize application")
        return False

def load_app_code(name: str) -> Optional[str]:
    try:
        if code := st.session_state.get('app_code_storage', {}).get(name):
            return code
            
        if not is_streamlit_cloud() and os.path.exists(name):
            with open(name, 'r', encoding='utf-8') as f:
                code = f.read()
                st.session_state.setdefault('app_code_storage', {})[name] = code
                return code
                
        return None
        
    except Exception as e:
        logger.error(f"Load error: {e}")
        return None

def edit_app(name: str) -> None:
    try:
        if not (content := load_app_code(name)):
            st.error(f"Failed to load {name}")
            return
            
        edited = st_ace(
            value=content,
            language="python",
            theme="monokai",
            height=600,
            key=f"editor_{name}"
        )
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Save"):
                if save_app_code(name, edited):
                    st.success("Saved!")
                    st.session_state.setdefault('version_history', []).append({
                        'timestamp': datetime.now().isoformat(),
                        'content': edited
                    })
                else:
                    st.error("Save failed")
                    
        with c2:
            if st.button("Discard"):
                st.session_state.editing = False
                st.experimental_rerun()
                
    except Exception as e:
        logger.error(f"Edit error: {e}")
        st.error("Edit failed")

def run_app(name: str) -> None:
    try:
        if not (module := load_module(name.replace('.py', ''))):
            st.error(f"Failed to load {name}")
            return
            
        st.session_state.current_app = name
        with st.spinner(f"Running {name}..."):
            try:
                module.main()
            except Exception as e:
                st.error(f"App crashed: {e}")
                cleanup_failed_app()
                
    except Exception as e:
        logger.error(f"Run error: {e}")
        st.error("Run failed")
        cleanup_failed_app()

def handle_app_error(error: Exception, app_name: str) -> None:
    """Centralized error handling with recovery options"""
    error_id = str(uuid.uuid4())
    logger.error(f"Error ID {error_id}: {str(error)}")
    
    st.error(f"""
        Error running {app_name} (ID: {error_id})
        
        Options:
        1. Try restarting the app
        2. Reset app state
        3. Contact support
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Restart App"):
            st.experimental_rerun()
    with col2:
        if st.button("Reset State"):
            cleanup_app_state()
            st.experimental_rerun()
    with col3:
        st.markdown("[Contact Support](mailto:support@example.com)")

def show_operation_progress(message: str, total_steps: int) -> None:
    """Show progress bar for long operations"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(step: int, step_message: str = "") -> None:
        progress = int(step * 100 / total_steps)
        progress_bar.progress(progress)
        status_text.text(f"{message}: {step_message}")
        
    return update_progress

@st.cache_data  # Add caching for better performance
def safe_import_requirements(app_code: str) -> Tuple[bool, List[str]]:
    """Safely analyze and import required packages"""
    try:
        # Extract imports
        import_pattern = r'^import\s+(\w+)|^from\s+(\w+)\s+import'
        matches = re.finditer(import_pattern, app_code, re.MULTILINE)
        required_packages = set()
        
        for match in matches:
            package = match.group(1) or match.group(2)
            if package not in ['streamlit', 'os', 'sys', 'typing']:
                required_packages.add(package)
        
        # Check each package
        missing_packages = []
        for package in required_packages:
            try:
                importlib.import_module(package)
            except ImportError:
                missing_packages.append(package)
                
        return len(missing_packages) == 0, missing_packages
        
    except Exception as e:
        logger.error(f"Package analysis error: {e}")
        return False, []

class SessionManager:
    @staticmethod
    def backup_session() -> None:
        """Create backup of session state"""
        try:
            backup_data = {
                k: v for k, v in st.session_state.items() 
                if not k.startswith('_') and isinstance(v, (str, int, float, bool, list, dict))
            }
            with open('session_backup.json', 'w') as f:
                json.dump(backup_data, f)
        except Exception as e:
            logger.error(f"Session backup failed: {e}")

    @staticmethod
    def restore_session() -> bool:
        """Restore session from backup"""
        try:
            if os.path.exists('session_backup.json'):
                with open('session_backup.json', 'r') as f:
                    backup_data = json.load(f)
                st.session_state.update(backup_data)
                return True
        except Exception as e:
            logger.error(f"Session restore failed: {e}")
        return False

class AppErrorHandler:
    MAX_ERRORS = 3
    ERROR_TIMEOUT = 300  # 5 minutes
    
    @staticmethod
    def handle_error(error: Exception, context: str) -> None:
        """Enhanced error handling with rate limiting"""
        try:
            current_time = time.time()
            error_count = st.session_state.get('error_count', 0)
            last_error_time = st.session_state.get('last_error_time', 0)
            
            # Reset error count if enough time has passed
            if current_time - last_error_time > AppErrorHandler.ERROR_TIMEOUT:
                error_count = 0
            
            # Update error tracking
            error_count += 1
            st.session_state.error_count = error_count
            st.session_state.last_error_time = current_time
            st.session_state.last_error = {
                'error': str(error),
                'context': context,
                'timestamp': datetime.now().isoformat()
            }
            
            # Log error
            error_id = str(uuid.uuid4())
            logger.error(f"Error ID {error_id} in {context}: {str(error)}")
            
            # Handle based on error count
            if error_count >= AppErrorHandler.MAX_ERRORS:
                st.error("""
                    Too many errors occurred. The app will restart.
                    Please contact support if the problem persists.
                """)
                cleanup_app_state()
                time.sleep(2)
                st.experimental_rerun()
            else:
                st.error(f"""
                    Error in {context} (ID: {error_id})
                    {str(error)}
                    
                    Try:
                    1. Refreshing the page
                    2. Clearing cache
                    3. Checking your inputs
                """)
                
        except Exception as e:
            logger.critical(f"Error handler failed: {e}")
            st.error("Critical error occurred")

class FileSystemManager:
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = {'.py', '.txt', '.csv', '.json'}
    
    @staticmethod
    def safe_file_operation(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                with FileLock("fs_operations.lock"):
                    return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"File operation failed: {e}")
                raise
        return wrapper
    
    @staticmethod
    @safe_file_operation
    def save_app_code(file_name: str, content: str) -> bool:
        """Safe file saving with validation"""
        try:
            # Validate file
            if len(content.encode()) > FileSystemManager.MAX_FILE_SIZE:
                raise ValueError("File too large")
                
            file_ext = Path(file_name).suffix
            if file_ext not in FileSystemManager.ALLOWED_EXTENSIONS:
                raise ValueError("Invalid file type")
                
            # Create backup
            backup_path = f"{file_name}.bak"
            if os.path.exists(file_name):
                shutil.copy2(file_name, backup_path)
                
            # Save file
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(content)
                
            # Update session state
            st.session_state.setdefault('app_code_storage', {})[file_name] = content
            
            # Remove backup
            if os.path.exists(backup_path):
                os.remove(backup_path)
                
            return True
            
        except Exception as e:
            logger.error(f"File save error: {e}")
            # Restore from backup
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, file_name)
            return False

# Add proper dependency checks
def check_dependencies() -> bool:
    """Check if all required packages are installed."""
    required_packages: Dict[str, str] = {
        'streamlit': 'streamlit',
        'openai': 'openai',
        'github': 'PyGithub',
        'dotenv': 'python-dotenv',
        'streamlit_ace': 'streamlit-ace'
    }
    
    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        st.error(f"Missing required packages: {', '.join(missing)}")
        st.info("Install missing packages with: pip install -r requirements.txt")
        return False
    return True

if not check_dependencies():
    st.stop()
if __name__ == "__main__":
    try:
        if ensure_dependencies():
            main()
        else:
            st.error("Failed to install dependencies")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        st.error(f"Critical error: {e}")


# Add at the top with other imports
import streamlit as st
import importlib
import os
import logging
import time
from PyGithub import Github  # Changed from: from github import Github
from dotenv import load_dotenv
from streamlit_ace import st_ace
from streamlit_option_menu import option_menu
import sys
from contextlib import contextmanager
import io
from datetime import datetime
import base64
import uuid
from importlib.metadata import distribution  # Replace pkg_resources
from typing import List, Optional, Dict, Any
import functools
from cryptography.fernet import Fernet
from filelock import FileLock
import glob
import re

# Add proper logger initialization
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI Autocoder Hub", layout="wide")

load_dotenv()

# Setup logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()
        ]
    )

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

@st.cache_resource
def load_module(module_name: str) -> Optional[Any]:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None

# Centralized session state initialization
def initialize_session_state():
    """Initialize all required session state variables"""
    defaults = {
        'current_app': None,
        'app_code_storage': {},
        'version_history': [],
        'current_code': '',
        'gh_token': None,
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
    
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

# Consolidated state management utilities
class StateManager:
    @staticmethod
    def get(key, default=None):
        """Safe session state access"""
        return st.session_state.get(key, default)
    
    @staticmethod
    def set(key, value):
        """Safe session state update"""
        st.session_state[key] = value
    
    @staticmethod
    def delete(key):
        """Safe session state cleanup"""
        if key in st.session_state:
            del st.session_state[key]

# Unified error handling decorator
def handle_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
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

# Consolidated file operations
class FileManager:
    @staticmethod
    @handle_errors
    def save_app_code(file_name: str, content: str) -> bool:
        """Universal save function with validation and error handling"""
        if not isinstance(content, str):
            raise ValueError("Content must be a string")
        
        if not file_name.endswith('.py'):
            file_name = f"{file_name}.py"
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_name) if os.path.dirname(file_name) else '.', exist_ok=True)
            
        StateManager.set(f"app_code_storage.{file_name}", content)
        
        if not is_streamlit_cloud():
            lock_path = f"{file_name}.lock"
            try:
                with FileLock(lock_path, timeout=10):
                    with open(file_name, 'w', encoding='utf-8') as f:
                        f.write(content)
            finally:
                if os.path.exists(lock_path):
                    try:
                        os.remove(lock_path)
                    except OSError:
                        pass
        return True
    
    @staticmethod
    @handle_errors
    def load_app_code(file_name: str) -> str:
        try:
            with open(file_name, 'r') as f:
                return f.read()
        except FileNotFoundError:
            return None

# Unified GitHub operations
class GitHubManager:
    def __init__(self):
        self.config = StateManager.get('settings', {}).get('github', {})
        self.client = None
        if self.config.get('token'):
            try:
                self.client = Github(self.config['token'])
                # Test connection
                self.client.get_user().login
            except Exception as e:
                logger.error(f"GitHub initialization error: {str(e)}")
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
                except Exception as e:
                    if "404" in str(e):  # File doesn't exist
                        repo.create_file(file, commit_msg, content)
                    else:
                        raise
            return True
        except Exception as e:
            logger.error(f"GitHub sync error: {str(e)}")
            raise

def is_streamlit_cloud():
    """Check if running on Streamlit Cloud"""
    return os.getenv('STREAMLIT_RUNTIME_ENV') == 'cloud'

def run_app_safely(module, app_name):
    try:
        with st.spinner("Running app..."):
            module.main()
    except Exception as e:
        st.error(f"App crashed: {str(e)}")
        # Add recovery mechanism
        st.session_state.current_app = None
        cleanup_failed_app()

def create_new_app():
    """Home page with app creation functionality"""
    st.title("AI Autocoder Hub")
    st.markdown("### Create New App")
    
    # App creation form
    with st.form("new_app_form"):
        app_name = st.text_input("App Name").strip()
        
        template_options = {
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
                FileManager.save_app_code(file_name, template_code)
                st.success(f"Created {file_name} successfully!")
                time.sleep(1)
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error creating app: {str(e)}")

def manage_dependencies():
    st.subheader("Dependency Management")
    
    # Define core dependencies that should always be present
    core_dependencies = [
        "streamlit",
        "streamlit-ace",
        "streamlit-option-menu",
        "PyGithub",
        "python-dotenv",
        "importlib-metadata"
    ]
    
    try:
        with open('requirements.txt', 'r') as f:
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
            with open('requirements.txt', 'w') as f:
                f.write('\n'.join(sorted(req_lines)))
            st.success("Requirements updated!")
            
            # Install requirements
            try:
                import subprocess
                subprocess.check_call(["pip", "install", "-r", "requirements.txt"])
                st.success("Dependencies installed successfully!")
            except Exception as e:
                st.error(f"Error installing dependencies: {str(e)}")

@handle_errors
def github_integration():
    st.subheader("GitHub Integration")
    
    token = st.session_state.get('GITHUB_TOKEN')
    if not token:
        st.warning("Please enter GitHub token in settings first")
        return
        
    try:
        g = Github(token)
        repo_name = f"{st.session_state.get('GITHUB_USERNAME')}/{st.session_state.get('GITHUB_REPO')}"
        repo = g.get_repo(repo_name)
        
        files_to_sync = st.multiselect(
            "Select Files to Sync",
            [f for f in os.listdir() if f.endswith('.py')]
        )
        
        commit_msg = st.text_input("Commit Message", "Update from Streamlit Hub")
        
        if st.button("Sync with GitHub"):
            with st.spinner("Syncing with GitHub..."):
                try:
                    for file in files_to_sync:
                        with open(file, 'r') as f:
                            content = f.read()
                        try:
                            # Check if file exists in repo
                            contents = repo.get_contents(file)
                            # Update existing file
                            repo.update_file(
                                contents.path,
                                commit_msg,
                                content,
                                contents.sha,
                                branch=repo.default_branch
                            )
                        except:
                            # Create new file
                            repo.create_file(
                                file,
                                commit_msg,
                                content,
                                branch=repo.default_branch
                            )
                    st.success("Successfully synced all files to GitHub!")
                except Exception as e:
                    st.error(f"Error during sync: {str(e)}")
                    st.exception(e)
    except Exception as e:
        st.error(f"GitHub Error: {str(e)}")
        st.exception(e)

def update_requirements(template):
    template_requirements = {
        "Basic": ["streamlit"],
        "Data Analysis": ["streamlit", "pandas", "plotly", "numpy"],
        "Machine Learning": ["streamlit", "pandas", "scikit-learn", "numpy"]
    }
    
    try:
        current_reqs = set()
        if os.path.exists('requirements.txt'):
            with open('requirements.txt', 'r') as f:
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
        
        with open('requirements.txt', 'w') as f:
            f.write('\n'.join(sorted(new_reqs)))
            
        # Install new requirements
        import subprocess
        subprocess.check_call(["pip", "install", "-r", "requirements.txt"])
            
    except Exception as e:
        st.error(f"Error updating requirements: {str(e)}")

def get_app_metadata(file_path):
    """Get metadata for an app file"""
    try:
        stats = os.stat(file_path)
        return {
            "size": f"{stats.st_size/1024:.1f} KB",
            "modified": datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M'),
            "lines": sum(1 for _ in open(file_path)),
            "has_main": "main()" in open(file_path).read()
        }
    except Exception as e:
        logger.error(f"Error getting metadata: {str(e)}")
        return {}

def show_breadcrumbs(selected, current_app=None):
    """Display navigation breadcrumbs"""
    crumbs = ["🏠 Home"]
    if selected != " Home":
        crumbs.append(selected)
    if current_app:
        crumbs.append(f"📱 {current_app}")
    
    st.markdown(" > ".join(crumbs))

def show_app_preview(app_path, with_edit=False):
    try:
        # Get content from session state first
        content = st.session_state.app_code_storage.get(app_path, '')
        
        # Fallback to file system if needed
        if not content and not is_streamlit_cloud():
            try:
                with open(app_path, 'r') as f:
                    content = f.read()
            except Exception as e:
                logger.warning(f"File read failed: {e}")
        
        if with_edit:
            col1, col2 = st.columns([3, 1])
            with col1:
                edited_code = st_ace(
                    value=content,
                    language="python",
                    theme="monokai",
                    height=400,
                    key=f"editor_{app_path}"  # Consistent key for state preservation
                )
            
            with col2:
                if st.button("Save Changes"):
                    save_app_code(app_path, edited_code)
                    st.success("Changes saved!")
                    time.sleep(1)
                    st.experimental_rerun()
        else:
            st.code(content, language="python")
            
    except Exception as e:
        st.error(f"Error in preview: {str(e)}")

# Add these new functions after the existing imports

def search_apps(apps, query):
    """Search apps by name or content"""
    if not query:
        return apps
    
    results = {}
    for name, path in apps.items():
        try:
            with open(path, 'r') as f:
                content = f.read().lower()
            if query.lower() in name.lower() or query.lower() in content:
                results[name] = path
        except Exception as e:
            logger.error(f"Error searching {name}: {str(e)}")
    return results

def clone_app(original_path, new_name):
    """Clone an existing app"""
    try:
        new_file_name = f"{new_name.lower().replace(' ', '_')}.py"
        new_path = os.path.join(os.getcwd(), new_file_name)
        if os.path.exists(new_path):
            return False, "App already exists"
        
        with open(original_path, 'r') as src, open(new_path, 'w') as dst:
            dst.write(src.read())
        return True, f"Created {new_file_name}"
    except Exception as e:
        return False, str(e)

def validate_requirements(requirements_text):
    """Validate requirements format and availability"""
    try:
        requirements = [r.strip() for r in requirements_text.split('\n') if r.strip()]
        invalid = []
        for req in requirements:
            try:
                package_name = req.split('==')[0].split('>=')[0].split('<=')[0].strip()
                # Simple import check instead of distribution check
                __import__(package_name)
            except ImportError:
                invalid.append(req)
        return not invalid, invalid
    except Exception as e:
        return False, str(e)

def show_diff(original, modified):
    """Show differences between two versions of code"""
    # Simplified diff view using basic comparison
    st.code("\n".join(
        f"- {line}" if line in original.splitlines() and line not in modified.splitlines()
        else f"+ {line}" if line in modified.splitlines() and line not in original.splitlines()
        else f"  {line}"
        for line in set(original.splitlines() + modified.splitlines())
    ))

def git_workflow(repo, branch='main'):
    """Enhanced GitHub workflow with branch support and history"""
    branches = [b.name for b in repo.get_branches()]
    selected_branch = st.selectbox("Select Branch", branches, index=branches.index('main') if 'main' in branches else 0)
    
    # Pull latest changes
    if st.button("Pull Latest Changes"):
        try:
            contents = repo.get_contents("")
            for content in contents:
                if content.path.endswith('.py'):
                    file_content = content.decoded_content.decode()
                    with open(content.path, 'w') as f:
                        f.write(file_content)
            st.success("Successfully pulled latest changes")
        except Exception as e:
            st.error(f"Error pulling changes: {str(e)}")

    # Show commit history
    with st.expander("Commit History"):
        commits = repo.get_commits()
        for commit in list(commits)[:5]:
            st.markdown(f"**{commit.commit.message}**")
            st.markdown(f"Author: {commit.commit.author.name}")
            st.markdown(f"Date: {commit.commit.author.date}")
            st.markdown("---")

def handle_template_customization():
    """Handle template customization and preview"""
    st.subheader("Template Customization")
    
    template = st.session_state.get('current_template', '')  # Changed from {} to ''
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
        st.session_state.current_template = modified_template  # Save changes
    
    with col2:
        st.markdown("### Preview")
        with st.expander("Template Preview", expanded=True):
            st.code(modified_template, language="python")

def version_control():
    """Basic version control for app editing"""
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

def save_settings(settings):
    """Save settings to session state and environment"""
    for key, value in settings.items():
        st.session_state[key] = value
        # Update environment variables
        if key in ['GITHUB_USERNAME', 'GITHUB_REPO', 'GITHUB_TOKEN']:
            os.environ[key] = value

def settings_page():
    """Settings configuration page"""
    st.title("Settings")
    
    # Default settings without hardcoded token
    default_settings = {
        'GITHUB_USERNAME': os.getenv('GITHUB_USERNAME', ''),
        'GITHUB_REPO': os.getenv('GITHUB_REPO', ''),
        'GITHUB_TOKEN': os.getenv('GITHUB_TOKEN', ''),
        'auto_save': True,
        'show_previews': True,
        'dark_mode': False
    }
    
    # Load current settings from session state or defaults
    current_settings = {
        key: st.session_state.get(key, default_settings[key])
        for key in default_settings
    }

    tab1, tab2 = st.tabs(["General", "GitHub"])
    
    with tab1:
        st.markdown("### General Settings")
        current_settings['auto_save'] = st.checkbox(
            "Auto Save", 
            value=current_settings['auto_save']
        )
        current_settings['show_previews'] = st.checkbox(
            "Show Code Previews", 
            value=current_settings['show_previews']
        )
        current_settings['dark_mode'] = st.checkbox(
            "Dark Mode", 
            value=current_settings['dark_mode']
        )
        
    with tab2:
        st.markdown("### GitHub Settings")
        
        # GitHub configuration with help text
        st.info("These settings are required for GitHub integration. Values will be saved as environment variables.")
        
        current_settings['GITHUB_USERNAME'] = st.text_input(
            "GitHub Username",
            value=current_settings['GITHUB_USERNAME'],
            help="Your GitHub username"
        )
        current_settings['GITHUB_REPO'] = st.text_input(
            "GitHub Repository",
            value=current_settings['GITHUB_REPO'],
            help="The name of your GitHub repository"
        )
        current_settings['GITHUB_TOKEN'] = st.text_input(
            "GitHub Token",
            value="",  # Do not display the token
            type="password",
            help="Your GitHub Personal Access Token"
        )
        
        # Add link to GitHub token creation
        st.markdown("""
        [Create a new GitHub token here](https://github.com/settings/tokens/new)
        > Required scopes: `repo`, `workflow`
        """)
        
        # Test GitHub connection
        if st.button("Test GitHub Connection"):
            try:
                g = Github(current_settings['GITHUB_TOKEN'])
                repo = g.get_repo(f"{current_settings['GITHUB_USERNAME']}/{current_settings['GITHUB_REPO']}")
                st.success(f"Successfully connected to {repo.full_name}")
            except Exception as e:
                st.error(f"GitHub connection failed: {str(e)}")
                st.exception(e)
    
    # Save button for all settings
    if st.button("Save Settings", type="primary"):
        save_settings(current_settings)
        st.success("Settings saved successfully!")
        time.sleep(1)
        st.experimental_rerun()

    # Show current configuration
    with st.expander("Current Configuration"):
        safe_settings = current_settings.copy()
        safe_settings['GITHUB_TOKEN'] = '****' if safe_settings['GITHUB_TOKEN'] else 'Not set'
        st.json(safe_settings)

def get_github_config():
    """Get GitHub configuration from environment variables or session state"""
    return {
        'username': st.session_state.get('GITHUB_USERNAME', os.getenv('GITHUB_USERNAME', '')),
        'repo': st.session_state.get('GITHUB_REPO', os.getenv('GITHUB_REPO', '')),
        'token': st.session_state.get('GITHUB_TOKEN', os.getenv('GITHUB_TOKEN', ''))
    }

def get_app_url(app_path):
    config = get_github_config()
    return f"https://share.streamlit.io/{config['username']}/{config['repo']}/main/{app_path}"

def ensure_dependencies():
    """Ensure all required packages are installed"""
    required_packages = [
        "PyGithub",  # This is the correct package name
        "streamlit-ace",
        "streamlit-option-menu",
        "python-dotenv"
    ]
    
    try:
        import subprocess
        for package in required_packages:
            try:
                if package == "PyGithub":
                    from PyGithub import Github  # Special case for PyGithub
                else:
                    __import__(package.replace('-', '_').lower())
            except ImportError:
                with st.spinner(f"Installing {package}..."):
                    subprocess.check_call(["pip", "install", package])
                st.success(f"Installed {package}")
        return True
    except Exception as e:
        st.error(f"Error installing dependencies: {str(e)}")
        return False

# Add error handling wrapper for session state access
def get_session_state(key, default=None):
    try:
        return st.session_state[key]
    except KeyError:
        if default is not None:
            st.session_state[key] = default
        return default

# Add type validation for data transformations
def validate_app_data(app_data):
    required_fields = {'name': str, 'content': str, 'metadata': dict}
    for field, field_type in required_fields.items():
        if not isinstance(app_data.get(field), field_type):
            raise TypeError(f"{field} must be of type {field_type}")
    return app_data

# Add proper API error handling wrapper
def api_call_wrapper(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Github.GithubException as e:
            st.error(f"GitHub API Error: {e.data.get('message', str(e))}")
        except Exception as e:
            st.error(f"API Error: {str(e)}")
        return None
    return wrapper

# Add callback return type validation
def validate_callback(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if result is not None and not isinstance(result, (bool, str, dict)):
            logger.warning(f"Callback {func.__name__} returned unexpected type: {type(result)}")
        return result
    return wrapper

# Add missing cleanup function referenced in main()
def cleanup_app_state():
    """Cleanup session state when app terminates"""
    try:
        # Add comprehensive cleanup
        temp_keys = [k for k in st.session_state.keys() if k.startswith('temp_')]
        for k in temp_keys:
            del st.session_state[k]
        
        # Clean temp files
        for f in glob.glob("*.tmp"):
            try:
                os.remove(f)
            except OSError:
                logger.warning(f"Failed to remove temp file: {f}")
        
        # Clear any temporary files
        temp_files = [f for f in os.listdir() if f.endswith('.tmp')]
        for f in temp_files:
            os.remove(f)
        
        # Clear sensitive data
        sensitive_keys = ['gh_token', 'current_code']
        for key in sensitive_keys:
            if key in st.session_state:
                del st.session_state[key]
                
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")

# Add missing error handling for module imports
def safe_import(module_name):
    """Safely import a module with error handling"""
    try:
        return __import__(module_name)
    except ImportError as e:
        logger.error(f"Failed to import {module_name}: {str(e)}")
        return None

# Add type validation for app data transformations
def transform_app_data(app_data):
    """Transform and validate app data"""
    if not isinstance(app_data, dict):
        raise TypeError("App data must be a dictionary")
        
    required = ['name', 'content', 'metadata']
    if not all(k in app_data for k in required):
        raise ValueError(f"Missing required fields: {required}")
        
    # Ensure consistent types
    app_data['name'] = str(app_data['name'])
    app_data['content'] = str(app_data['content'])
    if not isinstance(app_data['metadata'], dict):
        app_data['metadata'] = {}
        
    return app_data

# Add session state initialization
def initialize_session_state():
    """Initialize all required session state variables"""
    defaults = {
        'current_app': None,
        'app_code_storage': {},
        'version_history': [],
        'current_code': '',
        'gh_token': None,
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
    
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

# Add API response validation
def validate_github_response(response):
    """Validate GitHub API responses"""
    if not response:
        raise ValueError("Empty response from GitHub API")
    
    if isinstance(response, dict):
        if 'message' in response:
            raise ValueError(f"GitHub API error: {response['message']}")
    return response

# Add missing save_app_code function that's called but not defined
def save_app_code(file_name: str, content: str) -> bool:
    """Save app code to both session state and file system"""
    try:
        # Save to session state
        if 'app_code_storage' not in st.session_state:
            st.session_state.app_code_storage = {}
        st.session_state.app_code_storage[file_name] = content
        
        # Save to file system if not on Streamlit Cloud
        if not is_streamlit_cloud():
            lock_path = f"{file_name}.lock"
            with FileLock(lock_path):
                with open(file_name, 'w') as f:
                    f.write(content)
        return True
    except Exception as e:
        logger.error(f"Error saving app code: {str(e)}")
        return False

# Add missing main() function that's referenced but not defined
def main():
    """Main application entry point with comprehensive error handling"""
    if not initialize_app():
        st.error("Application failed to initialize. Please check the logs.")
        return

    try:
        # Navigation menu
        with st.sidebar:
            selected = option_menu(
                "AI Autocoder Hub",
                ["🏠 Home", "📱 Apps", "⚙️ Settings", "🔄 Sync", "📦 Dependencies"],
                icons=['house', 'app', 'gear', 'cloud-upload', 'box'],
                menu_icon="cast",
                default_index=0,
            )
        
        # Handle current app if running
        if st.session_state.get('current_app'):
            module = load_module(st.session_state.current_app)
            if module:
                with st.spinner("Running app..."):
                    run_app_safely(module, st.session_state.current_app)
                return
        
        # Main navigation logic with error handling
        navigation_map = {
            "🏠 Home": create_new_app,
            "📱 Apps": show_apps_page,
            "⚙️ Settings": settings_page,
            "🔄 Sync": github_integration,
            "📦 Dependencies": manage_dependencies
        }
        
        if selected in navigation_map:
            with st.spinner(f"Loading {selected}..."):
                navigation_map[selected]()
                
    except Exception as e:
        logger.error(f"Main execution error: {str(e)}")
        st.error("An unexpected error occurred. Please try again or contact support.")
        cleanup_app_state()

# Add missing show_apps_page function
def show_apps_page():
    """Apps management page with search and actions"""
    st.title("Manage Apps")
    
    # Search and filter
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("🔍 Search Apps", st.session_state.get('search_query', ''))
    with col2:
        sort_by = st.selectbox("Sort by", ["Name", "Last Modified", "Size"])
    
    # Get and filter apps
    apps = get_available_apps(search_query, sort_by)
    
    if not apps:
        st.info("No apps found. Create a new app to get started!")
        return
    
    # Display apps in grid
    for idx in range(0, len(apps), 3):
        cols = st.columns(3)
        for col_idx, app in enumerate(apps[idx:min(idx+3, len(apps))]):
            with cols[col_idx]:
                display_app_card(app)

def display_app_card(app):
    """Display individual app card with actions"""
    st.markdown(f"### {app['name']}")
    metadata = get_app_metadata(app['path'])
    
    st.markdown(f"""
        - Last modified: {metadata.get('modified', 'N/A')}
        - Size: {metadata.get('size', 'N/A')}
        - Lines: {metadata.get('lines', 'N/A')}
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("▶️ Run", key=f"run_{app['name']}"):
            st.session_state.current_app = app['path']
            st.experimental_rerun()
    with col2:
        if st.button("✏️ Edit", key=f"edit_{app['name']}"):
            show_app_preview(app['path'], with_edit=True)
    with col3:
        if st.button("🗑️ Delete", key=f"delete_{app['name']}"):
            delete_app(app['path'])
            st.experimental_rerun()

def sync_page():
    """GitHub synchronization page"""
    st.title("Sync with GitHub")
    
    if not validate_github_settings():
        st.warning("Please configure GitHub settings first")
        return
    
    # Get repository status
    repo_status = get_repo_status()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Local Changes")
        local_changes = repo_status['local_changes']
        if local_changes:
            for file, status in local_changes.items():
                st.checkbox(f"{file} ({status})", key=f"sync_{file}")
        else:
            st.info("No local changes detected")
    
    with col2:
        st.markdown("### Remote Changes")
        remote_changes = repo_status['remote_changes']
        if remote_changes:
            for file, status in remote_changes.items():
                st.checkbox(f"{file} ({status})", key=f"pull_{file}")
        else:
            st.info("No remote changes detected")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Push Selected"):
            push_selected_changes()
    with col2:
        if st.button("Pull Selected"):
            pull_selected_changes()

def dependencies_page():
    """Package management page"""
    st.title("Manage Dependencies")
    
    # Core dependencies that can't be removed
    core_deps = {
        "streamlit",
        "streamlit-ace",
        "streamlit-option-menu",
        "PyGithub",
        "python-dotenv"
    }
    
    col1, col2 = st.columns([2,1])
    
    with col1:
        st.markdown("### Current Dependencies")
        current_reqs = load_requirements()
        new_reqs = st_ace(
            value=current_reqs,
            language="text",
            theme="monokai",
            height=300
        )
    
    with col2:
        st.markdown("### Quick Add")
        common_packages = ["pandas", "numpy", "scikit-learn", "plotly", "matplotlib"]
        for package in common_packages:
            if st.button(f"Add {package}"):
                add_package(package, new_reqs)
        
        st.markdown("### Actions")
        if st.button("Save Changes"):
            save_requirements(new_reqs, core_deps)
        if st.button("Install All"):
            install_requirements()

def save_github_token(token: str):
    # Use environment variable or secure storage service
    key = Fernet.generate_key()
    f = Fernet(key)
    encrypted_token = f.encrypt(token.encode())
    st.session_state['gh_token_encrypted'] = encrypted_token

def validate_app_name(name: str) -> bool:
    pattern = r'^[a-zA-Z0-9_-]+$'
    return bool(re.match(pattern, name))

def get_available_apps(search_query: str = "", sort_by: str = "Name") -> List[dict]:
    """Get list of available apps with optional filtering and sorting"""
    apps = []
    try:
        # Get all Python files in directory
        for file in os.listdir():
            if file.endswith('.py') and file != 'streamlit_app.py':
                app_info = {
                    'name': file.replace('.py', '').replace('_', ' ').title(),
                    'path': file,
                    'metadata': get_app_metadata(file)
                }
                apps.append(app_info)
        
        # Apply search filter
        if search_query:
            apps = [app for app in apps if search_query.lower() in app['name'].lower()]
        
        # Apply sorting
        if sort_by == "Name":
            apps.sort(key=lambda x: x['name'])
        elif sort_by == "Last Modified":
            apps.sort(key=lambda x: x['metadata'].get('modified', ''), reverse=True)
        elif sort_by == "Size":
            apps.sort(key=lambda x: float(x['metadata'].get('size', '0').split()[0]), reverse=True)
            
        return apps
    except Exception as e:
        logger.error(f"Error getting apps: {str(e)}")
        return []

def delete_app(file_path: str) -> bool:
    """Delete an app and clean up associated resources"""
    try:
        # Remove from session state
        if file_path in st.session_state.app_code_storage:
            del st.session_state.app_code_storage[file_path]
        
        # Remove from file system
        if os.path.exists(file_path):
            os.remove(file_path)
            
        # Clean up any associated temporary files
        temp_path = f"{file_path}.tmp"
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return True
    except Exception as e:
        logger.error(f"Error deleting app {file_path}: {str(e)}")
        return False

def validate_github_settings() -> bool:
    """Validate GitHub configuration settings"""
    github_settings = st.session_state.settings.get('github', {})
    required_fields = ['username', 'repo', 'token']
    
    # Check all required fields are present and non-empty
    if not all(github_settings.get(field) for field in required_fields):
        return False
        
    try:
        # Test GitHub connection
        g = Github(github_settings['token'])
        repo = g.get_repo(f"{github_settings['username']}/{github_settings['repo']}")
        # Try to access repo (will raise exception if invalid)
        repo.get_contents("")
        return True
    except Exception as e:
        logger.error(f"GitHub validation error: {str(e)}")
        return False

def get_repo_status() -> dict:
    """Get status of local and remote repository changes"""
    status = {
        'local_changes': {},
        'remote_changes': {}
    }
    
    try:
        github_settings = st.session_state.settings['github']
        g = Github(github_settings['token'])
        repo = g.get_repo(f"{github_settings['username']}/{github_settings['repo']}")
        
        # Get remote files
        remote_files = {
            content.path: content.sha 
            for content in repo.get_contents("")
            if content.path.endswith('.py')
        }
        
        # Compare local files
        for file in os.listdir():
            if not file.endswith('.py'):
                continue
                
            with open(file, 'r') as f:
                content = f.read()
                
            if file in remote_files:
                # File exists in remote
                remote_content = repo.get_contents(file).decoded_content.decode()
                if content != remote_content:
                    status['local_changes'][file] = 'modified'
            else:
                status['local_changes'][file] = 'new'
                
        # Check for remote files not present locally
        for remote_file in remote_files:
            if not os.path.exists(remote_file):
                status['remote_changes'][remote_file] = 'deleted_locally'
                
        return status
    except Exception as e:
        logger.error(f"Error getting repo status: {str(e)}")
        return status

def push_selected_changes():
    """Push selected local changes to GitHub"""
    try:
        # Get selected files from session state
        selected = [
            key.replace('sync_', '') 
            for key in st.session_state.keys() 
            if key.startswith('sync_') and st.session_state[key]
        ]
        
        if not selected:
            st.warning("No files selected for push")
            return
            
        github_manager = GitHubManager()
        commit_msg = st.session_state.get('commit_message', 'Update from Streamlit Hub')
        
        if github_manager.sync_files(selected, commit_msg):
            st.success("Successfully pushed changes to GitHub")
        else:
            st.error("Failed to push changes")
    except Exception as e:
        st.error(f"Error pushing changes: {str(e)}")

def pull_selected_changes():
    """Pull selected remote changes from GitHub"""
    try:
        # Get selected files from session state
        selected = [
            key.replace('pull_', '') 
            for key in st.session_state.keys() 
            if key.startswith('pull_') and st.session_state[key]
        ]
        
        if not selected:
            st.warning("No files selected for pull")
            return
            
        github_settings = st.session_state.settings['github']
        g = Github(github_settings['token'])
        repo = g.get_repo(f"{github_settings['username']}/{github_settings['repo']}")
        
        for file in selected:
            content = repo.get_contents(file)
            with open(file, 'w') as f:
                f.write(content.decoded_content.decode())
                
        st.success("Successfully pulled changes from GitHub")
    except Exception as e:
        st.error(f"Error pulling changes: {str(e)}")

def load_requirements() -> str:
    """Load current requirements from requirements.txt"""
    try:
        if os.path.exists('requirements.txt'):
            with open('requirements.txt', 'r') as f:
                return f.read()
        return ""
    except Exception as e:
        logger.error(f"Error loading requirements: {str(e)}")
        return ""

def add_package(package: str, current_reqs: str) -> str:
    """Add a package to requirements if not already present"""
    try:
        requirements = set(current_reqs.split('\n'))
        if package not in requirements:
            requirements.add(package)
        return '\n'.join(sorted(requirements))
    except Exception as e:
        logger.error(f"Error adding package: {str(e)}")
        return current_reqs

# Fix initialization order and add error handling
def initialize_app():
    """Initialize application with proper error handling"""
    try:
        setup_logging()
        load_dotenv(override=True)
        initialize_session_state()
        return True
    except Exception as e:
        st.error(f"Failed to initialize app: {str(e)}")
        logger.error(f"Initialization error: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        if ensure_dependencies():
            main()
        else:
            st.error("Failed to install required dependencies. Please check the logs.")
    except Exception as e:
        st.error(f"Critical error: {str(e)}")
        logger.critical(f"Application crash: {str(e)}", exc_info=True)


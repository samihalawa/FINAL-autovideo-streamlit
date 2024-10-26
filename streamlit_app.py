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
import json
from typing import Optional, Dict, Any
import tempfile
import shutil

st.set_page_config(page_title="AI Autocoder Hub", layout="wide")

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
def load_module(module_name):
    try:
        if module_name in sys.modules:
            del sys.modules[module_name]
        return importlib.import_module(module_name)
    except Exception as e:
        logger.error(f"Error loading module {module_name}: {str(e)}")
        return None

@st.cache_data
def load_apps_from_directory():
    apps = {}
    for file in os.listdir():
        if file.endswith('.py') and file != 'streamlit_app.py':
            app_name = os.path.splitext(file)[0].replace('_', ' ').title()
            apps[app_name] = file
    return apps

def init_session_state():
    if 'app_states' not in st.session_state:
        st.session_state.app_states = {}
    if 'error_logs' not in st.session_state:
        st.session_state.error_logs = {}
    if 'last_successful_state' not in st.session_state:
        st.session_state.last_successful_state = None

def run_app_safely(module, app_name: str) -> None:
    try:
        # Save current state before running
        st.session_state.last_successful_state = st.session_state.app_states.get(app_name, {}).copy()
        
        with capture_streamlit_error() as captured:
            if hasattr(module, 'main'):
                module.main()
                # Update successful state
                st.session_state.app_states[app_name] = st.session_state.copy()
            else:
                raise AttributeError(f"No main() function found in {app_name}")
        
        error_output = captured.getvalue()
        if error_output:
            st.session_state.error_logs[app_name] = error_output
            with st.expander("Show App Errors/Logs"):
                st.code(error_output)
                
    except Exception as e:
        st.error(f"Error running {app_name}: {str(e)}")
        # Restore last successful state
        if st.session_state.last_successful_state:
            st.session_state.app_states[app_name] = st.session_state.last_successful_state
        with st.expander("Show Error Details"):
            st.exception(e)

def create_new_app():
    st.subheader("Create New App")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Create New")
        create_method = st.radio("Creation Method", ["Template", "Import from GitHub"])
    
    if create_method == "Template":
        templates = {
            "Basic": """import streamlit as st\n\ndef main():\n    st.title("New App")\n\nif __name__ == "__main__":\n    main()""",
            "Data Analysis": """import streamlit as st\nimport pandas as pd\nimport plotly.express as px\n\ndef main():\n    st.title("Data Analysis App")\n\nif __name__ == "__main__":\n    main()""",
            "Machine Learning": """import streamlit as st\nimport pandas as pd\nfrom sklearn.model_selection import train_test_split\n\ndef main():\n    st.title("ML App")\n\nif __name__ == "__main__":\n    main()"""
        }
        
        app_name = st.text_input("App Name")
        template = st.selectbox("Template", list(templates.keys()))
        
        st.session_state.current_template = templates[template]
        handle_template_customization()
        
        if st.button("Create App"):
            if not app_name:
                st.error("Please enter an app name")
                return None
                
            file_name = f"{app_name.lower().replace(' ', '_')}.py"
            if os.path.exists(file_name):
                st.error(f"App {file_name} already exists")
                return None
                
            try:
                with open(file_name, 'w') as f:
                    f.write(st.session_state.current_template)
                st.success(f"Created {file_name} successfully!")
                # Save app metadata
                metadata = {
                    "created": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "template": template,
                    "name": app_name,
                    "version": "1.0.0"
                }
                update_app_metadata(file_name, metadata)
                save_app_state(app_name, {"template": template})
                
                st.cache_data.clear()
                time.sleep(0.5)
                st.rerun()
                return file_name
            except Exception as e:
                st.error(f"Error creating app: {str(e)}")
                return None
    
    else:  # Import from GitHub
        if 'gh_token' not in st.session_state:
            st.error("Please set GitHub token in settings first")
            return None
            
        try:
            g = Github(st.session_state.gh_token)
            repo_url = st.text_input("GitHub Repository URL")
            if repo_url and st.button("Import"):
                repo_name = repo_url.split('/')[-2:]
                repo = g.get_repo('/'.join(repo_name))
                contents = repo.get_contents("")
                for content in contents:
                    if content.path.endswith('.py'):
                        with open(content.path, 'w') as f:
                            f.write(content.decoded_content.decode())
                st.success("Successfully imported files")
                st.rerun()
        except Exception as e:
            st.error(f"Error importing from GitHub: {str(e)}")
            return None

def manage_dependencies():
    st.subheader("Dependency Management")
    
    try:
        with open('requirements.txt', 'r') as f:
            current_reqs = f.read()
    except FileNotFoundError:
        current_reqs = ""
        
    col1, col2 = st.columns([2, 1])
    
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
            try:
                with open('requirements.txt', 'w') as f:
                    f.write(new_reqs)
                st.success("Requirements updated!")
            except Exception as e:
                st.error(f"Error saving requirements: {str(e)}")

def github_integration():
    st.subheader("GitHub Integration")
    
    token = st.session_state.get('gh_token')
    if not token:
        st.warning("Please enter GitHub token in settings first")
        return
        
    try:
        g = Github(token)
        user = g.get_user()
        repos = [repo.full_name for repo in user.get_repos()]
        
        col1, col2 = st.columns(2)
        with col1:
            repo_name = st.selectbox("Select Repository", repos)
            if repo_name:
                repo = g.get_repo(repo_name)
                git_workflow(repo)
        
        with col2:
            st.markdown("### Files to Sync")
            files_to_sync = st.multiselect(
                "Select Files",
                [f for f in os.listdir() if f.endswith('.py')]
            )
            
            commit_msg = st.text_input("Commit Message", "Update from Streamlit Hub")
            
            if st.button("Sync with GitHub"):
                try:
                    st.info("Pulling latest changes...")
                    git_workflow(repo)
                    
                    for file in files_to_sync:
                        try:
                            with open(file, 'r') as f:
                                content = f.read()
                            try:
                                contents = repo.get_contents(file)
                                repo.update_file(
                                    contents.path,
                                    commit_msg,
                                    content,
                                    contents.sha
                                )
                            except:
                                repo.create_file(
                                    file,
                                    commit_msg,
                                    content
                                )
                            st.success(f"Synced {file}")
                        except Exception as e:
                            st.error(f"Error syncing {file}: {str(e)}")
                except Exception as e:
                    st.error(f"Sync error: {str(e)}")
                    
    except Exception as e:
        st.error(f"GitHub Error: {str(e)}")

def update_requirements(template):
    template_requirements = {
        "Basic": ["streamlit"],
        "Data Analysis": ["streamlit", "pandas", "plotly"],
        "Machine Learning": ["streamlit", "pandas", "scikit-learn"]
    }
    
    try:
        current_reqs = set()
        if os.path.exists('requirements.txt'):
            with open('requirements.txt', 'r') as f:
                current_reqs = set(line.strip() for line in f.readlines())
        
        new_reqs = current_reqs.union(template_requirements.get(template, []))
        
        with open('requirements.txt', 'w') as f:
            f.write('\n'.join(sorted(new_reqs)))
            
    except Exception as e:
        st.error(f"Error updating requirements: {str(e)}")

def get_app_metadata(file_path):
    try:
        stats = os.stat(file_path)
        with open(file_path, 'r') as f:
            content = f.read()
        return {
            "size": f"{stats.st_size/1024:.1f} KB",
            "modified": datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M'),
            "lines": sum(1 for _ in open(file_path)),
            "has_main": "def main():" in content
        }
    except Exception as e:
        logger.error(f"Error getting metadata: {str(e)}")
        return {}

def show_breadcrumbs(selected, current_app=None):
    crumbs = ["🏠 Home"]
    if selected != "🏠 Home":
        crumbs.append(selected)
    if current_app:
        crumbs.append(f"📱 {current_app}")
    
    st.markdown(" > ".join(crumbs))

def show_app_preview(app_path):
    try:
        with open(app_path, 'r') as f:
            content = f.read()
        with st.expander("Preview Code"):
            st_ace(value=content, language="python", theme="monokai", readonly=True, height=200)
    except Exception as e:
        st.error(f"Error loading preview: {str(e)}")

def search_apps(apps, query):
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
    try:
        new_path = f"{new_name.lower().replace(' ', '_')}.py"
        if os.path.exists(new_path):
            return False, "App already exists"
        
        with open(original_path, 'r') as src, open(new_path, 'w') as dst:
            dst.write(src.read())
        return True, f"Created {new_path}"
    except Exception as e:
        return False, str(e)

def validate_requirements(requirements_text):
    try:
        requirements = [r.strip() for r in requirements_text.split('\n') if r.strip()]
        invalid = []
        for req in requirements:
            try:
                next(parse_requirements(req))
            except:
                invalid.append(req)
        return not invalid, invalid
    except Exception as e:
        return False, str(e)

def show_diff(original, modified):
    import difflib
    d = difflib.HtmlDiff()
    diff_html = d.make_file(original.splitlines(), modified.splitlines())
    st.markdown(diff_html, unsafe_allow_html=True)

def git_workflow(repo, branch='main'):
    branches = [b.name for b in repo.get_branches()]
    selected_branch = st.selectbox("Select Branch", branches, index=branches.index('main') if 'main' in branches else 0)
    
    if st.button("Pull Latest Changes"):
        try:
            contents = repo.get_contents("", ref=selected_branch)
            for content in contents:
                if content.path.endswith('.py'):
                    file_content = content.decoded_content.decode()
                    with open(content.path, 'w') as f:
                        f.write(file_content)
            st.success("Successfully pulled latest changes")
        except Exception as e:
            st.error(f"Error pulling changes: {str(e)}")

    with st.expander("Commit History"):
        commits = repo.get_commits(sha=selected_branch)
        for commit in list(commits)[:5]:
            st.markdown(f"**{commit.commit.message}**")
            st.markdown(f"Author: {commit.commit.author.name}")
            st.markdown(f"Date: {commit.commit.author.date}")
            st.markdown("---")

def handle_template_customization():
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

def version_control():
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

class ResourceManager:
    def __init__(self):
        self.active_resources: Dict[str, Any] = {}
        self.cleanup_queue = []

    def register(self, resource_id: str, resource: Any, cleanup_func=None):
        self.active_resources[resource_id] = resource
        if cleanup_func:
            self.cleanup_queue.append((resource_id, cleanup_func))

    def cleanup(self):
        for resource_id, cleanup_func in self.cleanup_queue:
            try:
                cleanup_func(self.active_resources.get(resource_id))
            except Exception as e:
                logger.error(f"Cleanup error for {resource_id}: {e}")
        self.cleanup_queue.clear()
        self.active_resources.clear()

class AppProcess:
    def __init__(self):
        self.processes = {}
        self.ports = {}
        self.resource_manager = ResourceManager()

    def start(self, app_path: str) -> int:
        if app_path in self.processes:
            self.stop(app_path)
        
        # Create a temporary directory for the app
        temp_dir = tempfile.mkdtemp()
        temp_app_path = os.path.join(temp_dir, os.path.basename(app_path))
        shutil.copy2(app_path, temp_app_path)
        
        # Run the app using streamlit run
        port = self._get_free_port()
        cmd = f"streamlit run {temp_app_path} --server.port={port}"
        
        # Use Popen to run the command
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        self.processes[app_path] = process
        self.ports[app_path] = port
        self.resource_manager.register(app_path, temp_dir, cleanup_func=shutil.rmtree)
        
        return port

    def stop(self, app_path):
        if app_path in self.processes:
            self.processes[app_path].terminate()
            del self.processes[app_path]
            del self.ports[app_path]
            self.resource_manager.cleanup()

    def _get_free_port(self):
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

def update_app_metadata(file_name, metadata):
    try:
        with open('streamlit_apps.json', 'r+') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
            data[file_name] = metadata
            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=4)
    except FileNotFoundError:
        with open('streamlit_apps.json', 'w') as f:
            json.dump({file_name: metadata}, f, indent=4)

def load_app_state(app_name: str) -> Optional[Dict]:
    try:
        with open('session_state.json', 'r') as f:
            states = json.load(f)
            return states.get(app_name)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def save_app_state(app_name: str, state: Dict) -> None:
    try:
        states = {}
        if os.path.exists('session_state.json'):
            with open('session_state.json', 'r') as f:
                states = json.load(f)
        states[app_name] = state
        with open('session_state.json', 'w') as f:
            json.dump(states, f)
    except Exception as e:
        logger.error(f"Failed to save app state: {e}")

def main():
    init_session_state()
    if 'current_app' not in st.session_state:
        st.session_state.current_app = None
    if 'app_manager' not in st.session_state:
        st.session_state.app_manager = AppProcess()

    with st.sidebar:
        selected = option_menu(
            "AI Autocoder Hub", 
            ["🏠 Home", "📱 Apps", "➕ Create", "📦 Dependencies", "🔄 Sync", "⚙️ Settings"],
            icons=['house', 'app', 'plus-circle', 'box', 'cloud-upload', 'gear'],
            menu_icon="code-slash",
            default_index=0,
        )
        
        st.markdown("---")
        
        if selected != "🏠 Home":
            apps = load_apps_from_directory()
            app_selected = st.selectbox(
                "Select App",
                list(apps.keys()),
                format_func=lambda x: f"📱 {x}"
            )
            if app_selected:
                st.session_state.current_app = app_selected
                
                app_path = apps[app_selected]
                metadata = get_app_metadata(app_path)
                st.markdown("### App Info")
                st.markdown(f"Size: {metadata.get('size', 'N/A')}")
                st.markdown(f"Modified: {metadata.get('modified', 'N/A')}")
                st.markdown(f"Lines: {metadata.get('lines', 'N/A')}")
                st.markdown(f"Has main(): {'✅' if metadata.get('has_main') else '❌'}")

        if st.button("🔄 Refresh"):
            with st.spinner("Refreshing..."):
                st.cache_data.clear()
                st.rerun()

    show_breadcrumbs(selected, st.session_state.current_app)

    if selected == "🏠 Home":
        st.title("Welcome to AI Autocoder Hub")
        apps = load_apps_from_directory()
        
        for app_name, app_path in apps.items():
            st.markdown("---")
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"### 📱 {app_name}")
                metadata = get_app_metadata(app_path)
                st.markdown(f"Last modified: {metadata.get('modified', 'N/A')} | Lines: {metadata.get('lines', 'N/A')}")
                
                with st.expander("Source Code"):
                    with open(app_path, 'r') as f:
                        code_content = f.read()
                    edited_code = st_ace(
                        value=code_content,
                        language="python",
                        theme="monokai",
                        height=300,
                        key=f"editor_{app_name}"
                    )
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Apply Changes", key=f"apply_{app_name}"):
                            try:
                                with open(app_path, 'w') as f:
                                    f.write(edited_code)
                                st.success("Changes saved!")
                                st.cache_data.clear()
                                st.cache_resource.clear()
                                time.sleep(0.5)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error saving: {str(e)}")
                    with col2:
                        if st.button("Revert", key=f"revert_{app_name}"):
                            st.rerun()
                    
            with col2:
                if st.button("👁️ Preview", key=f"preview_{app_name}"):
                    with st.expander("Code Preview", expanded=True):
                        st.code(code_content, language="python")
                    
            with col3:
                if st.button("▶️ Launch", key=f"launch_{app_name}"):
                    try:
                        port = st.session_state.app_manager.start(app_path)
                        time.sleep(1)  # Wait for server to start
                        app_url = f"http://localhost:{port}"
                        st.markdown(
                            f"""
                            <div style='text-align: center'>
                                <a href="{app_url}" target="_blank">
                                    <button style='padding: 8px 16px; background-color: #4CAF50; 
                                            color: white; border: none; border-radius: 4px;'>
                                        Open App in New Tab
                                    </button>
                                </a>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    except Exception as e:
                        st.error(f"Error launching app: {str(e)}")

    elif selected == "📱 Apps":
        if st.session_state.current_app:
            apps = load_apps_from_directory()
            app_path = apps.get(st.session_state.current_app)
            if app_path and os.path.exists(app_path):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if st.button("🔄 Restart App"):
                        st.cache_resource.clear()
                        st.rerun()
                with col2:
                    if st.button("📝 View Code"):
                        show_app_preview(app_path)
                with col3:
                    if st.button("📋 Clone App"):
                        new_name = st.text_input("New app name")
                        if new_name:
                            success, msg = clone_app(app_path, new_name)
                            if success:
                                st.success(msg)
                                update_app_metadata(new_name, get_app_metadata(app_path))
                            else:
                                st.error(msg)
                with col4:
                    if st.button("💾 Export"):
                        with open(app_path, 'r') as f:
                            content = f.read()
                        st.download_button(
                            "Download App",
                            content,
                            file_name=os.path.basename(app_path),
                            mime="text/plain"
                        )

                if st.checkbox("Enable Split View"):
                    col1, col2 = st.columns(2)
                    with col1:
                        with st.expander("Edit Code", expanded=True):
                            with open(app_path, 'r') as f:
                                original_code = f.read()
                            modified_code = st_ace(
                                value=original_code,
                                language="python",
                                theme="monokai",
                                height=400
                            )
                            if st.button("Save Changes"):
                                try:
                                    with open(app_path, 'w') as f:
                                        f.write(modified_code)
                                    st.success("Changes saved!")
                                    
                                    with st.expander("View Changes"):
                                        show_diff(original_code, modified_code)
                                except Exception as e:
                                    st.error(f"Error saving changes: {str(e)}")
                    
                    with col2:
                        st.markdown("### App Output")
                        try:
                            module = load_module(os.path.splitext(os.path.basename(app_path))[0])
                            if module:
                                run_app_safely(module, st.session_state.current_app)
                        except Exception as e:
                            st.error(f"Error loading app: {str(e)}")
                else:
                    try:
                        module = load_module(os.path.splitext(os.path.basename(app_path))[0])
                        if module:
                            run_app_safely(module, st.session_state.current_app)
                    except Exception as e:
                        st.error(f"Error loading app: {str(e)}")

    elif selected == "➕ Create":
        create_new_app()
        
    elif selected == "📦 Dependencies":
        manage_dependencies()
        if st.button("Validate Requirements"):
            try:
                with open('requirements.txt', 'r') as f:
                    reqs = f.read()
                valid, invalid = validate_requirements(reqs)
                if valid:
                    st.success("All requirements are valid!")
                else:
                    st.error("Invalid requirements found:")
                    st.write(invalid)
            except FileNotFoundError:
                st.warning("requirements.txt not found. Please create it first.")
        
    elif selected == "🔄 Sync":
        github_integration()
        
    elif selected == "⚙️ Settings":
        st.subheader("Settings")
        
        tab1, tab2 = st.tabs(["General", "GitHub"])
        
        with tab1:
            st.markdown("### General Settings")
            auto_save = st.checkbox("Auto Save", value=st.session_state.get('auto_save', True))
            show_previews = st.checkbox("Show Code Previews", value=st.session_state.get('show_previews', True))
            dark_mode = st.checkbox("Dark Mode", value=st.session_state.get('dark_mode', False))
            
            if st.button("Save General Settings"):
                st.session_state.auto_save = auto_save
                st.session_state.show_previews = show_previews
                st.session_state.dark_mode = dark_mode
                st.success("Settings saved!")
            
        with tab2:
            st.markdown("### GitHub Settings")
            default_token = os.getenv('GH_TOKEN') or ''
            gh_token = st.text_input(
                "GitHub Token", 
                type="password",
                value=default_token
            )
            if st.button("Save GitHub Token"):
                try:
                    # Validate token
                    g = Github(gh_token)
                    g.get_user().login
                    st.session_state['gh_token'] = gh_token
                    with open('.env', 'a') as f:
                        f.write(f"\nGH_TOKEN={gh_token}")
                    st.success("GitHub token saved and validated!")
                except Exception as e:
                    st.error(f"Invalid GitHub token: {str(e)}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Status")
    if st.session_state.current_app:
        st.sidebar.success(f"Running: {st.session_state.current_app}")
    else:
        st.sidebar.info("No app running")

if __name__ == "__main__":
    try:
        main()
    finally:
        if 'app_manager' in st.session_state:
            for app_path in list(st.session_state.app_manager.processes.keys()):
                st.session_state.app_manager.stop(app_path)


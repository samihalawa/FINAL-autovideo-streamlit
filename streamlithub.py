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
        if file.endswith('.py') and file != 'streamlithub.py':
            app_name = os.path.splitext(file)[0].replace('_', ' ').title()
            apps[app_name] = file
    return apps

def run_app_safely(module, app_name):
    st.markdown(f"### Running: {app_name}")
    
    if st.button("🏠 Back to Hub"):
        st.session_state.current_app = None
        st.experimental_rerun()
        return

    try:
        with capture_streamlit_error() as captured:
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

def create_new_app():
    st.subheader("Create New App")
    
    templates = {
        "Basic": """import streamlit as st\n\ndef main():\n    st.title("New App")\n\nif __name__ == "__main__":\n    main()""",
        "Data Analysis": """import streamlit as st\nimport pandas as pd\nimport plotly.express as px\n\ndef main():\n    st.title("Data Analysis App")\n\nif __name__ == "__main__":\n    main()""",
        "Machine Learning": """import streamlit as st\nimport pandas as pd\nfrom sklearn.model_selection import train_test_split\n\ndef main():\n    st.title("ML App")\n\nif __name__ == "__main__":\n    main()"""
    }
    
    col1, col2 = st.columns([2,1])
    with col1:
        app_name = st.text_input("App Name")
        template = st.selectbox("Template", list(templates.keys()))
        
    with col2:
        st.markdown("### Quick Settings")
        add_requirements = st.checkbox("Add to requirements.txt")
        create_test_file = st.checkbox("Create test file")
        
    code = st_ace(
        value=templates[template],
        language="python",
        theme="monokai",
        height=300
    )
    
    if st.button("Create App"):
        if not app_name:
            st.error("Please enter an app name")
            return
            
        file_name = f"{app_name.lower().replace(' ', '_')}.py"
        if os.path.exists(file_name):
            st.error(f"App {file_name} already exists")
            return
            
        try:
            with open(file_name, 'w') as f:
                f.write(code)
                
            if create_test_file:
                test_file = f"test_{file_name}"
                with open(test_file, 'w') as f:
                    f.write(f"""import unittest\nimport {app_name.lower().replace(' ', '_')}\n\nclass Test{app_name.replace(' ', '')}(unittest.TestCase):\n    def test_main(self):\n        pass""")
                    
            if add_requirements:
                update_requirements(template)
                
            st.success(f"Created {file_name} successfully!")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Error creating app: {str(e)}")

def manage_dependencies():
    st.subheader("Dependency Management")
    
    try:
        with open('requirements.txt', 'r') as f:
            current_reqs = f.read()
    except FileNotFoundError:
        current_reqs = ""
        
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
            with open('requirements.txt', 'w') as f:
                f.write(new_reqs)
            st.success("Requirements updated!")

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
        
        repo_name = st.selectbox("Select Repository", repos)
        files_to_sync = st.multiselect(
            "Files to Sync",
            [f for f in os.listdir() if f.endswith('.py')]
        )
        
        commit_msg = st.text_input("Commit Message", "Update from Streamlit Hub")
        
        if st.button("Sync with GitHub"):
            repo = g.get_repo(repo_name)
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
    if selected != "🏠 Home":
        crumbs.append(selected)
    if current_app:
        crumbs.append(f"📱 {current_app}")
    
    st.markdown(" > ".join(crumbs))

def show_app_preview(app_path):
    """Show app preview with syntax highlighting"""
    try:
        with open(app_path, 'r') as f:
            content = f.read()
        with st.expander("Preview Code"):
            st_ace(value=content, language="python", theme="monokai", readonly=True, height=200)
    except Exception as e:
        st.error(f"Error loading preview: {str(e)}")

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
        new_path = f"{new_name.lower().replace(' ', '_')}.py"
        if os.path.exists(new_path):
            return False, "App already exists"
        
        with open(original_path, 'r') as src, open(new_path, 'w') as dst:
            dst.write(src.read())
        return True, f"Created {new_path}"
    except Exception as e:
        return False, str(e)

def validate_requirements(requirements_text):
    """Validate requirements format and availability"""
    try:
        import pkg_resources
        requirements = [r.strip() for r in requirements_text.split('\n') if r.strip()]
        invalid = []
        for req in requirements:
            try:
                pkg_resources.require(req)
            except:
                invalid.append(req)
        return not invalid, invalid
    except Exception as e:
        return False, str(e)

def show_diff(original, modified):
    """Show differences between two versions of code"""
    import difflib
    d = difflib.HtmlDiff()
    diff_html = d.make_file(original.splitlines(), modified.splitlines())
    st.components.v1.html(diff_html, height=500, scrolling=True)

def main():
    if 'current_app' not in st.session_state:
        st.session_state.current_app = None

    # Sidebar navigation
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
                
                # Show app metadata in sidebar
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
                st.experimental_rerun()

    # Show breadcrumbs
    show_breadcrumbs(selected, st.session_state.current_app)

    # Main content area
    if selected == "🏠 Home":
        st.title("Welcome to AI Autocoder Hub")
        apps = load_apps_from_directory()
        
        for app_name, app_path in apps.items():
            st.markdown("---")
            col1, col2, col3, col4 = st.columns([3,1,1,1])
            with col1:
                st.markdown(f"### 📱 {app_name}")
                metadata = get_app_metadata(app_path)
                st.markdown(f"Last modified: {metadata.get('modified', 'N/A')} | Lines: {metadata.get('lines', 'N/A')}")
            with col2:
                if st.button("👁️ Preview", key=f"preview_{app_name}"):
                    show_app_preview(app_path)
            with col3:
                if st.button("▶️ Launch", key=f"launch_{app_name}"):
                    with st.spinner("Loading app..."):
                        st.session_state.current_app = app_name
                        st.experimental_rerun()
            with col4:
                if st.button("✏️ Edit", key=f"edit_{app_name}"):
                    st.session_state.current_app = app_name
                    selected = "📱 Apps"
                    st.experimental_rerun()
                    
    elif selected == "📱 Apps":
        if st.session_state.current_app:
            apps = load_apps_from_directory()
            app_path = apps.get(st.session_state.current_app)
            if app_path and os.path.exists(app_path):
                # Enhanced app controls
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if st.button("🔄 Restart App"):
                        st.cache_resource.clear()
                        st.experimental_rerun()
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

                # Split view for editing while running
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
                                with open(app_path, 'w') as f:
                                    f.write(modified_code)
                                st.success("Changes saved!")
                                
                                # Show diff
                                with st.expander("View Changes"):
                                    show_diff(original_code, modified_code)
                    
                    with col2:
                        st.markdown("### App Output")
                        try:
                            module = load_module(os.path.splitext(os.path.basename(app_path))[0])
                            if module:
                                run_app_safely(module, st.session_state.current_app)
                        except Exception as e:
                            st.error(f"Error loading app: {str(e)}")
                else:
                    # Regular app view
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
        # Add requirements validation
        if st.button("Validate Requirements"):
            with open('requirements.txt', 'r') as f:
                reqs = f.read()
            valid, invalid = validate_requirements(reqs)
            if valid:
                st.success("All requirements are valid!")
            else:
                st.error("Invalid requirements found:")
                st.write(invalid)
        
    elif selected == "🔄 Sync":
        github_integration()
        
    elif selected == "⚙️ Settings":
        st.subheader("Settings")
        
        tab1, tab2 = st.tabs(["General", "GitHub"])
        
        with tab1:
            st.markdown("### General Settings")
            auto_save = st.checkbox("Auto Save", value=True)
            show_previews = st.checkbox("Show Code Previews", value=True)
            dark_mode = st.checkbox("Dark Mode", value=False)
            
        with tab2:
            st.markdown("### GitHub Settings")
            gh_token = st.text_input("GitHub Token", 
                                   type="password",
                                   value=st.session_state.get('gh_token', ''))
            if gh_token:
                st.session_state['gh_token'] = gh_token

    # Add status indicator
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Status")
    if st.session_state.current_app:
        st.sidebar.success(f"Running: {st.session_state.current_app}")
    else:
        st.sidebar.info("No app running")

if __name__ == "__main__":
    main()

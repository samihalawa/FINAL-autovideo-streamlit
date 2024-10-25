# AutocoderAI - Advanced Streamlit Application for Code Optimization

import os
import ast
import re
import streamlit as st
import openai
import networkx as nx
from typing import Dict, List, Any
from difflib import unified_diff
import time
from streamlit_ace import st_ace
from streamlit_agraph import agraph, Node, Edge, Config
from aider.io import InputOutput
from aider import models, chat
from aider.coders import Coder
from aider.handlers import ConversationHandler
from aider.repomap import RepoMap
from openai import OpenAI
import subprocess
import sys
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
import concurrent.futures
from streamlit_folium import folium_static
import folium
import plotly.graph_objects as go
import json

# 1. Configuration
st.set_page_config(page_title="AutocoderAI", layout="wide")
DEFAULT_OPENAI_MODEL, DEFAULT_OPENAI_API_KEY = "gpt-4o-mini", os.environ.get("OPENAI_API_KEY", "")

# 2. Initialize session state
def init_session_state():
    defaults = {
        'script_sections': {}, 'optimized_sections': {}, 'final_script': "", 'progress': 0, 'logs': [],
        'function_list': [], 'current_function_index': 0, 'openai_api_key': DEFAULT_OPENAI_API_KEY,
        'openai_model': DEFAULT_OPENAI_MODEL, 'openai_endpoint': "https://api.openai.com/v1",
        'script_map': None, 'optimization_status': {}, 'error_log': [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# 3. Log with timestamp
def log(msg: str): st.session_state['logs'].append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# 4. Extract script sections
def extract_sections(script: str) -> Dict[str, Any]:
    try:
        tree = ast.parse(script)
    except SyntaxError as e:
        st.error(f"Syntax Error: {e}")
        return {}
    
    sections = {
        "package_installations": [], "imports": [], "settings": [],
        "function_definitions": {}, "class_definitions": {}, "global_code": []
    }
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            sections["imports"].append(ast.get_source_segment(script, node))
        elif isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
            sections["settings"].append(ast.get_source_segment(script, node))
        elif isinstance(node, ast.FunctionDef):
            sections["function_definitions"][node.name] = ast.get_source_segment(script, node)
            st.session_state['function_list'].append(node.name)
        elif isinstance(node, ast.ClassDef):
            sections["class_definitions"][node.name] = ast.get_source_segment(script, node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Attribute) and node.value.func.attr in ["system", "check_call", "run"]:
                for arg in node.value.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        match = re.findall(r'pip install ([\w\-\.\@]+)', arg.value)
                        if match:
                            sections["package_installations"].extend(match)
        elif isinstance(node, (ast.Expr, ast.Assign)):
            sections["global_code"].append(ast.get_source_segment(script, node))
    
    return sections

# 5. Generate requirements
def gen_requirements(pkgs: List[str]) -> str:
    reqs = []
    for pkg in sorted(set(pkgs)):
        try:
            ver = subprocess.check_output([sys.executable, '-m', 'pip', 'show', pkg]).decode().split('\n')[1].split(': ')[1]
            reqs.append(f"{pkg}=={ver}")
        except subprocess.CalledProcessError:
            reqs.append(pkg)
    return "\n".join(f"pip install {req}" for req in reqs)

# 6. Validate API key
def validate_api_key(key: str) -> bool: return bool(re.match(r'^sk-[a-zA-Z0-9]{48}$', key))

# 7. Set OpenAI credentials
def set_openai_creds(key: str, model: str, endpoint: str):
    try:
        openai.api_key, openai.api_base = key, endpoint
        st.session_state.update({'openai_api_key': key, 'openai_model': model, 'openai_endpoint': endpoint})
        openai.Model.list()
        log("OpenAI credentials set successfully.")
    except Exception as e:
        st.error(f"Error setting OpenAI credentials: {str(e)}")
        log(f"Error: {str(e)}")

# 8. Analyze and optimize code section
def analyze_and_optimize(content: str, name: str, coder: Coder) -> str:
    prompt = f"Optimize the following `{name}` section:\n\n```python\n{content}\n```\n\nConsider: time/space complexity, readability, PEP 8, modern Python features, error handling, type hints, imports, data structures, and caching."
    
    try:
        optimized_content = coder.edit(content, prompt)
        return optimized_content
    except Exception as e:
        st.error(f"Optimization error: {str(e)}")
        log(f"Error: {str(e)}")
        return content

# 9. Optimize all sections
def optimize_sections(coder: Coder):
    progress_bar = st.progress(0)
    total_steps = len(st.session_state['script_sections']['function_definitions']) + len(st.session_state['script_sections']['class_definitions']) + 3
    current_step = 0

    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_section = {
                executor.submit(analyze_and_optimize, content, section, coder): section
                for section, content in [
                    ('imports', '\n'.join(st.session_state['script_sections']['imports'])),
                    ('settings', '\n'.join(st.session_state['script_sections']['settings'])),
                    *[(f"function {func}", func_content) for func, func_content in st.session_state['script_sections']['function_definitions'].items()],
                    *[(f"class {cls}", cls_content) for cls, cls_content in st.session_state['script_sections']['class_definitions'].items()]
                ]
            }
            
            for future in concurrent.futures.as_completed(future_to_section):
                section = future_to_section[future]
                section_type = section.split()[0] + ('s' if section in ['imports', 'settings'] else '')
                if section_type not in st.session_state['optimized_sections']:
                    st.session_state['optimized_sections'][section_type] = {}
                if section in ['imports', 'settings']:
                    st.session_state['optimized_sections'][section_type] = content.splitlines()
                else:
                    st.session_state['optimized_sections'][section_type][section.split()[1]] = content
                current_step += 1
                progress_bar.progress(current_step / total_steps)
                st.session_state['optimization_status'][section] = "Completed"
                log(f"{section.capitalize()} optimized.")
    except Exception as e:
        st.error(f"Optimization error: {str(e)}")
        log(f"Error: {str(e)}")
    finally:
        progress_bar.empty()
        log("Optimization process finished.")

# 10. Assemble optimized script
def assemble_script() -> str:
    parts = []
    if st.session_state['script_sections']['package_installations']:
        parts.append("# Package Installations")
        parts.append(gen_requirements(st.session_state['script_sections']['package_installations']))
    parts.append("# Imports")
    parts.extend(st.session_state['optimized_sections']['imports'])
    parts.append("\n# Settings")
    parts.extend(st.session_state['optimized_sections']['settings'])
    for class_name, class_content in st.session_state['optimized_sections'].get('class_definitions', {}).items():
        parts.append(f"\n# Class: {class_name}")
        parts.append(class_content)
    for func_name, func_content in st.session_state['optimized_sections'].get('function_definitions', {}).items():
        parts.append(f"\n# Function: {func_name}")
        parts.append(func_content)
    if 'global_code' in st.session_state['script_sections']:
        parts.append("\n# Global Code")
        parts.extend(st.session_state['script_sections']['global_code'])
    return "\n\n".join(parts)

# 11. Display logs
def display_logs():
    st.header("📋 Optimization Logs")
    for log in st.session_state['logs']:
        if "Error" in log: st.error(log)
        elif "Optimizing" in log or "optimized" in log: st.success(log)
        else: st.info(log)

# 12. Generate script map
def gen_script_map() -> nx.DiGraph:
    return aider_chat.coder.repo_map.to_networkx()

# 13. Plot interactive graph
def plot_graph(G: nx.DiGraph):
    nodes = [Node(id=n, label=n, size=1000) for n in G.nodes()]
    edges = [Edge(source=e[0], target=e[1]) for e in G.edges()]
    config = Config(width=800, height=600, directed=True, physics=True, hierarchical=False)
    return agraph(nodes=nodes, edges=edges, config=config)

# 14. Highlight code changes
def highlight_changes(orig: str, opt: str) -> str:
    diff = unified_diff(orig.splitlines(), opt.splitlines(), lineterm='')
    return "<br>".join(f"<span style='background-color: {'#d4fcdc' if l.startswith('+ ') else '#fcdcdc' if l.startswith('- ') else 'transparent'}'>{l}</span>" for l in diff)

# 15. Display function changes
def display_func_changes():
    st.header("🔍 Code Changes")
    for item_type in ['function_definitions', 'class_definitions']:
        if item_type in st.session_state['optimized_sections']:
            for name, opt_content in st.session_state['optimized_sections'][item_type].items():
                with st.expander(f"{item_type.split('_')[0].capitalize()}: {name}", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Original")
                        st.code(st.session_state['script_sections'][item_type][name], language='python')
                    with col2:
                        st.subheader("Optimized")
                        st.code(opt_content, language='python')
                    st.markdown("### Diff")
                    st.markdown(highlight_changes(st.session_state['script_sections'][item_type][name], opt_content), unsafe_allow_html=True)

# 16. Display final script
def display_final_script():
    st.header("📄 Optimized Script")
    st.code(st.session_state['final_script'], language='python')
    st.download_button("💾 Download Optimized Script", st.session_state['final_script'], "optimized_script.py", "text/plain")

# 17. Main interface
def main_interface():
    st.title("AutocoderAI 🧑‍💻✨")
    st.markdown("**Advanced Python Script Optimizer using OpenAI's API**")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Script Input")
        script_input = st_ace(placeholder="Paste your Python script here...", language="python", theme="monokai", keybinding="vscode", font_size=14, min_lines=20, key="ace_editor")
        
        if st.button("🚀 Optimize Script", key="optimize_button"):
            if not script_input.strip():
                st.error("Please paste a Python script before proceeding.")
                return
            
            if not validate_api_key(st.session_state['openai_api_key']):
                st.error("Please enter a valid OpenAI API key in the sidebar.")
                return
            
            with st.spinner("Analyzing and optimizing script..."):
                st.session_state['script_sections'] = extract_sections(script_input)
                optimize_sections(aider_chat.coder)
                st.session_state['final_script'] = assemble_script()
                st.session_state['script_map'] = gen_script_map()
            
            st.success("Script optimization completed!")
    
    with col2:
        st.subheader("🗺️ Function Dependency Map")
        if st.session_state['script_map']:
            plot_interactive_graph(st.session_state['script_map'])
        else:
            st.info("Optimize a script to see the dependency map.")
    
    if st.session_state['final_script']:
        st.subheader("Optimized Script Sections")
        for section, content in st.session_state['optimized_sections'].items():
            if isinstance(content, dict):
                for subsection, subcontent in content.items():
                    st.session_state['optimized_sections'][section][subsection] = st.text_area(f"Edit {section} - {subsection}", subcontent, key=f"editor_{section}_{subsection}")
            else:
                st.session_state['optimized_sections'][section] = st.text_area(f"Edit {section}", "\n".join(content), key=f"editor_{section}")

        if st.button("Save Changes"):
            st.session_state['final_script'] = assemble_script()
            save_final_code(st.session_state['final_script'])

        if st.button("Save Final Code"):
            save_final_code(st.session_state['final_script'])

        display_collapsible_sections()
        display_side_by_side_diff()
        
    display_floating_action_button()

def plot_interactive_graph(G):
    m = folium.Map(zoom_start=2)
    nodes = list(G.nodes())
    edges = list(G.edges())
    
    for i, node in enumerate(nodes):
        folium.CircleMarker(
            location=[i*10, i*10],
            radius=5,
            popup=node,
            color="#3186cc",
            fill=True,
            fillColor="#3186cc",
        ).add_to(m)
    
    for edge in edges:
        folium.PolyLine(
            locations=[[nodes.index(edge[0])*10, nodes.index(edge[0])*10],
                       [nodes.index(edge[1])*10, nodes.index(edge[1])*10]],
            color="gray",
            weight=2,
            opacity=0.8
        ).add_to(m)
    
    folium_static(m)

def display_collapsible_sections():
    st.header("📊 Optimized Script Sections")
    for section, content in st.session_state['optimized_sections'].items():
        with st.expander(f"{section.capitalize()}"):
            st.code("\n".join(content) if isinstance(content, list) else content, language="python")

def display_side_by_side_diff():
    st.header("🔍 Code Changes")
    for item_type in ['function_definitions', 'class_definitions']:
        if item_type in st.session_state['optimized_sections']:
            for name, opt_content in st.session_state['optimized_sections'][item_type].items():
                with st.expander(f"{item_type.split('_')[0].capitalize()}: {name}", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Original")
                        st.code(st.session_state['script_sections'][item_type][name], language='python')
                    with col2:
                        st.subheader("Optimized")
                        st.code(opt_content, language='python')

def display_floating_action_button():
    st.markdown(
        """
        <style>
        .floating-button {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 999;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    with st.container():
        col1, col2, col3 = st.columns([1, 1, 1])
        with col3:
            if st.button("🚀 Optimize", key="float_optimize"):
                st.experimental_rerun()

# 18. Sidebar settings
def sidebar_settings():
    st.sidebar.header("⚙️ Settings")
    
    api_key = st.sidebar.text_input("OpenAI API Key", value=st.session_state['openai_api_key'], type="password", help="Enter your OpenAI API key.")
    model = st.sidebar.selectbox("OpenAI Model", ["gpt-3.5-turbo", "gpt-4", "gpt-4-0613", "gpt-4-32k"], index=2, help="Select the OpenAI model to use.")
    endpoint = st.sidebar.text_input("OpenAI Endpoint", value=st.session_state['openai_endpoint'], help="Enter the OpenAI API endpoint.")
    
    if st.sidebar.button("Apply Settings"):
        if validate_api_key(api_key):
            set_openai_creds(api_key, model, endpoint)
        else:
            st.sidebar.error("Invalid API Key")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Optimization Status")
    for section, status in st.session_state['optimization_status'].items():
        st.sidebar.text(f"{section}: {status}")

# 19. Run app
@st.cache(allow_output_mutation=True)
def run_app():
    try:
        init_session_state()
        sidebar_settings()
        main_interface()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        log(f"Error: {str(e)}")

# 20. Auto-scroll to function
def auto_scroll(func_name: str):
    js = f"""
    <script>
        function scrollToFunction(functionName) {{
            var elements = document.getElementsByTagName('*');
            for (var i = 0; i < elements.length; i++) {{
                if (elements[i].textContent.includes(functionName)) {{
                    elements[i].scrollIntoView({{behavior: 'smooth', block: 'center'}});
                    break;
                }}
            }}
        }}
        scrollToFunction("{func_name}");
    </script>
    """
    st.components.v1.html(js, height=0)

# 21. Enhance UI performance
@st.cache(ttl=3600)
def load_deps(): pass

# 22. Handle user prompts
def handle_prompts(prompt: str):
    try:
        response = aider_chat.run(prompt)
        return aider_chat.coder.get_edits(response)
    except Exception as e:
        st.error(f"Error generating code: {str(e)}")
        log(f"Error: {str(e)}")
        return None

# 23. Cache OpenAI responses
@st.cache(ttl=3600)
def cached_openai_call(prompt: str):
    return openai.ChatCompletion.create(
        model=st.session_state['openai_model'],
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

# 24. Handle network errors
def handle_network_errors(func):
    def wrapper(*args, **kwargs):
        max_retries, attempt = 3, 0
        while attempt < max_retries:
            try:
                return func(*args, **kwargs)
            except (openai.error.APIError, openai.error.RateLimitError, requests.exceptions.RequestException) as e:
                attempt += 1
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    st.warning(f"Network error. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    st.error(f"Max retries reached. Error: {str(e)}")
                    log(f"Error: {str(e)}")
    return wrapper

# 25. Tokenize and chunk large scripts
def tokenize_and_chunk(script: str) -> List[str]:
    encoder = tiktoken.encoding_for_model(st.session_state['openai_model'])
    tokens = encoder.encode(script)
    max_tokens = 4000
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=200,
        length_function=lambda x: len(encoder.encode(x))
    )
    
    return splitter.split_text(script)

# 26. Optimize large scripts in parallel
@handle_network_errors
def optimize_large_script(script: str):
    chunks = tokenize_and_chunk(script)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_chunk = {executor.submit(analyze_and_optimize, chunk, f"chunk_{i}"): i for i, chunk in enumerate(chunks)}
        optimized_chunks = []
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_index = future_to_chunk[future]
            try:
                optimized_chunk = future.result()
                optimized_chunks.append(optimized_chunk)
                st.success(f"Chunk {chunk_index + 1}/{len(chunks)} optimized")
            except Exception as e:
                st.error(f"Error optimizing chunk {chunk_index + 1}: {str(e)}")
    return "\n".join(optimized_chunks)

# 27. Generate documentation
def gen_docs(script: str):
    prompt = f"""
    Generate comprehensive documentation for the following Python script:

    ```python
    {script}
    
    Include: overview, function/class descriptions, parameters, return values, usage examples, and notes.
    Format in Markdown.

    Response format:
    {{
        "documentation": "string"
    }}

    Schema:
    {{
        "type": "object",
        "properties": {{
            "documentation": {{"type": "string"}}
        }},
        "required": ["documentation"]
    }}
    """
    
    try:
        response = openai.ChatCompletion.create(
            model=st.session_state['openai_model'],
            messages=[
                {"role": "system", "content": "You are an expert technical writer specializing in Python documentation."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return result['documentation']
    except Exception as e:
        st.error(f"Error generating documentation: {str(e)}")
        log(f"Error: {str(e)}")
        return None

# 28. Code review suggestions
def get_review_suggestions(script: str):
    prompt = f"""
    Review the following Python script:

    ```python
    {script}
    ```

    Provide: potential bugs, performance improvements, style recommendations, security considerations, and scalability advice.
    Format as a list of actionable items.

    Response format:
    {{
        "review_items": ["string"]
    }}

    Schema:
    {{
        "type": "object",
        "properties": {{
            "review_items": {{"type": "array", "items": {{"type": "string"}}}}
        }},
        "required": ["review_items"]
    }}
    """
    
    try:
        response = openai.ChatCompletion.create(
            model=st.session_state['openai_model'],
            messages=[
                {"role": "system", "content": "You are an experienced Python code reviewer with a keen eye for quality and best practices."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return result['review_items']
    except Exception as e:
        st.error(f"Error getting code review suggestions: {str(e)}")
        log(f"Error: {str(e)}")
        return None

# 29. Generate unit tests
def gen_unit_tests(script: str, coder: Coder):
    prompt = f"""
    Generate unit tests for the following Python script:

    ```python
    {script}
    ```

    Include: test cases for each function/method, edge cases, mocking, pytest fixtures, and clear test names/descriptions.
    Provide executable Python code.
    """
    
    try:
        unit_tests = coder.edit("", prompt)  # Start with an empty string and let Aider generate the tests
        return unit_tests
    except Exception as e:
        st.error(f"Error generating unit tests: {str(e)}")
        log(f"Error: {str(e)}")
        return None

# 30. Performance profiling
def profile_performance(script: str):
    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()
    exec(script)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    return s.getvalue()

# 31. Integrate aider.chat for optimization
def optimize_with_aider(script: str):
    try:
        prompt = "Optimize the following Python script, considering performance, readability, and best practices:\n\n" + script
        response = aider_chat.send_message(prompt)
        return response.content
    except Exception as e:
        st.error(f"Aider optimization error: {str(e)}")
        log(f"Error: {str(e)}")
        return script

# Add a function to save the final code
def save_final_code(code: str, filename: str = "optimized_script.py"):
    with open(filename, "w") as f:
        f.write(code)
    st.success(f"Saved optimized code to {filename}")

# Initialize aider components
io = InputOutput()
model = models.Model.create("gpt-4")  # Use GPT-4 as the default model
coder = Coder.create(main_model=model)
aider_chat = chat.Chat(io=io, coder=coder)

# Run the app
if __name__ == "__main__":
    run_app()

def generate_script_map(function_definitions: Dict[str, str]) -> nx.DiGraph:
    graph = nx.DiGraph()
    
    # Add nodes first
    for func_name in function_definitions:
        graph.add_node(func_name)
    
    # Then add edges based on function calls
    for func_name, func_code in function_definitions.items():
        for other_func in function_definitions:
            if other_func != func_name and other_func in func_code:
                graph.add_edge(func_name, other_func)
    
    return graph

def display_dependency_graph(graph: nx.DiGraph):
    if not graph.nodes():
        return
        
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=list(graph.nodes()),
            color="blue"
        ),
        link=dict(
            source=[list(graph.nodes()).index(edge[0]) for edge in graph.edges()],
            target=[list(graph.nodes()).index(edge[1]) for edge in graph.edges()],
            value=[1] * len(graph.edges())
    ))])
    
    fig.update_layout(title_text="Function Dependencies", font_size=10)
    st.plotly_chart(fig, use_container_width=True)

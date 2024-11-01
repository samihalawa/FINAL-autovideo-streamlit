import os, re, ast, json, threading, streamlit as st, openai, networkx as nx, matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
from difflib import unified_diff
from concurrent.futures import ThreadPoolExecutor, as_completed
from streamlit_ace import st_ace
import logging
import time
from openai import OpenAI
import plotly.graph_objects as go

# Logging function
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log(msg: str, level: str = "INFO") -> None:
    """Thread-safe logging function"""
    getattr(logging, level.lower())(msg)
    with threading.Lock():
        if 'logs' in st.session_state:
            st.session_state.logs.append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {level}: {msg}")

st.set_page_config(page_title="AutocoderAI", layout="wide")
openai.api_key = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4"

# 1. Remove hardcoded API key for security
openai.api_key = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"

# 2. Initialize session state with default values
def initialize_session_state():
    defaults = {
        'script_content': '', 'final_script': '', 'script_sections': {}, 
        'optimized_sections': {}, 'logs': [], 'function_list': [], 
        'progress': 0, 'current_function_index': 0, 'openai_model': OPENAI_MODEL
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# 3. Set OpenAI credentials
def set_openai_credentials(api_key: str, model: str):
    global client
    client = OpenAI(api_key=api_key)
    st.session_state['openai_model'] = model

# 4. Extract sections from script content
def extract_sections(script_content: str) -> Dict[str, Any]:
    try:
        tree = ast.parse(script_content)
    except SyntaxError as e:
        st.error(f"Syntax Error in script: {e}")
        return {}
    
    sections = {'package_installations': [], 'imports': [], 'settings': [], 'function_definitions': {}}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            sections['imports'].append(ast.get_source_segment(script_content, node))
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            sections['settings'].append(ast.get_source_segment(script_content, node))
        elif isinstance(node, ast.FunctionDef):
            sections['function_definitions'][node.name] = ast.get_source_segment(script_content, node)
            st.session_state['function_list'].append(node.name)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
            if node.value.func.attr in ["system", "check_call", "run"]:
                for arg in node.value.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        sections['package_installations'].extend(re.findall(r'pip install ([\w\-\.\@]+)', arg.value))
    return sections

# 5. Generate requirements string
def generate_requirements(packages: List[str]) -> str:
    return "\n".join(f"pip install {pkg}" for pkg in sorted(set(packages)))

# 6. Validate API key format
def validate_api_key(api_key: str) -> bool:
    return bool(api_key and api_key.startswith("sk-"))

# 7. Analyze and optimize code section
def analyze_and_optimize(section):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that optimizes Python code."},
                {"role": "user", "content": f"Analyze and optimize the following Python code section:\n\n{section}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        log(f"Error in analyze_and_optimize: {str(e)}", "ERROR")
        return section  # Return the original section if optimization fails

# 8. Optimize sections using ThreadPoolExecutor
def optimize_sections():
    sections_to_optimize = ['imports', 'settings'] + st.session_state['function_list']
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(analyze_and_optimize, 
                                   "\n".join(st.session_state['script_sections'][section]) if section in ['imports', 'settings'] else st.session_state['script_sections']['function_definitions'][section], 
                                   section): section for section in sections_to_optimize}
        
        progress_bar = st.progress(0)
        for i, future in enumerate(as_completed(futures)):
            section_name = futures[future]
            optimized_content = future.result()
            if section_name in ['imports', 'settings']:
                st.session_state['optimized_sections'][section_name] = optimized_content.splitlines()
            else:
                if 'function_definitions' not in st.session_state['optimized_sections']:
                    st.session_state['optimized_sections']['function_definitions'] = {}
                st.session_state['optimized_sections']['function_definitions'][section_name] = optimized_content
            progress_bar.progress((i + 1) / len(futures))
            st.session_state['logs'].append(f"Optimized {section_name}")
        progress_bar.empty()

# 9. Assemble optimized script
def assemble_script() -> str:
    script_parts = []
    if st.session_state['script_sections']['package_installations']:
        script_parts.append("# Package Installations\n" + generate_requirements(st.session_state['script_sections']['package_installations']))
    script_parts.extend([
        "# Imports\n" + "\n".join(st.session_state['optimized_sections']['imports']),
        "# Settings\n" + "\n".join(st.session_state['optimized_sections']['settings']),
        "\n\n".join(st.session_state['optimized_sections']['function_definitions'].values())
    ])
    return "\n\n".join(script_parts)

# 10. Generate script dependency map
def generate_script_map(function_definitions: Dict[str, str]) -> nx.DiGraph:
    graph = nx.DiGraph()
    function_names = list(function_definitions.keys())
    graph.add_nodes_from(function_names)
    for func_name, func_code in function_definitions.items():
        try:
            func_ast = ast.parse(func_code)
            for node in ast.walk(func_ast):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    called_func = node.func.id
                    if called_func in function_names and called_func != func_name:
                        graph.add_edge(func_name, called_func)
        except Exception as e:
            st.session_state['logs'].append(f"Error parsing function {func_name}: {e}")
    return graph

# 11. Plot dependency graph
def plot_graph(graph: nx.DiGraph):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000, font_size=8, arrows=True)
    plt.title("Function Dependency Map")
    st.pyplot(plt)

# 12. Highlight code changes
def highlight_code_changes(original: str, optimized: str) -> str:
    diff = unified_diff(original.splitlines(), optimized.splitlines(), lineterm='')
    return ''.join(f"<span style='background-color: {'#d4fcdc' if line.startswith('+ ') else '#fcdcdc' if line.startswith('- ') else 'transparent'}'>{line}</span><br>" for line in diff if not line.startswith('+++') and not line.startswith('---'))

# 13. Display function changes
def display_function_changes():
    st.header(" Function Changes")
    for func_name in st.session_state['function_list']:
        with st.expander(f"Function: {func_name}", expanded=False):
            st.markdown(highlight_code_changes(
                st.session_state['script_sections']['function_definitions'][func_name],
                st.session_state['optimized_sections']['function_definitions'][func_name]
            ), unsafe_allow_html=True)

# 14. Display final optimized script
def display_final_script():
    st.header("📄 Optimized Script")
    st.code(st.session_state['final_script'], language='python')
    st.download_button(label="💾 Download Optimized Script", data=st.session_state['final_script'], file_name="optimized_script.py", mime="text/plain")

# 15. Handle script input and optimization
def handle_script_input():
    st.title("AutocoderAI 🧑‍💻✨")
    st.markdown("**Automated Python Script Manager and Optimizer using OpenAI's API**")
    st.info("Paste your Python script below and click 'Optimize Script' to begin.")
    script_input = st_ace(placeholder="Paste your Python script here...", language="python", theme="monokai", keybinding="vscode", font_size=14, min_lines=20, key="ace_editor")
    if st.button("🚀 Optimize Script") and script_input.strip():
        st.session_state['script_content'] = script_input
        st.session_state['script_sections'] = extract_sections(script_input)
        optimize_sections()
        st.session_state['final_script'] = assemble_script()
        display_function_changes()
        display_final_script()
        graph = generate_script_map(st.session_state['optimized_sections']['function_definitions'])
        st.header("🗺️ Function Dependency Map")
        plot_graph(graph)
        st.header("📋 Logs")
        for log in st.session_state['logs']:
            st.write(log)
    elif not script_input.strip():
        st.error("Please paste a Python script before proceeding.")

# 16. Handle sidebar settings
def sidebar_settings():
    st.sidebar.header("⚙️ Settings")
    api_key = st.sidebar.text_input("OpenAI API Key", value=openai.api_key, type="password", help="Enter your OpenAI API key.")
    model = st.sidebar.text_input("OpenAI Model", value=st.session_state['openai_model'], help="Enter the OpenAI model to use.")
    if validate_api_key(api_key):
        set_openai_credentials(api_key, model)
    else:
        st.sidebar.error("Invalid API Key")

# 17. Handle user prompts for code generation
def handle_user_prompts():
    st.sidebar.header("💡 Generate Code from Prompt")
    prompt = st.sidebar.text_area("Enter a prompt to generate code:", height=100)
    if st.sidebar.button("Generate Code") and prompt.strip():
        generated_code = generate_code_from_prompt(prompt)
        st.session_state['script_content'] = generated_code
        st.session_state['script_sections'] = extract_sections(generated_code)
        st.success("Code generated from prompt.")
    elif not prompt.strip():
        st.sidebar.error("Please enter a prompt.")

# 18. Generate code from user prompt
def generate_code_from_prompt(prompt: str) -> str:
    full_prompt = f"""
    Write a Python script based on the following prompt:

    "{prompt}"

    Ensure that the script is self-contained and uses only standard libraries or libraries that can be installed with pip.

    Response format:
    {{
        "generated_code": "string"
    }}

    Schema:
    {{
        "type": "object",
        "properties": {{
            "generated_code": {{"type": "string"}}
        }},
        "required": ["generated_code"]
    }}
    """
    try:
        response = client.chat.completions.create(
            model=st.session_state['openai_model'],
            messages=[
                {"role": "system", "content": "You are a helpful assistant for generating Python code."},
                {"role": "user", "content": full_prompt}
            ],
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return result['generated_code']
    except Exception as e:
        st.error(f"Error during OpenAI API call: {e}")
        return ''

# 19. Highlight modified lines
def highlight_modified_lines(original: str, optimized: str) -> List[int]:
    return [i for i, (a, b) in enumerate(zip(original.splitlines(), optimized.splitlines())) if a != b]

# 20. Visualize line changes
def visualize_line_changes(function_name: str):
    original = st.session_state['script_sections']['function_definitions'][function_name]
    optimized = st.session_state['optimized_sections']['function_definitions'][function_name]
    modified_lines = highlight_modified_lines(original, optimized)
    st.code(optimized, language='python', line_numbers=True)
    st.markdown(f"Modified lines: {', '.join(map(str, modified_lines))}")

# 21. Save session state
def save_session_state():
    with open('session_state.json', 'w') as f:
        json.dump({k: v for k, v in st.session_state.items() if isinstance(v, (str, int, float, bool, list, dict))}, f)

# 22. Load session state
def load_session_state():
    if os.path.exists('session_state.json'):
        with open('session_state.json', 'r') as f:
            st.session_state.update(json.load(f))

# 23. Clear session state
def clear_session_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_session_state()

def identify_errors(script_content: str) -> List[Dict[str, Any]]:
    errors = []
    try:
        ast.parse(script_content)
    except SyntaxError as e:
        errors.append({
            "type": "SyntaxError",
            "message": str(e),
            "line": e.lineno,
            "offset": e.offset
        })
    return errors

def interactive_function_editor():
    st.header("🖊️ Interactive Function Editor")
    function_name = st.selectbox("Select a function to edit:", st.session_state['function_list'])
    
    if function_name:
        original_code = st.session_state['script_sections']['function_definitions'][function_name]
        optimized_code = st.session_state['optimized_sections']['function_definitions'][function_name]
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Function")
            st.code(original_code, language='python')
        with col2:
            st.subheader("Optimized Function")
            edited_code = st.text_area("Edit the optimized function:", value=optimized_code, height=300)
        
        if st.button("Update Function"):
            st.session_state['optimized_sections']['function_definitions'][function_name] = edited_code
            st.success(f"Function '{function_name}' updated successfully!")
            st.session_state['final_script'] = assemble_script()  # Reassemble the script with the updated function

# Replace profiling with simple timing
def profile_performance(script: str) -> str:
    start_time = time.time()
    try:
        exec(script)
        end_time = time.time()
        return f"Execution time: {end_time - start_time:.2f} seconds"
    except Exception as e:
        return f"Error: {str(e)}"

# Add unit test generation function
def generate_unit_tests(script: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates unit tests for Python code."},
                {"role": "user", "content": f"Generate unit tests for the following Python script:\n\n{script}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        log(f"Error generating unit tests: {str(e)}", "ERROR")
        return f"Error generating unit tests: {str(e)}"

# 24. Main function
def main():
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    
    initialize_session_state()
    load_session_state()
    sidebar_settings()
    handle_user_prompts()
    handle_script_input()
    
    if st.session_state['final_script']:
        interactive_function_editor()
        
        if st.button("Generate Unit Tests"):
            unit_tests = generate_unit_tests(st.session_state['final_script'])
            if unit_tests:
                st.header("📊 Generated Unit Tests")
                st.code(unit_tests, language='python')
        
        if st.button("Profile Performance"):
            profile_result = profile_performance(st.session_state['final_script'])
            st.header("🚀 Performance Profile")
            st.text(profile_result)
    
    if st.sidebar.button("Clear Session"):
        clear_session_state()
    save_session_state()

if __name__ == "__main__":
    main()

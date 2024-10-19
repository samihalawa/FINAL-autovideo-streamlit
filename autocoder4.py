# AutocoderAI - Enhanced Streamlit Application for Code Optimization

import os
import re
import streamlit as st
import openai
import networkx as nx
from typing import Dict, List, Any
import time
import tempfile
import faiss
from streamlit_ace import st_ace
from streamlit_agraph import agraph, Node, Edge, Config
from aider.io import InputOutput
from aider import models, chat
from aider.coders import Coder
import subprocess
from langchain import LLMChain, PromptTemplate
from langchain.experimental import BabyAGI
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
import logging
from openai import OpenAI
import plotly.graph_objects as go
import json

# Configuration
st.set_page_config(page_title="AutocoderAI", layout="wide")
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_ITERATIONS = 10

# Initialize session state
@st.cache_resource
class State:
    def __init__(self):
        self.script_sections = {}
        self.optimized_sections = {}
        self.final_script = ""
        self.progress = 0
        self.logs = []
        self.function_list = []
        self.current_function_index = 0
        self.openai_api_key = DEFAULT_OPENAI_API_KEY
        self.openai_model = DEFAULT_OPENAI_MODEL
        self.openai_endpoint = "https://api.openai.com/v1"
        self.script_map = None
        self.optimization_status = {}
        self.error_log = []
        self.chat_history = []
        self.iterations = DEFAULT_ITERATIONS

state = State()

# Logging function
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log(msg: str, level: str = "INFO"):
    getattr(logging, level.lower())(msg)
    state.logs.append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {level}: {msg}")

# Validate API key
def validate_api_key(key: str) -> bool:
    return bool(re.match(r'^sk-[a-zA-Z0-9]{48}$', key))

# Set OpenAI credentials
def set_openai_creds(key: str, model: str, endpoint: str):
    try:
        openai.api_key, openai.api_base = key, endpoint
        state.openai_api_key = key
        state.openai_model = model
        state.openai_endpoint = endpoint
        openai.Model.list()
        log("OpenAI credentials set successfully.")
    except openai.error.AuthenticationError:
        log("Authentication failed. Please check your API key.", "ERROR")
        st.error("Authentication failed. Please check your API key.")
    except openai.error.APIConnectionError:
        log("Unable to connect to the OpenAI API. Please check your internet connection or the API endpoint.", "ERROR")
        st.error("Unable to connect to the OpenAI API. Please check your internet connection or the API endpoint.")
    except Exception as e:
        log(f"Error setting OpenAI credentials: {str(e)}", "ERROR")
        st.error(f"Error setting OpenAI credentials: {str(e)}")

# Initialize Aider components
@st.cache_resource
def initialize_aider() -> Coder:
    io = InputOutput()
    model = models.Model.create(state.openai_model)
    coder = Coder.create(main_model=model)
    return coder

coder = initialize_aider()

# BabyAGI Task Creation
def get_task_creation_chain():
    prompt = PromptTemplate(
        input_variables=["result", "task_description", "incomplete_tasks"],
        template="You are an AI tasked with improving code. Given the result of the last task: {result}, the following task description: {task_description}, and incomplete tasks: {incomplete_tasks}, create new tasks to be completed in order to improve the code further. Return the tasks as a numbered list.",
    )
    return LLMChain(llm=coder.main_model, prompt=prompt)

# BabyAGI Task Prioritization
def get_task_prioritization_chain():
    prompt = PromptTemplate(
        input_variables=["task_names"],
        template="You are an AI tasked with prioritizing code improvement tasks. Given the following tasks: {task_names}, prioritize these tasks in the order they should be executed. Return the prioritized tasks as a numbered list.",
    )
    return LLMChain(llm=coder.main_model, prompt=prompt)

# BabyAGI Execution
def get_execution_chain():
    prompt = PromptTemplate(
        input_variables=["objective", "context", "task"],
        template="You are an AI tasked with improving code. Your objective is: {objective}. Given the context: {context}, complete the following task: {task}. Return the result of the task.",
    )
    return LLMChain(llm=coder.main_model, prompt=prompt)

# Supervisor Agent
def supervisor_review(changes, original_code):
    prompt = f"""
    As a senior code reviewer, analyze the following code changes:

    Original Code:    ```python
    {original_code}    ```

    Changed Code:    ```python
    {changes}    ```

    Provide a detailed review focusing on:
    1. Code quality improvements
    2. Potential bugs introduced
    3. Performance enhancements
    4. Readability and maintainability

    If the changes are acceptable, respond with "APPROVED". Otherwise, explain the issues and suggest further improvements.
    """
    
    review = coder.main_model.complete(prompt)
    return review

# Main optimization function
def optimize_and_debug(script_input, coder: Coder):
    st.write("Starting automated optimization and debugging process...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        embeddings_model = OpenAIEmbeddings()
        embedding_size = 1536
        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

        baby_agi = BabyAGI.from_llm(
            llm=coder.main_model,
            vectorstore=vectorstore,
            task_creation_chain=get_task_creation_chain(),
            task_prioritization_chain=get_task_prioritization_chain(),
            execution_chain=get_execution_chain(),
            verbose=True
        )

        objective = "Optimize and debug the given Python script"
        
        for i in range(state.iterations):
            progress = (i + 1) / state.iterations
            progress_bar.progress(progress)
            status_text.text(f"Iteration {i+1}/{state.iterations}")
            
            result = baby_agi({"objective": objective, "script": script_input})
            optimized_script = result['output']
            
            status_text.text(f"Iteration {i+1}/{state.iterations}: Applying Aider modifications")
            optimized_script = coder.edit(optimized_script, "Optimize and debug this entire script.")
            
            status_text.text(f"Iteration {i+1}/{state.iterations}: Supervisor review")
            review = supervisor_review(optimized_script, script_input)
            if "APPROVED" in review:
                st.success(f"Iteration {i+1} approved by supervisor.")
                script_input = optimized_script
            else:
                st.warning(f"Iteration {i+1} not approved. Supervisor comments:")
                st.write(review)
            
            # Update state
            state.final_script = script_input
            state.script_map = coder.get_repo_map()
            
            status_text.text(f"Iteration {i+1}/{state.iterations}: Running tests and linting")
            with st.spinner("Running tests and linting..."):
                test_result = run_tests(script_input)
                lint_result = run_linter(script_input)
            
            if test_result and "FAILED" not in test_result:
                st.success("All tests passed!")
            elif test_result:
                st.warning(f"Tests failed. Debugging in next iteration...")
            
            if lint_result:
                st.info("Linting results:")
                st.code(lint_result)

            if i == state.iterations - 1:
                st.info(f"Reached the maximum number of iterations ({state.iterations}). Stopping optimization process.")
                break

        progress_bar.progress(1.0)
        status_text.text("Optimization and debugging process completed.")

    except Exception as e:
        st.error(f"An error occurred during optimization: {str(e)}")
        log(f"Optimization error: {str(e)}")
        return script_input

    return script_input

def run_tests(script: str):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        temp_file.write(script)
        temp_file_path = temp_file.name
    
    try:
        result = subprocess.run(['pytest', temp_file_path], capture_output=True, text=True, timeout=30)
        return result.stdout
    except subprocess.TimeoutExpired:
        log("Error: Test execution timed out")
        return "Error: Test execution timed out"
    except FileNotFoundError:
        log("Error: pytest not found. Please install pytest.")
        return "Error: pytest not found. Please install pytest."
    except Exception as e:
        log(f"Error running tests: {str(e)}")
        return f"Error running tests: {str(e)}"
    finally:
        os.unlink(temp_file_path)

def run_linter(script: str):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        temp_file.write(script)
        temp_file_path = temp_file.name
    
    try:
        result = subprocess.run(['flake8', temp_file_path], capture_output=True, text=True, timeout=30)
        return result.stdout
    except subprocess.TimeoutExpired:
        log("Error: Linter execution timed out")
        return "Error: Linter execution timed out"
    except FileNotFoundError:
        log("Error: flake8 not found. Please install flake8.")
        return "Error: flake8 not found. Please install flake8."
    except Exception as e:
        log(f"Error running linter: {str(e)}")
        return f"Error running linter: {str(e)}"
    finally:
        os.unlink(temp_file_path)

def main_interface():
    st.title("AutocoderAI 🧑‍💻✨")
    st.markdown("**Advanced Python Script Optimizer using OpenAI's API, Aider, and LangChain**")
    
    uploaded_file = st.file_uploader("Upload a Python script", type="py")
    if uploaded_file is not None:
        script_input = uploaded_file.getvalue().decode("utf-8")
    else:
        script_input = st_ace(
            placeholder="Paste your Python script here...",
            language="python",
            theme="monokai",
            keybinding="vscode",
            font_size=14,
            min_lines=20,
            key="ace_editor"
        )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Optimize and Debug Script", key="optimize_debug_button"):
            if not script_input.strip():
                st.error("Please paste or upload a Python script before proceeding.")
                return
            
            if not validate_api_key(state.openai_api_key):
                st.error("Please enter a valid OpenAI API key in the sidebar.")
                return
            
            try:
                with st.status("Optimizing and debugging script...") as status:
                    optimized_script = optimize_and_debug(script_input, coder)
                    status.update(label="Optimization complete!", state="complete", expanded=False)
                
                st.subheader("Optimized Script")
                st.code(optimized_script, language="python")
                
                st.subheader("Changes")
                diff = get_diff(script_input, optimized_script)
                st.code(diff, language="diff")
            except Exception as e:
                st.error(f"An error occurred during optimization: {str(e)}")
                log(f"Optimization error: {str(e)}")
    
    with col2:
        state.iterations = st.slider("Number of iterations", min_value=1, max_value=50, value=DEFAULT_ITERATIONS)

    if state.script_map:
        st.subheader("🗺️ Function Dependency Map")
        plot_graph(state.script_map)
    
    display_logs()

    if state.final_script:
        st.subheader("Optimized Script Sections")
        for section, content in state.optimized_sections.items():
            if isinstance(content, dict):
                for subsection, subcontent in content.items():
                    state.optimized_sections[section][subsection] = st.text_area(f"Edit {section} - {subsection}", subcontent, key=f"editor_{section}_{subsection}")
            else:
                state.optimized_sections[section] = st.text_area(f"Edit {section}", "\n".join(content), key=f"editor_{section}")

        if st.button("Save Changes"):
            state.final_script = assemble_script()
            save_final_code(state.final_script)

        if st.button("Save Final Code"):
            save_final_code(state.final_script)

def display_logs():
    with st.expander("Logs", expanded=False):
        for log in state.logs:
            st.text(log)

def plot_graph(G: nx.DiGraph):
    nodes = [Node(id=n, label=n, size=1000) for n in G.nodes()]
    edges = [Edge(source=e[0], target=e[1]) for e in G.edges()]
    config = Config(width=800, height=600, directed=True, physics=True, hierarchical=False)
    return agraph(nodes=nodes, edges=edges, config=config)

def sidebar_settings():
    st.sidebar.header("⚙️ Settings")
    
    with st.sidebar.form("settings_form"):
        api_key = st.text_input(
            "OpenAI API Key",
            value=state.openai_api_key,
            type="password",
            help="Enter your OpenAI API key."
        )
        model = st.selectbox(
            "OpenAI Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-0613", "gpt-4-32k"],
            index=2,
            help="Select the OpenAI model to use."
        )
        endpoint = st.text_input(
            "OpenAI Endpoint",
            value=state.openai_endpoint,
            help="Enter the OpenAI API endpoint."
        )
        
        if st.form_submit_button("Apply Settings"):
            if validate_api_key(api_key):
                set_openai_creds(api_key, model, endpoint)
            else:
                st.error("Invalid API Key")

def run_app():
    try:
        sidebar_settings()
        main_interface()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        log(f"Unexpected Error: {str(e)}")

if __name__ == "__main__":
    run_app()

def analyze_and_optimize(section_content: str, section_name: str, api_key: str, model: str) -> str:
    """
    Uses OpenAI's API to analyze and optimize a given section of the code.
    """
    prompt = f"""
    You are an expert Python developer. Analyze the following `{section_name}` section of a Python script and optimize it. Ensure that:

    - No functions or essential components are removed.
    - Function names are not hallucinated or changed.
    - Logic is optimized for efficiency and clarity.
    - All inputs and outputs remain consistent.
    - If using ORMs like SQLAlchemy, schema integrity is maintained.
    - Improve error handling and add input validation where necessary.
    - Enhance code readability and add comments for complex logic.

    Here is the `{section_name}` section:

    ```python
    {section_content}
    ```

    Provide the optimized `{section_name}` section.
    """

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for optimizing Python code."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1500,
            n=1,
            stop=None,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        optimized_section = result.get("optimized_section", section_content)
        return optimized_section
    except Exception as e:
        st.error(f"Error during OpenAI API call: {e}")
        return section_content  # Return original if error occurs

# Add a function to save the final code
def save_final_code(code: str, filename: str = "optimized_script.py"):
    with open(filename, "w") as f:
        f.write(code)
    st.success(f"Saved optimized code to {filename}")

# Add this function near the other utility functions
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



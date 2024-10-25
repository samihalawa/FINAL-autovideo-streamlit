import streamlit as st
from autocoder import State, optimize_script, validate_api_key, save_final_code, generate_optimization_suggestion, generate_script_map
import plotly.graph_objects as go
from aider import chat
import time
from tqdm import tqdm
from typing import Dict, Any
import ast
import re

def extract_sections(script_content: str) -> Dict[str, Any]:
    # (Use the same implementation as in autocoder4.py)

def optimize_with_aider(script_input: str) -> str:
    try:
        aider_chat = chat.Chat(io=None, coder=None)  # Initialize Aider chat
        sections = extract_sections(script_input)
        optimized_sections = {}

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, (section_name, content) in enumerate(sections.items()):
            optimized_content = aider_chat.send_message(f"Optimize this Python code section:\n\n{content}")
            optimized_sections[section_name] = optimized_content.content
            progress = (i + 1) / len(sections)
            progress_bar.progress(progress)
            status_text.text(f"Optimizing section {i+1}/{len(sections)}: {section_name}")
            time.sleep(0.1)  # To allow for visual updates

        return "\n\n".join(optimized_sections.values())
    except Exception as e:
        st.error(f"Error during Aider optimization: {e}")
        return script_input

def animate_optimization(original_script: str, optimized_script: str):
    placeholder = st.empty()
    lines_original = original_script.split('\n')
    lines_optimized = optimized_script.split('\n')
    max_lines = max(len(lines_original), len(lines_optimized))
    
    for i in range(max_lines):
        current_original = '\n'.join(lines_original[:i+1])
        current_optimized = '\n'.join(lines_optimized[:i+1])
        
        col1, col2 = placeholder.columns(2)
        with col1:
            st.code(current_original, language='python')
        with col2:
            st.code(current_optimized, language='python')
        
        time.sleep(0.1)  # Adjust speed of animation

def main():
    st.title("AutocoderAI Massive Streamlit App")
    
    state = State()
    
    st.sidebar.subheader("OpenAI Settings")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    model = st.sidebar.selectbox("Model", ["gpt-3.5-turbo", "gpt-4"])
    
    if api_key:
        if validate_api_key(api_key):
            state.settings["openai_api_key"] = api_key
            state.settings["model"] = model
        else:
            st.sidebar.error("Invalid API Key")

    script_input = st.text_area("Enter your massive Python script here:", height=300)
    
    if st.button("🚀 Optimize with Aider"):
        with st.spinner("Optimizing script..."):
            optimized_script = optimize_with_aider(script_input)
        
        st.success("Optimization complete!")
        st.download_button(
            label="Download Optimized Script",
            data=optimized_script,
            file_name="optimized_massive_script.py",
            mime="text/plain"
        )

    # Add this check to avoid errors if script_sections is empty
    if state.script_sections.get("function_definitions"):
        graph = generate_script_map(state.script_sections.get("function_definitions", {}))
        fig = go.Figure(data=[go.Sankey(
            node = dict(
              pad = 15,
              thickness = 20,
              line = dict(color = "black", width = 0.5),
              label = list(graph.nodes()),
              color = "blue"
            ),
            link = dict(
              source = [list(graph.nodes()).index(edge[0]) for edge in graph.edges()],
              target = [list(graph.nodes()).index(edge[1]) for edge in graph.edges()],
              value = [1] * len(graph.edges())
        ))])
        fig.update_layout(title_text="Function Dependencies", font_size=10)
        st.plotly_chart(fig, use_container_width=True)
    
    user_input = st.text_input("Ask for optimization suggestions:")
    if user_input:
        suggestion = generate_optimization_suggestion(user_input)
        st.write(suggestion)

if __name__ == "__main__":
    main()

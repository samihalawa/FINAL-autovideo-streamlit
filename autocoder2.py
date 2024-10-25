import streamlit as st
from autocoder import State, optimize_script, validate_api_key, save_final_code, generate_optimization_suggestion
from aider import chat
import time
from langchain.chains.python.optimize import InputOutput, models, Coder

def main():
    st.title("AutocoderAI v2")
    
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

    script_input = st.text_area("Enter your Python script here:", height=300)
    
    if st.button("🚀 Optimize with Aider"):
        with st.spinner("Optimizing script..."):
            optimized_script = optimize_with_aider(script_input)
        
        st.success("Optimization complete!")
        st.code(optimized_script, language="python")
        
        if st.button("Save Optimized Code"):
            save_final_code(optimized_script)

def optimize_with_aider(script_input: str) -> str:
    try:
        io = InputOutput()
        model = models.Model.create("gpt-4")
        coder = Coder.create(main_model=model)
        aider_chat = chat.Chat(io=io, coder=coder)
        
        optimized_script = aider_chat.send_message(
            "Optimize this Python code for performance and readability:\n\n" + script_input
        )
        return optimized_script.content
    except Exception as e:
        st.error(f"Aider optimization failed: {str(e)}")
        return script_input

if __name__ == "__main__":
    main()

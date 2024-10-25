import streamlit as st
import sys
from io import StringIO
import traceback
from streamlit_ace import st_ace

def main():
    st.set_page_config(page_title="Python Debugger", layout="wide")
    st.title("Python Debugger 🐞")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Input Code")
        code = st_ace(
            placeholder="Enter Python code to debug...",
            language="python",
            theme="monokai",
            keybinding="vscode",
            font_size=14,
            min_lines=20,
            key="debug_editor"
        )

    with col2:
        st.subheader("Execution Output")
        if st.button("🚀 Run Code", key="run_button"):
            old_stdout = sys.stdout
            redirected_output = sys.stdout = StringIO()
            
            try:
                exec(code)
                output = redirected_output.getvalue()
                st.success("Code executed successfully!")
                st.code(output, language="python")
            except Exception as e:
                st.error("An error occurred during execution:")
                st.code(traceback.format_exc(), language="python")
            finally:
                sys.stdout = old_stdout

        st.subheader("Variable Inspector")
        if st.button("🔍 Inspect Variables", key="inspect_button"):
            local_vars = {}
            try:
                exec(code, {}, local_vars)
                for var, value in local_vars.items():
                    if not var.startswith("__"):
                        st.write(f"**{var}**: {type(value).__name__} = {value}")
            except Exception as e:
                st.error(f"Error during variable inspection: {str(e)}")

if __name__ == "__main__":
    main()

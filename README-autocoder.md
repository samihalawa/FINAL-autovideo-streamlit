# AI Code Enhancer

A Streamlit app that uses AI agents to enhance and analyze Python code. It allows users to input code, segment it into blocks, and process each block using AI-powered agents for editing, profiling, dependency checking, and verification.

## Features

- **Code Segmentation**: Segments input code into imports, functions, classes, and global code using the AST module.
- **AI Code Editing**: Uses OpenAI's GPT models to suggest and apply improvements to code blocks.
- **Profiling**: Profiles code blocks using `cProfile` to identify performance bottlenecks.
- **Dependency Checking**: Identifies unused imports in code blocks.
- **Verification**: Checks code blocks for syntax errors.
- **Undo/Redo**: Supports undoing and redoing changes to code blocks.
- **Graph Visualization**: Visualizes the code blocks and their order using a directed graph.
- **Settings**: Allows configuration of model settings, API keys, themes, and session timeouts.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/ai-code-enhancer.git

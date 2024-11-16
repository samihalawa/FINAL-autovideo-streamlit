#!/bin/bash

# Create virtual environment if it doesn't exist
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -e . 
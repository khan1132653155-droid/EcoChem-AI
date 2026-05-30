#!/bin/bash
set -e

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install streamlit
pip install rdkit
pip install pubchempy
pip install scikit-learn
pip install pandas
pip install matplotlib
pip install numpy
pip install joblib

# Install test dependency
pip install pytest

echo "All dependencies installed successfully!"

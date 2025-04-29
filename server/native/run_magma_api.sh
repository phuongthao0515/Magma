#!/bin/bash

# Set the working directory to the script's directory
cd "$(dirname "$0")"

# Get the path to the conda executable
CONDA_PATH="$(which conda)"
if [ -z "$CONDA_PATH" ]; then
    # If conda is not in PATH, try common locations
    if [ -f "$HOME/miniconda3/bin/conda" ]; then
        CONDA_PATH="$HOME/miniconda3/bin/conda"
    elif [ -f "$HOME/anaconda3/bin/conda" ]; then
        CONDA_PATH="$HOME/anaconda3/bin/conda"
    fi
fi

# Source conda.sh which is the proper way to use conda in scripts
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Could not find conda.sh. Please specify correct conda path in the script."
    exit 1
fi

# Activate the conda environment
conda activate magma

# Print environment information for debugging
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo "Current directory: $(pwd)"

# Clean installation approach that leverages pip's dependency resolver
cd ../..

# First install the package with core dependencies (no flash-attn)
echo "Installing core package..."
pip install -e .

# Then install server dependencies
echo "Installing server dependencies..." 
pip install -e ".[server]" || echo "Some server dependencies couldn't be installed, but we can continue"

# Try to install advanced dependencies (including flash-attn) but don't fail if it doesn't work
echo "Attempting to install advanced dependencies..."
pip install -e ".[advanced]" || echo "Advanced dependencies couldn't be installed, but basic functionality will still work"

cd server

# Run the FastAPI application
python main.py

#!/bin/bash

set -e

echo "========================================"
echo "Digital Image Processing Full Setup"
echo "========================================"

# -----------------------------
# CONFIG
# -----------------------------
CONDA_DIR="$HOME/miniconda3"
ENV_NAME="image-processing-cv"

# -----------------------------
# INSTALL MINICONDA (if needed)
# -----------------------------
echo
echo "[1/4] Checking Conda..."

if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

    bash miniconda.sh -b -p "$CONDA_DIR"

    CONDA_PATH="$CONDA_DIR/bin/conda"
else
    echo "Conda already installed."
    CONDA_PATH="conda"
fi

# -----------------------------
# INITIALIZE CONDA IN SCRIPT
# -----------------------------
echo
echo "[2/4] Initializing Conda..."

eval "$($CONDA_DIR/bin/conda shell.bash hook)"

# -----------------------------
# CREATE ENVIRONMENT
# -----------------------------
echo
echo "[3/4] Creating environment..."
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda env create -f environment.yml || echo "Environment already exists, skipping."
rm miniconda.sh
# -----------------------------
# TEST INSTALLATION
# -----------------------------
echo
echo "[4/4] Testing installation..."

conda activate "$ENV_NAME"

python -c "import numpy, cv2; print('Environment OK')"

# -----------------------------
# DONE
# -----------------------------
echo
echo "========================================"
echo "Setup complete!"
echo "========================================"

echo
echo "To use the environment later run:"
echo "conda activate $ENV_NAME"

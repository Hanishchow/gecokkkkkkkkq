# ============================================================================
# GEOCK Docker - Binding Affinity Prediction System
# ============================================================================
# 
# Supported commands:
#   - predict <smiles>     : Predict affinity from SMILES only
#   - dock <pdb> <smiles> : Full docking with PDB + SMILES
#   - shell                : Interactive bash shell
#
# Examples:
#   docker run geock predict "Cc1ccc(cc1)C(=O)O"
#   docker run geock dock /data/6p87.pdb "COc1ccccc1OCC(=O)N(C)[C@@H](c1cccs1)c1c[nH]c2ccccc12"
# ============================================================================

FROM python:3.11-slim-bookworm

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

# Add conda to PATH
ENV PATH=$CONDA_DIR/bin:$PATH

# Create environment and install packages
RUN conda create -n geock python=3.11 -y && \
    conda clean -afy

# Activate environment and install RDKit (required - must be conda)
RUN conda activate geock && \
    conda install -c conda-forge rdkit=2024.3.0 -y && \
    conda clean -afy

# Install Python packages via pip
RUN conda activate geock && \
    pip install --no-cache-dir \
        numpy>=1.24 \
        pandas>=2.0 \
        scipy>=1.10 \
        scikit-learn>=1.3 \
        xgboost>=2.0 \
        torch>=2.0 \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
        MDAnalysis>=2.10 \
        click>=8.0 \
        rich>=13.0 \
        tqdm>=4.65

# Create app directory
WORKDIR /app

# Copy GEOCK2 physics code
COPY geock2/ /app/geock2/

# Copy trained models
COPY models/ /app/models/

# Copy CLI scripts
COPY cli.py /app/cli.py
COPY dock.py /app/dock.py

# Create data directory for PDB files
RUN mkdir -p /data

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH
ENV GEOCK_HOME=/app

# Default command
ENTRYPOINT ["python", "/app/cli.py"]
CMD ["--help"]

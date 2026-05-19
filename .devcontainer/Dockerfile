# syntax=docker/dockerfile:1.7

# Use NVIDIA PyTorch base image with CUDA support
FROM nvcr.io/nvidia/pytorch:26.01-py3

COPY .devcontainer/certs/*.crt /usr/local/share/ca-certificates/

# Install curl and CA certificates
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set Node.js to trust the additional CA certificates
# NOTE: Fixes vscode extensions
ENV NODE_EXTRA_CA_CERTS=/usr/local/share/ca-certificates/CDC-G2.crt

# Install third-party dependencies from project metadata. Source code is mounted
# by the devcontainer, so dependency layers only need packaging metadata.
WORKDIR /workspaces/CONCH

COPY requirements-extra.txt .

RUN pip install --no-cache-dir \
        --constraint /etc/pip/constraint.txt \
        -r requirements-extra.txt

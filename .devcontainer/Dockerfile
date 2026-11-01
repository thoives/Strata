# Developer container for Strata (Zed or any editor that reads .devcontainer/).
# CPU tools only: the C++ engine builds here without a GPU. For CUDA builds see .devcontainer/README.md.
# The repo is not copied in: it is bind-mounted by the editor, so edits show up on the host at once.
ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 PYTHONUNBUFFERED=1

# Compilers and libraries the engine and the Python tools use (the same runtime libraries as the Dockerfile at the repo root),
# plus Python 3.12 (Ubuntu 24.04's default) with venv support.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential pkg-config ca-certificates curl git sudo unzip less \
        libgomp1 libatomic1 libnuma-dev zlib1g-dev \
        python3 python3-dev python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*


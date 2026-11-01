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

# ubuntu:24.04 ships a user called "ubuntu" with UID 1000. Replace it with a developer user of the same UID, so files
# written into the bind-mounted repo belong to the same user on the host.
RUN if id -u ubuntu >/dev/null 2>&1; then userdel -r ubuntu; fi \
    && groupadd --gid "$USER_GID" "$USERNAME" \
    && useradd --uid "$USER_UID" --gid "$USER_GID" --create-home --shell /bin/bash "$USERNAME" \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > "/etc/sudoers.d/$USERNAME" \
    && chmod 0440 "/etc/sudoers.d/$USERNAME"

USER $USERNAME
WORKDIR /home/$USERNAME

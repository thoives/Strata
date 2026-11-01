#!/usr/bin/env bash
# Runs once when the container is created (devcontainer.json: postCreateCommand).
# Makes a Python venv for the Terminal and its tests, outside the repo, so it never clashes with the .venv that setup.sh makes on the host.
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
venv="$HOME/.venv-strata"

if [ ! -x "$venv/bin/python" ]; then
    python3 -m venv "$venv"
fi

cd "$repo"
"$venv/bin/python" -m pip install --upgrade pip
# The pinned runtime packages, the build helpers (cmake, ninja), the optional monitor, and pytest for the test suites.
"$venv/bin/python" -m pip install -r requirements.txt -r requirements-build.txt -r requirements-monitor.txt pytest

echo
echo "Strata dev container is ready."
echo "  Python tests:   python -m pytest tests/terminal"
echo "  Setup tests:    python tools/test_setup_choices.py"
echo "  Engine (CPU):   cmake -S . -B build -G Ninja -DSTRATA_ENABLE_CUDA=OFF && cmake --build build --target strata"

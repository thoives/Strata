# Strata dev container (for Zed)

A ready-to-use development environment for working on Strata's Terminal, its tests and the C++ engine, in a
container. Your code stays on your computer: the editor mounts the repo into the container.

## What you need

- Docker (Docker Engine on Linux, Docker Desktop on Mac or Windows).
- Zed, with its Dev Containers support.

## Open it in Zed

1. Open the Strata folder in Zed.
2. Open the dev container from Zed's remote menu (or the command palette: "Open Dev Container").
3. The first start builds the image and runs `.devcontainer/post-create.sh`. It makes a Python venv in
   `~/.venv-strata` inside the container and installs the pinned packages from `requirements.txt`,
   `requirements-build.txt` and `requirements-monitor.txt`, plus pytest. This takes a few minutes.

The venv lives in the container, not in the repo. A `.venv` that `setup.sh` makes on your computer is left alone.

## Check that it works

```sh
python -m pytest tests/terminal          # the Terminal and Control tests
python tools/test_setup_choices.py       # setup's own tests; no GPU or downloads needed
```

## Build the engine (CPU)

```sh
cmake -S . -B build -G Ninja -DSTRATA_ENABLE_CUDA=OFF
cmake --build build --target strata
```

This builds the engine without a GPU. It does not check that the engine runs on a card.

## Build for NVIDIA cards (CUDA)

The default image has no CUDA toolkit. To build CUDA targets, use an NVIDIA base image and give the container the GPU:

1. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
2. In `.devcontainer/devcontainer.json`, add a `build.args` entry and a `runArgs` entry:

   ```json
   "build": {
     "dockerfile": "Dockerfile",
     "context": ".",
     "args": { "BASE_IMAGE": "nvidia/cuda:13.0.0-devel-ubuntu24.04" }
   },
   "runArgs": ["--gpus", "all", "--ulimit", "memlock=-1:-1", "--shm-size", "2g"]
   ```

3. Rebuild the container. Then build with `-DSTRATA_ENABLE_CUDA=ON` and your card's architecture
   (for example `-DCMAKE_CUDA_ARCHITECTURES=89` for RTX 40).

Leave `runArgs` out on a computer with no NVIDIA card, or the container will not start. The image for a release build is
the one at the repo root (`Dockerfile`, see `docs/DOCKER.md`); this container is for editing and testing.

## Not included

- AMD (HIP) and Intel (SYCL) toolchains. See `docs/AMD_HIP.md` and `docs/INTEL.md`.
- Metal (macOS only). See `docs/MAC.md`.
- Model downloads. A model takes tens of GB; run `strata launch` on a computer with the card, or see `docs/TERMINAL_LAUNCH.md`.

# Hierarchical Tsetlin Machine (CUDA) playground

This repo is built around [`PyHierarchicalTsetlinMachineCUDA`](https://github.com/cair/PyHierarchicalTsetlinMachineCUDA),
a Hierarchical Tsetlin Machine implementation that **requires an NVIDIA
GPU** (it imports `pycuda` and opens a CUDA context as soon as the library
module is imported). Everything here is set up to make it as easy as
possible to: get a working environment via [pixi](https://pixi.sh), check
what GPU hardware is actually available, pick which GPU(s) to use, and run
a first, fast, well-commented example to learn the library's API.

## Quick start

```bash
git clone --recurse-submodules <this repo>
cd hierarchicalTM_chem
# If you cloned without --recurse-submodules:
git submodule update --init --recursive

pixi run fix-submodules   # one-time native build fix for dev/tmu
pixi install              # base environment (works on any machine)

pixi run describe-hardware   # what GPU(s), if any, are visible?
pixi run select-gpu 0        # pick a GPU (only meaningful with >1 GPU)

pixi install -e gpu               # CUDA-only: pulls in pycuda + the TM library
pixi run -e gpu describe-hardware # confirm the library actually imports
pixi run -e gpu test-htm          # tiny, fast, first training run
```

`pixi run <task>` uses the `default` environment unless you pass `-e gpu`.
Only the `gpu` environment can actually train a Tsetlin Machine; the
default environment is enough for browsing code, running unit tests, and
GPU discovery/selection on any machine (e.g. a laptop with no GPU at all).

## Devcontainer

A devcontainer is provided for VS Code / Dev Containers:

```bash
chmod +x .devcontainer/build
./.devcontainer/build       # generates .devcontainer/devcontainer.json
```

Then "Reopen in Container". The build script detects an NVIDIA GPU on the
host and, if found, starts the container with `--gpus=all` (requires the
NVIDIA Container Toolkit on the host); `postStartCommand` then runs
`pixi install`, and additionally `pixi install -e gpu` only if `nvidia-smi`
is reachable *inside* the container. On a non-GPU machine you still get a
fully working container for everything except actually training a TM.

## Hardware assessment & GPU selection

`pixi run describe-hardware` (see `src/htm_env/hardware.py`) reports:
- every GPU `nvidia-smi` sees (name, free/total memory, utilization),
- any GPU selection you've persisted (see below),
- whether `PyHierarchicalTsetlinMachineCUDA` actually imports (this is the
  only way to know for sure the pycuda/driver/toolkit combination works).

If a machine has multiple GPUs (e.g. a shared server), pin which one(s)
`pixi run` should use:

```bash
pixi run select-gpu          # list GPUs + show current selection
pixi run select-gpu 0        # pin GPU 0
pixi run select-gpu 0,1      # pin GPUs 0 and 1
pixi run select-gpu --clear  # go back to "all GPUs visible"
```

This writes a small, gitignored `.gpu-device` file at the project root.
The `gpu` pixi environment's activation script (`scripts/activate_gpu.sh`)
reads it and exports `CUDA_VISIBLE_DEVICES` automatically on every
subsequent `pixi run -e gpu ...` / `pixi shell -e gpu`.

## Learning / testing the library

`examples/quickstart_htm.py` (run via `pixi run -e gpu test-htm`) trains a
plain (non-hierarchical) `TsetlinMachine` on a tiny noisy-XOR toy dataset
in a few seconds -- a fast way to confirm the environment works and to see
the library's basic `fit`/`predict`/`score` API. It's heavily commented;
read it alongside `dev/PyHierarchicalTsetlinMachineCUDA/tm.py` and that
submodule's own `examples/` directory (e.g. `MNISTHierarchicalDemo.py`) to
explore the hierarchical grouping API further.

## Tests

```bash
pixi run test
```

Runs `pytest` over `tests/`. Tests marked `@pytest.mark.gpu` are
auto-skipped unless `PyHierarchicalTsetlinMachineCUDA` actually imports
(see `tests/conftest.py`), so `pixi run test` is safe on any machine.

## Repo layout

- `src/htm_env/` -- GPU discovery/selection helpers (`hardware.py`).
- `examples/` -- small, standalone scripts for learning the TM library.
- `scripts/` -- one-time setup (`patch_native_deps.py`), the GPU picker
  CLI (`select_gpu.py`), and the pixi `gpu` environment's activation hook
  (`activate_gpu.sh`).
- `dev/` -- git submodules: `PyHierarchicalTsetlinMachineCUDA` (the CUDA
  TM library itself), `tmu` (CPU-only TM reference implementation, kept
  for comparison), `ChEMBL_Structure_Pipeline` (SMILES standardization).
- `data/` -- example datasets (`MOR.csv`, precomputed descriptor arrays).
- `tests/` -- hardware/environment sanity tests.

### Installing additional packages

```bash
pixi add <package-name>          # available on conda-forge
pixi add --pypi <package-name>   # pip-only package
```

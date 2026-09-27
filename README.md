# Hierarchical Tsetlin Machine (CUDA) playground

This repo is built around [`PyHierarchicalTsetlinMachineCUDA`](https://github.com/cair/PyHierarchicalTsetlinMachineCUDA),
a Hierarchical Tsetlin Machine implementation that **requires an NVIDIA
GPU** (it imports `pycuda` and opens a CUDA context as soon as the library
module is imported). Everything here is set up to make it as easy as
possible to: get a working CUDA environment via [pixi](https://pixi.sh), check
what GPU hardware is actually available, pick which GPU(s) to use, and run
a first, fast, well-commented example to learn the library's API.

## Quick start

```bash
git clone --recurse-submodules <this repo>
cd hierarchicalTM_chem
# If you cloned without --recurse-submodules:
git submodule update --init --recursive

pixi run fix-submodules   # one-time native build fix for dev/tmu
pixi install              # installs the CUDA backend in the default environment

pixi run describe-hardware   # what GPU(s), if any, are visible?
pixi run select-gpu 0        # pick a GPU (only meaningful with >1 GPU)

pixi run describe-hardware # confirm the library actually imports
pixi run test-htm          # tiny, fast, first training run
```

This project targets Linux with an NVIDIA GPU and CUDA toolkit. The default
Pixi environment includes `pycuda` and `PyHierarchicalTsetlinMachineCUDA`,
so `pixi run` and `pixi shell` both use the CUDA backend. In VS Code notebooks,
select `.pixi/envs/default/bin/python` as the kernel.

## Devcontainer

A devcontainer is provided for VS Code / Dev Containers:

```bash
chmod +x .devcontainer/build
./.devcontainer/build       # generates .devcontainer/devcontainer.json
```

Then "Reopen in Container". Docker starts the container with `--gpus=all`
(requires an NVIDIA GPU and the NVIDIA Container Toolkit on the host), and
`postStartCommand` runs `pixi install` for the single environment.

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
The Pixi activation script (`scripts/activate_gpu.sh`)
reads it and exports `CUDA_VISIBLE_DEVICES` automatically on every
subsequent `pixi run ...` / `pixi shell`.

## Learning / testing the library

`examples/quickstart_htm.py` (run via `pixi run test-htm`) trains a
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

Runs `pytest` over `tests/`. Tests marked `@pytest.mark.gpu` require a
working CUDA backend.

## Repo layout

- `src/htm_env/` -- GPU discovery/selection helpers (`hardware.py`).
- `examples/` -- small, standalone scripts for learning the TM library.
- `scripts/` -- one-time setup (`patch_native_deps.py`), the GPU picker
  CLI (`select_gpu.py`), and Pixi's activation hook
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

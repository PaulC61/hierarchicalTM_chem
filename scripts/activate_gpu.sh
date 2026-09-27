#!/usr/bin/env bash
# Activation hook for the default Pixi environment.
#
# Reads a gitignored `.gpu-device` file (written by `pixi run select-gpu`) at
# the project root and exports CUDA_VISIBLE_DEVICES accordingly, so every
# `pixi run` / `pixi shell` automatically uses whichever GPU(s) were selected.
# If no selection has been made yet, all GPUs remain visible.

GPU_FILE="${PIXI_PROJECT_ROOT:-$(pwd)}/.gpu-device"

if [ -f "$GPU_FILE" ]; then
    export CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES="$(tr -d '[:space:]' < "$GPU_FILE")"
    echo "[htm] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (from .gpu-device)"
else
    echo "[htm] No GPU selection found -- all detected GPUs are visible."
    echo "[htm] Run 'pixi run select-gpu <index>' to pin one."
fi

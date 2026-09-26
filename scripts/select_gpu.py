#!/usr/bin/env python
"""Interactively (or non-interactively) pick which GPU(s) `pixi run` should use.

Usage:
    pixi run select-gpu            # list detected GPUs, show current selection
    pixi run select-gpu 0          # pin CUDA_VISIBLE_DEVICES=0 for future runs
    pixi run select-gpu 0,1        # pin multiple GPUs
    pixi run select-gpu --clear    # remove the pinned selection (use all GPUs)

The selection is written to a gitignored `.gpu-device` file at the project
root. The pixi `gpu` environment's activation script reads that file and
exports `CUDA_VISIBLE_DEVICES` accordingly on every `pixi run` / `pixi shell`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from htm_env import hardware  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "devices",
        nargs="?",
        help="GPU index or comma-separated indices to pin, e.g. '0' or '0,1'.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Remove the persisted selection (fall back to all GPUs visible).",
    )
    args = parser.parse_args(argv)

    gpus = hardware.list_gpus()
    if not gpus:
        print("No NVIDIA GPUs detected (nvidia-smi missing, or found none).")
        if not args.clear:
            return 1

    if args.clear:
        hardware.clear_gpu_selection()
        print("Cleared GPU selection -- all detected GPUs will be visible.")
        return 0

    if gpus:
        print(f"Detected {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"  {gpu.summary()}")

    if args.devices is None:
        current = hardware.read_gpu_selection()
        print(f"\nCurrent selection: {current or '(none -- all GPUs visible)'}")
        print("Pass an index to pin one, e.g.: pixi run select-gpu 0")
        return 0

    valid_indices = {gpu.index for gpu in gpus}
    requested = [d.strip() for d in args.devices.split(",") if d.strip()]
    for d in requested:
        if not d.isdigit() or int(d) not in valid_indices:
            print(
                f"error: GPU index '{d}' not found. "
                f"Available indices: {sorted(valid_indices) or 'none'}"
            )
            return 1

    path = hardware.write_gpu_selection(",".join(requested))
    print(f"\nPersisted CUDA_VISIBLE_DEVICES={','.join(requested)} to {path}")
    print("Run `pixi run describe-hardware` to confirm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Auto-skip @pytest.mark.gpu tests unless a working CUDA TM backend is present.

This makes `pixi run test` safe to run on any machine (a laptop with no
GPU, a CI runner, or the actual CUDA server): GPU-only tests are collected
but skipped everywhere except where PyHierarchicalTsetlinMachineCUDA can
actually be imported.
"""

from __future__ import annotations

import pytest

from htm_env import hardware

_GPU_AVAILABLE, _GPU_DETAIL = hardware.gpu_backend_available()


def pytest_collection_modifyitems(config, items):
    if _GPU_AVAILABLE:
        return
    skip_gpu = pytest.mark.skip(reason=f"no working CUDA TM backend: {_GPU_DETAIL}")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)

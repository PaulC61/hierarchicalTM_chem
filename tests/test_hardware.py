"""Sanity tests for GPU hardware assessment -- run on any machine, no GPU required.

These verify `htm_env.hardware` degrades gracefully (returns empty/False,
never raises) when there is no NVIDIA GPU, and behaves correctly when
`nvidia-smi` output is present. Anything that needs a live CUDA GPU is
marked `@pytest.mark.gpu` and skipped by default.
"""

from __future__ import annotations

import subprocess

import pytest

from htm_env import hardware


def test_list_gpus_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(hardware.shutil, "which", lambda _: None)
    assert hardware.list_gpus() == []
    assert hardware.has_nvidia_gpu() is False


def test_list_gpus_parses_nvidia_smi_output(monkeypatch):
    monkeypatch.setattr(hardware, "_nvidia_smi_available", lambda: True)

    fake_stdout = "0, NVIDIA A100, 40960, 1024, 39936, 5\n"

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout=fake_stdout, stderr="")

    monkeypatch.setattr(hardware.subprocess, "run", fake_run)

    gpus = hardware.list_gpus()
    assert len(gpus) == 1
    assert gpus[0].index == 0
    assert gpus[0].name == "NVIDIA A100"
    assert gpus[0].memory_free_mib == 39936
    assert hardware.has_nvidia_gpu() is True


def test_gpu_selection_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(hardware, "GPU_SELECTION_FILE", tmp_path / ".gpu-device")
    assert hardware.read_gpu_selection() is None

    hardware.write_gpu_selection("0,1")
    assert hardware.read_gpu_selection() == "0,1"

    hardware.clear_gpu_selection()
    assert hardware.read_gpu_selection() is None


def test_gpu_backend_available_never_raises():
    # No GPU/pycuda in this test environment: should report unavailable,
    # not raise (the import itself may be missing or fail for many reasons).
    ok, detail = hardware.gpu_backend_available()
    assert isinstance(ok, bool)
    assert isinstance(detail, str)


def test_describe_does_not_raise():
    # Should produce a readable report on any machine, GPU or not.
    report = hardware.describe(check_import=False)
    assert "hardware report" in report.lower()


@pytest.mark.gpu
def test_gpu_backend_actually_available():
    """Only meaningful on a real CUDA machine with `pixi install -e gpu`."""
    ok, detail = hardware.gpu_backend_available()
    assert ok, detail

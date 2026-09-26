"""GPU discovery, selection, and readiness checks.

`PyHierarchicalTsetlinMachineCUDA` is CUDA-only: importing its `tm` module
opens a CUDA context immediately (`import pycuda.autoprimaryctx`). That means
hardware assessment must be done *before* importing the library, by shelling
out to `nvidia-smi`, and any actual import check must be done defensively.

This module provides:
- `list_gpus()`      -- structured info for every NVIDIA GPU nvidia-smi sees.
- `has_nvidia_gpu()` -- quick boolean probe.
- `gpu_backend_available()` -- attempts the real (context-creating) import.
- `describe()`       -- human-readable summary for `pixi run describe-hardware`.
- `read_gpu_selection()` / `write_gpu_selection()` -- persist which GPU
  index/indices `pixi run` should restrict itself to, via a small gitignored
  `.gpu-device` file at the project root (consumed by the pixi `gpu`
  environment's activation script to set `CUDA_VISIBLE_DEVICES`).
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GPU_SELECTION_FILE = PROJECT_ROOT / ".gpu-device"

_NVIDIA_SMI_FIELDS = (
    "index",
    "name",
    "memory.total",
    "memory.used",
    "memory.free",
    "utilization.gpu",
)


@dataclass
class GpuInfo:
    index: int
    name: str
    memory_total_mib: int
    memory_used_mib: int
    memory_free_mib: int
    utilization_pct: int

    def summary(self) -> str:
        return (
            f"[{self.index}] {self.name} - "
            f"{self.memory_free_mib}/{self.memory_total_mib} MiB free, "
            f"{self.utilization_pct}% util"
        )


def _nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def list_gpus() -> list[GpuInfo]:
    """Return structured info for every GPU `nvidia-smi` can see.

    Returns an empty list if `nvidia-smi` is missing or the call fails
    (no NVIDIA driver, no GPU, or running outside a container with GPU
    passthrough) rather than raising.
    """
    if not _nvidia_smi_available():
        return []
    query = ",".join(_NVIDIA_SMI_FIELDS)
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return []

    gpus: list[GpuInfo] = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(_NVIDIA_SMI_FIELDS):
            continue
        try:
            gpus.append(
                GpuInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    memory_total_mib=int(float(parts[2])),
                    memory_used_mib=int(float(parts[3])),
                    memory_free_mib=int(float(parts[4])),
                    utilization_pct=int(float(parts[5])),
                )
            )
        except ValueError:
            continue
    return gpus


def has_nvidia_gpu() -> bool:
    """Cheap check: is there at least one NVIDIA GPU visible to nvidia-smi?"""
    return len(list_gpus()) > 0


def gpu_backend_available() -> tuple[bool, str]:
    """Attempt the real, context-creating import of the CUDA TM backend.

    This is the only way to be sure the library will actually work (right
    driver/toolkit/pycuda combination), but it is expensive and creates a
    CUDA context as a side effect, so call it sparingly (e.g. once, in
    `describe()` or a smoke test) rather than on every pipeline step.
    """
    try:
        import PyHierarchicalTsetlinMachineCUDA.tm  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on host hardware
        return False, f"{type(exc).__name__}: {exc}"
    return True, "PyHierarchicalTsetlinMachineCUDA.tm imported successfully"


def read_gpu_selection() -> str | None:
    """Read the persisted GPU selection (contents of `.gpu-device`), if any."""
    if GPU_SELECTION_FILE.exists():
        value = GPU_SELECTION_FILE.read_text().strip()
        return value or None
    return None


def write_gpu_selection(devices: str) -> Path:
    """Persist a GPU selection (e.g. "0" or "0,2") for future `pixi run` calls.

    Written to a gitignored `.gpu-device` file at the project root, which the
    pixi `gpu` environment's activation script reads to export
    `CUDA_VISIBLE_DEVICES` on every subsequent `pixi run` / `pixi shell`.
    """
    GPU_SELECTION_FILE.write_text(devices.strip() + "\n")
    return GPU_SELECTION_FILE


def clear_gpu_selection() -> None:
    """Remove the persisted GPU selection (fall back to all GPUs visible)."""
    GPU_SELECTION_FILE.unlink(missing_ok=True)


def describe(check_import: bool = True) -> str:
    """Human-readable hardware summary for `pixi run describe-hardware`."""
    lines = ["Hierarchical Tsetlin Machine (CUDA) - hardware report", "=" * 55]

    if not _nvidia_smi_available():
        lines.append("nvidia-smi: NOT FOUND")
        lines.append(
            "  -> No NVIDIA driver detected. This project requires an "
            "NVIDIA GPU with drivers installed to run "
            "PyHierarchicalTsetlinMachineCUDA."
        )
        return "\n".join(lines)

    gpus = list_gpus()
    if not gpus:
        lines.append("nvidia-smi: found, but reported no GPUs.")
        return "\n".join(lines)

    lines.append(f"Detected {len(gpus)} GPU(s):")
    for gpu in gpus:
        lines.append(f"  {gpu.summary()}")

    selection = read_gpu_selection()
    if selection:
        lines.append(f"\nPersisted selection (.gpu-device): CUDA_VISIBLE_DEVICES={selection}")
    else:
        lines.append(
            "\nNo GPU selection persisted -- all detected GPUs are visible. "
            "Run `pixi run select-gpu <index>` to pin one, e.g. "
            "`pixi run select-gpu 0`."
        )

    if check_import:
        ok, detail = gpu_backend_available()
        status = "OK" if ok else "UNAVAILABLE"
        lines.append(f"\nPyHierarchicalTsetlinMachineCUDA import: {status}")
        lines.append(f"  {detail}")
        if not ok:
            lines.append(
                "  -> Run `pixi install -e gpu` (Linux + NVIDIA GPU required) "
                "to install pycuda + the CUDA TM library."
            )

    return "\n".join(lines)


def main() -> None:  # pragma: no cover - thin CLI wrapper
    print(describe())


if __name__ == "__main__":  # pragma: no cover
    main()

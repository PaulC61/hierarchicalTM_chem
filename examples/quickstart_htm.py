#!/usr/bin/env python
"""A minimal, heavily-commented first run of PyHierarchicalTsetlinMachineCUDA.

Purpose: give you something small and fast to run on a CUDA GPU to confirm
the environment works end-to-end, and to see the library's basic API in
action -- not a benchmark, just a "hello world".

Task: learn a noisy XOR-like pattern over a handful of boolean features.
This is the same toy problem the upstream library's own examples use
(see `dev/PyHierarchicalTsetlinMachineCUDA/examples/` once the submodule is
checked out), reproduced here in miniature.

Run it with:
    pixi run test-htm

Requires an NVIDIA GPU + `pixi install` (see `pixi run describe-hardware`
to check your setup).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from htm_env import hardware  # noqa: E402


def make_noisy_xor(n_samples: int, n_features: int, noise: float, seed: int = 42):
    """Boolean features; label = XOR of the first two features, with label noise."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=(n_samples, n_features)).astype(np.uint32)
    y = np.logical_xor(x[:, 0], x[:, 1]).astype(np.uint32)
    flip = rng.random(n_samples) < noise
    y[flip] = 1 - y[flip]
    return x, y


def main() -> int:
    ok, detail = hardware.gpu_backend_available()
    if not ok:
        print("PyHierarchicalTsetlinMachineCUDA is not usable in this environment:")
        print(f"  {detail}")
        print("\nRun `pixi run describe-hardware` for a full diagnostic.")
        return 1

    # Imported lazily and only after confirming it works above -- importing
    # this module creates a CUDA context as a side effect.
    from PyHierarchicalTsetlinMachineCUDA.tm import TsetlinMachine

    print("Generating a tiny noisy-XOR dataset (12 boolean features)...")
    x_train, y_train = make_noisy_xor(n_samples=5000, n_features=12, noise=0.05)
    x_test, y_test = make_noisy_xor(n_samples=1000, n_features=12, noise=0.0, seed=7)

    print("Constructing a (flat, non-hierarchical) TsetlinMachine...")
    # `hierarchy_structure=((0, 1),)` means: one group ("AND_GROUP" = 0),
    # depth 1 -- i.e. a plain, non-hierarchical Tsetlin Machine. See
    # `dev/PyHierarchicalTsetlinMachineCUDA/tm.py` for the full hierarchical
    # grouping API once you're ready to explore multi-level structures.
    tm = TsetlinMachine(
        number_of_clauses=100,
        T=15,
        s=3.9,
        hierarchy_structure=((0, 1),),
    )

    print("Training for 20 epochs (this is intentionally tiny/fast)...")
    start = time.time()
    tm.fit(x_train, y_train, epochs=20)
    elapsed = time.time() - start

    predictions = tm.predict(x_test)
    accuracy = float(np.mean(predictions == y_test))

    print(f"\nDone in {elapsed:.1f}s.")
    print(f"Test accuracy: {accuracy:.3f} (expect close to 1.0 on this clean test set)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

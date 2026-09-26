"""Small environment/hardware-assessment helpers for this project.

`PyHierarchicalTsetlinMachineCUDA` (the Hierarchical Tsetlin Machine library
this project is built around) only runs on an NVIDIA GPU: its `tm` module
unconditionally imports `pycuda` and opens a CUDA context on import. This
package answers "what GPU(s) does this machine have, and which one(s) should
`pixi run` use?" -- see `htm_env.hardware`.
"""

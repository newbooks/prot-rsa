# Implementation Specifications

The documents in this directory are implementation contracts for coding
agents. An implementation must satisfy the normative requirements and
acceptance criteria in the relevant specification. Broader design discussion
and future work remain in [`../implementation-plan.md`](../implementation-plan.md).

## Specifications

- [`command-line-interface.md`](command-line-interface.md): command options,
  supported input filenames, output-name derivation, help text, and CLI tests.
- [`constants_decision.md`](constants_decision.md): solvent probe, atomic-radius
  modes, Fe/unknown fallbacks, and loose hetero-component classifications.
- [`atom-sasa-naive.md`](atom-sasa-naive.md): deterministic 960-point
  Shrake–Rupley reference calculation of absolute per-atom SASA.
- [`atom-sasa-spatial.md`](atom-sasa-spatial.md): conservative `cKDTree`
  neighbor pruning for the first optimized atom-SASA production path.
- [`atom-sasa-vectorized-mask.md`](atom-sasa-vectorized-mask.md): bounded
  NumPy exposed-point masks over CSR neighbors without blocking a later Numba
  kernel.
- [`occlusion-ordered-neighbors.md`](occlusion-ordered-neighbors.md):
  deterministic largest-cap-first CSR traversal shared by NumPy and future
  compiled kernels.
- [`cache.md`](cache.md): invariant squared-radius and coordinate-component
  reuse inside the NumPy kernel without changing its future Numba boundary.

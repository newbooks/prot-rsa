# Complete-Burial Mask Specification

**Status:** Implemented

The implementation and initial synthetic benchmark results are recorded in
the repository README. Canonical-structure benchmarking remains part of the
broader release benchmark suite.

## Purpose and scope

Identify atoms whose entire probe-expanded sphere is enclosed by a larger
probe-expanded sphere before the surface-sampling kernel runs. Such atoms have
zero accessible surface area and need not construct or test sphere points.

This is a local optimization of the existing spatial/CSR pipeline. It must
preserve the full atom array, atom indices, CSR rows, output order, and public
API. The mask is metadata consumed by NumPy, Numba, and multiprocessing; it is
not a filtering operation that removes atoms from the
calculation.

This stage includes:

- deterministic complete-burial detection from validated coordinates, expanded
  radii, and existing CSR neighbors;
- a fixed-shape Boolean mask with one entry per input atom;
- zero-area handling in the numerical kernel; and
- focused boundary, unequal-radius, and output-equivalence tests.

The following are out of scope:

- changing KD-tree neighbor discovery or CSR membership/order;
- removing atoms or compacting coordinates and radii;
- changing the reference implementation;
- partial-occlusion classification;
- Numba, multiprocessing, or threading-specific code; and
- approximate geometry or reduced precision.

## Geometric contract

Let `R_i` and `R_j` be probe-expanded radii and `d(i,j)` the center distance.
Atom `i` is completely buried by neighbor `j` exactly when:

```text
d(i, j) + R_i <= R_j
```

The comparison is inclusive: internal tangency leaves no exposed surface and
must be marked buried. A larger sphere that only partially overlaps, touches
externally, or lies inside the target sphere must not mark the target buried.

The mask is:

```text
buried[i] = any(d(i, j) + R_i <= R_j for j in CSR_row(i))
```

Only existing CSR neighbors may be considered. The CSR builder remains
responsible for conservative pair discovery and boundary retention; the burial
stage must not search all atom pairs or build a second spatial index.

## Private interface

Add a private helper with this data-only contract (the exact name is not
normative):

```python
def _build_buried_mask(
    atom_coordinates: np.ndarray,
    expanded_radii: np.ndarray,
    neighbor_offsets: np.ndarray,
    neighbor_indices: np.ndarray,
) -> np.ndarray:
    """Return one complete-burial flag per atom."""
```

Inputs have already passed the shared validation path and must be contiguous
`float64` coordinates/radii and contiguous `np.intp` CSR arrays. The helper
must return a C-contiguous Boolean array of shape `(N,)`, in input atom order.
For empty or single-atom input, return all-false without distance work. Do not
mutate any input.

The helper must not read files, inspect atom records, call callbacks, or depend
on a backend-specific object. Its result must be reusable by every numerical
backend.

## Detection algorithm

For each target atom `i`:

1. Initialize `buried[i] = False`.
2. Traverse the existing CSR row from `neighbor_offsets[i]` through
   `neighbor_offsets[i + 1]` in its deterministic order.
3. For each neighbor `j`, calculate the center distance using the established
   `float64` coordinate arithmetic. Accumulate squared displacement in x, then
   y, then z; take one square root to obtain `d(i,j)`.
4. If `d(i,j) + expanded_radii[i] <= expanded_radii[j]`, set the flag and stop
   scanning that row.

The implementation may use an equivalent scalar formulation only when focused
boundary tests demonstrate the same classifications. Do not use a tolerance,
`isclose`, strict `<`, radius-only heuristics, or the occlusion ordering score
as a substitute for the specified enclosure condition. The ordering score may
put likely enclosing neighbors first, but it does not define burial semantics.

## Kernel integration

`atom_sasa_spatial()` remains responsible for validation, expanded-radius
construction, CSR construction, and mask construction. The existing numerical
kernel gains a private burial-mask input or an equivalent wrapper-level skip;
the public function signature and full `(N,)` output remain unchanged.

For each atom marked buried, the numerical kernel must write exactly `0.0` and
skip sphere-point construction and neighbor testing. Non-buried atoms retain
the current NumPy algorithm and arithmetic. If every atom is buried, the call
may return a zero array without allocating point-work buffers.

The reference implementation must not consume this mask. It remains the
unoptimized correctness oracle.

## Numerical behavior and invariants

The optimization must preserve the existing scientific result:

- a buried atom has zero SASA;
- an enclosing atom is evaluated normally;
- partial overlaps retain their sampled exposed-point result;
- external tangency does not imply burial;
- atom order and output shape remain unchanged;
- repeated calls are deterministic; and
- caller-owned arrays remain unchanged.

The optimized spatial output must remain numerically identical to the current
spatial output for all existing focused tests. The mask itself must match the
geometric condition for deliberately constructed boundary cases.

## Required tests

Add focused `pytest` coverage for:

1. empty and single-atom inputs;
2. a strictly enclosed atom;
3. internal tangency (`d + R_i == R_j`), which must be marked buried;
4. external tangency, which must not mark either atom buried;
5. partial overlap, which must not mark the target buried;
6. a smaller sphere inside a larger target, which must not mark the target
   buried;
7. unequal radii, multiple neighbors, and deterministic tie/order cases;
8. roundoff-adjacent boundaries using the repository’s established `float64`
   arithmetic;
9. equivalence of masked spatial output with the current unmasked spatial
   output and the reference result;
10. zero-area output for buried atoms while enclosing atoms remain nonzero;
11. custom sphere-point arrays, translation, permutation, repeated calls, and
    input non-mutation; and
12. all existing CSR and spatial tests remaining green.

Tests must not depend on private allocation identities or timing ratios.

## Benchmark requirements

Benchmark the spatial calculation with and without the burial shortcut using
identical warmed-up inputs. Include canonical structures, dense synthetic
structures with many enclosed atoms, sparse structures with no enclosed atoms,
and custom sphere-point counts.

Record atom count, buried-atom count, directed-neighbor count, mask-build time,
sampling-kernel time, total time, maximum absolute difference, MAE, and total
SASA difference. The optimization gate is no numerical difference from the
existing spatial output and no material regression on sparse or small inputs.
Report absolute timings when workload duration is below the repository’s
benchmark noise threshold.

## Acceptance criteria

This stage is complete when:

- the mask is computed only from validated arrays and existing CSR rows;
- no atoms are removed or reordered;
- complete burial uses the inclusive geometric condition;
- buried targets skip point construction and receive zero area;
- non-buried calculations retain existing numerical behavior;
- all focused and full tests pass; and
- benchmark conditions and results are documented before enabling the shortcut
  in production dispatch.

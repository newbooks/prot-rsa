# Spatially Pruned Atom SASA Specification

**Status:** Implemented

## Purpose and scope

Implement the first production optimization of the Shrake–Rupley atom-SASA
calculation. Use a three-dimensional spatial index to identify atoms whose
probe-expanded spheres can overlap, then test each surface sample only against
that atom's conservatively filtered neighbor list.

This stage is deliberately limited to spatial pruning. It must preserve the
calculation, validation, output, and deterministic behavior defined by
[`atom-sasa-naive.md`](atom-sasa-naive.md). The naive
`atom_sasa_reference()` function remains unchanged as the correctness oracle.

The following are out of scope for this stage:

- Numba or another JIT compiler;
- multiprocessing, threading, or GPU execution;
- reduced precision or approximate spatial queries;
- complete-burial shortcuts;
- neighbor reordering based on blocking likelihood;
- spherical-cap or exposed-mask kernels; and
- residue-SASA calculation.

These optimizations may be evaluated after the isolated benefit of spatial
pruning has been measured.

## Design rationale and legacy lessons

The previous PyMCCE implementation established the useful high-level pattern:
build a spatial tree, find nearby atoms, and avoid testing every surface point
against the entire structure. This implementation retains that idea but uses
SciPy's maintained `scipy.spatial.cKDTree` and a pair-specific overlap filter
that is conservative at floating-point boundaries.

Two legacy hazards must not be carried forward:

1. A search cutoff based on only one atom's radius, or only one copy of the
   probe radius, can omit a valid occluder. Neighbor eligibility depends on
   the sum of both probe-expanded radii.
2. A full `(sphere points, neighbors, 3)` temporary can make memory use grow
   unnecessarily. The initial spatial implementation retains the reference
   point-first traversal and its early exit when a point is blocked.

The spatial index is a conservative discovery mechanism only. Scientific
decisions are made by exact `float64` distance comparisons after discovery.

## Public numerical interface

Add the following reusable function to `protrsa.py`:

```python
def atom_sasa_spatial(
    coordinates,
    radii,
    *,
    probe_size: float = DEFAULT_PROBE_SIZE,
    sphere_points: np.ndarray | None = None,
) -> np.ndarray:
    """Return atom SASA using KD-tree-pruned neighbor lists."""
```

Its inputs, validation errors, units, return shape, return dtype, lack of
caller-owned mutation, and boundary behavior must match
`atom_sasa_reference()` exactly. Common input normalization and validation
should be factored into private helpers rather than maintained as two
independent contracts. This refactoring must not change the reference
algorithm or its public behavior.

After the benchmark gate in this specification is met,
`calculate_atom_sasa()` becomes the production dispatcher and calls
`atom_sasa_spatial()`. Until then, it continues to call
`atom_sasa_reference()`. `atom_sasa_reference()` remains directly importable
for verification and tests. No import-time tree construction, calculation,
worker creation, file access, or other side effect is permitted.

The neighbor-list builder is an internal implementation detail for this stage.
It may use a private helper returning compressed sparse row (CSR) arrays:

```text
offsets: shape (N + 1,), dtype np.intp
indices: shape (number_of_directed_neighbors,), dtype np.intp

neighbors(i) = indices[offsets[i]:offsets[i + 1]]
```

CSR is preferred over Python lists because it provides bounded contiguous
storage suitable for later compiled and parallel kernels. It does not become
public API in this stage.

`offsets[0]` must be zero, `offsets` must be nondecreasing, and
`offsets[-1] == len(indices)`. Empty and single-atom inputs therefore return
`offsets == [0]` and `offsets == [0, 0]`, respectively, with an empty
`indices` array.

## Required occluder candidates

For every atom `i`, define its probe-expanded radius:

```text
R[i] = radii[i] + probe_size
```

Distinct atoms `i` and `j` can occlude one another only if their closed
expanded spheres overlap or touch:

```text
squared_distance(coordinates[i], coordinates[j]) <= (R[i] + R[j])**2
```

Equality is included. Although two externally tangent spheres usually affect
at most one sampled point, excluding equality would contradict the reference
rule that a sample exactly on an occluder is blocked. The stored neighbor list
must be a conservative superset of this mathematical relation: harmless
near-boundary false positives are permitted, but false negatives are not.

The neighbor relationship must be symmetric, exclude self-indices, and store
each atom's neighbor indices in increasing input-index order. Sorting restores
the reference function's occluder traversal order and makes construction
deterministic regardless of the order returned by SciPy.

## KD-tree discovery

Construct one `scipy.spatial.cKDTree` from the validated, C-contiguous
`float64` coordinate array. Use exact Euclidean queries with `eps=0`; do not
request approximate neighbors.

The recommended initial implementation is:

1. Calculate `maximum_radius = max(R)` after the empty-input case has returned.
2. Set the global query cutoff to
   `np.nextafter(2 * maximum_radius, np.inf)` and call
   `query_pairs(cutoff, eps=0, output_type="ndarray")` to discover each
   conservative unordered pair once. Normalize SciPy's empty result to shape
   `(0, 2)` with dtype `np.intp`.
3. Calculate pairwise squared center distances for the returned pairs.
4. Retain pairs satisfying the pair-specific criterion above. Also retain a
   pair when its squared distance exceeds the computed squared cutoff only by
   ordinary `float64` roundoff. A suitable conservative test is exact `<=`
   followed by `np.isclose(..., rtol=8*np.finfo(np.float64).eps, atol=0.0)`.
5. Expand every retained unordered pair `(i, j)` into both CSR directions and
   sort each row by atom index.

`query_ball_point()` with per-atom conservative radii is an acceptable
alternative only if benchmarks show a material advantage and the same
conservative filter, symmetry, ordering, and tests are retained. Tree
construction and queries must remain serial in this stage so that spatial
pruning is measured independently from parallel execution.

The broad KD-tree cutoff and roundoff allowance are intentionally
conservative. No pair may be excluded merely because its individual radii are
smaller than the maximum or because a boundary calculation rounded outward.
Pair-specific filtering removes material false positives; microscopic
near-boundary false positives are harmless because the final sample-occlusion
comparison is unchanged.

## Occlusion calculation

For target atom `i`, iterate over the same sphere points in the same order as
the reference. Construct each Cartesian sample point using the same `float64`
operations:

```text
sample = coordinates[i] + R[i] * sphere_point
```

Test the sample only against `neighbors(i)`. A sample is blocked when:

```text
squared_distance(sample, coordinates[j]) <= R[j]**2
```

Stop at the first blocker. Calculate absolute SASA using the same exposed
count and actual sphere-point denominator as the reference. For an atom
without neighbors, set `exposed_count = point_count` without iterating over
the sphere points, then execute the same final arithmetic expression as the
reference:

```text
4 * pi * R[i]**2 * exposed_count / point_count
```

This no-neighbor shortcut is exact and does not require surface sampling.
Computing only the algebraically simplified `4 * pi * R[i]**2` expression is
not permitted because its `float64` result can differ by one or more bits from
the reference's multiply-then-divide operation order.

Do not allocate a temporary proportional to `P * K * 3`, where `P` is the
sphere-point count and `K` is the target's neighbor count. The implementation
may allocate the KD-tree pair array, pair-filter working arrays proportional
to the number of conservative pairs, CSR storage proportional to the number
of retained directed neighbors, and constant-size working values in the
point-first loop.

## Determinism and numerical equivalence

For the same validated inputs, `atom_sasa_spatial()` must produce the same
exposed-point count for every atom as `atom_sasa_reference()`. Because this
stage retains the reference sample construction and ascending occluder order,
tests should require bitwise-identical output with `np.array_equal`, not merely
an approximate tolerance.

The following invariants remain unchanged:

- sample points exactly on an expanded sphere are blocked;
- atom order determines output order only, not the physical result;
- translating all coordinates does not change SASA except for ordinary
  `float64` effects already permitted by the reference contract;
- caller-owned inputs are not mutated; and
- results are deterministic across repeated calls in one runtime.

If bitwise equivalence cannot be maintained on a supported platform, stop and
document the exact operation and observed difference before weakening the
criterion. Do not silently introduce a numerical tolerance.

## Edge cases

- Empty validated atom inputs return an empty `float64` array without building
  a tree or taking a maximum.
- One atom has an empty neighbor row and full isolated area.
- Distant atoms have empty neighbor rows.
- Coincident atom centers remain invalid under the reference input contract.
- Very unequal radii must use their pair-specific summed cutoff correctly.
- Externally tangent expanded spheres are retained as neighbors.
- Custom valid sphere-point arrays replace the default and use their own point
  count in the denominator.

## Required tests

Add focused `pytest` coverage for at least:

1. Empty, single-atom, distant-atom, partially overlapping, enclosed-atom,
   unequal-radius, and exactly tangent geometries.
2. A regression geometry in which omitting the second atom's radius or the
   second copy of the probe radius would miss a real occluder.
3. Symmetric neighbor membership, no self-neighbors, no duplicate neighbors,
   increasing neighbor-index order, and valid CSR offsets.
4. Rejection of material conservative KD-tree false positives while retaining
   exact and roundoff-adjacent boundary pairs.
5. Bitwise equality with `atom_sasa_reference()` for deterministic randomized
   structures spanning multiple atom counts, radii, probe sizes, and custom
   sphere-point counts.
6. Translation and atom-permutation invariants.
7. Repeated-call determinism and non-mutation of caller-owned arrays.
8. Every validation and exception case required by the reference contract.
9. After the benchmark gate is met, a test proving `calculate_atom_sasa()`
   dispatches to the optimized path rather than the reference path.

Tests must not assert private SciPy tree layout or pair-return ordering.

## Benchmark requirements

Before making the spatial path the production dispatcher, benchmark both
functions on the same process and inputs after one untimed warm-up:

- synthetic structures at several sizes, including sparse and protein-like
  local densities; and
- the repository's representative small, medium, and large protein
  structures when those fixtures are available.

Use at least five measured repetitions and report the median. Record:

- atom count and sphere-point count;
- retained directed-neighbor count and mean/max neighbors per atom;
- KD-tree and neighbor-list construction time;
- surface-occlusion time;
- total numerical-function time;
- reference time and speedup; and
- peak memory when practical.

Tiny structures may be slower because of tree setup. The spatial path should
show a material total-time improvement for representative medium and large
proteins, with a provisional target of at least `2x` over the pure reference.
If it does not, retain the implementation for analysis but do not switch
production dispatch until profiling identifies the bottleneck. Benchmark
results, rather than an assumed atom-count threshold, will determine whether a
small-input reference fallback is worthwhile.

Benchmarks are evidence, not unit tests: CI must not fail on wall-clock ratios.
Record the final comparable row in the README optimization table and record
the atom counts, dependency versions, hardware, timing breakdown, repetition
count, and median results in the implementation pull request or issue. Use the
README's 1LYZ, 1CA2, and 1UOR structures when available. If those files cannot
be obtained, use reproducible deterministic protein-like synthetic inputs for
the performance gate, record how they were generated, and state that the
canonical structure benchmark is still pending.

Implementation therefore proceeds in this order:

1. Implement and test `atom_sasa_spatial()` and the private CSR builder while
   leaving production dispatch unchanged.
2. Run and record the required benchmarks.
3. If the medium and large representative cases meet the material-improvement
   gate, switch `calculate_atom_sasa()` and add the dispatch test in the same
   change. If they do not, leave dispatch unchanged and report the measured
   blocker rather than claiming this stage complete.

## Acceptance criteria

This stage is complete when:

- `atom_sasa_reference()` remains a direct all-atoms oracle without spatial
  indexing or shortcuts;
- `atom_sasa_spatial()` satisfies the same public scientific and validation
  contract;
- conservative symmetric CSR neighbors are produced through `cKDTree`
  discovery and pair-specific filtering, with no required occluder omitted;
- optimized and reference outputs are bitwise identical for all focused test
  cases;
- focused and full test suites pass;
- benchmark results and conditions are recorded;
- representative medium and large inputs demonstrate a material improvement
  before production dispatch changes;
- `calculate_atom_sasa()` dispatches to `atom_sasa_spatial()` after that gate
  is met; and
- no multiprocessing, GPU, residue-SASA, or additional kernel optimization is
  introduced in this stage.

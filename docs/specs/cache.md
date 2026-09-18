# Cache

**Status:** Implemented

## Purpose and scope

Reduce repeated scalar indexing and arithmetic in the NumPy atom-SASA kernel by
caching values that are invariant throughout one calculation. This stage
caches squared probe-expanded radii and exposes the three atom-coordinate
components as strided column views. It must preserve the current exposed
mask algorithm and the exact scientific result.

This optimization is intentionally local to the numerical kernel. It does not
change spatial neighbor membership, CSR ordering, sample-point generation,
occlusion rules, or the public API.

The following are out of scope:

- Numba, multiprocessing, threading, or GPU execution;
- changing the five-array kernel boundary;
- batching all neighbors into an `N * P` or `P * K` temporary;
- reduced precision, approximate comparisons, or `fastmath`;
- atom reordering or output reordering; and
- caching values across separate public function calls.

## Existing kernel boundary

`atom_sasa_spatial()` and `calculate_atom_sasa()` retain their existing
signatures and dispatch. The private numerical kernel continues to consume
only these five arrays:

```text
atom_coordinates
expanded_radii
sphere_points
neighbor_offsets
neighbor_indices
```

All cache values are created inside the kernel call. No cache object, Python
list, callback, or additional metadata may cross this boundary. This keeps the
same inputs directly usable by a future Numba implementation.

## Required caches

At the beginning of `_atom_sasa_from_neighbors()`, after the empty-atom fast
path, create:

```python
expanded_radii_squared = np.empty_like(expanded_radii)
for atom_index, radius in enumerate(expanded_radii):
    expanded_radii_squared[atom_index] = radius ** 2

atom_x = atom_coordinates[:, 0]
atom_y = atom_coordinates[:, 1]
atom_z = atom_coordinates[:, 2]
```

The coordinate arrays must be strided views, not per-call copies. They are safe
because the shared input path supplies C-contiguous `float64` coordinates. The
radius cache is one `float64` value per atom and must not be recomputed inside
the neighbor loop. The scalar power operation is intentional: it must match the
reference expression `expanded_radii[neighbor] ** 2` exactly at boundaries.

Use `expanded_radii_squared[neighbor]` for the neighbor occlusion threshold
and `expanded_radii_squared[atom_index]` for the final target-area formula.
Cache the current target radius and squared radius in local scalar variables
when processing each atom.

## Coordinate-component distance calculation

Allocate `outside_y_squared` and `outside_z_squared` once per kernel call as
reusable `float64` buffers of shape `(P,)`. Replace the per-neighbor
subtraction of a gathered three-component center with
component-wise operations into the existing reusable displacement buffer:

```python
np.subtract(sample_points[:, 0], atom_x[neighbor], out=displacements[:, 0])
np.subtract(sample_points[:, 1], atom_y[neighbor], out=displacements[:, 1])
np.subtract(sample_points[:, 2], atom_z[neighbor], out=displacements[:, 2])
np.multiply(displacements[:, 0], displacements[:, 0], out=squared_distances)
np.multiply(
    displacements[:, 1],
    displacements[:, 1],
    out=outside_y_squared,
)
np.add(squared_distances, outside_y_squared, out=squared_distances)
np.multiply(
    displacements[:, 2],
    displacements[:, 2],
    out=outside_z_squared,
)
np.add(squared_distances, outside_z_squared, out=squared_distances)
```

The implementation may instead use dedicated one-dimensional displacement
buffers, provided memory remains `O(P)` and the arithmetic order is identical.
The x-squared value must be accumulated first, followed by y-squared and then
z-squared, matching the current explicit NumPy implementation. Do not replace
this with a reduction, `einsum`, or an unspecified `sum(axis=1)`.

The two additional squared-component buffers may be views or reusable
`float64` arrays of shape `(P,)`. They must be allocated at most once per
kernel call and reused for every neighbor. Existing Boolean mask buffers and
early exits remain unchanged.

## Numerical behavior

Caching is an implementation optimization only. It must preserve:

- the closed boundary comparison `squared_distance > radius_squared`;
- the current x/y/z subtraction, multiplication, and addition order;
- occlusion-ordered CSR traversal;
- exposed-point counts and complete-burial early exits; and
- output order, dtype, and values.

The optimized kernel must be bitwise identical to the pre-cache kernel and to
`atom_sasa_reference()` for focused deterministic tests. In particular, add
regression coverage for exact tangency, roundoff-adjacent boundaries, unequal
radii, complete burial, and custom sphere-point arrays. If precomputing the
radius cache changes any boundary classification on a supported platform, the
implementation must retain the original scalar operation for that comparison
and document the result.

Inputs must not be mutated. Empty and single-atom inputs must retain their
existing behavior; a no-neighbor call may avoid allocating point work buffers.

## Memory and complexity

The cache adds `O(N)` radius storage and three `O(1)` coordinate views. Point
work remains `O(P)`, and no allocation may scale with the number of neighbors
or with `N * P`.

The additional work is one radius-square pass over `N` atoms. The inner loop
must no longer square a neighbor radius or gather a three-component coordinate
vector for each neighbor.

## Required tests

Add or update focused `pytest` tests for:

1. Bitwise equality with the pre-cache kernel and
   `atom_sasa_reference()` on deterministic randomized structures.
2. Exact-tangent and roundoff-adjacent occlusion boundaries.
3. Unequal radii, occlusion ordering, complete burial, and disjoint neighbors.
4. Empty, single-atom, all-no-neighbor, non-contiguous, and custom-point inputs.
5. Input arrays remain unchanged after calculation.
6. Canonical 1LYZ, 1CA2, and 1UOR outputs retain exact baseline MAE of `0.0`
   Å².

Tests must not depend on implementation-specific object identities for the
coordinate views; numerical behavior and bounded allocation are the contract.

## Benchmark requirements

Measure the pre-cache and cached kernels using identical warmed-up calls and
report median times for 1LYZ, 1CA2, and 1UOR at 960 sphere points. Also retain
the dense, sparse, buried, and nondefault sphere-point cases used by the
occlusion-ordering benchmark.

Record total time, kernel time when separately measurable, directed-neighbor
count, speedup, maximum absolute difference, MAE, and total-SASA difference.

The provisional production gate is:

- all focused and full tests pass;
- canonical and synthetic outputs are bitwise identical;
- no measured case regresses by more than 5%; and
- the medium or large canonical case improves by at least 3% on the benchmark
  machine, or profiling demonstrates that the cached path removes the intended
  repeated operations without a measurable regression.

## Acceptance criteria

This stage is complete when:

- squared expanded radii are computed once per kernel call;
- atom coordinate components are reused without per-neighbor center gathers;
- all work remains bounded by `O(N + P)` outside the existing CSR arrays;
- the five-array kernel boundary and public APIs are unchanged;
- current arithmetic, boundary classification, and SASA results are preserved;
- focused and full test suites pass; and
- benchmark results and any observed speedup are documented in `README.md`.

# Occlusion-Ordered Neighbors Specification

**Status:** Implemented

## Purpose and scope

Order each atom's existing CSR neighbor row by the neighbor's expected ability
to block that atom's sampled surface points. Test likely strong occluders
first so the current vectorized exposed-point mask—and a later Numba
point-first kernel—can terminate earlier when an atom becomes completely
blocked.

This stage changes traversal order only. It must not change:

- input validation or atom normalization;
- probe-expanded radii;
- `cKDTree` discovery;
- conservative pair-specific neighbor membership;
- CSR offsets or directed-neighbor counts;
- Cartesian sample construction;
- squared-distance or boundary comparisons;
- exposed-point counts, SASA values, or output order; or
- any public Python or command-line interface.

The spatial index and CSR contract remain defined by
[`atom-sasa-spatial.md`](atom-sasa-spatial.md). The vectorized kernel remains
defined by
[`atom-sasa-vectorized-mask.md`](atom-sasa-vectorized-mask.md). This
specification supersedes only their requirement that neighbors within a CSR
row appear in increasing atom-index order.

The ordering score is a performance heuristic. It must never be used to add
or remove a neighbor, classify a sample, approximate a distance, or calculate
SASA.

## Motivation and measured opportunity

The vectorized mask currently checks CSR neighbors in atom-index order and
calls `np.any(exposed)` after each neighbor. Profiling 1LYZ showed roughly
37,000 early-exit scans. Atom-index order has no geometric meaning, so strong
occluders may be tested late.

A prototype that orders neighbors by expected blocked spherical-cap size was
bitwise identical to the current kernel and produced these median kernel
times:

| Structure | Index order (s) | Occlusion order (s) | Ordering cost (s) | Net improvement |
| --- | ---: | ---: | ---: | ---: |
| 1LYZ | 0.446890 | 0.328393 | 0.017086 | about 23% |
| 1CA2 | 0.892177 | 0.603739 | 0.034735 | about 28% |
| 1UOR | 1.987921 | 1.591169 | 0.049640 | about 18% |

These prototype measurements motivate the implementation but do not replace
the required final benchmarks.

## Geometric ordering score

For target atom `i` and an already accepted neighbor `j`, let:

```text
R_i = expanded radius of target i
R_j = expanded radius of neighbor j
d   = Euclidean center distance between i and j
```

The neighbor intersects directions on target `i` satisfying a spherical-cap
threshold. Define its ordering score as:

```text
score(i, j) = (d**2 + R_i**2 - R_j**2) / (2 * R_i * d)
```

This is the cosine threshold of the potentially blocked cap on target `i`.
Lower scores indicate larger potential caps and therefore higher priority.

Interpretation:

- `score <= -1`: neighbor `j` geometrically encloses the target surface;
- `-1 < score < 1`: neighbor blocks a nonzero spherical cap;
- `score == 1`: expanded spheres are externally tangent; and
- `score > 1`: a conservative false-positive neighbor or a sphere contained
  inside the target that cannot block its outer surface.

Do not clamp the score to `[-1, 1]`. Values outside that interval are useful
for ordering enclosing neighbors first and non-occluding conservative
candidates last.

Coincident centers are already rejected by the shared public validation, so
`d` is strictly positive. Expanded radii are finite and positive. The ordering
implementation must nevertheless calculate scores only after those existing
contracts have been satisfied; it must not become an alternate validation
path.

## Deterministic order

Within every target row, sort directed neighbors by:

```text
(score ascending, neighbor atom index ascending)
```

The atom index is the deterministic tie-breaker. Do not rely on the input
ordering of `cKDTree.query_pairs()`, an unstable sort, or platform-dependent
set iteration.

The same undirected pair generally has different directed scores because
`score(i, j)` and `score(j, i)` use different target and neighbor radii.
Calculate and sort both directed entries independently.

Repeated construction with identical arrays must return identical offsets and
indices in the same runtime. Small cross-platform rounding differences in
nearly tied scores are acceptable only because ordering cannot affect the
scientific result; exact score ties must still use the atom-index tie-breaker.

## Vectorized CSR construction

Preserve the existing exact directed-neighbor membership and offsets. After
the directed `rows` and `columns` arrays have been formed, calculate scores for
all directed entries with contiguous NumPy arrays.

A suitable implementation is:

```python
directed_displacements = (
    atom_coordinates[columns] - atom_coordinates[rows]
)
directed_distance_squared = np.sum(
    directed_displacements * directed_displacements,
    axis=1,
)
directed_distance = np.sqrt(directed_distance_squared)
scores = (
    directed_distance_squared
    + expanded_radii[rows] ** 2
    - expanded_radii[columns] ** 2
) / (
    2.0 * expanded_radii[rows] * directed_distance
)
order = np.lexsort((columns, scores, rows))
ordered_rows = rows[order]
ordered_indices = columns[order]
```

For `np.lexsort`, the final key is primary. Therefore `rows` groups target
atoms, `scores` orders each row by increasing occlusion threshold, and
`columns` breaks score ties by neighbor index.

The implementation may reuse pair distances already calculated during exact
neighbor filtering rather than recomputing them, provided each directed score
uses the correct target and neighbor radii and the resulting order satisfies
the same tests.

Recalculate offsets from `ordered_rows`, or retain the existing offsets after
proving that ordering does not change row counts. Returned CSR arrays remain:

```text
offsets: shape (N + 1,), dtype np.intp
indices: shape (M,), dtype np.intp
```

Do not store scores in the returned CSR representation. They are temporary
construction data and are not required by the occlusion kernel or future
Numba implementation.

## Numerical and scientific behavior

Neighbor ordering must be scientifically invisible. For each sample point,
blocking remains the Boolean union of the same per-neighbor comparisons:

```text
squared_distance(sample, center[j]) <= expanded_radius[j]**2
```

Changing the sequence of those independent comparisons cannot change whether
the union contains the sample. The current explicit NumPy component
multiplication and x/y/z addition order must remain unchanged.

An early exit is allowed only when every entry in the exposed mask is already
false. Later neighbors cannot make a blocked sample exposed.

Require bitwise equality with the current vectorized implementation and
`atom_sasa_reference()` for focused tests. Do not add a tolerance merely
because traversal order changed. If ordering changes any exposed-point count,
stop and identify the violated invariant before proceeding.

## Public and private interfaces

Do not add a public function or option. `atom_sasa_spatial()` and
`calculate_atom_sasa()` retain their existing signatures and dispatch.

The preferred implementation integrates ordering into
`_build_spatial_neighbors()` so every consumer receives the optimized CSR.
A separate private ordering helper is acceptable if it takes only numerical
arrays and returns deterministic `np.intp` CSR indices.

The five-array numerical kernel boundary remains unchanged:

```text
atom_coordinates
expanded_radii
sphere_points
neighbor_offsets
neighbor_indices
```

No score array, Python object, callback, or atom metadata may cross that
boundary.

## Numba compatibility

This optimization must directly support the later Numba stage:

- retain contiguous `np.intp` CSR offsets and indices;
- perform ordering once before entering the numerical kernel;
- do not sort inside the atom or sphere-point loops;
- do not introduce Python lists, object arrays, generators, or closures into
  the kernel inputs;
- retain the same five-array kernel signature; and
- keep the ordering heuristic independent of the backend.

A Numba point-first loop benefits from the same ordering because it can stop
testing each sample at its first blocker. The ordered CSR must therefore be
reusable by NumPy, Numba, and multiprocessing implementations.

## Memory and complexity

For `M` directed neighbors, ordering may use temporary arrays proportional to
`M`, including directed rows, columns, displacements, distances, scores, and
the sort permutation. It must not allocate data proportional to `N * P`,
`M * P`, or `P * K`.

Expected construction complexity is:

```text
O(M) score calculation + O(M log M) global lexicographic sort
```

Because target rows are part of the sort key, one global `np.lexsort()` is
preferred over a Python loop that sorts each row separately. A row-by-row
implementation is acceptable only if measured total performance is better and
the same deterministic order is produced.

## Required tests

Add or update focused `pytest` coverage for at least:

1. Neighbor membership, directed-neighbor count, offsets, symmetry, lack of
   self-neighbors, and absence of duplicates remain unchanged.
2. A target with multiple equal-radius neighbors at different distances sorts
   the closer/larger-cap neighbor first.
3. At equal scores, the smaller neighbor atom index appears first.
4. Directed order can differ between the two rows of the same unequal-radius
   pair or neighborhood.
5. A neighbor enclosing the target sorts ahead of partial occluders.
6. A conservative false positive or contained inner sphere sorts after true
   surface occluders.
7. Empty, single-atom, and all-no-neighbor inputs return the existing CSR
   representations without score calculation failures.
8. Exact-tangent and roundoff-adjacent neighbor membership remains unchanged.
9. Bitwise SASA equality with the unordered/index-ordered vectorized baseline
   and `atom_sasa_reference()` for deterministic randomized structures,
   custom sphere points, unequal radii, enclosure, and boundary geometries.
10. Bitwise equality for 1LYZ, 1CA2, and 1UOR against the existing
    `*.sas.baseline` files.
11. Translation and atom-permutation invariants, repeated-call determinism,
    and caller-input non-mutation remain satisfied.
12. The full existing suite passes unchanged.

Tests may inspect private CSR rows because their deterministic order is the
behavior introduced by this stage. Tests must not assert private cKDTree node
layout or timing ratios.

## Benchmark requirements

Before changing CSR order, capture the current vectorized-mask outputs and
five-repeat median timings. After implementation, benchmark identical inputs
with one untimed warm-up and at least five measured repetitions.

Measure:

- 1LYZ, 1CA2, and 1UOR at 960 sphere points;
- the documented dense synthetic cases;
- a sparse/no-neighbor case;
- an input containing completely buried atoms; and
- at least one nondefault sphere-point count.

Record:

- atom and sphere-point counts;
- directed-neighbor count and mean/max row lengths;
- ordering construction time;
- numerical-kernel and total `atom_sasa_spatial()` time;
- baseline time, ordered time, and speedup;
- how many neighbor evaluations and fully blocked early exits occur, when
  practical; and
- maximum absolute per-atom difference, MAE, and total-SASA difference.

Update the README optimization table with a separate
`Occlusion-ordered neighbors` row. Retain the cKDTree and vectorized-mask rows.

The provisional production gate is:

- bitwise equality for all focused tests;
- exact MAE of `0.0 Å²` against all three `*.sas.baseline` files;
- at least a 10% median total-function improvement on both 1CA2 and 1UOR after
  including ordering construction time; and
- no more than a 25% regression on any sparse or tiny benchmark whose baseline
  median is at least 10 ms. Below 10 ms, report absolute values without a ratio
  gate.

Wall-clock ratios are benchmark evidence and must not be CI assertions. If the
gate is not met, retain index-ordered CSR in production and report the measured
blocker.

## Acceptance criteria

This stage is complete when:

- exact neighbor membership and CSR offsets are unchanged;
- each CSR row is sorted by increasing occlusion score with atom-index
  tie-breaking;
- the ordering score is used only as a traversal heuristic;
- the five-array kernel interface is unchanged;
- current boundary arithmetic and bitwise SASA results are preserved;
- focused and full tests pass;
- required benchmark results and conditions are recorded;
- the README retains prior rows and adds the new optimization; and
- no Numba, multiprocessing, or scientific approximation is
  introduced.

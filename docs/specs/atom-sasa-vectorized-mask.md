# Vectorized Atom-SASA Mask Specification

**Status:** Implemented

## Purpose and scope

Replace the Python point-by-point occlusion loop in `atom_sasa_spatial()` with
a bounded NumPy exposed-point mask. The implementation must retain the
validated inputs, probe-expanded radii, `cKDTree` discovery, and deterministic
CSR neighbor lists defined by
[`atom-sasa-spatial.md`](atom-sasa-spatial.md).

This specification supersedes only the spatial specification's point-first
occlusion loop. Its validation, neighbor inclusion, public API, numerical
contract, output order, and production dispatch remain authoritative.
`atom_sasa_reference()` remains unchanged as the correctness oracle.

This stage includes:

- vectorized sample construction for one target atom at a time;
- neighbor-first occlusion over the target's existing CSR row;
- a reusable Boolean mask of currently exposed sphere points;
- reusable bounded NumPy work buffers; and
- isolation of the numerical kernel so it can later be replaced by Numba.

The following remain out of scope:

- Numba or another JIT compiler;
- multiprocessing, threading, or GPU execution;
- changes to KD-tree discovery or CSR layout;
- spherical-cap or dot-threshold reformulations;
- reduced precision, `fastmath`, or approximate comparisons;
- complete-burial shortcuts or neighbor reordering;
- caching across public function calls; and
- residue-SASA calculation.

The stage is intentionally an optimization of the existing algorithm, not a
new scientific method.

## Motivation and measured bottleneck

Profiling the current cKDTree implementation on 1LYZ attributes more than 99%
of runtime to the Python occlusion loop. One profiled 1LYZ calculation made
approximately 14.7 million point-versus-neighbor checks, while KD-tree and
neighbor-list construction took about 9 ms.

A prototype using the same coordinates, expanded radii, sphere points, and
CSR neighbors produced bitwise-identical 1LYZ output with these observed
single-run times on the development system:

| Kernel | Time (s) |
| --- | ---: |
| Current Python point-first loop | 18.12 |
| Scalar-coordinate Python loop | 10.14 |
| Bounded vectorized mask | 0.58 |

These measurements justify specifying the vectorized mask next. They are
development evidence, not final benchmark results.

## Public API and dispatch

Do not add or change a public numerical function. The existing interface
remains:

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

`calculate_atom_sasa()` must continue to dispatch to `atom_sasa_spatial()`.
The command-line interface and generated files must not change.

Factor the post-validation, post-neighbor-discovery calculation into a private
kernel boundary. Its inputs must be plain contiguous arrays and may follow
this shape:

```python
def _atom_sasa_from_neighbors(
    atom_coordinates: np.ndarray,
    expanded_radii: np.ndarray,
    sphere_points: np.ndarray,
    neighbor_offsets: np.ndarray,
    neighbor_indices: np.ndarray,
) -> np.ndarray:
    """Calculate atom SASA from validated arrays and CSR neighbors."""
```

The exact private name is not normative. The data-only boundary is normative:
the kernel must not build a tree, validate user input, read files, inspect CLI
arguments, or depend on Python atom-record objects. A later Numba kernel must
be able to consume the same five arrays without changing public APIs or CSR
construction.

## Required array properties

The kernel receives:

- `atom_coordinates`: C-contiguous `float64`, shape `(N, 3)`;
- `expanded_radii`: C-contiguous `float64`, shape `(N,)`;
- `sphere_points`: C-contiguous `float64`, shape `(P, 3)`;
- `neighbor_offsets`: `np.intp`, shape `(N + 1,)`; and
- `neighbor_indices`: `np.intp`, shape `(M,)`.

All validation remains in the shared input-validation path. CSR invariants
remain the responsibility of `_build_spatial_neighbors()`. The private kernel
does not need to repeat those checks.

The returned array must have shape `(N,)`, dtype `float64`, and input atom
order. Empty input returns an empty `float64` array.

## Reusable work buffers

For `P` sphere points, allocate these buffers at most once per kernel call and
reuse them for every target atom and neighbor:

| Buffer | Shape | Dtype | Purpose |
| --- | --- | --- | --- |
| `sample_points` | `(P, 3)` | `float64` | Cartesian samples for one target |
| `displacements` | `(P, 3)` | `float64` | Samples minus one neighbor center |
| `squared_distances` | `(P,)` | `float64` | Distance squared to one neighbor |
| `exposed` | `(P,)` | `bool` | Points not yet blocked |
| `outside_neighbor` | `(P,)` | `bool` | Current-neighbor comparison result |

At the default `P = 960`, these buffers require approximately 56 KiB,
excluding small array-object overhead. Working memory must be `O(P)` and must
not grow with the target's neighbor count.

Return empty input before allocating the work buffers. An implementation may
also defer allocation until the first atom with a nonempty CSR row, so a call
where every atom has no neighbors does not allocate unused point buffers.

Do not allocate:

- a `(P, K, 3)` or `(K, P, 3)` array;
- a complete sample-point array for every atom;
- a new displacement or Boolean array for every neighbor; or
- Python lists containing per-point or per-neighbor numerical results.

NumPy operations should use `out=` with the reusable buffers where supported.
Small scalar and iterator objects are acceptable.

## Vectorized mask algorithm

For each target atom `i` in input order:

1. Read its CSR interval:

   ```text
   start = neighbor_offsets[i]
   stop  = neighbor_offsets[i + 1]
   ```

2. If `start == stop`, set `exposed_count = P` without constructing sample
   points. Use the unchanged reference arithmetic for final SASA.

3. Otherwise construct all target samples in the reusable buffer:

   ```python
   np.multiply(sphere_points, expanded_radii[i], out=sample_points)
   np.add(sample_points, atom_coordinates[i], out=sample_points)
   exposed.fill(True)
   ```

4. For every neighbor position from `start` through `stop - 1`, preserving CSR
   order:

   ```python
   neighbor = neighbor_indices[position]
   np.subtract(
       sample_points,
       atom_coordinates[neighbor],
       out=displacements,
   )
   np.einsum(
       "ij,ij->i",
       displacements,
       displacements,
       out=squared_distances,
       optimize=False,
   )
   np.greater(
       squared_distances,
       expanded_radii[neighbor] ** 2,
       out=outside_neighbor,
   )
   np.logical_and(exposed, outside_neighbor, out=exposed)
   ```

   `outside_neighbor` is intentionally strict `>`. Therefore a sample with
   squared distance exactly equal to the neighbor's squared expanded radius is
   blocked, matching the reference `<=` rule.

5. After each neighbor, stop if `np.any(exposed)` is false. This is an exact
   early exit because later neighbors cannot make a blocked point exposed.

6. Set `exposed_count = int(np.count_nonzero(exposed))`.

7. Calculate the area with the reference operation order:

   ```text
   4.0 * pi * R[i]**2 * exposed_count / P
   ```

Do not simplify the no-neighbor expression to `4 * pi * R[i]**2`; preserving
the multiply-then-divide order is required for bitwise output equivalence.

The implementation may use an equally bounded NumPy formulation only if it
meets every numerical, memory, and benchmark requirement below. In
particular, selecting only currently exposed points with advanced indexing is
not preferred: the prototype was slower and creates variable temporary arrays
for every neighbor.

## Numerical behavior

All geometry remains `float64`. The vectorized kernel must retain:

- the same validated sphere points and their order;
- the same Cartesian sample definition;
- squared Euclidean distance comparisons;
- the rule that inside and boundary samples are blocked;
- the actual custom sphere-point count as denominator; and
- deterministic absolute SASA in Å².

The target result is bitwise equality with `atom_sasa_reference()` using
`np.array_equal`. Vectorized ufuncs and `np.einsum()` can have platform-specific
floating-point behavior near exact boundaries. Therefore tests must include
deliberate boundary geometries. If a supported platform produces a different
exposed-point classification, stop and document the exact geometry and
operations before changing the implementation or relaxing equality. Do not
silently add a numerical tolerance to the blocking comparison.

The spherical-cap identity and a dot-product threshold are explicitly
excluded in this stage because they rearrange the boundary-sensitive
floating-point calculation.

## Numba compatibility

This optimization must leave a direct path to a compiled kernel:

- retain the five-array private kernel interface;
- keep CSR offsets and indices as the only neighbor representation;
- keep validation and tree construction outside the numerical kernel;
- avoid closures, object arrays, dictionaries, atom-record objects, and
  callbacks in the kernel boundary; and
- do not cache mutable work buffers at module scope.

The later Numba implementation may replace the neighbor-first NumPy mask with
a point-first early-exit loop, because Numba removes Python-loop overhead. It
must not require changes to callers, validation, tree construction, or CSR
storage. The NumPy kernel remains a non-JIT optimized fallback and an
additional correctness comparator.

## Required tests

Add or update focused `pytest` coverage for at least:

1. Bitwise equality with `atom_sasa_reference()` for empty, single-atom,
   no-neighbor, partially overlapping, completely enclosed, unequal-radius,
   and exactly tangent geometries.
2. A custom sphere set containing a point exactly on an occluder's boundary;
   that point must be blocked.
3. Bitwise equality for deterministic randomized structures spanning multiple
   atom counts, radii, probe sizes, densities, and sphere-point counts.
4. Bitwise equality for the canonical 1LYZ, 1CA2, and 1UOR structures when
   those fixtures are available and running the reference is practical.
5. Translation, atom-permutation, repeated-call determinism, and caller-input
   non-mutation.
6. An atom whose first CSR neighbor blocks every point, exercising the mask
   early exit.
7. An atom with many neighbors where different neighbors block disjoint point
   subsets, proving the mask accumulates occlusion rather than replacing it.
8. A sphere-point count other than 960, including a very small custom set.
9. The existing validation and CSR-neighbor tests remain unchanged and pass.
10. `calculate_atom_sasa()` continues to dispatch through
    `atom_sasa_spatial()`.

Tests must not depend on timing ratios or private NumPy allocation internals.
Memory bounds should be established by implementation structure and benchmark
measurement, not brittle monkeypatches of NumPy constructors.

## Benchmark requirements

Benchmark the current cKDTree point-first implementation and the vectorized
mask with identical validated inputs. Use one untimed warm-up followed by at
least five measured repetitions and report medians.

Before replacing the point-first loop, capture its benchmark results and
output arrays for every required workload. After implementing the mask, run
the same benchmark driver and inputs, and compare against those saved arrays.
The old Python loop may be retained temporarily on the implementation branch
for A/B profiling, but remove the duplicate production implementation before
completion. The committed package must not select kernels through a hidden
environment variable or undocumented runtime switch.

Measure:

- 1LYZ, 1CA2, and 1UOR at 960 sphere points;
- the deterministic dense synthetic crossover cases already documented in
  the README;
- at least one sparse structure or synthetic input with many no-neighbor
  atoms; and
- at least one nondefault sphere-point count.

Record:

- atom and sphere-point counts;
- retained directed-neighbor count and mean/max neighbors;
- KD-tree/CSR construction time;
- vectorized occlusion time and total `atom_sasa_spatial()` time;
- current cKDTree baseline time, vectorized time, and speedup;
- maximum absolute per-atom difference, MAE, and total-SASA difference from
  the reference or current bitwise-equivalent implementation; and
- peak resident or allocator-observed memory when practical.

The current documented cKDTree baseline is 18, 37, and 84 seconds for the
small, medium, and large canonical structures. Update the README optimization
table with a separate vectorized-mask row; do not overwrite the cKDTree row.

The provisional production gate is:

- bitwise-identical focused-test output;
- zero measured MAE against the current implementation;
- for each sparse or tiny benchmark whose cKDTree baseline median is at least
  10 ms, a vectorized median no greater than `1.25x` that baseline; report
  absolute timings without applying a ratio gate below 10 ms, where timer and
  process noise dominate; and
- at least `5x` median total-function speedup on both 1CA2 and 1UOR.

If the gate is not met, keep the current production kernel and report the
profiled blocker. Wall-clock ratios are benchmark evidence and must not be CI
assertions.

## Acceptance criteria

This stage is complete when:

- the cKDTree and CSR implementation remains unchanged;
- `atom_sasa_spatial()` uses the bounded vectorized mask kernel;
- work buffers are allocated at most once per call and remain `O(P)`;
- no temporary scales as `P * K` or `N * P`;
- the private kernel boundary accepts only the five numerical arrays;
- bitwise equivalence and boundary behavior pass focused tests;
- all existing and new tests pass;
- required benchmark results and environment details are recorded;
- the README retains the cKDTree baseline and adds the vectorized result; and
- no Numba, parallel execution, GPU path, or new scientific approximation is
  introduced.

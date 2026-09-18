# Numba Atom-SASA Kernel Specification

**Status:** In progress

## Purpose and scope

Replace the Python/NumPy inner occlusion loop in the current spatial
implementation with an optional Numba-compiled CPU kernel. This is the next
optimization after the implemented cKDTree, vectorized-mask, occlusion-order,
and cache stages. The compiled kernel must consume the same validated arrays
and CSR neighbor representation, preserve the reference algorithm's boundary
semantics, and leave the NumPy kernel available as a correctness comparator and
portable fallback.

This stage includes:

- a private, data-only Numba kernel for atom occlusion;
- serial execution as the correctness baseline;
- an explicitly controlled `parallel=True`/`prange` variant for benchmarking;
- `backend="auto"`/`backend="cpu"` dispatch and capability checks that do not
  initialize Numba on import; and
- focused equivalence, boundary, and benchmark coverage.

The following are out of scope:

- multiprocessing or nested process/thread execution;
- changes to KD-tree discovery, CSR construction, neighbor ordering, or
  complete-burial detection;
- `float32`, `fastmath=True`, approximate comparisons, or altered sphere-point
  geometry; and
- removal of the existing NumPy implementation.

## Authoritative inputs and kernel boundary

The existing spatial signature gains an optional backend selector:

```python
def atom_sasa_spatial(
    coordinates,
    radii,
    *,
    probe_size=DEFAULT_PROBE_SIZE,
    sphere_points=None,
    backend="auto",
) -> np.ndarray:
    ...
```

After shared validation, KD-tree filtering, CSR ordering, and burial
detection, the private compiled kernel receives only contiguous numerical
arrays. The normative interface is:

```python
def _atom_sasa_numba_kernel(
    atom_coordinates: np.ndarray,
    expanded_radii: np.ndarray,
    sphere_points: np.ndarray,
    neighbor_offsets: np.ndarray,
    neighbor_indices: np.ndarray,
    buried: np.ndarray,
) -> np.ndarray:
    """Return per-atom SASA in input order."""
```

The exact Python wrapper name is not normative. The data-only boundary is:
the kernel must not validate user arguments, build a tree, read files, access
atom-record objects, call Python callbacks, or depend on global mutable work
buffers. The five numerical arrays from the existing NumPy boundary remain
unchanged, and the precomputed burial mask is supplied as a sixth Boolean
array. Complete burial is an upstream optimization: the Numba kernel consumes
the mask and must not independently reclassify burial.

Required properties before entering compiled code:

- `atom_coordinates`: C-contiguous `float64`, shape `(N, 3)`;
- `expanded_radii`: C-contiguous `float64`, shape `(N,)`;
- `sphere_points`: C-contiguous `float64`, shape `(P, 3)`;
- `neighbor_offsets`: C-contiguous `intp`, shape `(N + 1,)`;
- `neighbor_indices`: C-contiguous `intp`, shape `(M,)`.
- `buried`: C-contiguous `bool`, shape `(N,)`.

The wrapper owns conversion to these properties. The compiled function must
return a C-contiguous `float64` array of shape `(N,)`, preserve atom order,
and return an empty array for `N == 0`.

## Kernel algorithm

For each target atom `i`, in stable atom order:

1. If `buried[i]` is true, write `0.0` and continue.
2. Read `start = neighbor_offsets[i]` and
   `stop = neighbor_offsets[i + 1]`.
3. For each sphere point `p`, construct its target sample using the same
   multiplication and addition convention as the reference implementation.
4. Traverse CSR neighbors from `start` to `stop` in their existing
   occlusion-ordered sequence. For each neighbor `j`, compute the squared
   distance from the sample to `atom_coordinates[j]` and mark the sample
   blocked when:

   ```text
   squared_distance <= expanded_radii[j] ** 2
   ```

5. Stop testing neighbors for a point as soon as it is blocked. Count samples
   that remain exposed.
6. Write the area using the reference operation order:

   ```text
   4.0 * pi * expanded_radii[i] ** 2 * exposed_count / P
   ```

A point exactly on an expanded sphere is blocked. The implementation must not
replace the comparison with a tolerance, `isclose`, or a strict `<` test.
Squared-distance arithmetic must accumulate x, then y, then z, matching the
existing NumPy kernel. Do not use `fastmath`, reassociation, or an
unconstrained reduction.

The preferred Numba layout is point-first with scalar early exits: Numba
removes the Python-loop overhead that motivated the replacement of the
neighbor-first NumPy mask. A neighbor-first implementation is acceptable only
if it meets the same numerical and benchmark gates.

## Compilation and fallback behavior

Use `numba.njit(cache=True)` for the serial kernel. Compilation must be lazy:
importing `protrsa` and calling the reference implementation must not compile
or initialize Numba. The first spatial call may pay compilation cost; benchmark
warm-ups must exclude that cost.

`backend="auto"` and `backend="cpu"` both select the Numba CPU
implementation and fall back to the existing NumPy CPU kernel when Numba is
unavailable. No external runtime is initialized or probed.

If Numba is unavailable, disabled by an unsupported runtime, or compilation
fails, the call must use the existing NumPy kernel. Automatic fallback must
not hide numerical errors raised after successful compilation.

Do not add an undocumented environment-variable switch. If a temporary
developer switch is needed for A/B measurements, keep it outside the committed
production interface and remove it before completion.

## Parallel variant and thread control

Benchmark a separate `njit(parallel=True, cache=True)` kernel using `prange`
over independent target atoms. It must not mutate shared per-atom state or
reuse scratch buffers across parallel iterations. Per-target scratch storage
must be private to each iteration, with memory use and allocation behavior
reported in the benchmark.

Multiprocessing and its `workers` option are deferred to the multiprocessing
specification. Do not combine Numba `prange` with multiprocessing workers by
default; nested parallelism can oversubscribe the machine. Any Numba thread
count used for benchmarking must be explicit and restored after a call, but it
is not a new public `workers` parameter in this stage.

## Numerical and API contract

The Numba result must preserve the same scientific classification as the
existing cached NumPy kernel for supported `float64` inputs, including:

- exact tangency and roundoff-adjacent boundaries;
- unequal radii and complete burial;
- empty, single-atom, and all-no-neighbor structures;
- custom sphere-point arrays and nondefault point counts;
- occlusion-ordered neighbors and early exits; and
- translated, permuted, and repeated deterministic inputs.

Inputs must not be mutated. `calculate_atom_sasa()` continues to dispatch
through `atom_sasa_spatial()`, and output dtype, shape, ordering, and residue
aggregation remain unchanged. The NumPy kernel remains callable privately for
regression comparison and fallback; no new public numerical function is
required.

## Required tests

Add focused `pytest` coverage for:

1. serial Numba versus the cached NumPy kernel and reference outputs on
   deterministic randomized structures;
2. exact tangent and adjacent floating-point boundary geometries;
3. empty, single-atom, no-neighbor, partially overlapping, upstream-filtered,
   and unequal-radius cases;
4. custom sphere points, including a point exactly on an occluder boundary;
5. translation, atom permutation, repeated calls, and input non-mutation;
6. lazy compilation and import safety (no compilation on module import);
7. missing-Numba or compilation-failure fallback to the NumPy kernel; and
8. `backend="auto"` and `backend="cpu"` dispatch behavior; and
9. `calculate_atom_sasa()` and CLI outputs remaining unchanged.

Parallel tests must compare results to serial Numba and use a modest fixed
thread count. Tests must not require a particular speedup or machine-specific
thread scheduler behavior.

## Benchmark and acceptance gates

Use one untimed warm-up after compilation followed by at least five measured
calls on identical validated inputs. Measure the existing cached NumPy kernel,
serial Numba, parallel Numba, and (where available) four-process execution on
the canonical small, medium, and large structures plus dense and sparse
synthetic cases. Record atom count, point count, directed-neighbor count,
thread count, compilation exclusion, kernel time, total time, peak memory when
practical, maximum absolute difference, MAE, and total-SASA difference.

This stage is complete only when:

- all focused and full tests pass;
- serial and parallel Numba outputs preserve exposed/blocked classifications;
- fallback behavior is exercised and preserves outputs;
- deliberate tangent and near-boundary tests retain the specified blocked/
  exposed classification; small floating-point area differences are measured
  rather than assumed away;
- serial Numba materially improves representative medium or large workloads,
  or profiling documents why it is retained only as an optional path; and
- benchmark conditions and results are recorded in the README optimization
  table without replacing prior rows.

For every benchmark workload, record maximum absolute per-atom difference,
per-atom MAE, total-SASA absolute and relative difference, and any changed
boundary classification. Numerical tolerance selection is deferred; do not
fail this stage solely because Numba and NumPy differ by small floating-point
rounding amounts.

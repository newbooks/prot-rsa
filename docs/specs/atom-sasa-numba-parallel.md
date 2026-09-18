# Parallel Numba Atom-SASA Specification

**Status:** Planned

## Purpose and scope

Add a thread-parallel Numba CPU implementation of the existing atom-SASA
kernel. The serial Numba kernel defined by
[`atom-sasa-numba.md`](atom-sasa-numba.md) remains the correctness baseline.
Parallel execution must preserve atom order, burial-mask handling, CSR
neighbor ordering, and the scientific blocked/exposed classification while
allowing independent target atoms to execute concurrently.

This stage includes:

- a separate `numba.njit(parallel=True, cache=True)` kernel;
- `numba.prange` over independent target atoms;
- thread-count discovery, explicit temporary thread limits, and restoration;
- serial-versus-parallel equivalence tests; and
- representative performance benchmarks.

This stage does not include:

- multiprocessing;
- non-CPU execution;
- changes to KD-tree discovery, CSR construction, neighbor ordering, or burial
  detection;
- `float32`, `fastmath=True`, approximate comparisons, or altered sphere-point
  geometry; or
- changing the default `backend="auto"`/`backend="cpu"` dispatch before the
  benchmark gate is met.

## Kernel interface

The parallel kernel consumes the same six contiguous arrays as the serial
Numba kernel:

```python
def _atom_sasa_numba_parallel_kernel(
    atom_coordinates: np.ndarray,
    expanded_radii: np.ndarray,
    sphere_points: np.ndarray,
    neighbor_offsets: np.ndarray,
    neighbor_indices: np.ndarray,
    buried: np.ndarray,
) -> np.ndarray:
    """Return per-atom SASA in stable input order."""
```

The exact private name is not normative. Inputs are already validated and
contiguous. The kernel must not build trees, validate public arguments, read
files, access atom-record objects, invoke Python callbacks, or depend on
mutable module-level scratch buffers.

The result is a C-contiguous `float64` array of shape `(N,)`. Empty input
returns an empty array. The output index for atom `i` is always `i`, regardless
of execution order.

## Parallel algorithm

Use `numba.prange(atom_count)` for the outer target-atom loop. Each iteration
must:

1. Return `0.0` immediately for `buried[i]`.
2. Read only the target atom's CSR row and immutable input arrays.
3. Allocate or obtain scratch arrays private to that iteration, or use scalar
   locals with bounded temporary storage. Scratch storage must not be shared
   between iterations.
4. Traverse the existing CSR row in deterministic occlusion order.
5. Apply the same closed blocking comparison as serial Numba:

   ```text
   squared_distance <= expanded_radii[j] ** 2
   ```

6. Write only `surface_areas[i]`.

No reduction across atoms is required. Since each target is independent,
parallel execution must not use atom-order-dependent accumulation or shared
early-exit state. A point exactly on an expanded sphere remains blocked.

The implementation must not use `fastmath=True`. Squared distances accumulate
x, then y, then z. The final area uses the same operation order as the serial
kernel:

```text
4.0 * pi * expanded_radii[i] ** 2 * exposed_count / P
```

## Thread control

Thread control is an execution concern, not a multiprocessing worker count.
The public `workers` parameter and CLI option select the number of Numba CPU
threads, defaulting to `1`.

When a temporary thread count is requested:

1. Record `numba.get_num_threads()`.
2. Set the requested count with `numba.set_num_threads(n)`.
3. Execute the parallel kernel.
4. Restore the previous count in a `finally` block.

The runner must reject nonpositive counts and counts exceeding Numba's valid
thread range. If no count is requested, preserve the caller's current Numba
thread setting. Do not call `set_num_threads()` at import time.

Do not create multiprocessing workers around a parallel Numba call by default;
nested parallelism can oversubscribe the machine. Worker coordination belongs
to the later multiprocessing specification.

## Dispatch policy

`backend="auto"` and `backend="cpu"` select the parallel Numba CPU kernel with
the requested `workers` count, falling back to the serial Numba kernel and then
the NumPy kernel when necessary.

The implementation may later establish a measured atom-count crossover and
retain serial execution for tiny workloads, but the default implementation for
this stage is one Numba CPU thread. Callers may request parallel execution
explicitly with `workers=4` or another positive count.

## Numerical contract

Parallel and serial Numba must preserve the same exposed/blocked classification
for supported `float64` inputs, including:

- exact and roundoff-adjacent tangency;
- complete-burial mask entries;
- empty, single-atom, and all-no-neighbor inputs;
- unequal radii and custom sphere-point arrays;
- deterministic CSR ordering; and
- translated, permuted, and repeated inputs.

Small floating-point area differences between parallel Numba, serial Numba,
and NumPy must be measured and reported. This stage does not establish a final
numerical tolerance. It must not introduce a changed geometric comparison to
make parallel results appear closer.

## Required tests

Add focused `pytest` coverage for:

1. serial and parallel output shape, dtype, and atom ordering;
2. serial-versus-parallel equivalence on deterministic randomized structures;
3. exact tangent, near-boundary, unequal-radius, and buried-atom cases;
4. empty, single-atom, no-neighbor, and custom-point inputs;
5. repeated execution with fixed thread counts;
6. input non-mutation;
7. temporary thread-count restoration after success and after an exception;
8. rejection of invalid thread counts; and
9. import safety and lazy Numba compilation.

Tests must not require a machine-specific speedup or exact scheduler behavior.

## Benchmark requirements

Compile each kernel before timing. Use one untimed warm-up followed by at
least five measured calls for serial Numba and parallel Numba at explicitly
recorded thread counts. Include:

- 1LYZ, 1CA2, and 1UOR at 960 sphere points;
- dense and sparse synthetic structures;
- inputs with many buried atoms and inputs with none; and
- at least one custom sphere-point count.

Record atom count, point count, directed-neighbor count, buried count, thread
count, kernel time, total time, maximum absolute difference, MAE, total-SASA
difference, and any changed boundary classification. Report compilation time
separately and exclude it from timed medians.

The provisional performance gate is:

- all focused and full tests pass;
- serial and parallel classifications agree;
- no input mutation or thread-state leakage occurs; and
- parallel execution materially improves at least one representative medium or
  large workload without an unacceptable regression on sparse or small inputs.

Wall-clock ratios are benchmark evidence, not CI assertions. Record the
hardware, Python, Numba, NumPy, and SciPy versions with final results.

## Acceptance criteria

This stage is complete when:

- the private `prange` kernel exists and consumes the precomputed burial mask;
- scratch state is private to each parallel iteration;
- thread counts are controlled and restored safely;
- serial and parallel tests pass with the documented numerical observations;
- required benchmarks are recorded; and
- production dispatch uses one Numba CPU thread by default, with the public
  `workers` override documented and tested.

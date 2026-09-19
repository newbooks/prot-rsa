# prot-rsa Implementation Plan

## Objective

Build `prot-rsa` as a public PyPI application for calculating protein atom and residue solvent-accessible surface area (SASA). The main calculation application will live in one import-safe Python file, `protrsa.py`, which can be run directly, invoked through the installed `prot-rsa` command, or imported as the `protrsa` module. The atom-SASA comparison utility is a separate import-safe `compare_sas.py` script and installed `prot-rsa-compare` command.

The implementation will start from the previous PyMCCE SASA code while correcting known issues, defining a stable scientific contract, and adding validated CPU and multiprocessing execution paths.

## Interfaces

```bash
python protrsa.py structure.pdb
prot-rsa structure.cif.gz --mode SIDE --prob-size 1.40 --workers 4
prot-rsa-compare first.atom.sas second.atom.sas
```

```python
import protrsa

result = protrsa.calculate_sasa(...)
```

Importing `protrsa` or `compare_sas` must not parse command-line arguments,
create worker processes, initialize Numba, or perform calculations. Direct
execution will be protected by:

```python
if __name__ == "__main__":
    main()
```

The installed `prot-rsa-compare` entry point must resolve to
`compare_sas.compare_sas_main`; comparison code must not be imported into or
executed by `protrsa.main`.

## Basic program setup: input and output contract

The coding-level contract for this interface is maintained in
[`specs/command-line-interface.md`](specs/command-line-interface.md).

### Input file

The command accepts exactly one positional argument: the path to a protein
structure file. The parser infers the input format from the filename
extension. The first implementation will support:

- PDB files ending in `.pdb`
- mmCIF files ending in `.cif`
- gzip-compressed versions ending in `.pdb.gz` or `.cif.gz`

Extension matching should be case-insensitive. A missing file, unsupported
extension, malformed gzip stream, or invalid structure file must produce a
clear error on standard error and a nonzero exit status. Input decompression
must be streamed; the CLI must not create a persistent uncompressed copy.

### Command-line options

```text
usage: prot-rsa [OPTIONS] INPUT

positional arguments:
  INPUT                 PDB or mmCIF structure (.pdb, .cif, .pdb.gz, or .cif.gz)

options:
  --mode {ALL,SIDE} Atom selection mode (default: ALL)
  --prob-size FLOAT     Solvent probe radius in angstroms (default: 1.40)
  --workers INTEGER     Number of Numba CPU threads (default: 1)
  --sphere-points INTEGER
                        Number of deterministic sphere samples (minimum 122;
                        default: 960)
  --preserve-het        Preserve loose hetero-atoms (default: false)
  --use-h               Use hydrogen atoms supplied in the input file (default: false)
```

`--preserve-het` and `--use-h` are presence flags: omitting either flag keeps
its value false, while supplying it sets the value to true. `--mode` values
should be accepted case-insensitively and normalized to uppercase. The CLI
must reject a nonpositive probe size or worker count with a clear error.

The precise atom-selection rules for `SIDE`, and the definition of a "loose
hetero atom," remain scientific contract decisions. They must be specified and
tested before `SIDE` is implemented; the parser must not silently invent those
rules.

### Output files

Each successful run writes exactly two files alongside the input file:

- `<base>.atom.sas`, a TSV file containing atom solvent-accessible surface areas
- `<base>.res.sas`, a TSV file containing residue solvent-accessible surface
  areas and the planned Contextual Exposure Fraction (CEF)

`<base>` is the input path with the optional final `.gz` suffix removed,
followed by the `.pdb` or `.cif` suffix removed. For example:

| Input | Atom output | Residue output |
| --- | --- | --- |
| `protein.pdb` | `protein.atom.sas` | `protein.res.sas` |
| `protein.cif` | `protein.atom.sas` | `protein.res.sas` |
| `protein.pdb.gz` | `protein.atom.sas` | `protein.res.sas` |
| `/data/protein.cif.gz` | `/data/protein.atom.sas` | `/data/protein.res.sas` |

Both `.sas` files use tab-separated values. The initial atom-output columns,
units, precision, ordering, and atomic-overwrite policy are defined in
[`specs/atom-sasa-naive.md`](specs/atom-sasa-naive.md). Residue-output details
remain deferred, except for the CEF scientific definition below. Output files
are written only after parsing and calculation succeed.

## Known issues in the starting code

- `sas()` divides by the global `N_POINTS` instead of the number of supplied sphere points.
- Its neighbor-search cutoff appears to omit the neighboring atom probe radius and may miss valid occluders.
- Empty arrays cause operations such as `max()` to fail.
- Array shapes, lengths, finite values, and radii are not validated.
- `sas()` returns per-atom fractions, while `sasa_in_context()` returns only one total area.
- The implementation has no direct residue aggregation.
- The point-versus-neighbor temporary array in `sas()` can consume excessive memory.
- Degenerate and coincident atom cases are not defined consistently.
- Sampling resolution and convergence information are not exposed in results.
- Constants are imported from another package, so the code is not standalone.

## Phase 1: Define the scientific contract

Before implementing scientific behavior, decide and document:

1. The atomic radii set and element-to-radius mapping.
2. Whether explicitly present hydrogen atoms participate.
3. Whether waters, ions, ligands, and other non-protein atoms occlude protein surfaces.
4. Whether residue SASA means residue atoms exposed in the context of the complete structure.
5. Whether to report absolute SASA and the defined CEF metric together.
6. The detailed PDB and mmCIF model, alternate-location, and disorder rules.
7. The default sphere sampling method and resolution.
8. The expected units and numerical tolerance used for validation.

These choices are scientifically meaningful. Implementation must stop and request direction rather than silently selecting an ambiguous convention.

The provisional first-release scope is absolute SASA, a 1.4 Å probe,
deterministic sphere sampling, per-atom and per-residue output, residue values
evaluated in whole-structure context, and PDB/mmCIF input with gzip support.
Hydrogens are excluded unless `--use-h` is supplied. Loose hetero atoms are
excluded unless `--preserve-het` is supplied. This does not yet settle the
atomic radii set or the precise selection rules for hetero atoms and each
calculation mode.

The first relative-exposure metric is **Contextual Exposure Fraction (CEF)**,
not conventional RSA. For residue `i`:

```text
CEF[i] = SASA[i] in the complete protein
         / SASA[i] for the same residue conformation in isolation
```

The naked-residue denominator uses exactly the same residue atoms, selected
atomic radii, probe radius, sphere points, and `ALL` atom-selection rule as the
numerator, but removes all other residues. It therefore normalizes out
self-shielding caused by the residue's own conformation and measures the
fraction of its intrinsic surface retained in the complete protein. The
denominator is computed per residue instance, so terminal residues require no
special reference treatment. `SIDE` mode is deferred.

CEF should be bounded in `[0, 1]` within floating-point roundoff. The
implementation must not silently substitute a residue-type maximum-ASA table;
that would define conventional RSA rather than CEF.

The complete residue-level contract is defined in
[`specs/residue-cef.md`](specs/residue-cef.md).

## Phase 2: Define the public API

Provide composable array-based functions so callers do not need the structure parser:

```python
generate_sphere_points(n_points: int) -> np.ndarray

atom_sasa(
    coordinates,
    radii,
    *,
    probe_radius=1.4,
    sphere_points=None,
    workers=1,
    backend="auto",
) -> np.ndarray

residue_sasa(
    coordinates,
    radii,
    residue_ids,
    *,
    probe_radius=1.4,
    sphere_points=None,
    workers=1,
    backend="auto",
) -> dict

calculate_sasa(
    structure,
    *,
    probe_radius=1.4,
    n_points=None,
    workers=1,
    backend="auto",
):
    ...
```

A structured result will record per-atom fractions and areas, per-residue areas
and CEF values, total area, units, probe radius, sampling resolution, backend,
worker count, and input warnings. Public constants, result types, and
exceptions will be importable directly from `protrsa`.

## Phase 3: Validate and normalize input

- Require coordinates with shape `(N, 3)` and radii with shape `(N,)`.
- Require matching lengths and finite values.
- Require positive atomic radii and a nonnegative probe radius.
- Validate sphere points as a nonempty `(P, 3)` array.
- Define empty atom collections as valid empty/zero results.
- Convert inputs to deliberate contiguous NumPy dtypes.
- Never mutate caller-owned arrays.

## Phase 4: Implement the correctness reference

Create a straightforward CPU implementation:

1. Expand each atomic radius by the probe radius.
2. Place deterministic unit-sphere samples around each atom.
3. Identify samples inside any other expanded atomic sphere.
4. Calculate each atom's exposed fraction and area.
5. Aggregate atom areas into residues and a total separately.
6. For `ALL`, calculate each residue's naked-residue denominator and report
   CEF alongside the contextual residue SASA.

This implementation will be the oracle for optimized backends and remain available for testing. Boundary behavior, including a point exactly on another expanded sphere, will be explicit and identical across backends.

## Phase 5: Optimize the CPU implementation

Use `scipy.spatial.cKDTree` for neighbor discovery and a Numba-compiled kernel for point occlusion with neighbor lists stored in compressed sparse row form.

The optimized path will:

- Calculate SASA for every atom once in whole-structure context, then aggregate per-atom results by residue instead of recalculating SASA for every residue.
- Return per-atom fractions and areas.
- Use squared distances.
- Apply exact pair-specific expanded-radius checks after the conservative KD-tree query so false-positive candidates do not reach the expensive point-testing loop.
- Detect atoms completely enclosed by a larger expanded atomic sphere and assign zero accessible area without sampling their sphere points.
- Order neighbors deterministically by their likelihood of blocking samples, testing closer and more strongly overlapping atoms first so searches terminate earlier.
- Avoid the full `(sphere points × neighbors × 3)` temporary array.
- Handle atoms without neighbors directly.
- Support arbitrary validated sphere-point arrays.
- Preserve deterministic output ordering.
- Follow the reference implementation's mathematical conventions.

Candidate atoms overlap only when their center distance is less than the sum of their probe-expanded radii. Atom `i` is completely enclosed by atom `j` when:

```text
distance(i, j) + expanded_radius_i <= expanded_radius_j
```

Benchmark point-first traversal, neighbor-first traversal with an exposed-point mask, and a spherical-cap formulation using dot products instead of explicit Cartesian sample coordinates. Adopt an alternative only when it preserves documented boundary behavior and materially improves representative workloads.

Use Numba kernel caching and evaluate `parallel=True` with `prange` over independent atoms. Do not enable `fastmath=True` unless boundary-sensitive tests show no scientifically relevant change.

The KD-tree remains the initial production spatial index. Benchmark a uniform linked-cell implementation for very large or repeated calculations, but do not replace the simpler KD-tree without representative evidence.

Reuse immutable sphere points, expanded radii, spatial indexes, filtered and ordered CSR neighbors, complete-burial flags, and atom-to-residue mappings when inputs and parameters are unchanged. Cache invalidation must prevent stale reuse after coordinates, radii, probe radius, or sphere resolution changes.

Use `float64` for the reference and initial CPU implementation. Evaluate `float32` only with documented accuracy and performance tests, especially near occlusion boundaries.

## Phase 6: Add multiprocessing

Use multiprocessing when the workload benefits, with process-worker semantics
defined separately from the Numba CPU-thread `workers` option.

- Partition target atoms into stable contiguous chunks.
- Keep shared inputs read-only and minimize copies where practical.
- Preserve atom order when merging worker outputs.
- Use serial execution for workloads too small to offset process overhead.
- Support a separate process-count control when multiprocessing is introduced.
- Create processes only inside explicit calls and the guarded CLI path.

Serial and multiprocessing results must agree within the approved tolerance.

Benchmark four-process execution against a four-thread Numba `prange` kernel. Threads may avoid serialization and memory-copying overhead. Do not run multiple Numba threads inside every worker process by default, because nested parallelism can oversubscribe the machine. Control worker thread counts explicitly and establish measured crossover thresholds so small structures remain serial.

## Phase 7: Parse structures

Keep parsing separate from numerical calculation. The parser will produce coordinates, elements, radii, atom identifiers, residue identifiers, chain identifiers, and model/alternate-location information.

Support PDB and mmCIF input, including gzip-compressed `.pdb.gz` and `.cif.gz`
files, according to the basic input contract. Apply `--mode`, `--preserve-het`,
and `--use-h` consistently after parsing and before numerical calculation.
Unknown elements and ambiguous records will produce explicit errors or
documented warnings rather than silent guesses.

## Phase 9: Build the CLI

Required commands:

```bash
prot-rsa input.pdb
prot-rsa input.cif.gz --mode SIDE --prob-size 1.40 --workers 4
prot-rsa input.pdb --preserve-het --use-h
```

Implement the positional input and five options exactly as defined in the basic
program setup. A successful run writes the derived `.atom.sas` and `.res.sas`
files. Errors go to standard error, and invalid input or calculation failure
returns a nonzero exit status.

## Phase 10: Test and validate

Every new function will receive focused `pytest` coverage, and tests will run before its phase is complete.

### Numerical correctness

- One isolated atom
- Two distant atoms
- Partially overlapping expanded spheres
- A completely buried atom
- Target atoms occluded by background atoms
- Empty target and background collections
- Custom sphere-point arrays and counts
- Invalid shapes, lengths, values, and radii
- Coincident atom centers
- Atom-to-residue aggregation
- Deterministic repeated execution

### Backend equivalence

- Reference versus optimized CPU
- Serial versus four-worker multiprocessing
- Identical output ordering across backends

### Performance validation

- Measure KD-tree construction, neighbor-list construction, occlusion, parsing, aggregation, and parallel startup separately.
- Compare point-first, neighbor-first, and spherical-cap kernels.
- Compare serial Numba, four-thread Numba, and four-process execution.
- Measure peak memory as well as elapsed time.
- Benchmark small, medium, and large representative structures.
- Establish measured thresholds for serial and parallel selection.
- Treat material slowdowns as regressions unless justified by correctness or maintainability.

### Scientific validation

- Compare selected structures with an agreed independent SASA implementation.
- Record radii, probe, hydrogen, heteroatom, and sampling conventions.
- Test convergence as sphere-point resolution increases.
- Document expected discrepancies instead of relaxing tolerances without scientific justification.

### CLI and packaging

- Importing `protrsa` has no side effects.
- `python protrsa.py --help` succeeds.
- Installed `prot-rsa --help` succeeds.
- PDB and mmCIF inputs produce both expected output files.
- `.pdb.gz` and `.cif.gz` inputs use the correct output basename.
- Each CLI default and presence flag behaves according to the input contract.
- Invalid modes, extensions, probe sizes, and worker counts are rejected.
- Failed runs do not leave only one output file or partial output files.
- CLI failures return appropriate exit codes.
- Source and wheel distributions include `protrsa.py`, `README.md`, and `LICENSE`.
- Source and wheel distributions include `compare_sas.py`, and the
  `prot-rsa-compare` entry point resolves to that separate module.
- Built distributions pass PyPI metadata validation.

## Public-release requirements

Before the stable release:

- Document the algorithm, scientific conventions, limitations, and expected accuracy.
- Provide CLI and Python examples.
- State supported Python versions and platforms accurately.
- Publish benchmark conditions with performance results.
- Record user-visible and compatibility changes.
- Exclude private paths, credentials, unpublished data, and environment assumptions.

## Delivery sequence

1. Resolve scientific and input/output decisions.
2. Define constants, exceptions, result types, and validation.
3. Implement deterministic sphere generation.
4. Implement and test the reference algorithm.
5. Implement and test KD-tree and Numba CPU acceleration according to
   [`specs/atom-sasa-numba.md`](specs/atom-sasa-numba.md).
6. Add exact neighbor filtering, complete-burial masking, neighbor ordering,
   and residue aggregation according to
   [`specs/atom-sasa-burial.md`](specs/atom-sasa-burial.md) and
   [`specs/residue-cef.md`](specs/residue-cef.md).
7. Implement and test the Numba kernel according to
   [`specs/atom-sasa-numba.md`](specs/atom-sasa-numba.md), then benchmark
   kernel layouts.
8. Implement and benchmark Numba thread parallelism according to
   [`specs/atom-sasa-numba-parallel.md`](specs/atom-sasa-numba-parallel.md),
   without changing multiprocessing worker semantics.
9. Implement and test structure parsing and the CLI.
10. Run independent scientific validation and benchmarks.
11. Complete documentation and validate PyPI distributions.

No phase that depends on an unresolved scientific convention will proceed by assumption.

# Naive Atom SASA Reference Specification

**Status:** Ready for implementation

## Purpose and scope

Implement a deterministic, serial reference calculation of absolute
solvent-accessible surface area (SASA) for each atom. This implementation is
the correctness oracle for later optimized CPU, multiprocessing, and GPU
implementations.

This stage includes atom SASA only. It must not calculate or serialize residue
SASA, relative accessibility, exposed fractions, percentages, or normalized
values. Residue SASA will be specified only after optimized atom SASA has been
validated against this reference.

The eventual `.atom.sas` output is a tab-separated-values (TSV) file. Its
columns, headers, precision, ordering metadata, and overwrite behavior belong
in a separate atom-output specification. This numerical reference function
does not perform serialization.

The algorithm is the Shrake–Rupley dot-surface method described by Shrake and
Rupley (1973). The implementation must remain in the single import-safe
`protrsa.py` module.

## Structure input prerequisite

The same module must expose a reusable, import-safe structure reader for PDB,
mmCIF, and their gzip-compressed forms. Format selection follows the suffix
rules in [`command-line-interface.md`](command-line-interface.md). Reading
must not perform radius assignment, atom-mode filtering, loose-component
filtering, SASA calculation, or output serialization.

The reader must retain `ATOM` and `HETATM` records in input order and expose,
for each retained atom, its record type, source atom identifier, element, atom
name, alternate-location identifier, residue name, chain identifier, residue
sequence identifier, insertion code, Cartesian coordinates, occupancy, and
model number. Missing optional text values are normalized to the empty string;
missing occupancy is represented by `None`. For mmCIF identifiers, prefer the
`auth_*` value when it is present and otherwise use its corresponding
`label_*` value. A mmCIF `.` or `?` value is missing.

A structure containing more than one coordinate model must be rejected with a
clear error. A file without explicit model records is model 1. This stage must
not merge models or silently use only the first model.

Alternate locations are resolved consistently for the whole residue:

1. Atoms with no alternate-location identifier are shared and always retained.
2. For each residue containing nonblank alternate identifiers, calculate the
   mean occupancy separately for every identifier.
3. A label for which every occupancy is explicit ranks above a label with any
   missing occupancy. Among labels in the same completeness class, choose the
   greatest mean of the available occupancies; a label with no available
   occupancies has a mean below every numeric mean.
4. Occupancy ties prefer label `A`, then use lexical identifier order.
5. Retain the shared atoms and all atoms bearing the selected identifier,
   preserving their original relative order. Do not select alternates
   independently per atom, because that can create a hybrid residue that was
   not present in the structure.

PDB occupancy comes from columns 55–60 and the alternate identifier from
column 17. mmCIF uses `_atom_site.occupancy` and
`_atom_site.label_alt_id`. Malformed coordinate records, malformed gzip data,
missing required mmCIF atom-site fields, an empty atom collection, and multiple
models must raise a clear structure-reading error.

## Atom normalization and initial output

After alternate-location selection, normalize atoms before calling the
numerical oracle. With `--use-h` absent, discard atoms whose normalized element
is `H` or `D` and assign residue-aware ProtOr united-atom radii. With `--use-h`
present, retain supplied `H` and `D` atoms and assign every atom from
`EXPLICIT_ATOM_RADII`; never mix the two radius sets. Elements absent from the
selected table use `X` and `UNKNOWN_RADIUS = 2.00` Å. The program does not add
hydrogens.

For the initial ProtOr mapping, standard amino-acid nitrogen uses `N`; sulfur,
phosphorus, and selenium use their corresponding ProtOr entries; backbone and
side-chain carbonyl oxygen uses `CARBONYL_O`; and `SER OG`, `THR OG1`, and
`TYR OH` use `HYDROXYL_O`. Backbone `C`, side-chain amide/carboxyl carbon, and
substituted aromatic carbon use `TRIGONAL_C_NO_H`; aromatic carbons bearing one
hydrogen use `TRIGONAL_C_ONE_H`; other standard amino-acid carbons use
`TETRAHEDRAL_C`. Atoms not resolvable by these rules use the approved unknown
fallback.

Unless `--preserve-het` is present, discard `HETATM` records whose normalized
residue name belongs to `LOOSE_HETERO_COMPONENTS`. Retain other `HETATM`
records. With `--preserve-het`, retain both classes. Hydrogen filtering remains
independent and still applies unless `--use-h` is present.

The normalized internal record must retain the source atom metadata and add
the normalized element and assigned radius. Its coordinates and radii are
passed, in the same order, to `atom_sasa_reference()`.

The initial atom-only CLI supports `--mode ALL`; it must reject `SIDE` and
`KEY` clearly until their atom-selection rules are specified. It writes:

- `<base>.atom.sas`, a TSV with atom index, source record type and identifier,
  atom and residue identifiers, normalized element, radius in Å, and absolute
  SASA in Å²; and
- `<base>.pqr`, containing the same retained atoms and radii, sequential output
  serials, and the explicitly approved placeholder charge `0.000` for every
  atom.

No `.res.sas` file is written during this atom-only stage. Output order is the
normalized atom order. SASA is written to six decimal places and radii to three
decimal places. PQR output is a radius-bearing interchange file only; its zero
charges must not be interpreted as an electrostatic charge model. The CLI must
refuse to overwrite either output. If writing the pair fails, it must remove
any newly created partial output from that invocation.

References:

- Shrake and Rupley, *Environment and exposure to solvent of protein atoms:
  Lysozyme and insulin*, DOI `10.1016/0022-2836(73)90011-9`.
- MDTraj `shrake_rupley`, which uses a golden-section spiral and defaults to
  960 sphere points.
- Biopython `Bio.PDB.SASA.ShrakeRupley`, an independent implementation of the
  same algorithm.

## Normative constants

Add the following public constant to the constants section of `protrsa.py`:

```python
DEFAULT_SPHERE_POINTS = 960
```

The default probe size must be the existing `DEFAULT_PROBE_SIZE = 1.40` Å.
The calculation must use these constants directly as its defaults. Scientific
and classification constants defined in
[`constants_decision.md`](constants_decision.md) remain authoritative.

## Public numerical interface

Expose these reusable functions directly from `protrsa.py`:

```python
def generate_sphere_points(
    n_points: int = DEFAULT_SPHERE_POINTS,
) -> np.ndarray:
    """Return deterministic float64 unit-sphere points with shape (P, 3)."""


def atom_sasa_reference(
    coordinates,
    radii,
    *,
    probe_size: float = DEFAULT_PROBE_SIZE,
    sphere_points: np.ndarray | None = None,
) -> np.ndarray:
    """Return absolute per-atom SASA in square angstroms."""
```

Neither function may read files, inspect command-line arguments, create worker
processes, initialize a GPU runtime, write output, or mutate caller-owned
objects.

This stage does not connect the numerical function to `main()`. Until parsing,
atom filtering, radius assignment, and atom-output serialization are specified
and implemented, a non-help CLI invocation must continue to report that the
calculation is not implemented.

When `sphere_points` is `None`, `atom_sasa_reference()` must call
`generate_sphere_points(DEFAULT_SPHERE_POINTS)`. It must not depend on a
mutable module-level point array. A caller-supplied point array replaces the
default completely after validation.

## Deterministic sphere generation

Generate points with the midpoint Fibonacci/golden-angle construction. For
`k = 0, ..., P - 1`:

```text
z[k]         = 1 - 2 * (k + 0.5) / P
theta[k]     = k * pi * (3 - sqrt(5))
radial[k]    = sqrt(max(0, 1 - z[k]**2))
x[k]         = radial[k] * cos(theta[k])
y[k]         = radial[k] * sin(theta[k])
point[k]     = (x[k], y[k], z[k])
```

The result must:

- have shape `(P, 3)` and dtype `float64`;
- contain only finite values;
- place every point on the unit sphere within `1e-12` absolute and relative
  tolerance;
- be bitwise-identical across repeated calls in the same runtime and
  agree across supported platforms within `1e-14` absolute and relative
  tolerance;
- preserve the ordering implied by increasing `k`; and
- accept instances of `numbers.Integral`, including NumPy integer scalars;
  convert the accepted value with `int()` before array construction; and
- raise `TypeError` for Boolean or non-integral point counts and `ValueError`
  for integer values less than one.

There is no random rotation or random seed. The number 960 is the chosen
sampling resolution, not an icosahedral vertex count.

## Input contract

`atom_sasa_reference()` accepts array-like coordinate and radius inputs:

- `coordinates`: numeric Cartesian coordinates with shape `(N, 3)`, in Å;
- `radii`: assigned atomic radii with shape `(N,)`, in Å;
- `probe_size`: a finite scalar greater than or equal to zero, in Å; and
- optional `sphere_points`: a numeric, finite, nonempty `(P, 3)` array of unit
  vectors.

The function must convert working arrays with `np.asarray(..., dtype=np.float64)`
and make them C-contiguous when necessary, without mutating the original
inputs. Values that cannot be converted to `float64` must raise `TypeError`
(conversion exceptions may be caught and re-raised with context). After
conversion, it must reject:

- incorrect dimensions or shapes;
- different atom counts in `coordinates` and `radii`;
- NaN or infinite values;
- zero or negative atomic radii;
- a Boolean or non-`numbers.Real` probe size with `TypeError`;
- a negative or non-finite real probe size with `ValueError`;
- empty custom sphere-point arrays;
- custom points that are not unit vectors within `1e-12` absolute and relative
  tolerance; and
- two distinct atoms with exactly coincident coordinates, because absolute
  per-atom attribution is ambiguous for duplicate centers.

The accepted probe size must be converted with `float()` before calculation.
Coincident centers are tested after conversion to `float64`. Shape, finiteness,
radius, sphere-norm, and coincident-center validation failures must raise
`ValueError` with a message identifying the invalid argument or condition.

An empty atom collection represented by shapes `(0, 3)` and `(0,)` is valid
and must return an empty `float64` array. No maximum or reduction may be
performed before this case is handled.

Atom filtering and radius assignment happen before this numerical function.
The function must not interpret elements, atom names, residue names,
`--use-h`, `--preserve-het`, or `--mode`.

## Shrake–Rupley calculation

For each atom `i`, calculate its probe-expanded radius:

```text
R[i] = radii[i] + probe_size
```

For every unit-sphere point `u[k]`, calculate a sample point on atom `i`:

```text
p[i, k] = coordinates[i] + R[i] * u[k]
```

The point is blocked if it is inside or exactly on the probe-expanded sphere
of any other atom `j`:

```text
squared_distance(p[i, k], coordinates[j]) <= R[j]**2
```

The atom itself must be excluded by index, not by coordinate comparison. Stop
testing a point as soon as one blocking atom is found.

If `E[i]` of the `P` sample points remain exposed, return the absolute atom
SASA:

```text
atom_sasa[i] = 4 * pi * R[i]**2 * E[i] / P
```

The returned array must have shape `(N,)`, dtype `float64`, preserve input atom
order, and contain values in the closed interval
`[0, 4 * pi * R[i]**2]`, subject only to ordinary `float64` rounding.

The denominator must always be the actual number of supplied sphere points,
never the global default. The implementation must retain full `float64`
precision through the calculation.

For validation, two equal probe-expanded spheres of radius `R` whose centers
are separated by `d`, where `0 < d < 2R`, have the following analytical
exposed area per sphere:

```text
2 * pi * R**2 + pi * R * d
```

The sampled result is expected to approach this value as `P` increases; it is
not expected to equal it exactly at finite resolution.

## Deliberately naive implementation

The reference path must directly test each target atom's sphere points against
all other atoms. Its expected complexity is `O(N**2 * P)`.

Do not add any of the following to the reference function:

- KD-trees or other spatial indexes;
- precomputed neighbor lists;
- burial or overlap shortcuts;
- Numba or another JIT compiler;
- multiprocessing or threading;
- GPU execution;
- reduced precision;
- random sampling; or
- a full `(N, P, N, 3)` or `(P, N, 3)` points-by-neighbors temporary.

Small temporary arrays for one point or one target atom are acceptable. The
serial reference is intentionally exempt from parallel execution because its
purpose is to validate parallel and optimized implementations. Production
dispatch will use the optimized implementation once equivalence is proven.

## Required tests

Add focused `pytest` tests covering at least:

1. The default point count is exactly 960 and comes from
   `DEFAULT_SPHERE_POINTS`.
2. Generated point shape, `float64` dtype, finiteness, unit norms,
   determinism, and invalid point counts.
3. A single atom agrees with `4 * pi * (radius + probe_size)**2` within
   floating-point tolerance.
4. Multiple sufficiently distant atoms each retain their full isolated area.
5. At the default 960 points, two partially overlapping equal spheres each
   have less than their isolated area and each agree with the analytical
   exposed-cap area within 2% relative error. Choose geometry whose analytical
   area is nonzero. Their finite-sampling values need not be exactly equal
   because the ordered Fibonacci set is not necessarily inversion-symmetric.
   Do not require monotonic error reduction as point count increases or an
   error of at most one sample-point area; neither is guaranteed for every
   spherical cap sampled by a Fibonacci point set.
6. A completely enclosed atom has zero SASA while the enclosing atom remains
   exposed as dictated by the geometry.
7. Translating all coordinates leaves results unchanged within tight
   numerical tolerance.
8. Permuting atom order only applies the same permutation to the output.
9. Repeated calls return bitwise-identical results on the same platform.
10. A custom sphere-point count uses its own length as the denominator.
11. Boundary points at exactly an occluder's expanded radius are blocked.
12. Empty inputs return an empty `float64` result.
13. Every invalid shape, length, value, radius, probe size, sphere point, and
    coincident-center case raises the exception category specified by the
    input contract. In particular, Boolean probe sizes and Boolean or
    non-integer point counts must be rejected rather than accepted as Python
    integers.
14. Caller-owned coordinate, radius, and sphere-point arrays are unchanged.
15. The returned values are absolute areas in Å² rather than exposed
    fractions.

Tests must not depend on optional third-party SASA packages. Comparison with
MDTraj, Biopython, or FreeSASA may be added as an optional validation test but
cannot replace analytical and invariant-based tests.

## Convergence validation

Before treating 960 as a permanent production default, evaluate point counts:

```text
122, 240, 480, 960, 1920, 3840
```

Use the downloaded small, medium, and large benchmark structures. Treat the
3840-point calculation as the provisional high-resolution reference and
record, for each lower resolution:

- maximum absolute per-atom difference;
- root-mean-square per-atom difference;
- absolute total-SASA difference,
  `abs(sum(candidate_atom_sasa) - sum(reference_atom_sasa))`; and
- elapsed time.

This validation informs a later decision; it must not silently change
`DEFAULT_SPHERE_POINTS`.

## Acceptance criteria

The naive atom-SASA stage is complete when:

- both public functions satisfy this specification;
- results are deterministic absolute per-atom areas in Å²;
- the reference contains no optimization or parallel execution path;
- all focused tests and the full existing test suite pass; and
- no residue-SASA calculation or serialization is introduced.

# Constants and Classification Decisions

## Purpose

This document records approved scientific and input-classification constants
for `prot-rsa`. Coding agents must treat these values and sets as normative.
Changing them requires an explicit project decision and corresponding tests.

## Solvent probe

The default solvent probe radius is:

```python
DEFAULT_PROBE_SIZE = 1.40  # angstroms
```

The command-line option controlling this value is `--prob-size`. All radii and
coordinates are expressed in angstroms, and calculated surface areas are
expressed in square angstroms.

## Sphere sampling

The initial Shrake–Rupley sampling resolution is:

```python
DEFAULT_SPHERE_POINTS = 960
```

This value applies to the deterministic midpoint Fibonacci sphere defined in
[`atom-sasa-naive.md`](atom-sasa-naive.md). It must not be changed based on a
different sphere construction or a performance optimization without a new
decision supported by convergence results.

## Atomic-radius modes

### Default hydrogen-free mode

When `--use-h` is absent:

- Hydrogen and deuterium atoms supplied in the input must be excluded.
- Standard protein heavy atoms must use the ProtOr united-atom radii from
  Tsai et al. ProtOr radii account for hydrogens covalently attached to each
  heavy atom and therefore must be assigned by chemical atom type, not solely
  by element.
- The ProtOr assignment must be residue-and-atom-aware. Carbonyl, aromatic,
  and aliphatic carbon atoms must not all receive one elemental carbon radius.

The core ProtOr radii are:

| ProtOr united-atom type | Radius (Å) |
| --- | ---: |
| Trigonal carbon without hydrogen | 1.61 |
| Trigonal or aromatic carbon with one hydrogen | 1.76 |
| Tetrahedral carbon with one, two, or three hydrogens | 1.88 |
| Nitrogen | 1.64 |
| Carbonyl oxygen | 1.42 |
| Hydroxyl oxygen | 1.46 |
| Sulfur | 1.77 |
| Phosphorus extension | 1.80 |
| Selenium extension | 1.90 |

The ProtOr nitrogen radius applies to every nitrogen atom retained in
hydrogen-free mode, including nitrogen in nonstandard residues and retained
`HETATM` components.

The complete residue/atom-to-ProtOr-type mapping must be specified alongside
the structure parser before radius assignment is implemented.

### Explicit-hydrogen mode

When `--use-h` is present:

- Hydrogen and deuterium atoms supplied in the input must be retained.
- The program must not add missing hydrogen atoms.
- Every atom, including heavy atoms, must use a single-atom van der Waals
  radius. ProtOr radii must not be mixed with explicit hydrogens.
- The single-atom radii are the Mantina et al. 2009 main-group van der Waals
  radii, with the approved Fe and unknown-element extensions below.

The elements required by the initial protein and loose-component scope are:

| Element | Radius (Å) |
| --- | ---: |
| H, D | 1.10 |
| C | 1.70 |
| N | 1.55 |
| O | 1.52 |
| F | 1.47 |
| P | 1.80 |
| S | 1.80 |
| Cl | 1.75 |
| Se | 1.90 |
| Br | 1.83 |
| I | 1.98 |
| Li | 1.81 |
| Na | 2.27 |
| K | 2.75 |
| Rb | 3.03 |
| Cs | 3.43 |
| Fe | 2.00 |
| X | 2.00 |

Deuterium uses the hydrogen radius because it has the same electronic size for
this purpose.

### Iron and unresolved atoms

The approved iron and unresolved-atom constants are:

```python
IRON_RADIUS = 2.00    # angstroms
UNKNOWN_RADIUS = 2.00  # angstroms; element symbol X
```

An atom whose element or ProtOr atom type cannot be resolved must be normalized
to the synthetic element/type `X` and assigned `UNKNOWN_RADIUS`. The parser
must retain the original atom and residue names for output and diagnostics.
Radius fallback must be deterministic and must not depend on the execution
backend or worker count.

## Loose hetero-components

"Loose hetero-components" are complete non-polymer components normally
associated with solvent, simple salts, or crystallization conditions. The
classification applies to the entire component/residue, not to isolated atoms.

The initial approved sets are:

```python
WATER_COMPONENTS = frozenset({
    "HOH", "WAT", "H2O", "DOD", "SOL", "OH2", "TIP", "TIP3", "TIP4",
})

SIMPLE_ION_COMPONENTS = frozenset({
    "LI", "NA", "K", "RB", "CS",
    "F", "CL", "BR", "I",
    "NH4", "NO3", "CO3", "SO4", "PO4",
})

CRYSTALLIZATION_ADDITIVE_COMPONENTS = frozenset({
    "ACT", "ACY", "CIT", "TAR",
    "GOL", "EDO", "PEG", "PG4", "PGE", "MPD",
    "MOH", "EOH", "IPA", "DMS",
    "BME", "DTT",
    "TRS", "MES", "HEP", "BIC",
})

LOOSE_HETERO_COMPONENTS = (
    WATER_COMPONENTS
    | SIMPLE_ION_COMPONENTS
    | CRYSTALLIZATION_ADDITIVE_COMPONENTS
)
```

Component identifiers must be stripped of surrounding whitespace and
normalized to uppercase before membership testing.

Unless `--preserve-het` is supplied, all atoms belonging to a non-polymer
component in `LOOSE_HETERO_COMPONENTS` must be excluded. When
`--preserve-het` is supplied, these components must be retained. A component
that is part of a polymer or explicitly covalently linked to a polymer must not
be removed solely because its identifier appears in the loose-component set.

Transition metals such as `ZN`, `CU`, `MN`, `CO`, and `NI`, and biological
cofactors or nucleotides such as `HEM`, `FAD`, `FMN`, `NAD`, `NAP`, `ATP`,
`ADP`, `GDP`, and `GTP`, are not loose hetero-components.

## Required tests

Implementation tests must verify:

1. The default probe size is exactly `1.40` Å.
2. Hydrogen-free mode excludes H/D and uses residue-aware ProtOr assignments.
3. Explicit-hydrogen mode retains supplied H/D and uses the single-atom table
   for both hydrogen and heavy atoms.
4. Explicit-hydrogen mode never mixes ProtOr radii into the same calculation.
5. Fe and unresolved `X` atoms receive exactly `2.00` Å.
6. Every listed loose component is excluded by default and retained with
   `--preserve-het`.
7. Loose-component matching is case-insensitive and whitespace-insensitive.
8. Polymer or polymer-linked components are not discarded based only on their
   component identifier.
9. Radius and component classification results are identical across serial,
   multiprocessing and CPU execution.

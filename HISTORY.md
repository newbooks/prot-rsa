# Release history

## 1.0.0 — current release

`prot-rsa` is an importable Python module and command-line program for
calculating solvent-accessible surface area (SASA) for protein structures.

### Current capabilities

- Reads PDB, mmCIF, `.pdb.gz`, and `.cif.gz` structures.
- Resolves alternate locations and rejects multiple coordinate models.
- Calculates atom SASA with deterministic Fibonacci sphere sampling (960 points
  by default; configurable with `--sphere-points`, minimum 122).
- Uses cKDTree neighbor pruning, vectorized occlusion masks, ordered neighbors,
  cached geometric terms, complete-burial skipping, and a Numba CPU backend.
- Supports one or more Numba worker threads through `--workers`.
- Supports explicit-hydrogen mode (`--use-h`) and optional retained loose
  hetero-atoms (`--preserve-het`).
- Produces both complete-residue (`ALL`) and side-chain (`SIDE`) contextual
  exposure fractions (CEF).
- Uses `NA` for SIDE values of residues without retained side-chain atoms,
  including glycine under the canonical backbone definition.
- Exposes reusable calculation and parsing functions through `protrsa` without
  command-line or filesystem side effects on import.

The separate atom-file comparison helper is not part of this release.

### Output files and schemas

For an input such as `1LYZ.pdb`, the command writes:

```text
1LYZ.atom.sas
1LYZ.res.sas
1LYZ.pqr
```

`*.atom.sas` is a UTF-8 tab-separated file with this header:

```text
atom_index
record_type
source_id
atom_name
residue_name
chain_id
residue_sequence
insertion_code
element
radius_A
sasa_A2
```

`*.res.sas` is a UTF-8 tab-separated file with this header:

```text
residue_name
chain_id
residue_sequence
insertion_code
sasa_all
sasa_all_ref
sasa_all_ratio
sasa_side
sasa_side_ref
sasa_side_ratio
```

The first four residue fields identify the residue. The `*_all` values use all
normalized residue atoms. The `*_side` values exclude backbone atoms `N`,
`CA`, `C`, `O`, and `OXT`. Reference values are calculated with the selected
residue atoms isolated from the rest of the protein. Numeric residue values
are formatted to three decimal places; SIDE fields are `NA` when no SIDE atoms
remain.

`*.pqr` contains PDB-style records with normalized coordinates, a placeholder
charge of `0.000`, and the assigned radius. It has no tabular header.

### Expected performance

Representative one-thread timings on the canonical benchmark structures are
shown below. These are wall-clock medians and are hardware- and environment-
dependent; they are performance expectations rather than guarantees.

| Configuration | Sphere points | 1LYZ (1,001 atoms) | 1CA2 (2,040 atoms) | 1UOR (4,616 atoms) |
| --- | ---: | ---: | ---: | ---: |
| Naive reference | 960 | 551 s | 2,185 s | 11,634 s |
| cKDTree | 960 | 19.490 s | 40.644 s | 86.590 s |
| Vectorized mask | 960 | 0.609 s | 1.285 s | 2.820 s |
| Occlusion-ordered neighbors | 960 | 0.319 s | 0.630 s | 1.620 s |
| Cached NumPy kernel | 960 | 0.276 s | 0.536 s | 1.393 s |
| Numba CPU | 960 | 0.328 s | 0.388 s | 0.735 s |
| Numba CPU, reduced sampling | 480 | 0.358 s | 0.382 s | 0.494 s |

Halving the sphere-point count does not halve total wall-clock time because
validation, neighbor construction, and other fixed costs remain. The 480-point
results had atom-SASA MAE values of approximately 0.164, 0.145, and 0.184 Å²
per atom for 1LYZ, 1CA2, and 1UOR, respectively, relative to 960-point output.

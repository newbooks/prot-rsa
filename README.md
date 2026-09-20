# prot-rsa

Fast protein residue surface-area calculation in Python.

## Installation

Install the distribution from PyPI:

```bash
pip install prot-rsa
```

## Command-line use

Run the installed application with:

```bash
prot-rsa structure.pdb
# optionally choose a lower sampling resolution (minimum 122 points)
prot-rsa structure.pdb --sphere-points 480
```

Run `prot-rsa --help` to see the supported PDB/mmCIF input suffixes, calculation
options, defaults, and derived output filenames.

The command writes `<base>.atom.sas` and `<base>.res.sas` as TSV files, plus a
normalized `<base>.pqr` containing the selected radii and placeholder charge
`0.000`. The production calculation uses cKDTree neighbor pruning; the
deliberately naive serial implementation remains available as a correctness
reference.

## Python use

The application can also be imported as the `protrsa` module so its functions
and constants can be used directly:

```python
import protrsa
```

The distribution and command are named `prot-rsa`. The import name is
`protrsa` because Python module names cannot contain hyphens.

## Residue exposure

Residue solvent-accessible surface area will be calculated by summing the
already computed atom SASAs for each residue. The residue report will contain
both complete-residue (`ALL`) and side-chain (`SIDE`) **Contextual Exposure
Fraction (CEF)** values rather than conventional RSA:

```text
CEF = selected-atom SASA in the complete protein
      / SASA of the same selected residue conformation in isolation
```

The denominator is a naked-residue calculation using the same selected atoms,
atomic radii, probe radius, and sphere points, but without other residues
present. The compact output columns are `all`, `a_ref`, `a_cef`, `side`,
`s_ref`, and `s_cef`; leading `#` comment lines explain every field. CEF therefore represents
the fraction of the selected residue surface retained in its protein context.
It is distinct from conventional RSA, which normally uses a fixed residue-type
reference or maximum ASA.

Residues without retained side-chain atoms, such as glycine under the
canonical backbone definition, use `NA` for the three SIDE fields.

CEF lies in `[0, 1]` up to floating-point roundoff. The complete residue
output contract is documented in `docs/specs/residue-cef.md`.


## Optimization Comparison

This program targets the speed optimization of residue surface solvent exposure calculation.

The table below compares protein RSA calculation times under each
optimization. Execution times are reported in seconds; lower values are
better.

| Optimization | Sphere points | Time (small) | Time (medium) | Time (large) | Atom-SASA MAE vs `*.sas.baseline` (small / medium / large, Å²) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Naive | 960 | 551 | 2185 | 11634 | 0.000 / 0.000 / 0.000 |
| cKDTree | 960 | 19.490 | 40.644 | 86.590 | 0.000 / 0.000 / 0.000 |
| Vectorized mask | 960 | 0.609 | 1.285 | 2.820 | 0.000 / 0.000 / 0.000 |
| Occlusion-ordered neighbors | 960 | 0.319 | 0.630 | 1.620 | 0.000 / 0.000 / 0.000 |
| Cache | 960 | 0.276 | 0.536 | 1.393 | 0.000 / 0.000 / 0.000 |
| Numba CPU | 960 | 0.328 | 0.388 | 0.735 | 0.000 / 0.000 / 0.000 |
| Reduced points | 480 | 0.358 | 0.382 | 0.494 | 0.164 / 0.145 / 0.184 |

Benchmark structures are **small** — 1LYZ (129 residues); **medium** — 1CA2
(256 residues); **large** — 1UOR (580 residues). Residue counts are the numbers
of unique residues represented by `ATOM` records; waters, ions, and other
`HETATM` records are excluded.

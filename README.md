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
```

Run `prot-rsa --help` to see the supported PDB/mmCIF input suffixes, calculation
options, defaults, and derived output filenames.

The initial atom-only implementation writes `<base>.atom.sas` as TSV and a
normalized `<base>.pqr` containing the selected radii and placeholder charge
`0.000`. The production calculation uses cKDTree neighbor pruning; the
deliberately naive serial implementation remains available as a correctness
reference.

Compare two generated atom-SASA files with:

```bash
prot-rsa-compare first.atom.sas second.atom.sas
```

The comparison verifies every ordered atom-identity field and row before
reporting the mean absolute error between the two `sasa_A2` columns. Radius
and SASA values may differ. A missing,
malformed, reordered, or otherwise different atom row is reported as an error
instead of producing a potentially misaligned MAE.

## Python use

The application can also be imported as the `protrsa` module so its functions
and constants can be used directly:

```python
import protrsa
```

The distribution and command are named `prot-rsa`. The import name is
`protrsa` because Python module names cannot contain hyphens.

## Optimization Comparison

The table below compares protein RSA calculation times under each
optimization. Execution times are reported in seconds; lower values are
better.

| Optimization | Threads | Sphere points | Time (small) | Time (medium) | Time (large) | Atom-SASA MAE vs `*.sas.baseline` (Å²) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive | 1 | 960 | 551 | 2185 | 11634 | 0 |
| cKDTree | 1 | 960 | 18 | 37 | 84 | 0 |
| Vectorized mask | 1 | 960 | 0.61 | 1.29 | 2.82 | 0 |
| Occlusion-ordered neighbors | 1 | 960 | 0.32 | 0.63 | 1.62 | 0 |
| Cache | 1 | 960 | 0.28 | 0.56 | 1.45 | 0 |

Benchmark structures: **small** — 1LYZ (129 residues); **medium** — 1CA2
(256 residues); **large** — 1UOR (580 residues). Residue counts are the numbers
of unique residues represented by `ATOM` records; waters, ions, and other
`HETATM` records are excluded.

Benchmark results should be collected using the same input structures, probe
radius, sampling resolution, and hardware. Record the protein used for each
size category, the number of atoms and residues, software versions, and the
CPU and GPU models alongside the final results so that the comparison is
reproducible. Spatial timings are medians of five measured calls after one
warm-up. The benchmark retained 1,001 atoms for 1LYZ, 2,040 for 1CA2, and
4,616 for 1UOR after the documented default filtering.

| Structure | Directed neighbors (mean/max) | Tree/list time (s) | cKDTree time (s) | Vectorized time (s) | Vectorized speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1LYZ | 41,780 (41.74/70) | 0.004674 | 19.490129 | 0.608863 | 32.01x |
| 1CA2 | 88,946 (43.60/75) | 0.010259 | 40.643580 | 1.285390 | 31.62x |
| 1UOR | 190,354 (41.24/72) | 0.021744 | 86.589739 | 2.820431 | 30.70x |

Occlusion ordering was benchmarked against a fresh index-ordered vectorized
baseline using identical inputs and five measured calls after one warm-up.
The ordered CSR build time is included in the total.

| Structure | Index-ordered time (s) | Ordered CSR build (s) | Occlusion-ordered time (s) | Net speedup |
| --- | ---: | ---: | ---: | ---: |
| 1LYZ | 0.433064 | 0.013525 | 0.319389 | 1.36x |
| 1CA2 | 0.907509 | 0.020019 | 0.630230 | 1.44x |
| 1UOR | 1.969482 | 0.045665 | 1.619650 | 1.22x |

The cache optimization was compared with the uncached occlusion-ordered
kernel using identical CSR neighbors and five warmed-up calls. Median kernel
times were 0.316585 s versus 0.282086 s for 1LYZ (1.12x), 0.612894 s versus
0.549928 s for 1CA2 (1.11x), and 1.580235 s versus 1.412985 s for 1UOR
(1.12x). Cached results were bitwise identical to the uncached kernel.

The production-dispatch crossover was also evaluated with deterministic dense
synthetic grids using the same 960 sphere points. Coordinates were generated
from cubic grids with 2.1 Å spacing, with a repeating y-offset of 0.00, 0.17,
and 0.34 Å; atomic radii repeated five evenly spaced values from 1.45 through
1.80 Å.

| Atoms | Directed neighbors (mean/max) | Tree/list time (s) | Reference time (s) | cKDTree time (s) | Vectorized time (s) | Vectorized speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 27 | 636 (23.56/26) | 0.000156 | 0.371691 | 0.355585 | 0.010314 | 34.48x |
| 64 | 2,292 (35.81/58) | 0.000319 | 1.924054 | 1.081588 | 0.036195 | 29.88x |
| 125 | 5,616 (44.93/91) | 0.000686 | 7.239474 | 2.426148 | 0.080361 | 30.19x |

These measurements used one process on an AMD Ryzen 9 6900HX under Linux,
with NumPy 2.4.4 and SciPy 1.18.0. The spatial and reference results were
bitwise identical. The vectorized results were also bitwise identical to the
saved cKDTree outputs: maximum per-atom difference, MAE, and total-SASA
difference were all zero. Strict row-by-row comparisons of newly generated
vectorized outputs against `work/1LYZ.atom.sas.baseline`,
`work/1CA2.atom.sas.baseline`, and `work/1UOR.atom.sas.baseline` matched 1,001,
2,040, and 4,616 atoms respectively and produced an exact MAE of 0.0 Å² for
each structure. A sparse 125-atom input completed in 0.000211 s, and 1LYZ with
122 sphere points completed in 0.253982 s. The vectorized kernel clears its 5x
medium/large production gate by a wide margin. With occlusion ordering, the
dense 27-, 64-, and 125-atom cases completed in 0.007628, 0.022669, and
0.047146 s; the sparse 125-atom case completed in 0.000204 s; and the
122-point 1LYZ case completed in 0.153097 s. Ordered outputs remained bitwise
identical, and regenerated canonical SAS files retained exact 0.0 Å² MAE
against all three baselines.

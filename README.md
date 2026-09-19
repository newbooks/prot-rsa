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

Compare two generated atom-SASA files with the separate comparison utility:

```bash
prot-rsa-compare first.atom.sas second.atom.sas
# or, from a source checkout:
python compare_sas.py first.atom.sas second.atom.sas
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

## Residue exposure (planned)

Residue solvent-accessible surface area will be calculated by summing the
already computed atom SASAs for each residue. The first relative-exposure mode
will report **Contextual Exposure Fraction (CEF)** rather than conventional
RSA:

```text
CEF = residue SASA in the complete protein
      / SASA of the same residue conformation in isolation
```

The denominator is a naked-residue calculation using the same atoms, atomic
radii, probe radius, sphere points, and `ALL` atom-selection rule, but without
other residues present. CEF therefore represents the fraction of the residue's
intrinsic, same-conformation surface that remains exposed in its protein
context. It is distinct from conventional RSA, which normally uses a fixed
residue-type reference or maximum ASA. `SIDE` selection mode is deferred.

CEF is expected to lie in `[0, 1]` up to floating-point roundoff. The residue
output format and public API are still under development.

The first implementation computes one isolated-residue reference calculation
per residue. After one warm-up, one-thread medians at 960 sphere points were
0.090 s total for 1LYZ (129 residues), 0.118 s for 1CA2 (257 residues), and
0.269 s for 1UOR (580 residues). The denominator portion accounted for 0.059,
0.076, and 0.173 s respectively.

## Optimization Comparison

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

Benchmark results should be collected using the same input structures, probe
radius, sampling resolution, and hardware. Record the protein used for each
size category, the number of atoms and residues, software versions, and the
CPU model alongside the final results so that the comparison is
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

The Numba CPU backend was benchmarked through `atom_sasa_spatial(...,
backend="cpu")` on the same canonical structures. Numba compilation was
performed by one untimed warm-up before five measured calls; timings below are
the medians of those calls. Results were compared with the saved
`*.atom.sas.baseline` files. The precise MAEs were 1.38605884e-7, 1.19507407e-7,
and 1.54129259e-7 Å² for 1LYZ, 1CA2, and 1UOR, respectively; maximum absolute
errors were below 5.0e-7 Å² in all three cases. The Numba row reports total
`atom_sasa_spatial()` time, including validation, CSR construction, burial-mask
construction, and kernel execution. With the default one thread, the measured
medians were approximately 0.328 s, 0.388 s, and 0.735 s, respectively.
Four-thread execution remains available through `workers=4` for workloads
where it provides a measured benefit.

Reducing the sampling resolution from 960 to 480 points was also measured as
wall-clock time on the same host (one untimed warm-up followed by five timed
calls). The corresponding 960-point times were 0.030, 0.042, and 0.094 s for
1LYZ, 1CA2, and 1UOR; the 480-point times were 0.023, 0.041, and 0.074 s,
respectively. This is only a 1.27x speedup (approximately 21% less elapsed
time), demonstrating that halving the sphere-point count does not materially
reduce total wall-clock time because validation, neighbor construction, and
other fixed costs dominate. Against the 960-point output, the 480-point atom
SASA MAEs were 0.164, 0.145, and 0.184 Å² per atom, respectively.

The complete-burial mask was benchmarked with the cached kernel using 960
sphere points, one warm-up, and five measured calls. Dense synthetic cases
contained one large enclosing sphere and small targets; sparse cases used
separated unit-radius atoms. Masked and unmasked outputs were bitwise
identical in every case.

The burial-mask construction itself was also compared with its Python fallback
using four Numba threads, after one compilation warm-up and five measured
calls. The Numba mask was bitwise identical to the fallback and reduced mask
construction time by more than 700x on the canonical structures.

| Structure | Python mask (s) | Numba mask, 4 threads (s) | Speedup |
| --- | ---: | ---: | ---: |
| 1LYZ | 0.064351 | 0.000060 | 1074x |
| 1CA2 | 0.088435 | 0.000118 | 748x |
| 1UOR | 0.189883 | 0.000234 | 813x |

| Workload | Atoms | Buried | Directed neighbors | Unmasked (s) | Masked (s) | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense synthetic | 27 | 26 | 54 | 0.001185 | 0.000443 | 2.67x |
| Dense synthetic | 64 | 63 | 138 | 0.002850 | 0.001030 | 2.77x |
| Dense synthetic | 125 | 124 | 316 | 0.005660 | 0.002006 | 2.82x |
| Sparse synthetic | 27 | 0 | 0 | 0.000021 | 0.000022 | 0.95x |
| Sparse synthetic | 125 | 0 | 0 | 0.000086 | 0.000092 | 0.93x |

The synthetic coordinates were generated deterministically with NumPy seed
20260918. Timings were collected on Linux with NumPy 2.4.4 and SciPy 1.18.0;
sub-10-ms measurements are reported as absolute values because timer noise is
material at that scale. The mask removes the intended sampling work for dense
burial while adding negligible overhead to sparse inputs.

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

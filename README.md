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
`0.000`. The current calculation is the deliberately naive serial reference;
it is intended as a correctness baseline rather than the production-speed
backend.

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

| Optimization | Threads | Sphere points | Time (small) | Time (medium) | Time (large) | Atom-SASA MAE (Å²) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive | 1 | 960 | | | | 0 |

Benchmark results should be collected using the same input structures, probe
radius, sampling resolution, and hardware. Record the protein used for each
size category, the number of atoms and residues, software versions, and the
CPU and GPU models alongside the final results so that the comparison is
reproducible.

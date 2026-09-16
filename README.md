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

## Python use

The application can also be imported as the `protrsa` module so its functions
and constants can be used directly:

```python
import protrsa
```

The distribution and command are named `prot-rsa`. The import name is
`protrsa` because Python module names cannot contain hyphens.

## Optimization Comparison

The table below compares protein RSA calculation times for representative
protein sizes under each supported optimization. Execution times are reported
in seconds; lower values are better.

| Protein size | Reference CPU | Optimized CPU | Multiprocessing (4 workers) | GPU |
| --- | ---: | ---: | ---: | ---: |
| Small | — | — | — | — |
| Medium | — | — | — | — |
| Large | — | — | — | — |

Benchmark results should be collected using the same input structures, probe
radius, sampling resolution, and hardware. Record the protein used for each
size category, the number of atoms and residues, software versions, and the
CPU and GPU models alongside the final results so that the comparison is
reproducible.

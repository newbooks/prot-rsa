# Command-Line Interface Specification

## Purpose

Implement the initial `prot-rsa` command-line contract in the single
import-safe `protrsa.py` module. This specification covers argument parsing,
input filename validation, output filename derivation, and the information
shown by `--help`. Structure parsing, initial atom SASA, and current output
contents are defined by [`atom-sasa-naive.md`](atom-sasa-naive.md).

The words **must**, **should**, and **may** describe required, recommended, and
optional behavior, respectively.

## Invocation

Both entry points must expose the same interface:

```bash
prot-rsa INPUT [OPTIONS]
python protrsa.py INPUT [OPTIONS]
```

`protrsa.py` must expose reusable parser and path-handling functions without
parsing arguments or producing other side effects when imported. Command-line
execution must occur only through a guarded `main()` call.

This specification defines the CLI setup layer, not a temporary standalone
program milestone. Until the parsing, calculation, and serialization
specifications are implemented, an agent must test the parser and path helpers
directly and must not make a valid invocation create empty placeholder output
files or report a calculation as successful. Final `main()` orchestration will
connect this interface to those later components.

## Positional input

The command must accept exactly one positional argument named `INPUT`. It is a
path to one of these supported protein structure files:

- PDB: `.pdb`
- mmCIF: `.cif`
- gzip-compressed PDB: `.pdb.gz`
- gzip-compressed mmCIF: `.cif.gz`

Suffix matching must be case-insensitive. The format must be inferred only
from the complete supported suffix; content sniffing is outside this scope.
The input path may include directory components and multiple dots in its base
name.

Argument validation must reject:

- a missing `INPUT` argument;
- more than one positional input;
- a path that does not exist or is not a regular file; and
- a filename without one of the supported complete suffixes.

A malformed gzip stream or malformed structure content is a later parsing
error, not an argument-parser responsibility.

## Options and defaults

| Option | Parsed value | Default | Requirements |
| --- | --- | --- | --- |
| `--mode {ALL,SIDE,KEY}` | String normalized to uppercase | `ALL` | Accept the three values case-insensitively; reject all others. |
| `--prob-size FLOAT` | Floating-point number | `1.40` | Must be finite and greater than zero. |
| `--workers INTEGER` | Integer | `4` | Numba CPU thread count; must be greater than zero. |
| `--preserve-het` | Boolean presence flag | `False` | Supplying the flag sets the value to `True`; it takes no argument. |
| `--use-h` | Boolean presence flag | `False` | Supplying the flag sets the value to `True`; it takes no argument. |

`--prob-size` is the canonical public spelling. Do not silently add
`--probe-size` or `--probe-radius` aliases unless a later specification adds
them.

The parsed argument object must expose these attribute names and types:

| Attribute | Type |
| --- | --- |
| `input` | `pathlib.Path` |
| `mode` | `str` (`ALL`, `SIDE`, or `KEY`) |
| `prob_size` | `float` |
| `workers` | `int` |
| `preserve_het` | `bool` |
| `use_h` | `bool` |

Filesystem existence and regular-file validation may occur during argument
parsing or immediately afterward, but it must occur before structure parsing
or output creation.

The parser may accept options before or after `INPUT`. Repeating an option may
use the parser library's normal behavior, but the final parsed value must have
the type and validation described above.

## Output path derivation

A successful initial atom calculation produces two files alongside the input:

- `<base>.atom.sas`, a TSV file for atom solvent-accessible surface areas;
- `<base>.pqr`, normalized atoms with radii and placeholder zero charges.

`<base>.res.sas` remains the reserved residue-SASA path but is not written by
the atom-only stage.

Derive `<base>` by removing a final `.gz` suffix when present and then removing
the final `.pdb` or `.cif` suffix. Suffix removal must follow the same
case-insensitive rules as input validation. Preserve the input directory and
all earlier parts of the filename.

Examples:

| Input | Atom output | PQR output | Reserved residue output |
| --- | --- | --- | --- |
| `protein.pdb` | `protein.atom.sas` | `protein.pqr` | `protein.res.sas` |
| `protein.cif.gz` | `protein.atom.sas` | `protein.pqr` | `protein.res.sas` |
| `model.v2.PDB.GZ` | `model.v2.atom.sas` | `model.v2.pqr` | `model.v2.res.sas` |
| `/data/set/protein.cif` | `/data/set/protein.atom.sas` | `/data/set/protein.pqr` | `/data/set/protein.res.sas` |

Output paths must be derived by a reusable pure function. The function must
not create, truncate, or otherwise modify either file. The atom-SASA stage
atomically replaces existing `.atom.sas` and `.pqr` outputs as defined in
[`atom-sasa-naive.md`](atom-sasa-naive.md). The `.sas` suffix does not change
the format: SASA files use tab-separated values rather than comma-separated
values.

## Help message

`prot-rsa --help` and `python protrsa.py --help` must exit successfully without
requiring `INPUT`. Their help output must communicate all of the following,
although exact whitespace and parser-generated punctuation may vary:

```text
usage: prot-rsa [-h] [--mode {ALL,SIDE,KEY}] [--prob-size FLOAT]
                [--workers INTEGER] [--preserve-het] [--use-h]
                INPUT

Calculate atom solvent-accessible surface areas for a protein structure.

positional arguments:
  INPUT                 PDB or mmCIF input file; .pdb, .cif, .pdb.gz, and
                        .cif.gz are supported

options:
  -h, --help            show this help message and exit
  --mode {ALL,SIDE,KEY}
                        atom selection mode (default: ALL)
  --prob-size FLOAT     solvent probe radius in angstroms (default: 1.40)
  --workers INTEGER     number of Numba CPU threads (default: 1)
  --preserve-het        preserve loose hetero-atoms (default: false)
  --use-h               use hydrogen atoms supplied in the input file
                        (default: false)

output files:
  <base>.atom.sas       atom solvent-accessible surface areas
  <base>.pqr            normalized atoms, radii, and zero charges
  <base>.res.sas        reserved for the later residue-SASA stage

The output files are written alongside INPUT. <base> is INPUT with .gz, when
present, and then .pdb or .cif removed.
```

The displayed program name should match the active entry point where practical.
The installed command must display `prot-rsa`. A custom help formatter may be
used to show defaults and the `output files` section.

## Errors and exit status

Argument and input-path errors must:

- explain the invalid value or path;
- write the diagnostic to standard error;
- not produce a Python traceback for ordinary user errors;
- return a nonzero exit status; and
- create no output files.

Successful `--help` and a successful `ALL` atom calculation return status `0`.
`SIDE` and `KEY` must fail clearly until their scientific selection rules are
specified.

## Required implementation tests

Add focused `pytest` tests for reusable functions and use a subprocess or the
chosen parser library's test mechanism for CLI behavior. At minimum, test:

1. Every documented default.
2. Each valid `--mode` spelling in uppercase and lowercase.
3. Rejection of an unknown mode.
4. Rejection of zero, negative, infinite, and NaN probe sizes.
5. Rejection of zero and negative worker counts.
6. Both Boolean flags independently and together.
7. Every supported uncompressed and compressed suffix, including uppercase.
8. Rejection of unsupported and incomplete suffixes such as `.txt`, `.gz`,
   and `.mmcif`.
9. Output derivation for relative paths, absolute paths, and bases containing
   multiple dots.
10. Missing, extra, nonexistent, and non-file positional inputs.
11. `--help` success through both entry points.
12. Help text coverage for every option, default, supported input suffix, both
    output suffixes, and the output-location rule.
13. Importing `protrsa` has no command-line or filesystem side effects.

Tests should assert required help content rather than the entire formatted
help string so harmless wrapping differences across supported Python versions
do not cause failures.

## Acceptance criteria

This specification is complete when:

- the public command and direct-script entry point implement the same parser;
- parsing and output-path derivation are reusable without CLI side effects;
- all required values, defaults, validation rules, and help content match this
  specification; and
- no placeholder calculation results or empty output files are introduced; and
- the focused tests and the full existing test suite pass.

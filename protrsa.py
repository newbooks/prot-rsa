"""Command-line interface for protein solvent-accessible surface calculations."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence


DEFAULT_MODE = "ALL"
DEFAULT_PROBE_SIZE = 1.40
DEFAULT_WORKERS = 4
MODES = ("ALL", "SIDE", "KEY")
SUPPORTED_INPUT_SUFFIXES = (".pdb", ".cif", ".pdb.gz", ".cif.gz")


def _strip_structure_suffix(input_path: Path) -> Path:
    """Return *input_path* without its supported structure-file suffix."""

    name = input_path.name
    lowercase_name = name.lower()
    for suffix in sorted(SUPPORTED_INPUT_SUFFIXES, key=len, reverse=True):
        if lowercase_name.endswith(suffix):
            return input_path.with_name(name[: -len(suffix)])
    supported = ", ".join(SUPPORTED_INPUT_SUFFIXES)
    raise ValueError(f"unsupported input extension for '{input_path}'; expected {supported}")


def _input_file(value: str) -> Path:
    """Validate an input argument and preserve its user-supplied path form."""

    path = Path(value)
    try:
        _strip_structure_suffix(path)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not path.is_file():
        raise argparse.ArgumentTypeError(
            f"input path does not exist or is not a regular file: '{path}'"
        )
    return path


def _positive_finite_float(value: str) -> float:
    """Parse a finite floating-point value greater than zero."""

    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected a floating-point number: '{value}'") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and greater than zero")
    return parsed


def _positive_int(value: str) -> int:
    """Parse an integer greater than zero."""

    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected an integer: '{value}'") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def build_parser(*, prog: str | None = None) -> argparse.ArgumentParser:
    """Build and return the shared command-line argument parser."""

    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Calculate atom and residue solvent-accessible surface areas for a "
            "protein structure."
        ),
        epilog=(
            "output files:\n"
            "  <base>.atom.sas       atom solvent-accessible surface areas\n"
            "  <base>.res.sas        residue solvent-accessible surface areas\n\n"
            "The output files are written alongside INPUT. <base> is INPUT with "
            ".gz, when present, and then .pdb or .cif removed."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        metavar="INPUT",
        type=_input_file,
        help="PDB or mmCIF input file; .pdb, .cif, .pdb.gz, and .cif.gz are supported",
    )
    parser.add_argument(
        "--mode",
        type=str.upper,
        choices=MODES,
        default=DEFAULT_MODE,
        help="atom selection mode (default: ALL)",
    )
    parser.add_argument(
        "--prob-size",
        type=_positive_finite_float,
        default=DEFAULT_PROBE_SIZE,
        metavar="FLOAT",
        help="solvent probe radius in angstroms (default: 1.40)",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=DEFAULT_WORKERS,
        metavar="INTEGER",
        help="number of worker processes (default: 4)",
    )
    parser.add_argument(
        "--preserve-het",
        action="store_true",
        help="preserve loose hetero-atoms (default: false)",
    )
    parser.add_argument(
        "--use-h",
        action="store_true",
        help="use hydrogen atoms supplied in the input file (default: false)",
    )
    return parser


def parse_args(
    argv: Sequence[str] | None = None,
    *,
    prog: str | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments using the public CLI contract."""

    return build_parser(prog=prog).parse_args(argv)


def derive_output_paths(input_path: str | Path) -> tuple[Path, Path]:
    """Return atom and residue output paths without modifying the filesystem."""

    base = _strip_structure_suffix(Path(input_path))
    atom_output = base.with_name(f"{base.name}.atom.sas")
    residue_output = base.with_name(f"{base.name}.res.sas")
    return atom_output, residue_output


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line entry point."""

    parser = build_parser()
    parser.parse_args(argv)
    parser.exit(
        1,
        "prot-rsa: error: SASA calculation and output serialization are not "
        "implemented yet\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())

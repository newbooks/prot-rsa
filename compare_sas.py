"""Compare two atom-SASA TSV files and report their mean absolute error."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from protrsa import ATOM_IDENTITY_COLUMNS, ATOM_SASA_COLUMNS


def _read_atom_sasa_tsv(
    input_path: str | Path,
) -> tuple[list[tuple[str, ...]], np.ndarray]:
    """Read atom identities and SASA values from one atom-SASA TSV."""

    path = Path(input_path)
    identities: list[tuple[str, ...]] = []
    sasa_values: list[float] = []
    with path.open(mode="r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file, delimiter="\t")
        if reader.fieldnames != list(ATOM_SASA_COLUMNS):
            expected = "\t".join(ATOM_SASA_COLUMNS)
            actual = "\t".join(reader.fieldnames or [])
            raise ValueError(
                f"'{path}' has an unexpected header; expected {expected!r}, "
                f"found {actual!r}"
            )
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(
                row[column] is None for column in ATOM_SASA_COLUMNS
            ):
                raise ValueError(f"'{path}' has a malformed row on line {line_number}")
            identities.append(tuple(row[column] for column in ATOM_IDENTITY_COLUMNS))
            try:
                sasa = float(row["sasa_A2"])
            except ValueError as error:
                raise ValueError(
                    f"'{path}' has an invalid sasa_A2 value on line {line_number}"
                ) from error
            if not math.isfinite(sasa):
                raise ValueError(
                    f"'{path}' has a non-finite sasa_A2 value on line {line_number}"
                )
            sasa_values.append(sasa)

    if not identities:
        raise ValueError(f"'{path}' contains no atom rows")
    return identities, np.asarray(sasa_values, dtype=np.float64)


def compare_atom_sasa_files(
    first_path: str | Path,
    second_path: str | Path,
) -> tuple[int, float]:
    """Verify matching atom rows and return their count and SASA MAE."""

    first_identities, first_sasa = _read_atom_sasa_tsv(first_path)
    second_identities, second_sasa = _read_atom_sasa_tsv(second_path)
    if len(first_identities) != len(second_identities):
        raise ValueError(
            "atom count mismatch: "
            f"'{first_path}' has {len(first_identities)} rows, "
            f"'{second_path}' has {len(second_identities)} rows"
        )
    for row_number, (first_atom, second_atom) in enumerate(
        zip(first_identities, second_identities),
        start=1,
    ):
        if first_atom != second_atom:
            differing_columns = [
                column
                for column, first_value, second_value in zip(
                    ATOM_IDENTITY_COLUMNS, first_atom, second_atom
                )
                if first_value != second_value
            ]
            raise ValueError(
                f"atom mismatch at data row {row_number}; differing columns: "
                + ", ".join(differing_columns)
            )
    mae = float(np.mean(np.abs(first_sasa - second_sasa)))
    return len(first_identities), mae


def compare_sas_main(argv: Sequence[str] | None = None) -> int:
    """Run the atom-SASA comparison command-line interface."""

    parser = argparse.ArgumentParser(
        prog="prot-rsa-compare",
        description="Verify matching atom rows and calculate SASA MAE.",
    )
    parser.add_argument("first", type=Path, metavar="FIRST.atom.sas")
    parser.add_argument("second", type=Path, metavar="SECOND.atom.sas")
    arguments = parser.parse_args(argv)
    try:
        atom_count, mae = compare_atom_sasa_files(arguments.first, arguments.second)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Atoms matched: {atom_count}")
    print(f"SASA MAE: {mae:.6f} Å²")
    return 0


if __name__ == "__main__":
    raise SystemExit(compare_sas_main())

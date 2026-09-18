"""Protein solvent-accessible surface calculations and command-line interface."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import gzip
import math
import numbers
import os
from pathlib import Path
import sys
import tempfile
import time
from types import MappingProxyType
from typing import Iterable, Sequence

import numpy as np
from scipy.spatial import cKDTree


DEFAULT_MODE = "ALL"
DEFAULT_PROBE_SIZE = 1.40
DEFAULT_SPHERE_POINTS = 960
DEFAULT_WORKERS = 4
DEFAULT_PRESERVE_HET = False
DEFAULT_USE_H = False
MODES = ("ALL", "SIDE", "KEY")
SUPPORTED_INPUT_SUFFIXES = (".pdb", ".cif", ".pdb.gz", ".cif.gz")
ATOM_SASA_COLUMNS = (
    "atom_index",
    "record_type",
    "source_id",
    "atom_name",
    "residue_name",
    "chain_id",
    "residue_sequence",
    "insertion_code",
    "element",
    "radius_A",
    "sasa_A2",
)
ATOM_IDENTITY_COLUMNS = ATOM_SASA_COLUMNS[:-2]

PROTOR_RADII = MappingProxyType(
    {
        "TRIGONAL_C_NO_H": 1.61,
        "TRIGONAL_C_ONE_H": 1.76,
        "TETRAHEDRAL_C": 1.88,
        "N": 1.64,
        "CARBONYL_O": 1.42,
        "HYDROXYL_O": 1.46,
        "S": 1.77,
        "P": 1.80,
        "SE": 1.90,
    }
)

IRON_RADIUS = 2.00
UNKNOWN_RADIUS = IRON_RADIUS
EXPLICIT_ATOM_RADII = MappingProxyType(
    {
        "H": 1.10,
        "D": 1.10,
        "C": 1.70,
        "N": 1.55,
        "O": 1.52,
        "F": 1.47,
        "P": 1.80,
        "S": 1.80,
        "CL": 1.75,
        "SE": 1.90,
        "BR": 1.83,
        "I": 1.98,
        "LI": 1.81,
        "NA": 2.27,
        "K": 2.75,
        "RB": 3.03,
        "CS": 3.43,
        "FE": IRON_RADIUS,
        "X": UNKNOWN_RADIUS,
    }
)

WATER_COMPONENTS = frozenset(
    {"HOH", "WAT", "H2O", "DOD", "SOL", "OH2", "TIP", "TIP3", "TIP4"}
)
SIMPLE_ION_COMPONENTS = frozenset(
    {
        "LI",
        "NA",
        "K",
        "RB",
        "CS",
        "F",
        "CL",
        "BR",
        "I",
        "NH4",
        "NO3",
        "CO3",
        "SO4",
        "PO4",
    }
)
CRYSTALLIZATION_ADDITIVE_COMPONENTS = frozenset(
    {
        "ACT",
        "ACY",
        "CIT",
        "TAR",
        "GOL",
        "EDO",
        "PEG",
        "PG4",
        "PGE",
        "MPD",
        "MOH",
        "EOH",
        "IPA",
        "DMS",
        "BME",
        "DTT",
        "TRS",
        "MES",
        "HEP",
        "BIC",
    }
)
LOOSE_HETERO_COMPONENTS = (
    WATER_COMPONENTS
    | SIMPLE_ION_COMPONENTS
    | CRYSTALLIZATION_ADDITIVE_COMPONENTS
)


class StructureReadError(ValueError):
    """Raised when a supported structure file cannot be read unambiguously."""


@dataclass(frozen=True, slots=True)
class AtomRecord:
    """Normalized atom information read from one PDB or mmCIF coordinate row."""

    record_type: str
    serial: str
    element: str
    atom_name: str
    altloc: str
    residue_name: str
    chain_id: str
    residue_sequence: str
    insertion_code: str
    x: float
    y: float
    z: float
    occupancy: float | None
    model: int
    atom_name_raw: str = ""


@dataclass(frozen=True, slots=True)
class NormalizedAtom:
    """An atom retained for calculation with its normalized element and radius."""

    source: AtomRecord
    element: str
    radius: float


STANDARD_AMINO_ACIDS = frozenset(
    {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    }
)
PROTOR_TRIGONAL_C_NO_H = frozenset(
    {
        ("ARG", "CZ"), ("ASN", "CG"), ("ASP", "CG"), ("GLN", "CD"),
        ("GLU", "CD"), ("HIS", "CG"), ("PHE", "CG"), ("TRP", "CG"),
        ("TRP", "CD2"), ("TRP", "CE2"), ("TYR", "CG"), ("TYR", "CZ"),
    }
)
PROTOR_TRIGONAL_C_ONE_H = frozenset(
    {
        ("HIS", "CD2"), ("HIS", "CE1"), ("PHE", "CD1"), ("PHE", "CD2"),
        ("PHE", "CE1"), ("PHE", "CE2"), ("PHE", "CZ"), ("TRP", "CD1"),
        ("TRP", "CE3"), ("TRP", "CZ2"), ("TRP", "CZ3"), ("TRP", "CH2"),
        ("TYR", "CD1"), ("TYR", "CD2"), ("TYR", "CE1"), ("TYR", "CE2"),
    }
)
PROTOR_HYDROXYL_O = frozenset({("SER", "OG"), ("THR", "OG1"), ("TYR", "OH")})


def _missing_cif_value(value: str) -> str:
    """Normalize an mmCIF missing-value marker to an empty string."""

    return "" if value in {".", "?"} else value


def _four_character_atom_name(atom_name: str, element: str) -> str:
    """Construct the PDB-style four-character representation of an atom name."""

    name = atom_name[:4]
    if len(name) >= 4:
        return name
    if len(element) == 1:
        return f" {name:<3}"
    return f"{name:<4}"


def _tokenize_mmcif(text: str) -> list[str]:
    """Tokenize the subset of CIF 1.1 syntax needed to locate atom-site loops."""

    tokens: list[str] = []
    lines = text.splitlines()
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        if line.startswith(";"):
            value_lines = [line[1:]]
            line_index += 1
            while line_index < len(lines) and not lines[line_index].startswith(";"):
                value_lines.append(lines[line_index])
                line_index += 1
            if line_index == len(lines):
                raise StructureReadError("unterminated semicolon-delimited mmCIF value")
            tokens.append("\n".join(value_lines))
            line_index += 1
            continue

        position = 0
        while position < len(line):
            while position < len(line) and line[position].isspace():
                position += 1
            if position == len(line) or line[position] == "#":
                break
            if line[position] in {"'", '"'}:
                quote = line[position]
                position += 1
                start = position
                while position < len(line):
                    if line[position] == quote and (
                        position + 1 == len(line) or line[position + 1].isspace()
                    ):
                        break
                    position += 1
                if position == len(line):
                    raise StructureReadError("unterminated quoted mmCIF value")
                tokens.append(line[start:position])
                position += 1
            else:
                start = position
                while position < len(line) and not line[position].isspace():
                    position += 1
                tokens.append(line[start:position])
        line_index += 1
    return tokens


def _parse_optional_float(value: str, *, field: str) -> float | None:
    """Parse an optional finite structure-file number."""

    normalized = _missing_cif_value(value.strip())
    if not normalized:
        return None
    try:
        parsed = float(normalized)
    except ValueError as error:
        raise StructureReadError(f"invalid {field}: {value!r}") from error
    if not math.isfinite(parsed):
        raise StructureReadError(f"{field} must be finite")
    return parsed


def _parse_model(value: str) -> int:
    """Parse a positive coordinate-model identifier."""

    normalized = _missing_cif_value(value.strip())
    if not normalized:
        return 1
    try:
        model = int(normalized)
    except ValueError as error:
        raise StructureReadError(f"invalid model identifier: {value!r}") from error
    if model < 1:
        raise StructureReadError("model identifier must be greater than zero")
    return model


def _select_alternate_locations(atoms: list[AtomRecord]) -> list[AtomRecord]:
    """Choose one residue-consistent alternate-location label by occupancy."""

    residue_labels: dict[tuple[str, str, str, str], dict[str, list[float | None]]] = {}
    for atom in atoms:
        if not atom.altloc:
            continue
        residue_key = (
            atom.chain_id,
            atom.residue_sequence,
            atom.insertion_code,
            atom.residue_name,
        )
        residue_labels.setdefault(residue_key, {}).setdefault(atom.altloc, []).append(
            atom.occupancy
        )

    selected: dict[tuple[str, str, str, str], str] = {}
    for residue_key, labels in residue_labels.items():
        def ranking(
            item: tuple[str, list[float | None]],
        ) -> tuple[int, float, int, str]:
            label, occupancies = item
            numeric = [value for value in occupancies if value is not None]
            complete = int(len(numeric) == len(occupancies))
            mean = sum(numeric) / len(numeric) if numeric else -math.inf
            return -complete, -mean, int(label != "A"), label

        selected[residue_key] = min(labels.items(), key=ranking)[0]

    retained: list[AtomRecord] = []
    for atom in atoms:
        if not atom.altloc:
            retained.append(atom)
            continue
        residue_key = (
            atom.chain_id,
            atom.residue_sequence,
            atom.insertion_code,
            atom.residue_name,
        )
        if atom.altloc == selected[residue_key]:
            retained.append(atom)
    return retained


def _parse_pdb(lines: Iterable[str]) -> list[AtomRecord]:
    """Parse PDB coordinate records without performing scientific filtering."""

    atoms: list[AtomRecord] = []
    explicit_model_count = 0
    models: set[int] = set()
    current_model = 1
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip("\r\n")
        record = line[0:6].strip().upper()
        if record == "MODEL":
            current_model = _parse_model(line[10:14])
            explicit_model_count += 1
            if explicit_model_count > 1:
                raise StructureReadError("multiple coordinate models are not supported")
            continue
        if record not in {"ATOM", "HETATM"}:
            continue
        if len(line) < 54:
            raise StructureReadError(f"PDB coordinate record on line {line_number} is too short")
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError as error:
            raise StructureReadError(
                f"invalid PDB coordinates on line {line_number}"
            ) from error
        if not all(math.isfinite(value) for value in (x, y, z)):
            raise StructureReadError(f"PDB coordinates on line {line_number} must be finite")
        occupancy = _parse_optional_float(line[54:60], field="PDB occupancy")
        models.add(current_model)
        if len(models) > 1:
            raise StructureReadError("multiple coordinate models are not supported")
        atoms.append(
            AtomRecord(
                record_type=record,
                serial=line[6:11].strip(),
                element=line[76:78].strip().upper() if len(line) >= 78 else "",
                atom_name=line[12:16].strip(),
                altloc=line[16:17].strip(),
                residue_name=line[17:20].strip(),
                chain_id=line[21:22].strip(),
                residue_sequence=line[22:26].strip(),
                insertion_code=line[26:27].strip(),
                x=x,
                y=y,
                z=z,
                occupancy=occupancy,
                model=current_model,
                atom_name_raw=line[12:16],
            )
        )
    return atoms


def _parse_mmcif(text: str) -> list[AtomRecord]:
    """Parse atom-site loops from an mmCIF document."""

    tokens = _tokenize_mmcif(text)
    atom_rows: list[dict[str, str]] = []
    index = 0
    while index < len(tokens):
        if tokens[index].lower() != "loop_":
            index += 1
            continue
        index += 1
        headers: list[str] = []
        while index < len(tokens) and tokens[index].startswith("_"):
            headers.append(tokens[index].lower())
            index += 1
        if not headers:
            raise StructureReadError("mmCIF loop has no column names")
        width = len(headers)
        values: list[str] = []
        while index < len(tokens):
            token_lower = tokens[index].lower()
            at_row_boundary = len(values) % width == 0
            if at_row_boundary and (
                token_lower == "loop_"
                or token_lower == "stop_"
                or token_lower == "global_"
                or token_lower.startswith("data_")
                or token_lower.startswith("save_")
                or tokens[index].startswith("_")
            ):
                break
            values.append(tokens[index])
            index += 1
        if len(values) % width:
            raise StructureReadError("mmCIF loop contains an incomplete row")
        if headers[0].startswith("_atom_site."):
            for offset in range(0, len(values), width):
                atom_rows.append(dict(zip(headers, values[offset : offset + width])))
        if index < len(tokens) and tokens[index].lower() == "stop_":
            index += 1

    if not atom_rows:
        return []

    def required(row: dict[str, str], name: str) -> str:
        if name not in row:
            raise StructureReadError(f"mmCIF atom-site loop is missing {name}")
        return row[name]

    def preferred(row: dict[str, str], auth: str, label: str) -> str:
        auth_value = _missing_cif_value(row.get(auth, ""))
        if auth_value:
            return auth_value
        return _missing_cif_value(row.get(label, ""))

    atoms: list[AtomRecord] = []
    models: set[int] = set()
    for row_number, row in enumerate(atom_rows, start=1):
        record = _missing_cif_value(required(row, "_atom_site.group_pdb")).upper()
        if record not in {"ATOM", "HETATM"}:
            continue
        x_value = required(row, "_atom_site.cartn_x")
        y_value = required(row, "_atom_site.cartn_y")
        z_value = required(row, "_atom_site.cartn_z")
        try:
            x = float(x_value)
            y = float(y_value)
            z = float(z_value)
        except ValueError as error:
            raise StructureReadError(
                f"invalid mmCIF coordinates in atom-site row {row_number}"
            ) from error
        if not all(math.isfinite(value) for value in (x, y, z)):
            raise StructureReadError(
                f"mmCIF coordinates in atom-site row {row_number} must be finite"
            )
        model = _parse_model(row.get("_atom_site.pdbx_pdb_model_num", "1"))
        models.add(model)
        if len(models) > 1:
            raise StructureReadError("multiple coordinate models are not supported")
        atom_name = preferred(row, "_atom_site.auth_atom_id", "_atom_site.label_atom_id")
        residue_name = preferred(
            row, "_atom_site.auth_comp_id", "_atom_site.label_comp_id"
        )
        chain_id = preferred(row, "_atom_site.auth_asym_id", "_atom_site.label_asym_id")
        residue_sequence = preferred(
            row, "_atom_site.auth_seq_id", "_atom_site.label_seq_id"
        )
        if not atom_name or not residue_name:
            raise StructureReadError("mmCIF atom-site row lacks an atom or residue name")
        element = _missing_cif_value(row.get("_atom_site.type_symbol", "")).upper()
        atoms.append(
            AtomRecord(
                record_type=record,
                serial=_missing_cif_value(row.get("_atom_site.id", "")),
                element=element,
                atom_name=atom_name,
                altloc=_missing_cif_value(row.get("_atom_site.label_alt_id", "")),
                residue_name=residue_name,
                chain_id=chain_id,
                residue_sequence=residue_sequence,
                insertion_code=_missing_cif_value(
                    row.get("_atom_site.pdbx_pdb_ins_code", "")
                ),
                x=x,
                y=y,
                z=z,
                occupancy=_parse_optional_float(
                    row.get("_atom_site.occupancy", ""), field="mmCIF occupancy"
                ),
                model=model,
                atom_name_raw=_four_character_atom_name(atom_name, element),
            )
        )
    return atoms


def read_structure(input_path: str | Path) -> list[AtomRecord]:
    """Read one PDB or mmCIF model and resolve residue alternate locations."""

    path = Path(input_path)
    _strip_structure_suffix(path)
    lowercase_name = path.name.lower()
    is_gzip = lowercase_name.endswith(".gz")
    is_pdb = lowercase_name.endswith(".pdb") or lowercase_name.endswith(".pdb.gz")
    opener = gzip.open if is_gzip else open
    try:
        with opener(path, mode="rt", encoding="utf-8") as input_file:
            if is_pdb:
                atoms = _parse_pdb(input_file)
            else:
                atoms = _parse_mmcif(input_file.read())
    except StructureReadError:
        raise
    except (OSError, UnicodeError) as error:
        raise StructureReadError(f"could not read structure file '{path}': {error}") from error
    if not atoms:
        raise StructureReadError("structure contains no ATOM or HETATM coordinate records")
    return _select_alternate_locations(atoms)


def _normalized_element(atom: AtomRecord) -> str:
    """Return the element, falling back to PDB atom-name alignment."""

    if atom.element:
        return atom.element.upper()
    atom_name_field = atom.atom_name_raw or atom.atom_name
    inferred = atom_name_field[:2].strip().upper().lstrip("0123456789")
    return inferred or "X"


def _protor_radius(atom: AtomRecord, element: str) -> float:
    """Assign a ProtOr united-atom radius or the approved unknown fallback."""

    residue = atom.residue_name.upper()
    atom_name = atom.atom_name.upper()
    if element == "N":
        return PROTOR_RADII["N"]
    if element == "S":
        return PROTOR_RADII["S"]
    if element == "P":
        return PROTOR_RADII["P"]
    if element == "SE":
        return PROTOR_RADII["SE"]
    if residue not in STANDARD_AMINO_ACIDS:
        return UNKNOWN_RADIUS
    if element == "O":
        if (residue, atom_name) in PROTOR_HYDROXYL_O:
            return PROTOR_RADII["HYDROXYL_O"]
        return PROTOR_RADII["CARBONYL_O"]
    if element == "C":
        if atom_name == "C" or (residue, atom_name) in PROTOR_TRIGONAL_C_NO_H:
            return PROTOR_RADII["TRIGONAL_C_NO_H"]
        if (residue, atom_name) in PROTOR_TRIGONAL_C_ONE_H:
            return PROTOR_RADII["TRIGONAL_C_ONE_H"]
        return PROTOR_RADII["TETRAHEDRAL_C"]
    return UNKNOWN_RADIUS


def normalize_atoms(
    atoms: Sequence[AtomRecord],
    *,
    use_h: bool = DEFAULT_USE_H,
    preserve_het: bool = DEFAULT_PRESERVE_HET,
) -> list[NormalizedAtom]:
    """Filter atoms and assign the selected explicit or united-atom radii."""

    if not isinstance(use_h, (bool, np.bool_)):
        raise TypeError("use_h must be Boolean")
    if not isinstance(preserve_het, (bool, np.bool_)):
        raise TypeError("preserve_het must be Boolean")
    normalized: list[NormalizedAtom] = []
    for atom in atoms:
        element = _normalized_element(atom)
        if not use_h and element in {"H", "D"}:
            continue
        if (
            atom.record_type == "HETATM"
            and atom.residue_name.strip().upper() in LOOSE_HETERO_COMPONENTS
            and not preserve_het
        ):
            continue
        radius = (
            EXPLICIT_ATOM_RADII.get(element, UNKNOWN_RADIUS)
            if use_h
            else _protor_radius(atom, element)
        )
        resolved_element = element
        if (use_h and element not in EXPLICIT_ATOM_RADII) or (
            not use_h and radius == UNKNOWN_RADIUS
        ):
            resolved_element = "X"
        normalized.append(
            NormalizedAtom(source=atom, element=resolved_element, radius=radius)
        )
    if not normalized:
        raise StructureReadError("no atoms remain after hydrogen and HETATM filtering")
    return normalized


def calculate_atom_sasa(
    atoms: Sequence[NormalizedAtom],
    *,
    probe_size: float = DEFAULT_PROBE_SIZE,
    sphere_points: np.ndarray | None = None,
) -> np.ndarray:
    """Calculate absolute SASA for normalized atoms with spatial pruning."""

    coordinates = np.asarray(
        [(atom.source.x, atom.source.y, atom.source.z) for atom in atoms],
        dtype=np.float64,
    ).reshape((-1, 3))
    radii = np.asarray([atom.radius for atom in atoms], dtype=np.float64)
    return atom_sasa_spatial(
        coordinates,
        radii,
        probe_size=probe_size,
        sphere_points=sphere_points,
    )


def _write_new_text(output_path: str | Path, text: str) -> None:
    """Atomically create or replace one text file."""

    path = Path(output_path)
    try:
        existing_mode = path.stat().st_mode & 0o777
    except FileNotFoundError:
        existing_mode = None
    with tempfile.TemporaryDirectory(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as temporary_directory:
        temporary_path = Path(temporary_directory) / path.name
        with temporary_path.open(mode="x", encoding="utf-8") as output_file:
            output_file.write(text)
        if existing_mode is not None:
            temporary_path.chmod(existing_mode)
        os.replace(temporary_path, path)


def write_pqr(output_path: str | Path, atoms: Sequence[NormalizedAtom]) -> None:
    """Write normalized atoms as PQR with the approved zero placeholder charge."""

    lines: list[str] = []
    for serial, atom in enumerate(atoms, start=1):
        source = atom.source
        residue_sequence = f"{source.residue_sequence}{source.insertion_code}"
        lines.append(
            f"{source.record_type:<6}{serial:5d} {source.atom_name:>4} "
            f"{source.residue_name:>3} {source.chain_id[:1]:1}{residue_sequence:>4}    "
            f"{source.x:8.3f}{source.y:8.3f}{source.z:8.3f}"
            f" {0.0:7.3f} {atom.radius:6.3f}\n"
        )
    _write_new_text(output_path, "".join(lines))


def write_atom_sasa_tsv(
    output_path: str | Path,
    atoms: Sequence[NormalizedAtom],
    atom_sasa: Sequence[float],
) -> None:
    """Write normalized atom metadata and absolute SASA as TSV."""

    if len(atoms) != len(atom_sasa):
        raise ValueError("atoms and atom_sasa must have the same length")
    lines = ["\t".join(ATOM_SASA_COLUMNS) + "\n"]
    for index, (atom, sasa) in enumerate(zip(atoms, atom_sasa), start=1):
        source = atom.source
        lines.append(
            f"{index}\t{source.record_type}\t{source.serial}\t{source.atom_name}\t"
            f"{source.residue_name}\t{source.chain_id}\t{source.residue_sequence}\t"
            f"{source.insertion_code}\t{atom.element}\t{atom.radius:.3f}\t"
            f"{float(sasa):.6f}\n"
        )
    _write_new_text(output_path, "".join(lines))


def _read_atom_sasa_tsv(
    input_path: str | Path,
) -> tuple[list[tuple[str, ...]], np.ndarray]:
    """Read atom identities and SASA values from one prot-rsa TSV file."""

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


def generate_sphere_points(
    n_points: int = DEFAULT_SPHERE_POINTS,
) -> np.ndarray:
    """Return deterministic float64 unit-sphere points with shape ``(P, 3)``."""

    if isinstance(n_points, (bool, np.bool_)) or not isinstance(
        n_points, numbers.Integral
    ):
        raise TypeError("n_points must be an integer")
    point_count = int(n_points)
    if point_count < 1:
        raise ValueError("n_points must be greater than zero")

    indices = np.arange(point_count, dtype=np.float64)
    z_coordinates = 1.0 - 2.0 * (indices + 0.5) / point_count
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    angles = indices * golden_angle
    radial = np.sqrt(np.maximum(0.0, 1.0 - z_coordinates * z_coordinates))

    points = np.empty((point_count, 3), dtype=np.float64)
    points[:, 0] = radial * np.cos(angles)
    points[:, 1] = radial * np.sin(angles)
    points[:, 2] = z_coordinates
    return points


def atom_sasa_reference(
    coordinates,
    radii,
    *,
    probe_size: float = DEFAULT_PROBE_SIZE,
    sphere_points: np.ndarray | None = None,
) -> np.ndarray:
    """Return deterministic absolute per-atom SASA in square angstroms."""

    atom_coordinates, atom_radii, probe_radius, points = _prepare_atom_sasa_inputs(
        coordinates,
        radii,
        probe_size=probe_size,
        sphere_points=sphere_points,
    )

    atom_count = atom_coordinates.shape[0]
    if atom_count == 0:
        return np.empty(0, dtype=np.float64)

    expanded_radii = atom_radii + probe_radius
    if not np.all(np.isfinite(expanded_radii)):
        raise ValueError("probe-expanded radii must be finite")

    point_count = points.shape[0]
    surface_areas = np.empty(atom_count, dtype=np.float64)
    for atom_index in range(atom_count):
        exposed_count = 0
        for unit_point in points:
            sample_point = (
                atom_coordinates[atom_index]
                + expanded_radii[atom_index] * unit_point
            )
            blocked = False
            for other_index in range(atom_count):
                if other_index == atom_index:
                    continue
                displacement = sample_point - atom_coordinates[other_index]
                squared_distance = float(np.dot(displacement, displacement))
                if squared_distance <= expanded_radii[other_index] ** 2:
                    blocked = True
                    break
            if not blocked:
                exposed_count += 1
        surface_areas[atom_index] = (
            4.0
            * math.pi
            * expanded_radii[atom_index] ** 2
            * exposed_count
            / point_count
        )

    return surface_areas


def _prepare_atom_sasa_inputs(
    coordinates,
    radii,
    *,
    probe_size: float,
    sphere_points: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Validate and normalize inputs shared by atom-SASA implementations."""

    if isinstance(probe_size, (bool, np.bool_)) or not isinstance(
        probe_size, numbers.Real
    ):
        raise TypeError("probe_size must be a real number")
    probe_radius = float(probe_size)
    if not math.isfinite(probe_radius) or probe_radius < 0.0:
        raise ValueError("probe_size must be finite and greater than or equal to zero")

    try:
        atom_coordinates = np.ascontiguousarray(coordinates, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError("coordinates must contain values convertible to float64") from error
    try:
        atom_radii = np.ascontiguousarray(radii, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError("radii must contain values convertible to float64") from error

    if atom_coordinates.ndim != 2 or atom_coordinates.shape[1:] != (3,):
        raise ValueError("coordinates must have shape (N, 3)")
    if atom_radii.ndim != 1:
        raise ValueError("radii must have shape (N,)")
    if atom_coordinates.shape[0] != atom_radii.shape[0]:
        raise ValueError("coordinates and radii must contain the same number of atoms")
    if not np.all(np.isfinite(atom_coordinates)):
        raise ValueError("coordinates must contain only finite values")
    if not np.all(np.isfinite(atom_radii)):
        raise ValueError("radii must contain only finite values")
    if np.any(atom_radii <= 0.0):
        raise ValueError("radii must contain only positive values")

    atom_count = atom_coordinates.shape[0]
    if atom_count > 0 and np.unique(atom_coordinates, axis=0).shape[0] != atom_count:
        raise ValueError("distinct atoms must not have coincident coordinates")

    if sphere_points is None:
        points = generate_sphere_points(DEFAULT_SPHERE_POINTS)
    else:
        try:
            points = np.ascontiguousarray(sphere_points, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise TypeError(
                "sphere_points must contain values convertible to float64"
            ) from error
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError("sphere_points must have shape (P, 3)")
        if points.shape[0] == 0:
            raise ValueError("sphere_points must not be empty")
        if not np.all(np.isfinite(points)):
            raise ValueError("sphere_points must contain only finite values")
        point_norms = np.linalg.norm(points, axis=1)
        if not np.allclose(point_norms, 1.0, rtol=1e-12, atol=1e-12):
            raise ValueError("sphere_points must contain unit vectors")

    return atom_coordinates, atom_radii, probe_radius, points


def _build_spatial_neighbors(
    atom_coordinates: np.ndarray,
    expanded_radii: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic CSR neighbors for overlapping expanded spheres."""

    atom_count = atom_coordinates.shape[0]
    if atom_count < 2:
        return (
            np.zeros(atom_count + 1, dtype=np.intp),
            np.empty(0, dtype=np.intp),
        )

    maximum_radius = float(np.max(expanded_radii))
    query_cutoff = np.nextafter(2.0 * maximum_radius, np.inf)
    pairs = np.asarray(
        cKDTree(atom_coordinates).query_pairs(
            query_cutoff,
            eps=0.0,
            output_type="ndarray",
        ),
        dtype=np.intp,
    ).reshape((-1, 2))
    if pairs.shape[0] == 0:
        return (
            np.zeros(atom_count + 1, dtype=np.intp),
            np.empty(0, dtype=np.intp),
        )

    displacements = atom_coordinates[pairs[:, 0]] - atom_coordinates[pairs[:, 1]]
    squared_distances = np.einsum("ij,ij->i", displacements, displacements)
    summed_radii = expanded_radii[pairs[:, 0]] + expanded_radii[pairs[:, 1]]
    squared_cutoffs = summed_radii * summed_radii
    keep = (squared_distances <= squared_cutoffs) | np.isclose(
        squared_distances,
        squared_cutoffs,
        rtol=8.0 * np.finfo(np.float64).eps,
        atol=0.0,
    )
    pairs = pairs[keep]
    if pairs.shape[0] == 0:
        return (
            np.zeros(atom_count + 1, dtype=np.intp),
            np.empty(0, dtype=np.intp),
        )

    rows = np.concatenate((pairs[:, 0], pairs[:, 1]))
    columns = np.concatenate((pairs[:, 1], pairs[:, 0]))
    order = np.lexsort((columns, rows))
    rows = rows[order]
    indices = np.ascontiguousarray(columns[order], dtype=np.intp)
    counts = np.bincount(rows, minlength=atom_count)
    offsets = np.empty(atom_count + 1, dtype=np.intp)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return offsets, indices


def _atom_sasa_from_neighbors(
    atom_coordinates: np.ndarray,
    expanded_radii: np.ndarray,
    sphere_points: np.ndarray,
    neighbor_offsets: np.ndarray,
    neighbor_indices: np.ndarray,
) -> np.ndarray:
    """Calculate atom SASA from validated arrays and CSR neighbors."""

    atom_count = atom_coordinates.shape[0]
    if atom_count == 0:
        return np.empty(0, dtype=np.float64)

    point_count = sphere_points.shape[0]
    surface_areas = np.empty(atom_count, dtype=np.float64)
    if neighbor_indices.size:
        sample_points = np.empty_like(sphere_points)
        displacements = np.empty_like(sphere_points)
        squared_distances = np.empty(point_count, dtype=np.float64)
        exposed = np.empty(point_count, dtype=np.bool_)
        outside_neighbor = np.empty(point_count, dtype=np.bool_)

    for atom_index in range(atom_count):
        neighbor_start = int(neighbor_offsets[atom_index])
        neighbor_stop = int(neighbor_offsets[atom_index + 1])
        if neighbor_start == neighbor_stop:
            exposed_count = point_count
        else:
            np.multiply(
                sphere_points,
                expanded_radii[atom_index],
                out=sample_points,
            )
            np.add(sample_points, atom_coordinates[atom_index], out=sample_points)
            exposed.fill(True)
            for neighbor_position in range(neighbor_start, neighbor_stop):
                other_index = neighbor_indices[neighbor_position]
                np.subtract(
                    sample_points,
                    atom_coordinates[other_index],
                    out=displacements,
                )
                np.einsum(
                    "ij,ij->i",
                    displacements,
                    displacements,
                    out=squared_distances,
                    optimize=False,
                )
                np.greater(
                    squared_distances,
                    expanded_radii[other_index] ** 2,
                    out=outside_neighbor,
                )
                np.logical_and(exposed, outside_neighbor, out=exposed)
                if not np.any(exposed):
                    break
            exposed_count = int(np.count_nonzero(exposed))
        surface_areas[atom_index] = (
            4.0
            * math.pi
            * expanded_radii[atom_index] ** 2
            * exposed_count
            / point_count
        )

    return surface_areas


def atom_sasa_spatial(
    coordinates,
    radii,
    *,
    probe_size: float = DEFAULT_PROBE_SIZE,
    sphere_points: np.ndarray | None = None,
) -> np.ndarray:
    """Return atom SASA using KD-tree-pruned neighbor lists."""

    atom_coordinates, atom_radii, probe_radius, points = _prepare_atom_sasa_inputs(
        coordinates,
        radii,
        probe_size=probe_size,
        sphere_points=sphere_points,
    )

    atom_count = atom_coordinates.shape[0]
    if atom_count == 0:
        return np.empty(0, dtype=np.float64)

    expanded_radii = atom_radii + probe_radius
    if not np.all(np.isfinite(expanded_radii)):
        raise ValueError("probe-expanded radii must be finite")

    neighbor_offsets, neighbor_indices = _build_spatial_neighbors(
        atom_coordinates,
        expanded_radii,
    )
    return _atom_sasa_from_neighbors(
        atom_coordinates,
        expanded_radii,
        points,
        neighbor_offsets,
        neighbor_indices,
    )


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
            "Calculate atom solvent-accessible surface areas for a protein "
            "structure."
        ),
        epilog=(
            "output files:\n"
            "  <base>.atom.sas       atom solvent-accessible surface areas\n"
            "  <base>.pqr            normalized atoms, radii, and zero charges\n"
            "  <base>.res.sas        reserved for the later residue-SASA stage\n\n"
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
        help=f"atom selection mode (default: {DEFAULT_MODE})",
    )
    parser.add_argument(
        "--prob-size",
        type=_positive_finite_float,
        default=DEFAULT_PROBE_SIZE,
        metavar="FLOAT",
        help=(
            "solvent probe radius in angstroms "
            f"(default: {DEFAULT_PROBE_SIZE:.2f})"
        ),
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=DEFAULT_WORKERS,
        metavar="INTEGER",
        help=f"number of worker processes (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--preserve-het",
        action="store_true",
        default=DEFAULT_PRESERVE_HET,
        help=(
            "preserve loose hetero-atoms "
            f"(default: {str(DEFAULT_PRESERVE_HET).lower()})"
        ),
    )
    parser.add_argument(
        "--use-h",
        action="store_true",
        default=DEFAULT_USE_H,
        help=(
            "use hydrogen atoms supplied in the input file "
            f"(default: {str(DEFAULT_USE_H).lower()})"
        ),
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


def derive_pqr_path(input_path: str | Path) -> Path:
    """Return the normalized PQR output path without modifying the filesystem."""

    base = _strip_structure_suffix(Path(input_path))
    return base.with_name(f"{base.name}.pqr")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line entry point."""

    start_time = time.perf_counter()
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.mode != "ALL":
        parser.error("SIDE and KEY atom-selection modes are not implemented yet")
    atom_output, _ = derive_output_paths(arguments.input)
    pqr_output = derive_pqr_path(arguments.input)
    try:
        source_atoms = read_structure(arguments.input)
        atoms = normalize_atoms(
            source_atoms,
            use_h=arguments.use_h,
            preserve_het=arguments.preserve_het,
        )
        atom_sasa = calculate_atom_sasa(atoms, probe_size=arguments.prob_size)
    except (StructureReadError, ValueError) as error:
        parser.error(str(error))
    try:
        write_atom_sasa_tsv(atom_output, atoms, atom_sasa)
        write_pqr(pqr_output, atoms)
    except OSError as error:
        parser.error(f"could not write output files: {error}")
    elapsed_time = time.perf_counter() - start_time
    print(f"Total elapsed time: {elapsed_time:.3f} seconds", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

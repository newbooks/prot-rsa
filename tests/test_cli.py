from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import protrsa


def make_input(tmp_path: Path, name: str = "protein.pdb") -> Path:
    input_path = tmp_path / name
    input_path.write_text("", encoding="utf-8")
    return input_path


def test_defaults(tmp_path: Path) -> None:
    input_path = make_input(tmp_path)

    arguments = protrsa.parse_args([str(input_path)])

    assert arguments.input == input_path
    assert arguments.mode == protrsa.DEFAULT_MODE
    assert arguments.prob_size == protrsa.DEFAULT_PROBE_SIZE
    assert arguments.workers == protrsa.DEFAULT_WORKERS
    assert arguments.preserve_het is protrsa.DEFAULT_PRESERVE_HET
    assert arguments.use_h is protrsa.DEFAULT_USE_H


def test_scientific_constants_match_decisions() -> None:
    assert protrsa.DEFAULT_PROBE_SIZE == 1.40
    assert protrsa.IRON_RADIUS == 2.00
    assert protrsa.UNKNOWN_RADIUS == protrsa.IRON_RADIUS
    assert protrsa.EXPLICIT_ATOM_RADII["FE"] == protrsa.IRON_RADIUS
    assert protrsa.EXPLICIT_ATOM_RADII["X"] == protrsa.UNKNOWN_RADIUS
    assert protrsa.EXPLICIT_ATOM_RADII["H"] == 1.10
    assert protrsa.EXPLICIT_ATOM_RADII["D"] == protrsa.EXPLICIT_ATOM_RADII["H"]
    assert protrsa.LOOSE_HETERO_COMPONENTS == (
        protrsa.WATER_COMPONENTS
        | protrsa.SIMPLE_ION_COMPONENTS
        | protrsa.CRYSTALLIZATION_ADDITIVE_COMPONENTS
    )


def test_probe_default_and_help_derive_from_constant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = make_input(tmp_path)
    monkeypatch.setattr(protrsa, "DEFAULT_PROBE_SIZE", 1.25)

    parser = protrsa.build_parser(prog="prot-rsa")

    assert parser.parse_args([str(input_path)]).prob_size == 1.25
    assert "default: 1.25" in parser.format_help()


@pytest.mark.parametrize(
    ("supplied", "expected"),
    [("ALL", "ALL"), ("all", "ALL"), ("SiDe", "SIDE"), ("key", "KEY")],
)
def test_modes_are_case_insensitive(
    tmp_path: Path, supplied: str, expected: str
) -> None:
    input_path = make_input(tmp_path)

    arguments = protrsa.parse_args([str(input_path), "--mode", supplied])

    assert arguments.mode == expected


def test_unknown_mode_is_rejected(tmp_path: Path) -> None:
    input_path = make_input(tmp_path)

    with pytest.raises(SystemExit) as error:
        protrsa.parse_args([str(input_path), "--mode", "backbone"])

    assert error.value.code != 0


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_invalid_probe_sizes_are_rejected(tmp_path: Path, value: str) -> None:
    input_path = make_input(tmp_path)

    with pytest.raises(SystemExit) as error:
        protrsa.parse_args([str(input_path), "--prob-size", value])

    assert error.value.code != 0


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "many"])
def test_invalid_worker_counts_are_rejected(tmp_path: Path, value: str) -> None:
    input_path = make_input(tmp_path)

    with pytest.raises(SystemExit) as error:
        protrsa.parse_args([str(input_path), "--workers", value])

    assert error.value.code != 0


@pytest.mark.parametrize(
    ("flags", "preserve_het", "use_h"),
    [
        ([], False, False),
        (["--preserve-het"], True, False),
        (["--use-h"], False, True),
        (["--preserve-het", "--use-h"], True, True),
    ],
)
def test_boolean_flags(
    tmp_path: Path,
    flags: list[str],
    preserve_het: bool,
    use_h: bool,
) -> None:
    input_path = make_input(tmp_path)

    arguments = protrsa.parse_args([*flags, str(input_path)])

    assert arguments.preserve_het is preserve_het
    assert arguments.use_h is use_h


@pytest.mark.parametrize(
    "name",
    [
        "protein.pdb",
        "protein.cif",
        "protein.pdb.gz",
        "protein.cif.gz",
        "protein.PDB",
        "protein.CIF",
        "protein.PDB.GZ",
        "protein.CIF.GZ",
    ],
)
def test_supported_input_suffixes(tmp_path: Path, name: str) -> None:
    input_path = make_input(tmp_path, name)

    assert protrsa.parse_args([str(input_path)]).input == input_path


@pytest.mark.parametrize("name", ["protein.txt", "protein.gz", "protein.mmcif"])
def test_unsupported_input_suffixes_are_rejected(tmp_path: Path, name: str) -> None:
    input_path = make_input(tmp_path, name)

    with pytest.raises(SystemExit) as error:
        protrsa.parse_args([str(input_path)])

    assert error.value.code != 0


@pytest.mark.parametrize(
    ("input_name", "atom_name", "residue_name"),
    [
        ("protein.pdb", "protein.atom.sas", "protein.res.sas"),
        ("protein.cif.gz", "protein.atom.sas", "protein.res.sas"),
        ("model.v2.PDB.GZ", "model.v2.atom.sas", "model.v2.res.sas"),
    ],
)
def test_output_path_derivation(
    tmp_path: Path,
    input_name: str,
    atom_name: str,
    residue_name: str,
) -> None:
    input_path = tmp_path / "inputs" / input_name

    atom_path, residue_path = protrsa.derive_output_paths(input_path)

    assert atom_path == input_path.with_name(atom_name)
    assert residue_path == input_path.with_name(residue_name)


def test_output_path_derivation_rejects_unsupported_suffix() -> None:
    with pytest.raises(ValueError, match="unsupported input extension"):
        protrsa.derive_output_paths("protein.txt")


def test_relative_output_path_derivation() -> None:
    atom_path, residue_path = protrsa.derive_output_paths("inputs/protein.pdb.gz")

    assert atom_path == Path("inputs/protein.atom.sas")
    assert residue_path == Path("inputs/protein.res.sas")


def test_missing_input_is_rejected() -> None:
    with pytest.raises(SystemExit) as error:
        protrsa.parse_args([])

    assert error.value.code != 0


def test_extra_input_is_rejected(tmp_path: Path) -> None:
    first_input = make_input(tmp_path, "one.pdb")
    second_input = make_input(tmp_path, "two.pdb")

    with pytest.raises(SystemExit) as error:
        protrsa.parse_args([str(first_input), str(second_input)])

    assert error.value.code != 0


def test_nonexistent_input_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        protrsa.parse_args([str(tmp_path / "missing.pdb")])

    assert error.value.code != 0


def test_directory_input_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "directory.pdb"
    directory.mkdir()

    with pytest.raises(SystemExit) as error:
        protrsa.parse_args([str(directory)])

    assert error.value.code != 0


def test_help_contains_required_contract() -> None:
    help_text = protrsa.build_parser(prog="prot-rsa").format_help()

    for required_text in (
        "usage: prot-rsa",
        "INPUT",
        ".pdb",
        ".cif",
        ".pdb.gz",
        ".cif.gz",
        "--mode {ALL,SIDE,KEY}",
        "--prob-size FLOAT",
        "--workers INTEGER",
        "--preserve-het",
        "--use-h",
        "preserve loose hetero-atoms",
        "use hydrogen atoms supplied in the input file",
        "default: ALL",
        "default: 1.40",
        "default: 4",
        "default: false",
        "<base>.atom.sas",
        "<base>.pqr",
        "<base>.res.sas",
        "written alongside INPUT",
    ):
        assert required_text in help_text


def test_direct_script_help_succeeds() -> None:
    project_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, str(project_root / "protrsa.py"), "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--prob-size FLOAT" in result.stdout
    assert result.stderr == ""


def test_empty_structure_fails_without_outputs(tmp_path: Path) -> None:
    input_path = make_input(tmp_path)

    with pytest.raises(SystemExit) as error:
        protrsa.main([str(input_path)])

    assert error.value.code != 0
    assert not (tmp_path / "protein.atom.sas").exists()
    assert not (tmp_path / "protein.res.sas").exists()
    assert not (tmp_path / "protein.pqr").exists()

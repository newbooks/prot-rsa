from __future__ import annotations

import gzip
from pathlib import Path

import pytest

import protrsa


def pdb_atom(
    serial: int,
    atom_name: str,
    *,
    altloc: str = "",
    residue_name: str = "ALA",
    chain_id: str = "A",
    residue_sequence: int = 1,
    x: float = 1.0,
    y: float = 2.0,
    z: float = 3.0,
    occupancy: float | None = 1.0,
    element: str = "C",
    record_type: str = "ATOM",
) -> str:
    occupancy_text = "      " if occupancy is None else f"{occupancy:6.2f}"
    return (
        f"{record_type:<6}{serial:5d} {atom_name:>4}{altloc:1}{residue_name:>3} "
        f"{chain_id:1}{residue_sequence:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
        f"{occupancy_text}{20.0:6.2f}          {element:>2}\n"
    )


def mmcif_text(rows: str) -> str:
    return f"""data_test
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
{rows}#
"""


def write_text_or_gzip(path: Path, text: str) -> None:
    if path.name.lower().endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8") as output:
            output.write(text)
    else:
        path.write_text(text, encoding="utf-8")


@pytest.mark.parametrize("name", ["model.pdb", "model.PDB.GZ"])
def test_read_pdb_and_gzip(name: str, tmp_path: Path) -> None:
    path = tmp_path / name
    write_text_or_gzip(
        path,
        pdb_atom(1, "N", element="N")
        + pdb_atom(
            2,
            "O",
            residue_name="HOH",
            residue_sequence=2,
            record_type="HETATM",
            element="O",
            occupancy=None,
        ),
    )

    atoms = protrsa.read_structure(path)

    assert atoms == [
        protrsa.AtomRecord(
            "ATOM", "1", "N", "N", "", "ALA", "A", "1", "", 1.0, 2.0, 3.0,
            1.0, 1, "   N"
        ),
        protrsa.AtomRecord(
            "HETATM", "2", "O", "O", "", "HOH", "A", "2", "", 1.0, 2.0,
            3.0, None, 1, "   O"
        ),
    ]


@pytest.mark.parametrize("name", ["model.cif", "model.CIF.GZ"])
def test_read_mmcif_and_gzip_with_author_identifiers(name: str, tmp_path: Path) -> None:
    path = tmp_path / name
    text = mmcif_text(
        "ATOM 1 N N . GLY X 9 ? 1.0 2.0 3.0 0.75 42 GLY A 'N' 1\n"
    )
    write_text_or_gzip(path, text)

    atoms = protrsa.read_structure(path)

    assert atoms == [
        protrsa.AtomRecord(
            "ATOM", "1", "N", "N", "", "GLY", "A", "42", "", 1.0, 2.0, 3.0,
            0.75, 1, " N  "
        )
    ]


def test_pdb_atom_name_field_is_retained_exactly(tmp_path: Path) -> None:
    path = tmp_path / "names.pdb"
    alpha_carbon = pdb_atom(1, "CA", element="C")
    calcium = pdb_atom(2, "CA", element="CA", record_type="HETATM")
    path.write_text(alpha_carbon + calcium, encoding="utf-8")

    atoms = protrsa.read_structure(path)

    assert atoms[0].atom_name == atoms[1].atom_name == "CA"
    assert atoms[0].atom_name_raw == alpha_carbon[12:16]
    assert atoms[1].atom_name_raw == calcium[12:16]


@pytest.mark.parametrize("suffix", ["pdb", "cif"])
def test_residue_consistent_highest_mean_occupancy_is_selected(
    suffix: str, tmp_path: Path
) -> None:
    path = tmp_path / f"alternate.{suffix}"
    if suffix == "pdb":
        text = (
            pdb_atom(1, "N")
            + pdb_atom(2, "CA", altloc="A", occupancy=0.60)
            + pdb_atom(3, "CB", altloc="A", occupancy=0.60)
            + pdb_atom(4, "CA", altloc="B", occupancy=0.40)
            + pdb_atom(5, "CB", altloc="B", occupancy=0.90)
        )
    else:
        text = mmcif_text(
            "ATOM 1 N N . ALA A 1 ? 0 0 0 1.00 1 ALA A N 1\n"
            "ATOM 2 C CA A ALA A 1 ? 1 0 0 0.60 1 ALA A CA 1\n"
            "ATOM 3 C CB A ALA A 1 ? 1 1 0 0.60 1 ALA A CB 1\n"
            "ATOM 4 C CA B ALA A 1 ? 2 0 0 0.40 1 ALA A CA 1\n"
            "ATOM 5 C CB B ALA A 1 ? 2 1 0 0.90 1 ALA A CB 1\n"
        )
    write_text_or_gzip(path, text)

    atoms = protrsa.read_structure(path)

    assert [atom.serial for atom in atoms] == ["1", "4", "5"]
    assert [atom.altloc for atom in atoms] == ["", "B", "B"]


def test_complete_occupancy_label_beats_label_with_missing_value(tmp_path: Path) -> None:
    path = tmp_path / "missing-occupancy.pdb"
    path.write_text(
        pdb_atom(1, "CA", altloc="A", occupancy=0.40)
        + pdb_atom(2, "CB", altloc="A", occupancy=0.40)
        + pdb_atom(3, "CA", altloc="B", occupancy=0.90)
        + pdb_atom(4, "CB", altloc="B", occupancy=None),
        encoding="utf-8",
    )

    atoms = protrsa.read_structure(path)

    assert [atom.altloc for atom in atoms] == ["A", "A"]


def test_alternate_selection_groups_atom_and_hetatm_in_one_residue(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mixed-records.pdb"
    path.write_text(
        pdb_atom(1, "CA", altloc="A", occupancy=0.40)
        + pdb_atom(2, "CB", altloc="B", occupancy=0.80, record_type="HETATM"),
        encoding="utf-8",
    )

    atoms = protrsa.read_structure(path)

    assert [atom.serial for atom in atoms] == ["2"]
    assert [atom.altloc for atom in atoms] == ["B"]


def test_occupancy_tie_prefers_a_then_lexical_order(tmp_path: Path) -> None:
    path = tmp_path / "tie.pdb"
    path.write_text(
        pdb_atom(1, "CA", altloc="C", occupancy=0.50)
        + pdb_atom(2, "CA", altloc="B", occupancy=0.50)
        + pdb_atom(3, "CA", altloc="A", occupancy=0.50),
        encoding="utf-8",
    )

    assert [atom.altloc for atom in protrsa.read_structure(path)] == ["A"]


@pytest.mark.parametrize("suffix", ["pdb", "cif"])
def test_multiple_models_are_rejected(suffix: str, tmp_path: Path) -> None:
    path = tmp_path / f"multi.{suffix}"
    if suffix == "pdb":
        text = (
            "MODEL        1\n"
            + pdb_atom(1, "N")
            + "ENDMDL\nMODEL        2\n"
            + pdb_atom(2, "N")
        )
    else:
        text = mmcif_text(
            "ATOM 1 N N . GLY A 1 ? 0 0 0 1 1 GLY A N 1\n"
            "ATOM 2 N N . GLY A 1 ? 0 0 0 1 1 GLY A N 2\n"
        )
    write_text_or_gzip(path, text)

    with pytest.raises(protrsa.StructureReadError, match="multiple coordinate models"):
        protrsa.read_structure(path)


@pytest.mark.parametrize(
    ("name", "text", "match"),
    [
        ("empty.pdb", "HEADER empty\n", "no ATOM or HETATM"),
        ("short.pdb", "ATOM      1\n", "too short"),
        ("empty.cif", "data_empty\n#\n", "no ATOM or HETATM"),
        (
            "missing-column.cif",
            "data_bad\nloop_\n_atom_site.group_PDB\n_atom_site.id\nATOM 1\n#\n",
            "missing _atom_site.cartn_x",
        ),
    ],
)
def test_malformed_or_empty_structures_are_rejected(
    name: str, text: str, match: str, tmp_path: Path
) -> None:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")

    with pytest.raises(protrsa.StructureReadError, match=match):
        protrsa.read_structure(path)


def test_malformed_gzip_is_reported_as_structure_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.pdb.gz"
    path.write_bytes(b"not gzip data")

    with pytest.raises(protrsa.StructureReadError, match="could not read"):
        protrsa.read_structure(path)


def test_cli_reports_structure_error_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "empty.pdb"
    path.write_text("HEADER empty\n", encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        protrsa.main([str(path)])

    captured = capsys.readouterr()
    assert error.value.code != 0
    assert "structure contains no ATOM or HETATM" in captured.err
    assert "Traceback" not in captured.err

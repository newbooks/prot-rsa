from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import protrsa


def atom(
    atom_name: str,
    element: str,
    *,
    residue_name: str = "ALA",
    record_type: str = "ATOM",
    serial: str = "1",
    atom_name_raw: str = "",
) -> protrsa.AtomRecord:
    return protrsa.AtomRecord(
        record_type, serial, element, atom_name, "", residue_name, "A", "1", "",
        0.0, 0.0, 0.0, 1.0, 1, atom_name_raw,
    )


def test_default_normalization_filters_hydrogen_and_loose_hetatm() -> None:
    source = [
        atom("CA", "C", serial="1"),
        atom("HA", "H", serial="2"),
        atom("D1", "D", serial="3"),
        atom("O", "O", residue_name="HOH", record_type="HETATM", serial="4"),
        atom("C1", "C", residue_name="LIG", record_type="HETATM", serial="5"),
    ]

    normalized = protrsa.normalize_atoms(source)

    assert [item.source.serial for item in normalized] == ["1", "5"]
    assert normalized[0].radius == protrsa.PROTOR_RADII["TETRAHEDRAL_C"]
    assert normalized[1].radius == protrsa.UNKNOWN_RADIUS
    assert normalized[1].element == "X"


def test_blank_element_fallback_distinguishes_alpha_carbon_and_calcium() -> None:
    alpha_carbon = atom("CA", "", atom_name_raw=" CA ")
    calcium = atom(
        "CA", "", residue_name="CA", record_type="HETATM", serial="2",
        atom_name_raw="CA  "
    )

    normalized = protrsa.normalize_atoms([alpha_carbon, calcium])

    assert [item.source.atom_name for item in normalized] == ["CA", "CA"]
    assert [item.element for item in normalized] == ["C", "X"]


def test_use_h_retains_hydrogen_and_uses_only_explicit_radii() -> None:
    normalized = protrsa.normalize_atoms(
        [atom("CA", "C"), atom("HA", "H", serial="2")], use_h=True
    )

    assert [item.radius for item in normalized] == [
        protrsa.EXPLICIT_ATOM_RADII["C"],
        protrsa.EXPLICIT_ATOM_RADII["H"],
    ]


def test_preserve_het_retains_loose_component() -> None:
    normalized = protrsa.normalize_atoms(
        [atom("O", "O", residue_name="HOH", record_type="HETATM")],
        preserve_het=True,
    )
    assert len(normalized) == 1


@pytest.mark.parametrize(
    ("residue", "atom_name", "expected_type"),
    [
        ("ALA", "C", "TRIGONAL_C_NO_H"),
        ("PHE", "CG", "TRIGONAL_C_NO_H"),
        ("PHE", "CZ", "TRIGONAL_C_ONE_H"),
        ("ALA", "CA", "TETRAHEDRAL_C"),
        ("SER", "OG", "HYDROXYL_O"),
        ("ASP", "OD1", "CARBONYL_O"),
    ],
)
def test_residue_aware_protor_assignment(
    residue: str, atom_name: str, expected_type: str
) -> None:
    normalized = protrsa.normalize_atoms(
        [atom(atom_name, atom_name[0], residue_name=residue)]
    )
    assert normalized[0].radius == protrsa.PROTOR_RADII[expected_type]


def test_calculation_uses_normalized_radius() -> None:
    normalized = protrsa.normalize_atoms([atom("CA", "C")])
    sasa = protrsa.calculate_atom_sasa(
        normalized,
        probe_size=0.0,
        sphere_points=np.array([[1.0, 0.0, 0.0]]),
    )
    assert sasa[0] == pytest.approx(4.0 * math.pi * normalized[0].radius**2)


def test_writers_emit_zero_charge_radius_and_sasa(tmp_path: Path) -> None:
    normalized = protrsa.normalize_atoms([atom("CA", "C")])
    pqr_path = tmp_path / "model.pqr"
    sasa_path = tmp_path / "model.atom.sas"

    protrsa.write_pqr(pqr_path, normalized)
    protrsa.write_atom_sasa_tsv(sasa_path, normalized, [12.3456789])

    pqr_text = pqr_path.read_text(encoding="utf-8")
    assert "  0.000 " in pqr_text
    assert f" {normalized[0].radius:6.3f}" in pqr_text
    sasa_text = sasa_path.read_text(encoding="utf-8")
    assert "sasa_A2" in sasa_text
    assert "12.345679" in sasa_text


def test_writer_removes_file_when_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "partial.pqr"
    original_open = Path.open

    class FailingFile:
        def __init__(self, path: Path, *args: object, **kwargs: object) -> None:
            self.file = original_open(path, *args, **kwargs)  # type: ignore[arg-type]

        def __enter__(self) -> "FailingFile":
            return self

        def write(self, text: str) -> None:
            self.file.write(text[:5])
            raise OSError("simulated disk full")

        def __exit__(self, *args: object) -> None:
            self.file.close()

    def failing_open(path: Path, *args: object, **kwargs: object) -> FailingFile:
        return FailingFile(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)

    with pytest.raises(OSError, match="disk full"):
        protrsa.write_pqr(output_path, protrsa.normalize_atoms([atom("CA", "C")]))

    assert not output_path.exists()


def test_pqr_path_derivation() -> None:
    assert protrsa.derive_pqr_path("inputs/model.cif.gz") == Path("inputs/model.pqr")


def test_cli_writes_atom_sasa_and_pqr_for_single_atom(tmp_path: Path) -> None:
    input_path = tmp_path / "one.pdb"
    input_path.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00"
        "           C  \n",
        encoding="utf-8",
    )

    exit_status = protrsa.main([str(input_path)])

    assert exit_status == 0
    atom_output = tmp_path / "one.atom.sas"
    pqr_output = tmp_path / "one.pqr"
    assert atom_output.is_file()
    assert pqr_output.is_file()
    assert not (tmp_path / "one.res.sas").exists()
    assert "sasa_A2" in atom_output.read_text(encoding="utf-8")
    assert "0.000" in pqr_output.read_text(encoding="utf-8")


def test_cli_refuses_to_overwrite_output(tmp_path: Path) -> None:
    input_path = tmp_path / "one.pdb"
    input_path.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00"
        "           C  \n",
        encoding="utf-8",
    )
    output_path = tmp_path / "one.atom.sas"
    output_path.write_text("keep me\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        protrsa.main([str(input_path)])

    assert output_path.read_text(encoding="utf-8") == "keep me\n"
    assert not (tmp_path / "one.pqr").exists()

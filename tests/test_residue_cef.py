from __future__ import annotations

from pathlib import Path

import numpy as np

import protrsa


def make_atom(
    *,
    serial: str,
    atom_name: str,
    residue_name: str,
    chain_id: str,
    residue_sequence: str,
    insertion_code: str = "",
    x: float,
    y: float = 0.0,
    z: float = 0.0,
) -> protrsa.NormalizedAtom:
    source = protrsa.AtomRecord(
        record_type="ATOM",
        serial=serial,
        element="C",
        atom_name=atom_name,
        altloc="",
        residue_name=residue_name,
        chain_id=chain_id,
        residue_sequence=residue_sequence,
        insertion_code=insertion_code,
        x=x,
        y=y,
        z=z,
        occupancy=1.0,
        model=1,
        atom_name_raw=f"{atom_name:>4}",
    )
    return protrsa.NormalizedAtom(source=source, element="C", radius=1.0)


def test_residue_cef_groups_atoms_and_preserves_first_seen_order() -> None:
    atoms = [
        make_atom(
            serial="1",
            atom_name="CA",
            residue_name="ALA",
            chain_id="A",
            residue_sequence="1",
            x=0.0,
        ),
        make_atom(
            serial="2",
            atom_name="CA",
            residue_name="GLY",
            chain_id="A",
            residue_sequence="2",
            x=2.0,
        ),
        make_atom(
            serial="3",
            atom_name="CB",
            residue_name="ALA",
            chain_id="A",
            residue_sequence="1",
            x=0.0,
            z=3.0,
        ),
    ]
    points = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    coordinates = np.array([(a.source.x, a.source.y, a.source.z) for a in atoms])
    radii = np.array([a.radius for a in atoms])
    atom_sasa = protrsa.atom_sasa_spatial(
        coordinates, radii, probe_size=0.0, sphere_points=points, backend="cpu"
    )

    records = protrsa.calculate_residue_sasa(
        atoms, atom_sasa, probe_size=0.0, sphere_points=points, backend="cpu"
    )

    assert [(record.residue_name, record.residue_sequence) for record in records] == [
        ("ALA", "1"),
        ("GLY", "2"),
    ]
    assert records[0].sasa_inprotein < records[0].sasa_reference
    assert 0.0 <= records[0].cef < 1.0
    assert 0.0 <= records[1].cef < 1.0


def test_residue_cef_writer_uses_required_schema_and_three_decimals(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "1LYZ.res.sas"
    records = [
        protrsa.ResidueCefRecord(
            residue_name="ALA",
            chain_id="A",
            residue_sequence="7",
            insertion_code="B",
            sasa_inprotein=12.3456,
            sasa_reference=23.4567,
            cef=0.52501,
        )
    ]

    protrsa.write_residue_cef_tsv(output_path, records)

    assert output_path.read_text(encoding="utf-8") == (
        "residue_name\tchain_id\tresidue_sequence\tinsertion_code\t"
        "sasa_inprotein\tsasa_reference\tcef\n"
        "ALA\tA\t7\tB\t12.346\t23.457\t0.525\n"
    )


def test_residue_cef_keeps_same_residue_occlusion_in_reference() -> None:
    atoms = [
        make_atom(
            serial="1",
            atom_name="CA",
            residue_name="ALA",
            chain_id="",
            residue_sequence="X",
            x=0.0,
        ),
        make_atom(
            serial="2",
            atom_name="CB",
            residue_name="ALA",
            chain_id="",
            residue_sequence="X",
            x=2.0,
        ),
    ]
    points = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    coordinates = np.array([(a.source.x, a.source.y, a.source.z) for a in atoms])
    radii = np.array([a.radius for a in atoms])
    atom_sasa = protrsa.atom_sasa_spatial(
        coordinates, radii, probe_size=0.0, sphere_points=points, backend="cpu"
    )

    records = protrsa.calculate_residue_sasa(
        atoms, atom_sasa, probe_size=0.0, sphere_points=points, backend="cpu"
    )

    assert len(records) == 1
    assert records[0].sasa_inprotein == records[0].sasa_reference
    assert records[0].cef == 1.0

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
    assert records[0].sasa_all < records[0].sasa_all_ref
    assert 0.0 <= records[0].sasa_all_ratio < 1.0
    assert 0.0 <= records[1].sasa_all_ratio < 1.0
    assert records[0].sasa_side is not None
    assert records[1].sasa_side is None
    assert records[1].sasa_side_ref is None
    assert records[1].sasa_side_ratio is None


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
            sasa_all=12.3456,
            sasa_all_ref=23.4567,
            sasa_all_ratio=0.52501,
            sasa_side=5.4321,
            sasa_side_ref=9.8765,
            sasa_side_ratio=0.55001,
        )
    ]

    protrsa.write_residue_cef_tsv(output_path, records)

    assert output_path.read_text(encoding="utf-8") == (
        "# res: residue name\n"
        "# chain: chain identifier\n"
        "# seq: residue sequence identifier\n"
        "# ins: insertion code\n"
        "# all: in-protein SASA of all residue atoms, Å²\n"
        "# a_ref: isolated-residue reference SASA of all atoms, Å²\n"
        "# a_cef: all-atom contextual exposure fraction\n"
        "# side: in-protein side-chain SASA, Å²\n"
        "# s_ref: isolated side-chain reference SASA, Å²\n"
        "# s_cef: side-chain contextual exposure fraction\n"
        "res\tchain\tseq\tins\tall\ta_ref\ta_cef\t"
        "side\ts_ref\ts_cef\n"
        "ALA\tA\t7\tB\t12.346\t23.457\t0.525\t5.432\t9.877\t0.550\n"
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
    assert records[0].sasa_all == records[0].sasa_all_ref
    assert records[0].sasa_all_ratio == 1.0
    assert records[0].sasa_side < records[0].sasa_side_ref
    assert records[0].sasa_side_ratio == 0.5

from __future__ import annotations

from pathlib import Path

import pytest

import protrsa


HEADER = "\t".join(protrsa.ATOM_SASA_COLUMNS) + "\n"


def sas_row(*, atom_index: int, atom_name: str, sasa: str) -> str:
    return (
        f"{atom_index}\tATOM\t{atom_index}\t{atom_name}\tALA\tA\t1\t\tC\t"
        f"1.880\t{sasa}\n"
    )


def sas_row_with_radius(
    *, atom_index: int, atom_name: str, radius: str, sasa: str
) -> str:
    return (
        f"{atom_index}\tATOM\t{atom_index}\t{atom_name}\tALA\tA\t1\t\tC\t"
        f"{radius}\t{sasa}\n"
    )


def write_sas(path: Path, *rows: str) -> None:
    path.write_text(HEADER + "".join(rows), encoding="utf-8")


def test_compare_atom_sasa_files_returns_count_and_mae(tmp_path: Path) -> None:
    first = tmp_path / "first.atom.sas"
    second = tmp_path / "second.atom.sas"
    write_sas(
        first,
        sas_row(atom_index=1, atom_name="CA", sasa="10.0"),
        sas_row(atom_index=2, atom_name="CB", sasa="20.0"),
    )
    write_sas(
        second,
        sas_row(atom_index=1, atom_name="CA", sasa="12.0"),
        sas_row(atom_index=2, atom_name="CB", sasa="17.0"),
    )

    atom_count, mae = protrsa.compare_atom_sasa_files(first, second)

    assert atom_count == 2
    assert mae == pytest.approx(2.5)


def test_compare_command_prints_atom_count_and_mae(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = tmp_path / "first.atom.sas"
    second = tmp_path / "second.atom.sas"
    write_sas(first, sas_row(atom_index=1, atom_name="CA", sasa="1.25"))
    write_sas(second, sas_row(atom_index=1, atom_name="CA", sasa="2.00"))

    exit_status = protrsa.compare_sas_main([str(first), str(second)])

    assert exit_status == 0
    assert capsys.readouterr().out == "Atoms matched: 1\nSASA MAE: 0.750000 Å²\n"


def test_compare_allows_different_atomic_radii(tmp_path: Path) -> None:
    first = tmp_path / "first.atom.sas"
    second = tmp_path / "second.atom.sas"
    write_sas(
        first,
        sas_row_with_radius(
            atom_index=1, atom_name="CA", radius="1.880", sasa="10.0"
        ),
    )
    write_sas(
        second,
        sas_row_with_radius(
            atom_index=1, atom_name="CA", radius="1.700", sasa="12.0"
        ),
    )

    atom_count, mae = protrsa.compare_atom_sasa_files(first, second)

    assert atom_count == 1
    assert mae == pytest.approx(2.0)


def test_compare_rejects_atom_count_mismatch(tmp_path: Path) -> None:
    first = tmp_path / "first.atom.sas"
    second = tmp_path / "second.atom.sas"
    write_sas(first, sas_row(atom_index=1, atom_name="CA", sasa="1.0"))
    write_sas(
        second,
        sas_row(atom_index=1, atom_name="CA", sasa="1.0"),
        sas_row(atom_index=2, atom_name="CB", sasa="2.0"),
    )

    with pytest.raises(ValueError, match="atom count mismatch"):
        protrsa.compare_atom_sasa_files(first, second)


def test_compare_reports_first_atom_metadata_mismatch(tmp_path: Path) -> None:
    first = tmp_path / "first.atom.sas"
    second = tmp_path / "second.atom.sas"
    write_sas(first, sas_row(atom_index=1, atom_name="CA", sasa="1.0"))
    write_sas(second, sas_row(atom_index=1, atom_name="CB", sasa="1.0"))

    with pytest.raises(ValueError, match=r"data row 1.*atom_name"):
        protrsa.compare_atom_sasa_files(first, second)


@pytest.mark.parametrize(
    ("contents", "match"),
    [
        ("wrong\theader\n1\t2\n", "unexpected header"),
        (HEADER, "contains no atom rows"),
        (HEADER + sas_row(atom_index=1, atom_name="CA", sasa="nan"), "non-finite"),
        (HEADER + sas_row(atom_index=1, atom_name="CA", sasa="bad"), "invalid"),
    ],
)
def test_compare_rejects_invalid_sas_file(
    tmp_path: Path, contents: str, match: str
) -> None:
    first = tmp_path / "first.atom.sas"
    second = tmp_path / "second.atom.sas"
    first.write_text(contents, encoding="utf-8")
    write_sas(second, sas_row(atom_index=1, atom_name="CA", sasa="1.0"))

    with pytest.raises(ValueError, match=match):
        protrsa.compare_atom_sasa_files(first, second)

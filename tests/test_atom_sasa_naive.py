from __future__ import annotations

import math

import numpy as np
import pytest

import protrsa


def test_default_sphere_points_are_deterministic_unit_vectors() -> None:
    first = protrsa.generate_sphere_points()
    second = protrsa.generate_sphere_points()

    assert protrsa.DEFAULT_SPHERE_POINTS == 960
    assert first.shape == (protrsa.DEFAULT_SPHERE_POINTS, 3)
    assert first.dtype == np.float64
    assert np.all(np.isfinite(first))
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, rtol=1e-12, atol=1e-12)
    assert np.array_equal(first, second)


@pytest.mark.parametrize("n_points", [True, False, 1.5, "12", None])
def test_sphere_point_count_rejects_non_integers(n_points: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        protrsa.generate_sphere_points(n_points)  # type: ignore[arg-type]


@pytest.mark.parametrize("n_points", [0, -1])
def test_sphere_point_count_rejects_nonpositive_integers(n_points: int) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        protrsa.generate_sphere_points(n_points)


def test_sphere_point_count_accepts_numpy_integer() -> None:
    assert protrsa.generate_sphere_points(np.int64(7)).shape == (7, 3)


def test_single_atom_has_full_absolute_surface_area() -> None:
    result = protrsa.atom_sasa_reference([[0.0, 0.0, 0.0]], [1.7])

    expected = 4.0 * math.pi * (1.7 + protrsa.DEFAULT_PROBE_SIZE) ** 2
    np.testing.assert_allclose(result, [expected], rtol=1e-15, atol=1e-15)
    assert result.dtype == np.float64


def test_distant_atoms_have_full_surface_area() -> None:
    coordinates = [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]]
    radii = [1.0, 2.0]

    result = protrsa.atom_sasa_reference(coordinates, radii)

    expanded = np.asarray(radii) + protrsa.DEFAULT_PROBE_SIZE
    np.testing.assert_allclose(result, 4.0 * math.pi * expanded**2)


def test_overlapping_equal_spheres_approach_analytical_area() -> None:
    radius = 2.0
    distance = 2.0
    result = protrsa.atom_sasa_reference(
        [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]],
        [radius, radius],
        probe_size=0.0,
    )

    isolated = 4.0 * math.pi * radius**2
    analytical = 2.0 * math.pi * radius**2 + math.pi * radius * distance
    assert np.all(result < isolated)
    np.testing.assert_allclose(result, analytical, rtol=0.02)


def test_enclosed_atom_has_zero_surface_area() -> None:
    result = protrsa.atom_sasa_reference(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        [3.0, 1.0],
        probe_size=0.0,
    )

    assert result[0] == pytest.approx(4.0 * math.pi * 3.0**2)
    assert result[1] == 0.0


def test_translation_and_atom_permutation_preserve_results() -> None:
    coordinates = np.array([[0.0, 0.0, 0.0], [2.0, 0.2, 0.0], [0.5, 3.0, 0.1]])
    radii = np.array([1.5, 1.2, 1.8])
    original = protrsa.atom_sasa_reference(coordinates, radii)

    translated = protrsa.atom_sasa_reference(coordinates + [10.0, -7.0, 2.0], radii)
    permutation = np.array([2, 0, 1])
    permuted = protrsa.atom_sasa_reference(coordinates[permutation], radii[permutation])

    np.testing.assert_allclose(translated, original, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(permuted, original[permutation])


def test_repeated_calculations_are_bitwise_identical() -> None:
    coordinates = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    radii = [1.5, 1.5]

    first = protrsa.atom_sasa_reference(coordinates, radii)
    second = protrsa.atom_sasa_reference(coordinates, radii)

    assert np.array_equal(first, second)


def test_custom_points_control_denominator_and_boundary_is_blocked() -> None:
    points = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    result = protrsa.atom_sasa_reference(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        [1.0, 1.0],
        probe_size=0.0,
        sphere_points=points,
    )

    assert result[0] == pytest.approx(2.0 * math.pi)
    assert result[1] == pytest.approx(2.0 * math.pi)


def test_empty_atom_collection_returns_float64_array() -> None:
    result = protrsa.atom_sasa_reference(np.empty((0, 3)), np.empty((0,)))

    assert result.shape == (0,)
    assert result.dtype == np.float64


def test_empty_atom_collection_still_validates_custom_points() -> None:
    with pytest.raises(ValueError, match="empty"):
        protrsa.atom_sasa_reference(
            np.empty((0, 3)),
            np.empty((0,)),
            sphere_points=np.empty((0, 3)),
        )


@pytest.mark.parametrize(
    ("coordinates", "radii", "match"),
    [
        ([0.0, 0.0, 0.0], [1.0], "coordinates"),
        ([[0.0, 0.0]], [1.0], "coordinates"),
        ([[0.0, 0.0, 0.0]], [[1.0]], "radii"),
        ([[0.0, 0.0, 0.0]], [1.0, 2.0], "same number"),
        ([[math.nan, 0.0, 0.0]], [1.0], "finite"),
        ([[0.0, 0.0, 0.0]], [math.inf], "finite"),
        ([[0.0, 0.0, 0.0]], [0.0], "positive"),
        ([[0.0, 0.0, 0.0]], [-1.0], "positive"),
        ([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [1.0, 2.0], "coincident"),
    ],
)
def test_invalid_atom_inputs_raise_value_error(
    coordinates: object, radii: object, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        protrsa.atom_sasa_reference(coordinates, radii)


@pytest.mark.parametrize(
    ("coordinates", "radii", "match"),
    [
        ([["not-a-number", 0.0, 0.0]], [1.0], "coordinates"),
        ([[0.0, 0.0, 0.0]], ["not-a-number"], "radii"),
    ],
)
def test_unconvertible_atom_inputs_raise_type_error(
    coordinates: object, radii: object, match: str
) -> None:
    with pytest.raises(TypeError, match=match):
        protrsa.atom_sasa_reference(coordinates, radii)


@pytest.mark.parametrize("probe_size", [True, np.bool_(False), "1.4", None])
def test_invalid_probe_types_raise_type_error(probe_size: object) -> None:
    with pytest.raises(TypeError, match="probe_size"):
        protrsa.atom_sasa_reference([[0.0, 0.0, 0.0]], [1.0], probe_size=probe_size)  # type: ignore[arg-type]


@pytest.mark.parametrize("probe_size", [-1.0, math.nan, math.inf, -math.inf])
def test_invalid_probe_values_raise_value_error(probe_size: float) -> None:
    with pytest.raises(ValueError, match="probe_size"):
        protrsa.atom_sasa_reference([[0.0, 0.0, 0.0]], [1.0], probe_size=probe_size)


@pytest.mark.parametrize(
    ("points", "exception", "match"),
    [
        ([1.0, 0.0, 0.0], ValueError, "shape"),
        (np.empty((0, 3)), ValueError, "empty"),
        ([[math.nan, 0.0, 0.0]], ValueError, "finite"),
        ([[2.0, 0.0, 0.0]], ValueError, "unit vectors"),
        ([["not-a-number", 0.0, 0.0]], TypeError, "float64"),
    ],
)
def test_invalid_custom_sphere_points(
    points: object, exception: type[Exception], match: str
) -> None:
    with pytest.raises(exception, match=match):
        protrsa.atom_sasa_reference(
            [[0.0, 0.0, 0.0]], [1.0], sphere_points=points  # type: ignore[arg-type]
        )


def test_inputs_are_not_mutated() -> None:
    coordinates = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    radii = np.array([1.0, 1.0])
    points = protrsa.generate_sphere_points(12)
    coordinates_before = coordinates.copy()
    radii_before = radii.copy()
    points_before = points.copy()

    protrsa.atom_sasa_reference(
        coordinates, radii, probe_size=0.0, sphere_points=points
    )

    np.testing.assert_array_equal(coordinates, coordinates_before)
    np.testing.assert_array_equal(radii, radii_before)
    np.testing.assert_array_equal(points, points_before)

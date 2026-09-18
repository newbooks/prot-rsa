from __future__ import annotations

import math

import numpy as np
import pytest

import protrsa


def test_empty_and_single_atom_neighbor_csr() -> None:
    empty_offsets, empty_indices = protrsa._build_spatial_neighbors(
        np.empty((0, 3), dtype=np.float64),
        np.empty(0, dtype=np.float64),
    )
    single_offsets, single_indices = protrsa._build_spatial_neighbors(
        np.zeros((1, 3), dtype=np.float64),
        np.array([2.0]),
    )

    np.testing.assert_array_equal(empty_offsets, [0])
    np.testing.assert_array_equal(single_offsets, [0, 0])
    assert empty_offsets.dtype == np.intp
    assert single_offsets.dtype == np.intp
    assert empty_indices.dtype == np.intp
    assert single_indices.dtype == np.intp
    assert empty_indices.size == single_indices.size == 0


def test_neighbor_csr_is_symmetric_sorted_and_exactly_filtered() -> None:
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ]
    )
    expanded_radii = np.array([2.0, 2.0, 4.0, 2.0])

    offsets, indices = protrsa._build_spatial_neighbors(
        coordinates, expanded_radii
    )
    rows = [indices[offsets[i] : offsets[i + 1]].tolist() for i in range(4)]

    assert offsets.dtype == np.intp
    assert indices.dtype == np.intp
    np.testing.assert_array_equal(offsets, [0, 1, 3, 4, 4])
    assert rows == [[1], [0, 2], [1], []]
    assert all(row == sorted(set(row)) for row in rows)


def test_tangent_expanded_spheres_are_neighbors() -> None:
    coordinates = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])

    offsets, indices = protrsa._build_spatial_neighbors(
        coordinates, np.array([2.0, 2.0])
    )

    np.testing.assert_array_equal(offsets, [0, 1, 2])
    np.testing.assert_array_equal(indices, [1, 0])


def test_second_probe_radius_is_included_in_neighbor_cutoff() -> None:
    coordinates = [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
    radii = [1.0, 1.0]
    points = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])

    result = protrsa.atom_sasa_spatial(
        coordinates,
        radii,
        probe_size=1.4,
        sphere_points=points,
    )
    reference = protrsa.atom_sasa_reference(
        coordinates,
        radii,
        probe_size=1.4,
        sphere_points=points,
    )

    np.testing.assert_array_equal(result, reference)
    assert np.all(result < 4.0 * math.pi * 2.4**2)


def test_vectorized_mask_accumulates_disjoint_neighbor_occlusion() -> None:
    coordinates = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]
    radii = [1.0, 1.0, 1.0]
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )

    result = protrsa.atom_sasa_spatial(
        coordinates,
        radii,
        probe_size=0.0,
        sphere_points=points,
    )
    reference = protrsa.atom_sasa_reference(
        coordinates,
        radii,
        probe_size=0.0,
        sphere_points=points,
    )

    np.testing.assert_array_equal(result, reference)
    assert result[0] == pytest.approx(2.0 * math.pi)


def test_vectorized_mask_handles_first_neighbor_complete_block() -> None:
    coordinates = [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]]
    radii = [1.0, 3.0, 1.0]
    points = protrsa.generate_sphere_points(17)

    result = protrsa.atom_sasa_spatial(
        coordinates,
        radii,
        probe_size=0.0,
        sphere_points=points,
    )
    reference = protrsa.atom_sasa_reference(
        coordinates,
        radii,
        probe_size=0.0,
        sphere_points=points,
    )

    np.testing.assert_array_equal(result, reference)
    assert result[0] == 0.0


@pytest.mark.parametrize("atom_count", [0, 1, 2, 8, 24])
@pytest.mark.parametrize("probe_size", [0.0, 1.4, 2.25])
def test_spatial_is_bitwise_equal_to_reference_for_randomized_inputs(
    atom_count: int, probe_size: float
) -> None:
    random = np.random.default_rng(1729 + atom_count)
    coordinates = random.uniform(-8.0, 8.0, size=(atom_count, 3))
    radii = random.uniform(1.0, 2.5, size=atom_count)
    points = protrsa.generate_sphere_points(37)

    result = protrsa.atom_sasa_spatial(
        coordinates,
        radii,
        probe_size=probe_size,
        sphere_points=points,
    )
    reference = protrsa.atom_sasa_reference(
        coordinates,
        radii,
        probe_size=probe_size,
        sphere_points=points,
    )

    np.testing.assert_array_equal(result, reference)


def test_spatial_handles_enclosure_and_unequal_radii() -> None:
    coordinates = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [9.0, 0.0, 0.0]]
    radii = [3.0, 1.0, 2.0]

    result = protrsa.atom_sasa_spatial(coordinates, radii, probe_size=0.0)
    reference = protrsa.atom_sasa_reference(coordinates, radii, probe_size=0.0)

    np.testing.assert_array_equal(result, reference)
    assert result[1] == 0.0


def test_spatial_translation_permutation_and_repeat_invariants() -> None:
    coordinates = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.2, 0.0], [0.5, 3.0, 0.1]]
    )
    radii = np.array([1.5, 1.2, 1.8])
    points = protrsa.generate_sphere_points(53)
    original = protrsa.atom_sasa_spatial(
        coordinates, radii, sphere_points=points
    )
    repeated = protrsa.atom_sasa_spatial(
        coordinates, radii, sphere_points=points
    )
    translated = protrsa.atom_sasa_spatial(
        coordinates + [10.0, -7.0, 2.0], radii, sphere_points=points
    )
    permutation = np.array([2, 0, 1])
    permuted = protrsa.atom_sasa_spatial(
        coordinates[permutation], radii[permutation], sphere_points=points
    )

    np.testing.assert_array_equal(repeated, original)
    np.testing.assert_allclose(translated, original, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(permuted, original[permutation])


def test_spatial_does_not_mutate_inputs() -> None:
    coordinates = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    radii = np.array([1.0, 1.0])
    points = protrsa.generate_sphere_points(12)
    coordinates_before = coordinates.copy()
    radii_before = radii.copy()
    points_before = points.copy()

    protrsa.atom_sasa_spatial(
        coordinates, radii, probe_size=0.0, sphere_points=points
    )

    np.testing.assert_array_equal(coordinates, coordinates_before)
    np.testing.assert_array_equal(radii, radii_before)
    np.testing.assert_array_equal(points, points_before)


@pytest.mark.parametrize(
    ("coordinates", "radii", "probe_size", "sphere_points", "exception", "match"),
    [
        ([0.0, 0.0, 0.0], [1.0], 1.4, None, ValueError, "coordinates"),
        ([[0.0, 0.0]], [1.0], 1.4, None, ValueError, "coordinates"),
        ([[0.0, 0.0, 0.0]], [[1.0]], 1.4, None, ValueError, "radii"),
        ([[0.0, 0.0, 0.0]], [1.0, 2.0], 1.4, None, ValueError, "same number"),
        ([[math.nan, 0.0, 0.0]], [1.0], 1.4, None, ValueError, "finite"),
        ([[0.0, 0.0, 0.0]], [math.inf], 1.4, None, ValueError, "finite"),
        ([[0.0, 0.0, 0.0]], [0.0], 1.4, None, ValueError, "positive"),
        (
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [1.0, 2.0],
            1.4,
            None,
            ValueError,
            "coincident",
        ),
        ([[0.0, 0.0, 0.0]], [1.0], True, None, TypeError, "probe_size"),
        ([[0.0, 0.0, 0.0]], [1.0], -1.0, None, ValueError, "probe_size"),
        (
            [[0.0, 0.0, 0.0]],
            [1.0],
            1.4,
            np.empty((0, 3)),
            ValueError,
            "empty",
        ),
        ([[0.0, 0.0, 0.0]], [1.0], 1.4, [[2.0, 0.0, 0.0]], ValueError, "unit"),
        (
            [["not-a-number", 0.0, 0.0]],
            [1.0],
            1.4,
            None,
            TypeError,
            "coordinates",
        ),
    ],
)
def test_spatial_validation_matches_reference_contract(
    coordinates: object,
    radii: object,
    probe_size: object,
    sphere_points: object,
    exception: type[Exception],
    match: str,
) -> None:
    with pytest.raises(exception, match=match):
        protrsa.atom_sasa_spatial(
            coordinates,
            radii,
            probe_size=probe_size,  # type: ignore[arg-type]
            sphere_points=sphere_points,  # type: ignore[arg-type]
        )

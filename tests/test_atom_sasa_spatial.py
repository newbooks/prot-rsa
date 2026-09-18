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
    assert rows == [[1], [2, 0], [1], []]
    assert all(len(row) == len(set(row)) for row in rows)


def test_neighbors_are_ordered_by_largest_occlusion_cap() -> None:
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    expanded_radii = np.array([2.0, 2.0, 2.0])

    offsets, indices = protrsa._build_spatial_neighbors(
        coordinates, expanded_radii
    )

    assert indices[offsets[0] : offsets[1]].tolist() == [2, 1]


def test_equal_occlusion_scores_use_neighbor_index_tiebreaker() -> None:
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    expanded_radii = np.array([2.0, 2.0, 2.0])

    offsets, indices = protrsa._build_spatial_neighbors(
        coordinates, expanded_radii
    )

    assert indices[offsets[0] : offsets[1]].tolist() == [1, 2]


def test_enclosing_neighbor_precedes_partial_and_contained_neighbors() -> None:
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ]
    )
    expanded_radii = np.array([2.0, 4.0, 2.0, 0.5])

    offsets, indices = protrsa._build_spatial_neighbors(
        coordinates, expanded_radii
    )

    assert indices[offsets[0] : offsets[1]].tolist() == [1, 2, 3]


def test_tangent_expanded_spheres_are_neighbors() -> None:
    coordinates = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])

    offsets, indices = protrsa._build_spatial_neighbors(
        coordinates, np.array([2.0, 2.0])
    )

    np.testing.assert_array_equal(offsets, [0, 1, 2])
    np.testing.assert_array_equal(indices, [1, 0])


def test_buried_mask_marks_internal_tangency_only() -> None:
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ]
    )
    expanded_radii = np.array([1.0, 3.0, 1.0, 1.0])
    offsets, indices = protrsa._build_spatial_neighbors(
        coordinates, expanded_radii
    )

    buried = protrsa._build_buried_mask(
        coordinates, expanded_radii, offsets, indices
    )

    # Atom 0 is internally tangent to atom 1; atom 2 is only externally
    # tangent to atom 1, and atom 3 has no neighbors.
    np.testing.assert_array_equal(buried, [True, False, False, False])


def test_burial_mask_preserves_spatial_result_and_skips_buried_atom() -> None:
    coordinates = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [9.0, 0.0, 0.0]])
    radii = np.array([3.0, 1.0, 2.0])
    points = protrsa.generate_sphere_points(97)
    atom_coordinates, atom_radii, probe_radius, normalized_points = (
        protrsa._prepare_atom_sasa_inputs(
            coordinates, radii, probe_size=0.0, sphere_points=points
        )
    )
    expanded_radii = atom_radii + probe_radius
    offsets, indices = protrsa._build_spatial_neighbors(
        atom_coordinates, expanded_radii
    )
    buried = protrsa._build_buried_mask(
        atom_coordinates, expanded_radii, offsets, indices
    )
    masked = protrsa._atom_sasa_from_neighbors(
        atom_coordinates,
        expanded_radii,
        normalized_points,
        offsets,
        indices,
        buried,
    )
    unmasked = protrsa._atom_sasa_from_neighbors(
        atom_coordinates,
        expanded_radii,
        normalized_points,
        offsets,
        indices,
    )

    np.testing.assert_array_equal(masked, unmasked)
    assert masked[1] == 0.0


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


def test_vectorized_mask_preserves_dot_boundary_rounding() -> None:
    coordinates = [
        [0.0, 0.0, 0.0],
        [3.1675033383994773, 6.2549486055980381, 3.0807866885653379],
    ]
    radii = [1.0, 7.301623066215285]
    points = np.array([[1.0, 0.0, 0.0]])

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


def test_spatial_preserves_reference_for_boundary_sensitive_random_inputs() -> None:
    random = np.random.default_rng(123)
    points = protrsa.generate_sphere_points(37)

    for _ in range(128):
        coordinates = random.uniform(-5.0, 5.0, size=(8, 3))
        radii = random.uniform(0.5, 3.0, size=8)
        result = protrsa.atom_sasa_spatial(
            coordinates, radii, sphere_points=points
        )
        reference = protrsa.atom_sasa_reference(
            coordinates, radii, sphere_points=points
        )

        np.testing.assert_allclose(result, reference, rtol=1e-14, atol=1e-12)


def test_spatial_handles_enclosure_and_unequal_radii() -> None:
    coordinates = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [9.0, 0.0, 0.0]]
    radii = [3.0, 1.0, 2.0]

    result = protrsa.atom_sasa_spatial(coordinates, radii, probe_size=0.0)
    reference = protrsa.atom_sasa_reference(coordinates, radii, probe_size=0.0)

    np.testing.assert_array_equal(result, reference)
    assert result[1] == 0.0


def test_backend_auto_and_cpu_agree_with_numpy_fallback() -> None:
    coordinates = np.array([[0.0, 0.0, 0.0], [2.0, 0.3, 0.0], [8.0, 0.0, 0.0]])
    radii = np.array([1.5, 1.2, 1.0])
    points = protrsa.generate_sphere_points(73)

    automatic = protrsa.atom_sasa_spatial(
        coordinates, radii, sphere_points=points, backend="auto"
    )
    cpu = protrsa.atom_sasa_spatial(
        coordinates, radii, sphere_points=points, backend="cpu"
    )
    np.testing.assert_allclose(automatic, cpu, rtol=1e-14, atol=1e-12)


def test_spatial_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="backend must be 'auto' or 'cpu'"):
        protrsa.atom_sasa_spatial([[0.0, 0.0, 0.0]], [1.0], backend="invalid")


def test_parallel_workers_restore_numba_thread_count() -> None:
    numba = pytest.importorskip("numba")
    coordinates = np.array([[0.0, 0.0, 0.0], [2.0, 0.3, 0.0]])
    radii = np.array([1.5, 1.2])
    original = numba.get_num_threads()
    try:
        protrsa.atom_sasa_spatial(
            coordinates,
            radii,
            sphere_points=protrsa.generate_sphere_points(31),
            workers=1,
        )
        assert numba.get_num_threads() == original
    finally:
        numba.set_num_threads(original)


def test_parallel_workers_require_positive_integer() -> None:
    with pytest.raises(ValueError, match="workers must be a positive integer"):
        protrsa.atom_sasa_spatial(
            [[0.0, 0.0, 0.0]], [1.0], workers=0
        )


def test_calculate_atom_sasa_validates_workers_even_for_empty_input() -> None:
    with pytest.raises(TypeError, match="workers must be a positive integer"):
        protrsa.calculate_atom_sasa([], workers=True)
    with pytest.raises(ValueError, match="workers must be a positive integer"):
        protrsa.atom_sasa_spatial(
            np.empty((0, 3)), np.empty(0), workers=0
        )


def test_empty_input_rejects_workers_above_numba_limit() -> None:
    numba = pytest.importorskip("numba")
    with pytest.raises(ValueError, match="workers must not exceed Numba's thread limit"):
        protrsa.atom_sasa_spatial(
            np.empty((0, 3)),
            np.empty(0),
            workers=numba.config.NUMBA_NUM_THREADS + 1,
        )


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

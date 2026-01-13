"""
Tests for pdyna.io module.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from pdyna.io import process_lat, process_lat_reverse


class TestProcessLat:
    """Tests for the process_lat function."""

    def test_cubic_lattice(self, sample_lattice_matrix):
        """Test lattice processing for cubic cell."""
        result = process_lat(sample_lattice_matrix)

        # Should return shape (1, 6)
        assert result.shape == (1, 6)

        # For cubic: a=b=c=6, alpha=beta=gamma=90
        assert_array_almost_equal(result[0, :3], [6.0, 6.0, 6.0])
        assert_array_almost_equal(result[0, 3:], [90.0, 90.0, 90.0])

    def test_orthorhombic_lattice(self):
        """Test lattice processing for orthorhombic cell."""
        lattice = np.array([
            [5.0, 0.0, 0.0],
            [0.0, 6.0, 0.0],
            [0.0, 0.0, 7.0]
        ])
        result = process_lat(lattice)

        assert_array_almost_equal(result[0, :3], [5.0, 6.0, 7.0])
        assert_array_almost_equal(result[0, 3:], [90.0, 90.0, 90.0])


class TestProcessLatReverse:
    """Tests for the process_lat_reverse function."""

    def test_cubic_cellpar(self, sample_cellpar):
        """Test reverse lattice processing for cubic cell."""
        result = process_lat_reverse(sample_cellpar)

        # Should return 3x3 matrix
        assert result.shape == (3, 3)

        # For cubic cell, should be diagonal
        assert_array_almost_equal(np.diag(result), [6.0, 6.0, 6.0])

    def test_scalar_input(self):
        """Test with scalar input (cubic cell)."""
        result = process_lat_reverse(5.0)
        assert result.shape == (3, 3)
        assert_array_almost_equal(np.diag(result), [5.0, 5.0, 5.0])

    def test_single_value_list(self):
        """Test with single value list (cubic cell)."""
        result = process_lat_reverse([5.0])
        assert result.shape == (3, 3)
        assert_array_almost_equal(np.diag(result), [5.0, 5.0, 5.0])

    def test_three_values(self):
        """Test with three values (orthorhombic cell)."""
        result = process_lat_reverse([5.0, 6.0, 7.0])
        assert result.shape == (3, 3)
        assert_array_almost_equal(np.diag(result), [5.0, 6.0, 7.0])

    def test_triclinic_cellpar(self, sample_triclinic_cellpar):
        """Test reverse lattice processing for triclinic cell."""
        result = process_lat_reverse(sample_triclinic_cellpar)

        # Should return 3x3 matrix
        assert result.shape == (3, 3)

        # First vector should be along x-axis
        assert_array_almost_equal(result[0], [5.0, 0.0, 0.0])

    def test_roundtrip_cubic(self, sample_cellpar):
        """Test that process_lat(process_lat_reverse(x)) returns x."""
        lattice_matrix = process_lat_reverse(sample_cellpar)
        recovered = process_lat(lattice_matrix)

        assert_array_almost_equal(recovered[0], sample_cellpar)

    def test_roundtrip_triclinic(self, sample_triclinic_cellpar):
        """Test roundtrip for triclinic cell."""
        lattice_matrix = process_lat_reverse(sample_triclinic_cellpar)
        recovered = process_lat(lattice_matrix)

        assert_array_almost_equal(recovered[0], sample_triclinic_cellpar, decimal=5)


class TestChemicalFromFormula:
    """Tests for the chemical_from_formula function."""

    def test_known_formula(self):
        """Test formula lookup for known compounds."""
        from pdyna.io import chemical_from_formula
        from pymatgen.core import Structure, Lattice

        # Create a simple CsPbI3 structure
        lattice = Lattice.cubic(6.0)
        struct = Structure(
            lattice,
            ["Cs", "Pb", "I", "I", "I"],
            [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
        )

        result = chemical_from_formula(struct)
        assert result == r'CsPbI$_{3}$'

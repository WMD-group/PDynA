"""
Tests for pdyna.structural module utility functions.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal


class TestCoordinateConversions:
    """Tests for coordinate conversion functions."""

    def test_frac_to_cart_cubic(self):
        """Test fractional to Cartesian conversion for cubic cell."""
        from pdyna.structural import get_cart_from_frac

        frac_coords = np.array([[0.5, 0.5, 0.5]])
        lattice = np.array([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0]
        ])

        cart_coords = get_cart_from_frac(frac_coords, lattice)

        assert_array_almost_equal(cart_coords, [[5.0, 5.0, 5.0]])

    def test_cart_to_frac_cubic(self):
        """Test Cartesian to fractional conversion for cubic cell."""
        from pdyna.structural import get_frac_from_cart

        cart_coords = np.array([[5.0, 5.0, 5.0]])
        lattice = np.array([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0]
        ])

        frac_coords = get_frac_from_cart(cart_coords, lattice)

        assert_array_almost_equal(frac_coords, [[0.5, 0.5, 0.5]])

    def test_roundtrip_conversion(self):
        """Test that frac->cart->frac returns original coordinates."""
        from pdyna.structural import get_cart_from_frac, get_frac_from_cart

        original_frac = np.array([[0.25, 0.33, 0.75]])
        lattice = np.array([
            [6.0, 0.0, 0.0],
            [0.0, 6.0, 0.0],
            [0.0, 0.0, 6.0]
        ])

        cart = get_cart_from_frac(original_frac, lattice)
        recovered_frac = get_frac_from_cart(cart, lattice)

        assert_array_almost_equal(recovered_frac, original_frac)

    def test_frac_to_cart_triclinic(self):
        """Test conversion with non-orthogonal cell."""
        from pdyna.structural import get_cart_from_frac
        from pdyna.io import process_lat_reverse

        # Create a triclinic cell
        cellpar = np.array([5.0, 6.0, 7.0, 80.0, 85.0, 75.0])
        lattice = process_lat_reverse(cellpar)

        frac_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        cart_coords = get_cart_from_frac(frac_coords, lattice)

        # First atom at origin
        assert_array_almost_equal(cart_coords[0], [0.0, 0.0, 0.0])

        # Second atom should be at lattice vector a
        assert_array_almost_equal(cart_coords[1], lattice[0])

        # Third atom should be at lattice vector b
        assert_array_almost_equal(cart_coords[2], lattice[1])

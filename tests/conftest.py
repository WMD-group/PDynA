"""
Pytest configuration and fixtures for PDynA tests.
"""

import pytest
import numpy as np


@pytest.fixture
def sample_lattice_matrix():
    """Return a sample cubic lattice matrix."""
    return np.array([
        [6.0, 0.0, 0.0],
        [0.0, 6.0, 0.0],
        [0.0, 0.0, 6.0]
    ])


@pytest.fixture
def sample_cellpar():
    """Return sample cell parameters (a, b, c, alpha, beta, gamma)."""
    return np.array([6.0, 6.0, 6.0, 90.0, 90.0, 90.0])


@pytest.fixture
def sample_triclinic_cellpar():
    """Return sample triclinic cell parameters."""
    return np.array([5.0, 6.0, 7.0, 80.0, 85.0, 75.0])

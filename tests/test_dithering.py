import pytest
import numpy as np
from src.simple_dithering import _recursive_bayer_dithering

_BAYER_DITHERING_1 = np.array([[0., 8., 2., 10.],
                               [12.,  4., 14., 6.],
                               [3., 11., 1.,  9.],
                               [15.,  7., 13.,  5.]], dtype=np.float64)/16

_BAYER_DITHERING_0 = np.array([[0, 2], [3, 1]], dtype=np.float64)/4


def test_recursive_bayer_dithering():
    assert np.array_equal(
        _recursive_bayer_dithering(level=1),
        _BAYER_DITHERING_1
    )
    assert np.array_equal(
        _recursive_bayer_dithering(level=0),
        _BAYER_DITHERING_0
    )

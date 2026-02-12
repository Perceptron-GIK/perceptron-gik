"""Tests for the IMU mathlib module.

Tests cover:
- normalise: vector normalisation
- I: identity matrix creation
- F: state transition Jacobian
- G: gyroscope noise mapping
- skew: skew-symmetric matrix
- rotate: quaternion rotation matrix
- filter_signal: butterworth filter
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "imu", "v1"))
from mathlib import normalise, I, F, G, skew, rotate, H_helper, H, filter_signal


class TestNormalise:
    def test_unit_vector(self):
        v = np.array([3.0, 4.0, 0.0])
        result = normalise(v)
        np.testing.assert_almost_equal(np.linalg.norm(result), 1.0)

    def test_already_normalised(self):
        v = np.array([1.0, 0.0, 0.0])
        result = normalise(v)
        np.testing.assert_array_almost_equal(result, v)

    def test_column_vector(self):
        v = np.array([[0.0], [3.0], [4.0]])
        result = normalise(v)
        np.testing.assert_almost_equal(np.linalg.norm(result), 1.0)


class TestIdentity:
    def test_identity_3(self):
        np.testing.assert_array_equal(I(3), np.eye(3))

    def test_identity_4(self):
        np.testing.assert_array_equal(I(4), np.eye(4))


class TestStateTransition:
    def test_F_at_zero_gyro(self):
        """When gyroscope reads zero, F should be identity."""
        gt = np.array([[0.0, 0.0, 0.0]]).T
        result = F(gt, 0.01)
        np.testing.assert_array_almost_equal(result, np.eye(4))

    def test_F_shape(self):
        gt = np.array([[1.0, 2.0, 3.0]]).T
        result = F(gt, 0.01)
        assert result.shape == (4, 4)


class TestG:
    def test_G_shape(self):
        q = np.array([[1.0, 0.0, 0.0, 0.0]]).T
        result = G(q)
        assert result.shape == (4, 3)


class TestSkew:
    def test_skew_symmetric(self):
        x = np.array([[1.0, 2.0, 3.0]]).T
        S = skew(x)
        np.testing.assert_array_almost_equal(S, -S.T)

    def test_skew_zero(self):
        x = np.array([[0.0, 0.0, 0.0]]).T
        S = skew(x)
        np.testing.assert_array_almost_equal(S, np.zeros((3, 3)))


class TestRotate:
    def test_identity_quaternion(self):
        """Identity quaternion [1,0,0,0] should give identity rotation."""
        q = np.array([[1.0, 0.0, 0.0, 0.0]]).T
        R = rotate(q)
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_rotation_shape(self):
        q = normalise(np.array([[1.0, 1.0, 0.0, 0.0]]).T)
        R = rotate(q)
        assert R.shape == (3, 3)


class TestFilterSignal:
    def test_lowpass_preserves_dc(self):
        """A constant signal should pass through a lowpass filter unchanged."""
        data = [np.ones((100, 3))]
        filtered = filter_signal(data, dt=0.01, cof=10, btype='lowpass')
        np.testing.assert_array_almost_equal(filtered[0], data[0], decimal=2)

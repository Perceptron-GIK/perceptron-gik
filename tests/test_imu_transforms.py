"""Tests for IMU transforms module (Issue #8 fix).

Tests cover:
- Default rotation matrices are valid orthonormal matrices
- transform_finger_data applies correct rotation
- Identity rotation (index finger default ~0°) preserves input
- set_finger_rotation validates input
- Unknown finger names raise ValueError
"""

import numpy as np
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.imu.transforms import (
    transform_finger_data,
    set_finger_rotation,
    FINGER_ROTATIONS,
    _rot_z,
)


class TestRotZ:
    """Tests for the Z-axis rotation helper."""

    def test_zero_rotation_is_identity(self):
        R = _rot_z(0.0)
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_90_degree_rotation(self):
        R = _rot_z(90.0)
        v = np.array([1.0, 0.0, 0.0])
        result = R @ v
        np.testing.assert_array_almost_equal(result, [0.0, 1.0, 0.0])

    def test_rotation_is_orthonormal(self):
        for angle in [0, 30, 45, 60, 90, 180, -45]:
            R = _rot_z(angle)
            np.testing.assert_array_almost_equal(R @ R.T, np.eye(3))
            assert abs(np.linalg.det(R) - 1.0) < 1e-10


class TestFingerRotations:
    """Tests for the default finger rotation matrices."""

    def test_all_fingers_present(self):
        expected = {"thumb", "index", "middle", "ring", "pinky"}
        assert set(FINGER_ROTATIONS.keys()) == expected

    def test_all_matrices_are_3x3(self):
        for name, R in FINGER_ROTATIONS.items():
            assert R.shape == (3, 3), f"{name} rotation is {R.shape}"

    def test_all_matrices_are_orthonormal(self):
        for name, R in FINGER_ROTATIONS.items():
            np.testing.assert_array_almost_equal(
                R @ R.T, np.eye(3), err_msg=f"{name} is not orthonormal"
            )


class TestTransformFingerData:
    """Tests for transform_finger_data()."""

    def test_index_finger_near_identity(self):
        """Index finger has 0° offset, so transform should be near identity."""
        accel, gyro = transform_finger_data("index", 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        np.testing.assert_array_almost_equal(accel, [1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(gyro, [0.0, 1.0, 0.0])

    def test_z_component_preserved(self):
        """Z-axis rotation should preserve the Z component."""
        for finger in FINGER_ROTATIONS:
            accel, gyro = transform_finger_data(finger, 0.0, 0.0, 5.0, 0.0, 0.0, 3.0)
            assert abs(accel[2] - 5.0) < 1e-10
            assert abs(gyro[2] - 3.0) < 1e-10

    def test_magnitude_preserved(self):
        """Rotation should preserve vector magnitude."""
        for finger in FINGER_ROTATIONS:
            accel, gyro = transform_finger_data(finger, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
            np.testing.assert_almost_equal(
                np.linalg.norm(accel),
                np.linalg.norm([1.0, 2.0, 3.0]),
            )
            np.testing.assert_almost_equal(
                np.linalg.norm(gyro),
                np.linalg.norm([4.0, 5.0, 6.0]),
            )

    def test_unknown_finger_raises(self):
        with pytest.raises(ValueError, match="Unknown finger"):
            transform_finger_data("wrist", 0, 0, 0, 0, 0, 0)

    def test_case_insensitive(self):
        """Finger names should be case-insensitive."""
        accel1, _ = transform_finger_data("Thumb", 1, 0, 0, 0, 0, 0)
        accel2, _ = transform_finger_data("thumb", 1, 0, 0, 0, 0, 0)
        np.testing.assert_array_equal(accel1, accel2)


class TestSetFingerRotation:
    """Tests for set_finger_rotation()."""

    def test_override_rotation(self):
        """Setting a custom rotation should be used in subsequent transforms."""
        custom_R = np.eye(3)  # Identity
        set_finger_rotation("thumb", custom_R)
        accel, gyro = transform_finger_data("thumb", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        np.testing.assert_array_almost_equal(accel, [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(gyro, [4.0, 5.0, 6.0])
        # Restore default
        from src.imu.transforms import _DEFAULT_FINGER_ANGLES
        set_finger_rotation("thumb", _rot_z(_DEFAULT_FINGER_ANGLES["thumb"]))

    def test_invalid_finger_raises(self):
        with pytest.raises(ValueError, match="Unknown finger"):
            set_finger_rotation("elbow", np.eye(3))

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="3×3"):
            set_finger_rotation("thumb", np.eye(4))

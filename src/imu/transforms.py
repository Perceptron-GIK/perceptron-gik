"""Per-finger IMU local-to-global axis transformation.

Each finger IMU is mounted at a different orientation on the glove.
This module provides rotation matrices that map each sensor's local
(x, y, z) frame into a common *glove-global* reference frame so that
all finger readings share the same coordinate convention.

The rotation angles below are *initial calibration defaults* and
should be updated after measuring each glove's actual mounting
geometry.

Usage
-----
>>> from src.imu.transforms import transform_finger_data
>>> global_accel, global_gyro = transform_finger_data(
...     "thumb", ax, ay, az, gx, gy, gz
... )
"""

import numpy as np
from typing import Tuple, Dict

# ---------------------------------------------------------------------------
# Per-finger rotation matrices  (local → global)
# ---------------------------------------------------------------------------
# Each 3×3 matrix R satisfies:  v_global = R @ v_local
#
# Convention
# ----------
#   * The *base* IMU defines the global frame (identity rotation).
#   * Finger rotations are expressed as simple axis permutations /
#     sign flips that approximate the physical mounting angle.
#   * Update these matrices after measuring the actual mounting
#     orientation of each finger IMU on the glove.
# ---------------------------------------------------------------------------

def _rot_z(deg: float) -> np.ndarray:
    """Return the 3×3 rotation matrix for a rotation around Z by *deg* degrees."""
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [ c, -s, 0.0],
        [ s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


# Default per-finger rotation offsets (degrees around Z-axis)
# These approximate the splay angle of each finger relative to
# the hand's forward (index-finger) direction.
_DEFAULT_FINGER_ANGLES: Dict[str, float] = {
    "thumb":  -60.0,
    "index":    0.0,
    "middle":  10.0,
    "ring":    20.0,
    "pinky":   30.0,
}

# Pre-computed rotation matrices
FINGER_ROTATIONS: Dict[str, np.ndarray] = {
    name: _rot_z(angle) for name, angle in _DEFAULT_FINGER_ANGLES.items()
}


def set_finger_rotation(finger: str, rotation_matrix: np.ndarray) -> None:
    """Override the rotation matrix for a given finger.

    Args:
        finger: One of 'thumb', 'index', 'middle', 'ring', 'pinky'.
        rotation_matrix: A 3×3 orthonormal rotation matrix.

    Raises:
        ValueError: If *finger* is not recognised or the matrix shape is wrong.
    """
    finger = finger.lower()
    if finger not in FINGER_ROTATIONS:
        raise ValueError(f"Unknown finger '{finger}'. "
                         f"Expected one of {list(FINGER_ROTATIONS)}")
    rotation_matrix = np.asarray(rotation_matrix, dtype=float)
    if rotation_matrix.shape != (3, 3):
        raise ValueError(f"Rotation matrix must be 3×3, got {rotation_matrix.shape}")
    FINGER_ROTATIONS[finger] = rotation_matrix


def transform_finger_data(
    finger: str,
    ax: float, ay: float, az: float,
    gx: float, gy: float, gz: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform a single finger's IMU reading from local to global frame.

    Args:
        finger: One of 'thumb', 'index', 'middle', 'ring', 'pinky'.
        ax, ay, az: Accelerometer readings in the sensor's local frame.
        gx, gy, gz: Gyroscope readings in the sensor's local frame.

    Returns:
        A tuple (accel_global, gyro_global), each a length-3 numpy array.

    Raises:
        ValueError: If *finger* is not recognised.
    """
    finger = finger.lower()
    if finger not in FINGER_ROTATIONS:
        raise ValueError(f"Unknown finger '{finger}'. "
                         f"Expected one of {list(FINGER_ROTATIONS)}")
    R = FINGER_ROTATIONS[finger]
    accel_local = np.array([ax, ay, az])
    gyro_local  = np.array([gx, gy, gz])
    return R @ accel_local, R @ gyro_local

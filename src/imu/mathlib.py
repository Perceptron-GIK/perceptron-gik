import numpy as np
import scipy.signal

# Normalisation
def normalise(x):
    try:
        return x / np.linalg.norm(x)
    except:
        return x

# Identity matrix
def I(n):
    return np.eye(n)

# State transition Jacobian matrix
def F(gt, dt):
    g = gt.t[0]
    Omega = np.array([[0, -g[0], -g[1], -g[2]], [g[0], 0, g[2], -g[1]],
                      [g[1], -g[2], 0, g[0]], [g[2], g[1], -g[0], 0]])
    return I(4) + 0.5*dt*Omega

# Map gyroscope noise into quaternion perturbations
def G(q):
    q = q.T[0]
    return 0.5 * np.array([[-q[1], -q[2], -q[3]], [q[0], -q[3], q[2]],
                           [q[3], q[0], -q[1]], [-q[2], q[1], q[0]]])

# Take a 3D column vector and return its skew-symmetric matrix
def skew(x):
    x = x.T[0]
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

# Rotate quaternion matrix from world to body frame
def rotate(q):
    qv = q[1:4, :] # Vector component
    qc = q[0] # Scalar component
    return (qc**2 - qv.T @ qv) * I(3) - 2 * qc * skew(qv) + 2 * qv @ qv.T

# Helper function for computing the measurement Jacobian matrix
def H_helper(q, v):
    x, y, z = v.T[0][0], v.T[0][1], v.T[0][2]
    q0, q1, q2, q3 = q.T[0][0], q.T[0][1], q.T[0][2], q.T[0][3]
    h = np.array([
        [q0*x - q3*y + q2*z, q1*x + q2*y + q3*z, -q2*x + q1*y + q0*z, -q3*x - q0*y + q1*z],
        [q3*x + q0*y - q1*z, q2*x - q1*y - q0*z, q1*x + q2*y + q3*z, q0*x - q3*y + q2*z],
        [-q2*x + q1*y +q0*z, q3*x + q0*y - q1*z, -q0*x + q3*y - q2*z, q1*x + q2*y + q3*z]
    ])
    return 2*h

# Compute the measurement Jacobian matrix
def H(q, grv, magv):
    H1 = H_helper(q, grv)
    H2 = H_helper(q, magv)
    return np.vstack((-H1, H2))

# Apply butterworth filtering
def filter_signal(data, dt=0.01, cof=10, btype='lowpass', order=1):
    filtered = []
    num, den = scipy.signal.butter(order, cof, fs=1/dt, btype=btype)
    for d in data:
        d = scipy.signal.filtfilt(num, den, d, axis=0)
        filtered.append(d)
    return filtered
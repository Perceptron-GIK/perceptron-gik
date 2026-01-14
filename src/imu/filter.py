import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("imu_raw.csv")
ax_raw, ay_raw, az_raw = df["ax"].to_numpy(dtype=float), df["ay"].to_numpy(dtype=float), df["az"].to_numpy(dtype=float)
t = df.index

"""
Function that applies Kalman filtering to raw IMU data
- raw: A time series of raw IMU data
- Q: Scalar process noise variance
- R: Measurement noise variance
- x0: Initial state
- p0: Initial state variance
"""
def kalman(raw, Q=1e-4, R=1e-3, x0=None, p0=1.0):
    x = raw[0] if not x0 else float(x0) # Use first measurement as an initial guess of the state
    P = float(p0) # Predicted state variance
    filtered = np.zeros_like(raw) # Empty array to store the filtered state estimate

    for t, v in enumerate(raw):
        x_pred = x
        p_pred = P + Q # Add process noise variance to predicted state variance

        """
        Updating Kalman gain K and filtered state variance P:
        - If measurement noise R is large, K is small, i.e. the filter trusts the model more. P drops by only a small amount.
        - If measurement noise R is small, K is large, i.e. the filter trusts the raw measurement more. P drops significantly.
        """
        K = p_pred/(p_pred + R)
        x = x_pred + K*(v - x_pred) # Filtered state
        P = (1 - K)*p_pred # Update predicted state variance

        filtered[t] = x 
    
    return filtered

ax_filtered, ay_filtered, az_filtered = kalman(ax_raw), kalman(ay_raw), kalman(az_raw)

plt.plot(t, ax_raw, label="ax (raw)")
plt.plot(t, ax_filtered, label="ax (filtered)")

# plt.plot(t, ay_raw, label="ay (raw)")
# plt.plot(t, ay_filtered, label="ay (filtered)")

# plt.plot(t, az_raw, label="az (raw)")
# plt.plot(t, az_filtered, label="az (filtered)")

plt.legend()
plt.show()
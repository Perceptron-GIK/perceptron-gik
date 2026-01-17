import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

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

ax_k, ay_k, az_k = kalman(ax_raw), kalman(ay_raw), kalman(az_raw)
a_k = np.column_stack((t, ax_k, ay_k, az_k)) # Concatenate 3 axes into 1 array
df_k = pd.DataFrame(a_k, columns=["t", "ax_k", "ay_k", "az_k"])
df_k.to_csv("imu_kalman.csv", index=False) # Write filtered data to a CSV file

# Savitzky-Golay filtering: window_length must be odd and >= polyorder + 2
ax_sg = savgol_filter(ax_raw, window_length=11, polyorder=3)
ay_sg = savgol_filter(ay_raw, window_length=11, polyorder=3)
az_sg = savgol_filter(az_raw, window_length=11, polyorder=3)
a_sg = np.column_stack((t, ax_sg, ay_sg, az_sg)) # Concatenate 3 axes into 1 array
df_k = pd.DataFrame(a_k, columns=["t", "ax_sg", "ay_sg", "az_sg"])
df_k.to_csv("imu_savgol.csv", index=False) # Write filtered data to a CSV file

# plt.plot(t, ax_raw, label="ax (raw)")
# plt.plot(t, ax_k, label="ax (kalman)")
# plt.plot(t, ax_sg, label="ax (savgol)")

# plt.plot(t, ay_raw, label="ay (raw)")
# plt.plot(t, ay_k, label="ay (kalman)")
# plt.plot(t, ay_sg, label="ay (savgol)")

# plt.plot(t, az_raw, label="az (raw)")
# plt.plot(t, az_k, label="az (kalman)")
# plt.plot(t, az_sg, label="az (savgol)")

# plt.legend()
# plt.show()
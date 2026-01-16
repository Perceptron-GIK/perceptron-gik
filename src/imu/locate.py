import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

df = pd.read_csv("imu_savgol.csv")
t, ax_in, ay_in, az_in = df["t"].values, df["ax_sg"].values, df["ay_sg"].values, df["az_sg"].values
t *= 0.001 # Convert time to seconds

# Helper function to remove bias from IMU data
def remove_bias(data):
    data = np.asarray(data, dtype=float).copy()
    delta = np.diff(data, prepend=data[0])
    bias_acc = [] # Empty list to store all potential bias values

    for k in range(len(data)):
        if delta[k] <= 0.01:
            bias_acc.append(data[k])
    
    bias = sum(bias_acc) / len(bias_acc) # Estimate the bias as an average of all potential bias values
    data -= bias # Remove bias
    return data

# Helper function to extract peaks using a sliding window
def extract_peaks(data, th):
    data = np.asarray(data, dtype=float).copy()
    output = np.zeros_like(data)

    max_idx, _ = find_peaks(data, height=th, distance=5)
    min_idx, _ = find_peaks(-data, height=th, distance=5)
    output[max_idx] = data[max_idx]
    output[min_idx] = data[min_idx]

    return output

# Main function to process IMU data
def process(data, high_th):
    processed = extract_peaks(remove_bias(data), high_th)
    return processed

# Thresholds should be different for accelerometer, gyroscope and magnetometer
a_th = 0.5
# g_th = None
# m_th = None 
ax, ay, az = process(ax_in, a_th), process(ay_in, a_th), process(az_in, a_th)

dt = np.diff(t, prepend=t[0])
dt[0] = dt[1]

vx = np.zeros_like(ax)
vy = np.zeros_like(ay)
for k in range(1, len(t)):
    vx[k] = vx[k-1] + ax[k] * dt[k]
    vy[k] = vy[k-1] + ay[k] * dt[k]

x = np.zeros_like(vx)
y = np.zeros_like(vy)
for k in range(1, len(t)):
    x[k] = x[k-1] + vx[k] * dt[k]
    y[k] = y[k-1] + vy[k] * dt[k]

plt.plot(t, ax_in, label="ax_in")
plt.plot(t, ax, label="ax")
# plt.plot(t, ay, label="ay")
# plt.plot(t, vx, label="vx")
# plt.plot(t, vy, label="vy")
# plt.plot(t, x, label="x")
# plt.plot(t, y, label="y")
plt.legend()
plt.show()
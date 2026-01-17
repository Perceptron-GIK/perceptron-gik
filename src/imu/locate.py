import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

df = pd.read_csv("imu_savgol.csv")
t, ax, ay, az = df["t"].values, df["ax_sg"].values, df["ay_sg"].values, df["az_sg"].values
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

# Helper function to extract peaks from IMU data using a sliding window
def extract_peaks(data, th):
    data = np.asarray(data, dtype=float).copy()
    output = np.zeros_like(data)

    max_idx, _ = find_peaks(data, height=th, distance=5)
    min_idx, _ = find_peaks(-data, height=th, distance=5)
    output[max_idx] = data[max_idx]
    output[min_idx] = data[min_idx]

    return output

# Helper function to zero out velocity during periods of constant acceleration
def zero_vel(acc, vel):
    vel = np.asarray(vel, dtype=float).copy()
    for i in range(2, len(acc)-2):
        if not acc[i-2:i+2].any():
            vel[i-2:i+2] = 0
    return vel

# Main function to process a 1D series of IMU data
def process(data, t, th, label):
    dt = np.diff(t, prepend=t[0])
    dt[0] = dt[1]

    acc = extract_peaks(remove_bias(data), th) # Process acceleration data from IMU
    vel = np.zeros_like(acc)
    for k in range(1, len(t)):
        vel[k] = vel[k-1] + acc[k]*dt[k]
    vel = zero_vel(acc, vel)
    pos = np.zeros_like(vel)
    for k in range(1, len(t)):
        pos[k] = pos[k-1] + vel[k]*dt[k]

    # plt.plot(t, data, label="input")
    # plt.plot(t, acc, label="acc "+label)
    # plt.plot(t, vel, label="vel "+label)
    plt.plot(t, pos, label="pos "+label)
    plt.legend()
    plt.show()

# Thresholds should be different for accelerometer, gyroscope and magnetometer
a_th = 0.5
# g_th = None
# m_th = None 

process(ax, t, a_th, label="x")
# process(ay, t, a_th, label="y")
# process(az, t, a_th, label="z")
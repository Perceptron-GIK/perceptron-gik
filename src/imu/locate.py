import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Helper function to extract event maxima using a sliding window
def extract_maxima(data, high_th, low_th):
    data = np.asarray(data, dtype=float).copy()
    output = np.zeros_like(data)
    in_event = False
    start_idx, max_idx, max_val = None, None, None
    
    for i in range(len(data)):
        if not in_event:
            if abs(data[i]) > high_th: # Check if value exceeds threshold
                in_event = True
                start_idx = i
                max_idx = i
                max_val = data[i]
        else:
            if abs(data[i]) > abs(max_val): # Update new maximum
                max_idx = i
                max_val = data[i]
            if abs(data[i]) < low_th: # If value falls below threshold
                output[max_idx] = max_val # Store maximum point
                in_event = False # Reset
                max_idx, max_val = None, None
    
    if in_event: # If event runs until the last data point
        output[max_idx] = max_val
    
    return output

# Main function to process IMU data
def process(data, high_th, low_th):
    processed = extract_maxima(remove_bias(data), high_th, low_th)
    return processed

# Thresholds should be different for accelerometer, gyroscope and magnetometer
high_a, low_a = 0.5, 0.1
# high_g, low_g = None, None
# high_m, low_m = None, None
ax, ay, az = process(ax_in, high_a, low_a), process(ay_in, high_a, low_a), process(az_in, high_a, low_a)
print(ax)

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("imu_savgol.csv")
t, ax_in, ay_in, az_in = df["t"].values, df["ax_sg"].values, df["ay_sg"].values, df["az_sg"].values
t *= 0.001 # Convert time to seconds

# Helper function to remove bias from the IMU data
def remove_bias(data):
    data = np.asarray(data, dtype=float).copy()
    delta = np.diff(data, prepend=data[0])
    bias_acc = [] # Empty list to store all potential bias values

    for k in range(len(data)):
        if delta[k] <= 0.01:
            bias_acc.append(data[k])
    
    bias = sum(bias_acc) / len(bias_acc) # Estimate the bias as an average of all potential bias values
    data -= bias
    return data

ax, ay, az = remove_bias(ax_in), remove_bias(ay_in), remove_bias(az_in)

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

# plt.plot(t, ax, label="ax")
# plt.plot(t, ay, label="ay")
# plt.plot(t, vx, label="vx")
# plt.plot(t, vy, label="vy")
plt.plot(t, x, label="x")
plt.plot(t, y, label="y")
plt.legend()
plt.show()
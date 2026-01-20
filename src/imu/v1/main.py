import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mathlib import *

class IMUTracker:
    def __init__(self, sr, use_mag=True):
        self.sr = sr # Sampling rate in Hz
        self.dt = 1/sr
        self.use_mag = use_mag # Boolean flag indicating whether magnetometer data is used
    
    def initialise(self, data, noise_coefficient={'g':100, 'a': 100, 'm': 10}):
        '''
        Initialisation for downstream EKF algorithm

        @param data: (, 9) ndarray
        @param noise_coefficient: sensor noise = variance magnitude*noise coefficient

        Return: a list of initialisation values used by the EKF algorithm
        (grv, grm, magv, gyro_noise, gyro_bias, acc_noise, mag_noise)
        '''

        a = data[:, 1:4]
        g = data[:, 4:7]

        if data.shape[1] >= 10:
            m = data[:, 7:10]
        else:
            m = None

        grv = -a.mean(axis=0) # Gravity vector
        grv = grv[:, np.newaxis]
        grm = np.linalg.norm(grv) # Gravity magnitude

        if self.use_mag and m is not None:
            magv = m.mean(axis=0) # Magnetic field vector
            magv = normalise(magv)[:, np.newaxis]
        else:
            magv = np.array([[1.0, 0.0, 0.0]]).T # Dummy unit vector

        # Compute noise covariance
        avar = a.var(axis=0)
        gvar = g.var(axis=0)

        # Compute sensor noise
        gyro_noise = np.linalg.norm(gvar)*noise_coefficient['g']
        gyro_bias = g.mean(axis=0)
        acc_noise = np.linalg.norm(avar)*noise_coefficient['a']

        if self.use_mag and m is not None:
            mvar = m.var(axis=0)
            mag_noise = np.linalg.norm(mvar)*noise_coefficient['m']
        else:
            mag_noise = 0.0

        return (grv, grm, magv, gyro_noise, gyro_bias, acc_noise, mag_noise)
    
    def track_attitude(self, data, init_tuple):
        '''
        Remove the effect of gravity from acceleration data
        Transform acceleration data into world frame
        Track device orientation relative to the world frame

        @param data: (, 9) ndarray
        @param init_tuple: Initialisation values for the EKF algorithm
        (grv, grm, magv, gyro_noise, gyro_bias, acc_noise, mag_noise)

        Return: (a_world, ori_x, ori_y, ori_z)
        '''

        grv, grm, magv, gyro_noise, gyro_bias, acc_noise, mag_noise = init_tuple
        a = data[:, 1:4]
        g = data[:, 4:7] - gyro_bias

        if self.use_mag and data.shape[1] >= 10:
            m = data[:, 7:10]
        else:
            m = None

        nSamples = np.shape(data)[0]

        # Empty lists to store acceleration and orientation data
        a_world, ori_x, ori_y, ori_z = [], [], [], []

        P = 1e-10*I(4) # State covariance matrix
        q = np.array([[1, 0, 0, 0]]).T # Initial quaternion state, assume alignment with world frame
        ori_ini = I(3) # Initial orientation, assume alignment with world frame
        
        # Extended Kalman Filter (EKF)
        t = 0
        while t < nSamples:
            at = a[t, np.newaxis].T
            gt = g[t, np.newaxis].T

            if self.use_mag and m is not None:
                mt = normalise(m[t, np.newaxis].T)

            Ft = F(gt, self.dt) # State transition Jacobian matrix
            Gt = G(q) # Map gyroscope noise into quaternion perturbations
            Q = (gyro_noise*self.dt)**2 * Gt @ Gt.T # Process noise covariance
            q = normalise(Ft @ q) # Update quaternion state
            P = Ft @ P @ Ft.T + Q # Update state covariance

            pred_a = normalise(-rotate(q) @ grv) # Predicted acceleration

            if self.use_mag and m is not None:
                pred_m = normalise(-rotate(q) @ magv) # Predicted magnetic field

                # Difference between actual and predicted vectors
                residual = np.vstack((normalise(at), mt)) - np.vstack((pred_a, pred_m))

                Ra = [(acc_noise/np.linalg.norm(at))**2 + (1 - grm/np.linalg.norm(at))**2]*3 # Accelerometer noise
                Rm = [mag_noise**2]*3 # Magnetometer noise
                R = np.diag(Ra + Rm)

                Ht = H(q, grv, magv) # Measurement Jacobian matrix
            else:
                residual = normalise(at) - pred_a
                Ra = [(acc_noise/np.linalg.norm(at))**2 + (1 - grm/np.linalg.norm(at))**2]*3
                R = np.diag(Ra)
                H_full = H(q, grv, magv)
                Ht = H_full[0:3, :] # Extract accelerometer component

            S = Ht @ P @ Ht.T + R # Residual covariance
            K = P @ Ht.T @ np.linalg.inv(S) # Kalman gain matrix

            q = q + K @ residual # Update quaternion state
            P = P - K @ Ht @ P # Update state covariance

            q = normalise(q)
            P = 0.5*(P + P.T) # Ensure P is symmetrical

            conj = -I(4)
            conj[0, 0]  = 1 # Quaternion conjugation matrix
            a_world_t = rotate(conj @ q) @ at + grv
            a_world.append(a_world_t.T[0])

            ori_t = rotate(conj @ q) @ ori_ini
            ori_x.append(ori_t.T[0, :])
            ori_y.append(ori_t.T[1, :])
            ori_z.append(ori_t.T[2, :])

            t += 1
        
        a_world = np.array(a_world)
        ori_x = np.array(ori_x)
        ori_y = np.array(ori_y)
        ori_z = np.array(ori_z)
        return (a_world, ori_x, ori_y, ori_z) # Orientation is returned for debugging or visualisation purposes only
    
    def remove_acc_drift(self, a_world, threshold=0.2, filter=False, cof=(0.01, 15)):
        '''
        Remove drift in acceleration data, assuming that the 
        device is stationary at the start and end of measurement.
        Optionally, pass the final acceleration data through a bandpass filter.
        
        @param a_world: Acceleration data from Kalman filter output
        @param threshold: Threshold to detect the start and end of motion
        @param cof: Bandpass filter cutoff frequencies

        Return: Corrected and optionally filtered acceleration data
        '''

        nSamples = np.shape(a_world)[0]
        t_start = 0
        for t in range(nSamples):
            at = a_world[t]
            if np.linalg.norm(at) > threshold:
                t_start = t
                break
        
        t_end = 0
        for t in range(nSamples-1, -1, -1):
            at = a_world[t]
            if np.linalg.norm(at-a_world[-1]) > threshold:
                t_end = t
                break
        
        drift = a_world[t_end:].mean(axis=0)
        drift_rate = drift/(t_end - t_start) # Assuming drift accumulates linearly

        # Remove drift during period of motion
        for i in range(t_end - t_start):
            a_world[t_start+i] -= (i+1)*drift_rate

        # Remove drift after motion
        for i in range(nSamples - t_end):
            a_world[t_end+1] -= drift

        # Optional bandpass filtering
        if filter:
            filtered_a_world = filter_signal([a_world], dt=self.dt, cof=cof, btype='bandpass')[0]
            return filtered_a_world
        else:
            return a_world
        
    def zupt(self, a_world, threshold):
        '''
        Apply Zero Velocity Update (ZUPT) algorithm to obtain velocities from acceleration data
        
        @param: a_world: Acceleration data with drift removed
        @param: threshold: Threshold for motion detection

        Return: Velocity data
        '''

        nSamples = np.shape(a_world)[0]
        velocities = []
        prev_t = -1 # Previous stationary time
        stationary = False # Flag indicating whether the device is stationary

        vt = np.zeros((3, 1))
        t = 0
        while t < nSamples:
            at = a_world[t, np.newaxis].T

            if np.linalg.norm(at) < threshold:
                if not stationary: # End of motion
                    pred_v = vt + at*self.dt 
                    drift_rate = pred_v / (t- prev_t)
                    for i in range(t - prev_t - 1): # Remove drift from every period of motion
                        velocities[prev_t + 1 + i] -= (i+1)*drift_rate.T[0]

                vt = np.zeros((3, 1)) # Set velocity to zero
                prev_t = t
                stationary = True
            else:
                vt = vt + at*self.dt 
                stationary = False

            velocities.append(vt.T[0])
            t += 1
        
        velocities = np.array(velocities)
        return velocities
    
    def track_position(self, a_world, velocities):
        '''
        Obtain position from acceleration and velocity data

        @param a_world: Acceleration data with drift removed
        @param velocities: Velocity data from ZUPT output

        Return: 3D coordinates in world frame
        '''

        nSamples = np.shape(a_world)[0]
        positions = []
        
        pt = np.array([[0, 0, 0]]).T # Set the starting position as the origin
        t = 0
        while t < nSamples:
            at = a_world[t, np.newaxis].T
            vt = velocities[t, np.newaxis].T
            pt = pt + vt*self.dt + 0.5*at*self.dt**2 # Kinematic constant acceleration equation
            positions.append(pt.T[0])
            t += 1
        
        positions = np.array(positions)
        return positions

# Read raw IMU data from CSV file
# TODO: Check if removing first and last few samples help with performance after introducing starting and ending gesture
def read_data(filepath):
    data = pd.read_csv(filepath, skiprows=0, skipfooter=0, engine='python').to_numpy()
    return data

'''
Tunable Parameters:
| Parameter | Function |
|-----------|----------|
| noise_coefficient | initialise |
| threshold | remove_acc_drift, zupt |
| filter | remove_acc_drift |
| cof | remove_acc_drift |
'''

# Main function
def run(use_mag=True):
    df = pd.read_csv("../data/Left_1.csv")
    t = pd.to_datetime(df["time_stamp"], format="%H:%M:%S.%f").iloc[:].diff().dt.total_seconds()*1000 # Time in ms from start of data collection
    t.iloc[0] = 0
    sr = 1/(t.mean()*0.001) # Sampling rate in Hz
    data = pd.concat([t, df.iloc[:, 2:8]], axis=1).to_numpy()
    
    tracker = IMUTracker(sr=sr, use_mag=use_mag)
    init_tuple = tracker.initialise(data, noise_coefficient={'g': 10, 'a': 35, 'm': 10})

    a_world, *_ = tracker.track_attitude(data, init_tuple)
    a_world_processed = tracker.remove_acc_drift(a_world, threshold=0.2, filter=True, cof=(0.1, 5))
    v = tracker.zupt(a_world_processed, threshold=0.2)
    p = tracker.track_position(a_world_processed, v)

    plt.plot(np.arange(0, t.shape[0], 1), data[:, 1], label="acc x (raw)")
    plt.plot(np.arange(0, a_world_processed.shape[0], 1), a_world_processed[:, 0], label="acc x (processed)")
    plt.plot(np.arange(0, v.shape[0], 1), v[:, 0], label='vel x')
    plt.plot(np.arange(0, p.shape[0], 1), p[:, 0], label="pos x")
    # plt.plot(np.arange(0, p.shape[0], 1), p[:, 1], label="pos y")
    # plt.plot(np.arange(0, p.shape[0], 1), p[:, 2], label="pos z")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run(use_mag=False)
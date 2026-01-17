import numpy as np
from mathlib import *

class IMUTracker:
    def __init__(self, sampling):
        super().__init__() # TODO: Check if this is necessary
        self.sampling = sampling # Sampling rate of the IMU in Hz, TODO: Calculate sampling rate from IMU data
        self.dt = 1/sampling
    
    def initialise(self, data, noise_coefficient={'g':100, 'a': 100, 'm': 10}):
        '''
        Initialisation for downstream EKF algorithm

        @param data: (, 9) ndarray
        @param noise_coefficient: sensor noise = variance magnitude*noise coefficient

        Return: a list of initialisation values used by the EKF algorithm
        (grv, grm, magv, gyro_noise, gyro_bias, acc_noise, mag_noise)
        '''

        a = data[:, 0:3]
        g = data[:, 3:6]
        m = data[:, 6:9]

        grv = -a.mean(axis=0) # Gravity vector
        grv = grv[:, np.newaxis]
        grm = np.linalg.norm(grv) # Gravity magnitude

        magv = m.mean(axis=0) # Magnetic field vector
        magv = normalise(magv)[:, np.newaxis]

        # Compute noise covariance
        avar = a.var(axis=0)
        gvar = g.var(axis=0)
        mvar = m.var(axis=0)

        # Compute sensor noise
        gyro_noise = np.linalg.norm(gvar)*noise_coefficient['g']
        gyro_bias = g.mean(axis=0)
        acc_noise = np.linalg.norm(avar)*noise_coefficient['a']
        mag_noise = np.linalg.norm(mvar)*noise_coefficient['m']
        
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
        a = data[:, 0:3]
        g = data[:, 3:6] - gyro_bias
        m = data[:, 6:9]
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
            mt = normalise(m[t, np.newaxis].T)

            Ft = F(gt, self.dt) # State transition Jacobian matrix
            Gt = G(q) # Map gyroscope noise into quaternion perturbations
            Q = (gyro_noise*self.dt)**2 * Gt @ Gt.T # Process noise covariance
            q = normalise(Ft @ q) # Update quaternion state
            P = Ft @ P @ Ft.T + Q # Update state covariance

            pred_a = normalise(-rotate(q) @ grv) # Predicted acceleration
            pred_m = normalise(-rotate(q) @ magv) # Predicted magnetic field

            # Difference between actual and predicted vectors
            residual = np.vstack((normalise(at), mt)) - np.vstack((pa, pm))

            Ra = [(acc_noise/np.linalg.norm(at))**2 + (1 - grm/np.linalg.norm(at))**2]*3 # Accelerometer noise
            Rm = [mag_noise**2]*3 # Magnetometer noise
            R = np.diag(Ra + Rm)

            Ht = H(q, grv, magv) # Measurement Jacobian matrix
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
        return (a_world, ori_x, ori_y, ori_z)
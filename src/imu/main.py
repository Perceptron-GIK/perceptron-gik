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
    
"""
GIK Preprocessing Pipeline for Real-Time Inference

1. IMU signal filtering (using src/imu/main.py)
2. Data alignment between left and right IMU sensors/FSRs (using src/pipeline/align.py)
3. Dataset creation and export for inference

"""

import numpy as np
import pandas as pd
import os
import sys
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Optional, Tuple, List, Dict, Any, Union

# Custom imports
from src.imu.main import IMUTracker
from src.pipeline.align import AlignData

IMU_SAMPLING_RATE = 100.0
IMU_COLS = [0, 6, 13, 20, 27, 34]
IMU_IDX_TO_PART = {
    0: "base",
    6: "thumb",
    13: "index",
    20: "middle",
    27: "ring",
    34: "pinky"
}

def filter_imu_data(data: np.ndarray) -> np.ndarray:
    '''
    Apply IMU filtering to a single window of IMU data from one hand
    Return the processed array
    '''
    timestamps = data[:, -1]
    time_rel = timestamps - timestamps[0]

    tracker = IMUTracker(sr=IMU_SAMPLING_RATE, use_mag=False)
    filtered_data = data.copy()

    for imu_col in IMU_COLS:
        imu_data = np.column_stack([time_rel, data[:, imu_col:imu_col+6]])

        try:
            init_tuple = tracker.initialise(imu_data)
            if imu_col == 0: # Use the base IMU as a reference for the keyboard frame
                R0_ref, a, *_ = tracker.track_attitude(imu_data, init_tuple)
            else:
                _, a, *_ = tracker.track_attitude(imu_data, init_tuple, R0_ref=R0_ref)

            a_p = tracker.remove_acc_drift(a, threshold=0.2, filter=True, cof=(0.1, 5))
            vel = tracker.zupt(a_p, threshold=0.2)
            pos = tracker.track_position(a, vel)

            filtered_data[:, imu_col] = np.nan_to_num(a[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
            filtered_data[:, imu_col+1] = np.nan_to_num(a[:, 1], nan=0.0, posinf=0.0, neginf=0.0)
            filtered_data[:, imu_col+2] = np.nan_to_num(a[:, 2], nan=0.0, posinf=0.0, neginf=0.0)
            filtered_data[:, imu_col+3] = np.nan_to_num(pos[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
            filtered_data[:, imu_col+4]= np.nan_to_num(pos[:, 1], nan=0.0, posinf=0.0, neginf=0.0)
            filtered_data[:, imu_col+5] = np.nan_to_num(pos[:, 2], nan=0.0, posinf=0.0, neginf=0.0)    

        except Exception as e:
            print(f"Warning: Filtering failed for {IMU_IDX_TO_PART[imu_col]} IMU")
            filtered_data[:, imu_col+3] = np.zeros((data.shape[0], 1))
            filtered_data[:, imu_col+4] = np.zeros((data.shape[0], 1))
            filtered_data[:, imu_col+5] = np.zeros((data.shape[0], 1))

    return filtered_data

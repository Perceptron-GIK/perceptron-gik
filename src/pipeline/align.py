"""
GIK Data Alignment for Real-Time Inference

1. Data alignment between left and right IMU sensors/FSRs
2. Outputs combined data from both hands

"""

import numpy as np
from typing import Optional, Tuple, Dict, Any

class RealTimeData:
    def __init__(self, data: np.ndarray):
        self.data = np.array(data, dtype=np.float32)
    
    @property
    def sorted_data(self):
        idx = np.argsort(self.data[:, -1])
        return self.data[idx]
    
class AlignData:
    '''
    Aligns data from left and right IMU sensors/FSRs
    '''
    def __init__(
        self,
        left_data: Optional[np.ndarray] = None,
        right_data: Optional[np.ndarray] = None
    ):
        if left_data is None and right_data is None:
            raise ValueError("No data provided.")
        self.left = RealTimeData(left_data) if left_data is not None else None
        self.right = RealTimeData(right_data) if right_data is not None else None

    @property
    def has_left(self) -> bool:
        return self.left is not None
    
    @property
    def has_right(self) -> bool:
        return self.right is not None
    
    @staticmethod
    def _pad_to_length(data: np.ndarray, target_len: int) -> np.ndarray:
        """
        Zero-pad or truncate sequence to target_len 
        Output shape: (target_len, features)
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        n = len(data)
        if n == 0:
            return np.zeros((target_len, data.shape[1]), dtype=np.float64)
        if n >= target_len:
            return data[:target_len].astype(np.float64)
        out = np.zeros((target_len, data.shape[1]), dtype=np.float64)
        out[:n] = data
        return out
    
    @staticmethod
    def _combine_hands(
        right: Optional[np.ndarray], 
        left: Optional[np.ndarray], 
        max_len: int
    ) -> Optional[np.ndarray]:
        has_r = right is not None and len(right) > 0
        has_l = left is not None and len(left) > 0
        if not has_r and not has_l:
            return None
        if has_r and not has_l:
            return AlignData._pad_to_length(right, max_len)
        if has_l and not has_r:
            return AlignData._pad_to_length(left, max_len)
        return np.concatenate([
            AlignData._pad_to_length(right, max_len),
            AlignData._pad_to_length(left, max_len),
        ], axis=1)
    
    def align(
        self,
        max_seq_length: int=100,
        filter_fn: Optional[callable] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        '''
        Returns (samples, metadata) for a single window of data
        '''
        if self.has_right:
            self.right.data = self.right.sorted_data
            if filter_fn is not None:
                self.right.data = filter_fn(self.right.data)

        if self.has_left:
            self.left.data = self.left.sorted_data
            if filter_fn is not None:
                self.left.data = filter_fn(self.left.data)

        left_win, right_win = None, None

        if self.has_left and self.has_right:
            left_start, left_end = self.left.data[0, -1], self.left.data[-1, -1]
            right_start, right_end = self.right.data[0, -1], self.right.data[-1, -1]

            overlap_start = max(left_start, right_start)
            overlap_end = min(left_end, right_end)

            left_mask = (self.left.data[:, -1] >= overlap_start) & (self.left.data[:, -1] <= overlap_end)
            right_mask = (self.right.data[:, -1] >= overlap_start) & (self.right.data[:, -1] <= overlap_end)

            left_arr = self.left.data[left_mask][:, :-1]
            left_win = left_arr if len(left_arr) > 0 else np.zeros((1, self.left.data.shape[1] - 1))

            right_arr = self.right.data[right_mask][:, :-1]
            right_win = right_arr if len(right_arr) > 0 else np.zeros((1, self.right.data.shape[1] - 1))

        combined = self._combine_hands(
            left=left_win,
            right=right_win,
            max_len=max_seq_length
        )

        metadata = {
            'num_hands': (1 if self.has_right else 0) + (1 if self.has_left else 0),
            'has_left': self.has_left,
            'has_right': self.has_right,
            'feat_dim': combined.shape[1],
            'max_seq_length': max_seq_length
        }

        return combined, metadata
    
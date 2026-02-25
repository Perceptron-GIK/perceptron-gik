from collections import deque
import numpy as np
import torch

class SlidingWindow:
    def __init__(self, window_size: int, feature_dim: int):
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.buffer = deque(maxlen=window_size)

    def add(self, sample: np.ndarray):
        self.buffer.append(sample) # (feature_dim, )

    def is_ready(self) -> bool:
        return len(self.buffer) == self.window_size

    def get_tensor(self) -> torch.Tensor:
        arr = np.array(self.buffer, dtype=np.float32)
        tensor = torch.from_numpy(arr)
        return tensor.unsqueeze(0) # (1, window_size, feature_dim)

    def reset(self):
        self.buffer.clear()
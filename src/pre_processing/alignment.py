"""
GIK Data Alignment: align IMU sensor data with keyboard events for labeled training samples.
Outputs samples and labels as characters; mapping (char -> index or vector) is done in the Dataset.

Usage:
    from src.pre_processing.alignment import Preprocessing
    from src.Constants.char_to_key import CHAR_TO_INDEX

    preprocessor = Preprocessing(data_dir="data/", keyboard_file="K.csv", left_file="L.csv", right_file="R.csv")
    samples, labels, prev_labels, metadata = preprocessor.align()
    # labels / prev_labels are lists of str (characters)
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any
from src.Constants.char_to_key import CHAR_TO_INDEX, SPECIAL_KEY_MAP

NON_FEATURE_COLS = {'sample_id', 'time_stamp'}


class TrainingData:
    """Load and manage CSV data; expose feature columns and time-sorted frame."""

    def __init__(self, filename: str, data_dir: str):
        self._data_dir = data_dir
        self.file_names = [filename]
        self.df = pd.read_csv(self._path(filename))

    def _path(self, filename: str) -> str:
        p = os.path.join(self._data_dir, filename)
        assert os.path.isfile(p), f"File not found: {p}"
        return p

    @property
    def feature_columns(self) -> List[str]:
        return [c for c in self.df.columns if c not in NON_FEATURE_COLS]

    @property
    def sorted_df(self) -> pd.DataFrame:
        self.df = self.df.sort_values('time_stamp').reset_index(drop=True)
        return self.df

    def add_data(self, filename: str):
        self.df = pd.concat([self.df, pd.read_csv(self._path(filename))], axis=0, ignore_index=True)
        self.file_names.append(filename)


class Preprocessing:
    """Align IMU and keyboard data into (samples, labels, metadata)."""

    @staticmethod
    def _pad_to_length(data: np.ndarray, target_len: int) -> np.ndarray:
        """Zero-pad or truncate sequence to target_len. Shape (target_len, features)."""
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
        right: Optional[np.ndarray], left: Optional[np.ndarray], max_len: int
    ) -> Optional[np.ndarray]:
        has_r = right is not None and len(right) > 0
        has_l = left is not None and len(left) > 0
        if not has_r and not has_l:
            return None
        if has_r and not has_l:
            return Preprocessing._pad_to_length(right, max_len)
        if has_l and not has_r:
            return Preprocessing._pad_to_length(left, max_len)
        t = min(max(len(right), len(left)), max_len)
        return np.concatenate([
            Preprocessing._pad_to_length(right, max_len),
            Preprocessing._pad_to_length(left, max_len),
        ], axis=1)

    @staticmethod
    def _char_from_key(name) -> Optional[str]:
        """Convert keyboard event name to char.
            Basically only useful for the special charecters like space, and backspace"""
        if name is None or (isinstance(name, float) and pd.isna(name)):
            return None
        if not isinstance(name, str):
            name = str(name)
        k = name.lower()
        return SPECIAL_KEY_MAP[k] if k in SPECIAL_KEY_MAP else k

    def __init__(
        self,
        data_dir: str,
        keyboard_file: str,
        left_file: Optional[str] = None,
        right_file: Optional[str] = None,
    ):
        if right_file is None and left_file is None:
            raise ValueError("At least one of left_file or right_file must be provided")
        self.data_dir = data_dir
        self.right = TrainingData(right_file, data_dir) if right_file else None
        self.left = TrainingData(left_file, data_dir) if left_file else None
        self.keyboard = TrainingData(keyboard_file, data_dir)

    @property
    def has_right(self) -> bool:
        return self.right is not None

    @property
    def has_left(self) -> bool:
        return self.left is not None

    def align(
        self,
        max_seq_length: int = 100,
        filter_func: Optional[callable] = None,
    ) -> Tuple[List[np.ndarray], List[str], List[str], Dict[str, Any]]:
        """Returns (samples, labels, prev_labels, metadata). labels and prev_labels are characters (str). prev_labels use '' for no previous."""
        samples, labels, prev_labels = [], [], []
        key_events = (
            self.keyboard.df[self.keyboard.df['event_type'] == 'down']
            .sort_values('time')
            .reset_index(drop=True)
        )

        right_cols, left_cols = [], []
        if self.has_right:
            self.right.df = self.right.sorted_df
            if filter_func is not None:
                self.right.df = filter_func(self.right.df)
            right_cols = self.right.feature_columns
        if self.has_left:
            self.left.df = self.left.sorted_df
            if filter_func is not None:
                self.left.df = filter_func(self.left.df)
            left_cols = self.left.feature_columns

        skipped_chars = {}
        last_char = None
        for i in range(len(key_events) - 1):
            cur_t, next_t = key_events.iloc[i]['time'], key_events.iloc[i + 1]['time']
            next_char = self._char_from_key(key_events.iloc[i + 1]['name'])
            if next_char is None or next_char not in CHAR_TO_INDEX:
                key = '<nan>' if next_char is None else next_char
                skipped_chars[key] = skipped_chars.get(key, 0) + 1
                continue

            prev_label = last_char if last_char is not None else ''

            right_win = None
            if self.has_right:
                mask = (self.right.df['time_stamp'] >= cur_t) & (self.right.df['time_stamp'] < next_t)
                arr = self.right.df.loc[mask, right_cols].values
                right_win = arr if len(arr) > 0 else np.zeros((1, len(right_cols)))

            left_win = None
            if self.has_left:
                mask = (self.left.df['time_stamp'] >= cur_t) & (self.left.df['time_stamp'] < next_t)
                arr = self.left.df.loc[mask, left_cols].values
                left_win = arr if len(arr) > 0 else np.zeros((1, len(left_cols)))

            combined = self._combine_hands(right_win, left_win, max_seq_length)
            if combined is not None:
                samples.append(combined)
                labels.append(next_char)
                prev_labels.append(prev_label)
                last_char = next_char

        if skipped_chars:
            print(f"Skipped characters : {skipped_chars}")

        n_right, n_left = len(right_cols), len(left_cols)
        feat_dim = n_right + n_left
        metadata = {
            'num_samples': len(samples),
            'num_hands': (1 if self.has_right else 0) + (1 if self.has_left else 0),
            'has_right': self.has_right,
            'has_left': self.has_left,
            'feat_dim': feat_dim,
            'features_per_hand': n_right or n_left,
            'max_seq_length': max_seq_length,
            'skipped_chars': skipped_chars,
        }
        return samples, labels, prev_labels, metadata

    def get_class_distribution(self, labels: List[str]) -> Dict[str, int]:
        """Count per character."""
        dist = {}
        for char in labels:
            dist[char] = dist.get(char, 0) + 1
        return dict(sorted(dist.items(), key=lambda x: -x[1]))

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
from src.Constants.char_to_key import CHAR_TO_INDEX, CHAR_TO_INDEX_4, SPECIAL_KEY_MAP

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
    def _pad_df_to_length(df: pd.DataFrame, target_len: int) -> pd.DataFrame:
        """Zero-pad or truncate dataframe rows to target_len."""
        cols = list(df.columns)
        arr = df.to_numpy(dtype=np.float64)
        n_rows, n_cols = arr.shape if arr.ndim == 2 else (len(df), 0)

        if n_rows >= target_len:
            out = arr[:target_len]
        else:
            out = np.zeros((target_len, n_cols), dtype=np.float64)
            if n_rows > 0:
                out[:n_rows] = arr
        return pd.DataFrame(out, columns=cols)

    @staticmethod
    def _combine_hands(
        right_df: Optional[pd.DataFrame], left_df: Optional[pd.DataFrame],
        right_cols: List[str], left_cols: List[str], max_len: int,) -> Optional[pd.DataFrame]:
        if len(left_cols) == 0 and len(right_cols) == 0:
            return None

        if left_df is None:
            left_df = pd.DataFrame(columns=left_cols)
        if right_df is None:
            right_df = pd.DataFrame(columns=right_cols)

        left_block = left_df.reindex(columns=left_cols, fill_value=0.0).add_suffix("_L")
        right_block = right_df.reindex(columns=right_cols, fill_value=0.0).add_suffix("_R")

        target_len = max_len if max_len != -1 else max(6, len(left_block), len(right_block))
        left_padded = Preprocessing._pad_df_to_length(left_block, target_len)
        right_padded = Preprocessing._pad_df_to_length(right_block, target_len)
        return pd.concat([left_padded, right_padded], axis=1)

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
        max_seq_length: int = 10,
        filter_func: Optional[callable] = None,
        context_prev_windows: int = 0,
        context_future_windows: int = 0,
        char_to_index: Optional[Dict[str, int]] = None,
    ) -> Tuple[List[np.ndarray], List[str], List[str], Dict[str, Any]]:
        """Returns (samples, labels, prev_labels, metadata). labels and prev_labels are characters (str). prev_labels use '' for no previous.
        char_to_index: optional mapping (default CHAR_TO_INDEX). Use CHAR_TO_INDEX_4 for 4-class row-based mapping."""
        vocab = char_to_index if char_to_index is not None else CHAR_TO_INDEX
        context_prev_windows = max(0, int(context_prev_windows))
        context_future_windows = max(0, int(context_future_windows))
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
                self.right.df = filter_func(self.right.df,'R')
            right_cols = self.right.feature_columns
        if self.has_left:
            self.left.df = self.left.sorted_df
            if filter_func is not None:
                self.left.df = filter_func(self.left.df,'L')
            left_cols = self.left.feature_columns

        if len(left_cols) == 0 and len(right_cols) > 0:
            left_cols = list(right_cols)
        if len(right_cols) == 0 and len(left_cols) > 0:
            right_cols = list(left_cols)

        skipped_chars = {}
        combined_col_names = []
        last_char = None
        for i in range(len(key_events) - 1):
            start_evt_idx = max(0, i - context_prev_windows)
            end_evt_idx = min(len(key_events) - 1, (i + 1) + context_future_windows)
            context_start_t = key_events.iloc[start_evt_idx]['time']
            context_end_t = key_events.iloc[end_evt_idx]['time']
            next_char = self._char_from_key(key_events.iloc[i + 1]['name'])
            if next_char is None or next_char not in vocab:
                key = '<nan>' if next_char is None else next_char
                skipped_chars[key] = skipped_chars.get(key, 0) + 1
                continue

            prev_label = last_char if last_char is not None else ''

            right_win = None
            if self.has_right:
                mask = (self.right.df['time_stamp'] > context_start_t) & (self.right.df['time_stamp'] <= context_end_t)
                window_right_df = self.right.df.loc[mask, right_cols]
                if len(window_right_df) > 0:
                    right_win = window_right_df
                else:
                    right_win = pd.DataFrame(columns=right_cols)

            left_win = None
            if self.has_left:
                mask = (self.left.df['time_stamp'] > context_start_t) & (self.left.df['time_stamp'] <= context_end_t)
                window_left_df = self.left.df.loc[mask, left_cols]
                if len(window_left_df) > 0:
                    left_win = window_left_df
                else:
                    left_win = pd.DataFrame(columns=left_cols)

            combined = self._combine_hands(
                right_df=right_win,
                left_df=left_win,
                right_cols=right_cols,
                left_cols=left_cols,
                max_len=max_seq_length,
            )
            if combined is not None:
                if not combined_col_names:
                    combined_col_names = list(combined.columns)
                samples.append(combined.to_numpy(dtype=np.float64))
                labels.append(next_char)
                prev_labels.append(prev_label)
                last_char = next_char

        if skipped_chars:
            print(f"Skipped characters : {skipped_chars}")

        n_right, n_left = len(right_cols), len(left_cols)
        feat_dim = len(combined_col_names)
        metadata = {
            'num_samples': len(samples),
            'num_hands': (1 if self.has_right else 0) + (1 if self.has_left else 0),
            'has_right': self.has_right,
            'has_left': self.has_left,
            'combined_col_names': combined_col_names,
            'feat_dim': feat_dim,
            'features_per_hand': max(n_right, n_left),
            'max_seq_length': max_seq_length,
            'skipped_chars': skipped_chars,
            'context_prev_windows': context_prev_windows,
            'context_future_windows': context_future_windows,
        }
        return samples, labels, prev_labels, metadata

    def get_class_distribution(self, labels: List[str]) -> Dict[str, int]:
        """Count per character."""
        dist = {}
        for char in labels:
            dist[char] = dist.get(char, 0) + 1
        return dict(sorted(dist.items(), key=lambda x: -x[1]))

"""
GIK Data Augmentation: apply transformations to training data for sample diversity.
Generates different variants of the same sample while keeeping the same label, to help the model generalize better.

This only applies to the training dataset

We can have a config to randomly apply one of the following augmentations to each sample during training:
- Gaussian Noise: Add small random noise to the sensor readings to simulate measurement errors.
- Scaling: Randomly scale the sensor readings by a factor within a specified range to simulate variations
- Random Uniform Noise: Add random uniform noise to the sensor readings to introduce variability without biasing in a specific direction.

Each of them have equal probability

"""

import torch
import numpy as np
from torch.utils.data import Dataset


class GIKAugmentationsPerFeature:

    def __init__(self, gauss_std=0.01,scale_range=(0.9, 1.1),uni_low=-0.01 ,uni_high=0.01): # Not setting these too high 
        self.gauss_std = gauss_std
        self.scale_range = scale_range
        self.uni_low = uni_low
        self.uni_high = uni_high

    def gaussian_noise(self, feature_tensor):
        noise = torch.randn_like(feature_tensor)*self.gauss_std
        return feature_tensor + noise

    def scaling(self, feature_tensor):
        scale = torch.empty(1, device=feature_tensor.device).uniform_(*self.scale_range).item()
        return feature_tensor * scale

    def random_uniform_noise(self, feature_tensor):
        noise = torch.empty_like(feature_tensor).uniform_(self.uni_low, self.uni_high)
        return feature_tensor + noise

    def __call__(self, sample):
        """pick gaussian, scaling, uniform with equal prob."""
        augmented = sample.clone()

        num_features = augmented.shape[0]

        for i in range(num_features):

            choice = np.random.randint(0, 4)
            
            if choice == 0:
                augmented[i] = self.gaussian_noise(augmented[i])
            elif choice == 1:
                augmented[i] = self.scaling(augmented[i])
            elif choice == 2:
                augmented[i] = self.random_uniform_noise(augmented[i])
            else:
                pass

        return augmented

class AugmentedDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        augment=None,
        use_augmentation=True,
        synthetic_multiplier=0,
        precompute_synthetic=False,  # Set False by default!
        device=None,
        regression=False,
        reverse_time_prob=0.0,
    ):
        self.base = base_dataset
        self.augment = augment
        self.use_augmentation = use_augmentation
        self.regression = regression
        self.synthetic_multiplier = max(0, int(synthetic_multiplier))
        self.precompute_synthetic = bool(precompute_synthetic)
        self.reverse_time_prob = float(max(0.0, min(1.0, reverse_time_prob)))

        # Synthetic buffer (stored in memory)
        self.synthetic_samples = []
        self.synthetic_labels = []
        self.synthetic_raw_labels = []

        if self.precompute_synthetic and self.synthetic_multiplier > 0 and use_augmentation and augment is not None:
            self._build_synthetic_buffer(self.synthetic_multiplier, device)

    def _resolve_base_item_index(self, idx):
        """Map index in a Subset-wrapped dataset back to the original dataset index."""
        if hasattr(self.base, "indices"):
            base_idx = self.base.indices[idx]
            root = self.base.dataset
            while hasattr(root, "indices"):
                base_idx = root.indices[base_idx]
                root = root.dataset
            return root, int(base_idx)
        return self.base, int(idx)

    @staticmethod
    def _label_to_index(y):
        if torch.is_tensor(y):
            if y.ndim > 0:
                return int(torch.argmax(y).item())
            return int(y.item())
        return int(y)

    @staticmethod
    def _index_to_label(template_y, idx):
        if torch.is_tensor(template_y):
            if template_y.ndim > 0:
                one_hot = torch.zeros_like(template_y, dtype=torch.float32)
                one_hot[int(idx)] = 1.0
                return one_hot
            return torch.tensor(int(idx), dtype=torch.long)
        return int(idx)

    @staticmethod
    def _reverse_sequence_only(x):
        x_rev = x.clone()
        valid = (x.abs().sum(dim=-1) > 0)
        n_valid = int(valid.sum().item())
        if n_valid > 1:
            x_rev[:n_valid] = torch.flip(x[:n_valid], dims=[0])
        if n_valid < x.shape[0]:
            x_rev[n_valid:] = 0.0
        return x_rev

    def _reverse_with_transition_swap(self, x, y, idx):
        """Reverse temporal order and swap transition labels: prev=i, curr=j -> prev=j, curr=i."""
        root_dataset, raw_idx = self._resolve_base_item_index(idx)
        prev_labels = getattr(root_dataset, "_prev_labels", None)
        labels = getattr(root_dataset, "_labels", None)
        char_to_idx = getattr(root_dataset, "_char_to_index", None)
        num_classes = getattr(root_dataset, "_num_classes", None)

        if not (isinstance(prev_labels, list) and isinstance(labels, list) and isinstance(char_to_idx, dict)):
            return x, y
        if raw_idx < 0 or raw_idx >= len(labels):
            return x, y

        prev_char = prev_labels[raw_idx]
        curr_char = labels[raw_idx]
        if (not prev_char) or (prev_char not in char_to_idx) or (curr_char not in char_to_idx):
            return x, y

        prev_idx = int(char_to_idx[prev_char])
        curr_idx = int(char_to_idx[curr_char])
        x_rev = self._reverse_sequence_only(x)
        y_rev = self._index_to_label(y, prev_idx)

        if isinstance(num_classes, int) and num_classes > 0 and x_rev.shape[-1] >= num_classes:
            prev_feat = x_rev[:, -num_classes:]
            probe = prev_feat[0]
            in_range = bool(((probe >= -0.05) & (probe <= 1.05)).all().item())
            near_one = bool(abs(float(probe.sum().item()) - 1.0) < 0.2)
            if in_range and near_one:
                new_prev = torch.zeros((num_classes,), dtype=x_rev.dtype, device=x_rev.device)
                new_prev[curr_idx] = 1.0
                valid = (x_rev.abs().sum(dim=-1) > 0)
                n_valid = int(valid.sum().item())
                if n_valid > 0:
                    x_rev[:n_valid, -num_classes:] = new_prev.unsqueeze(0).expand(n_valid, -1)
                if n_valid < x.shape[0]:
                    x_rev[n_valid:, -num_classes:] = 0.0
        return x_rev, y_rev

    def _build_synthetic_buffer(self, multiplier, device):
        """Generate multiplier augmented copies per base sample."""
        for idx in range(len(self.base)):
            if self.regression:
                x_orig, y, raw_label = self.base[idx]  # ← LOCAL var, not self.char
            else:
                x_orig, y = self.base[idx]
            
            if not torch.is_tensor(x_orig):
                x_orig = torch.as_tensor(x_orig)
            if device is not None:
                x_orig = x_orig.to(device)

            for _ in range(multiplier):
                x_aug = self.augment(x_orig)
                if (not self.regression) and (self.reverse_time_prob > 0.0):
                    if np.random.rand() < self.reverse_time_prob:
                        x_aug = self._reverse_sequence_only(x_aug)
                self.synthetic_samples.append(x_aug.clone())
                self.synthetic_labels.append(y)  # y should be scalar/tensor
                if self.regression:
                    self.synthetic_raw_labels.append(raw_label)  # ← LOCAL var

    def __len__(self):
        virtual_syn = 0
        if (
            self.synthetic_multiplier > 0
            and self.use_augmentation
            and self.augment is not None
            and not self.precompute_synthetic
        ):
            virtual_syn = len(self.base) * self.synthetic_multiplier
        return len(self.base) + virtual_syn + len(self.synthetic_samples)

    def __getitem__(self, idx):
        if idx < len(self.base):
            # Original base sample (with optional online aug)
            if self.regression:
                x, y, raw_label = self.base[idx]
            else:
                x, y = self.base[idx]

            if self.use_augmentation and self.augment is not None:
                x = self.augment(x)

            if (not self.regression) and self.reverse_time_prob > 0.0 and (np.random.rand() < self.reverse_time_prob):
                x, y = self._reverse_with_transition_swap(x, y, idx)

            if self.regression:
                return x, y, raw_label
            return x, y

        virtual_syn_start = len(self.base)
        virtual_syn_end = virtual_syn_start
        if (
            self.synthetic_multiplier > 0
            and self.use_augmentation
            and self.augment is not None
            and not self.precompute_synthetic
        ):
            virtual_syn_end += len(self.base) * self.synthetic_multiplier

        if virtual_syn_start <= idx < virtual_syn_end:
            # On-the-fly synthetic samples, without storing a huge buffer.
            base_idx = (idx - virtual_syn_start) % len(self.base)
            if self.regression:
                x, y, raw_label = self.base[base_idx]
            else:
                x, y = self.base[base_idx]
            x = self.augment(x)
            if (not self.regression) and self.reverse_time_prob > 0.0 and (np.random.rand() < self.reverse_time_prob):
                x, y = self._reverse_with_transition_swap(x, y, base_idx)
            if self.regression:
                return x, y, raw_label
            return x, y
        else:
            # Precomputed synthetic sample (no further augmentation)
            syn_idx = idx - virtual_syn_end
            if self.regression:
                return (
                    self.synthetic_samples[syn_idx],
                    self.synthetic_labels[syn_idx],
                    self.synthetic_raw_labels[syn_idx],
                )
            return self.synthetic_samples[syn_idx], self.synthetic_labels[syn_idx]


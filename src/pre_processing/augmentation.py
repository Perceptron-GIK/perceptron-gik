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

    def __init__(self, gauss_std=0.02,scale_range=(0.8, 1.2),uni_low=-0.02 ,uni_high=0.02): # Not setting these too high 
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
        synthetic_multiplier=0,  # ← NEW: copies per base sample
        precompute_synthetic=True,  # ← NEW: build buffer now?
        device=None,
    ):
        self.base = base_dataset
        self.augment = augment
        self.use_augmentation = use_augmentation

        # Synthetic buffer (stored in memory)
        self.synthetic_samples = []
        self.synthetic_labels = []

        if precompute_synthetic and synthetic_multiplier > 0 and augment is not None:
            self._build_synthetic_buffer(synthetic_multiplier, device)

    def _build_synthetic_buffer(self, multiplier, device):
        """Generate multiplier augmented copies per base sample."""
        for idx in range(len(self.base)):
            x_orig, y = self.base[idx]
            if not torch.is_tensor(x_orig):
                x_orig = torch.as_tensor(x_orig)
            if device is not None:
                x_orig = x_orig.to(device)

            for _ in range(multiplier):
                x_aug = self.augment(x_orig)
                self.synthetic_samples.append(x_aug.clone())
                self.synthetic_labels.append(y)

    def __len__(self):
        return len(self.base) + len(self.synthetic_samples)

    def __getitem__(self, idx):
        if idx < len(self.base):
            # Original base sample (with optional online aug)
            x, y = self.base[idx]
            if self.use_augmentation and self.augment is not None:
                x = self.augment(x)
            return x, y
        else:
            # Precomputed synthetic sample (no further augmentation)
            syn_idx = idx - len(self.base)
            return self.synthetic_samples[syn_idx], self.synthetic_labels[syn_idx]

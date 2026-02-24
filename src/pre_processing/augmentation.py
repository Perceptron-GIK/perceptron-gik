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


class GIKAugmentations:

    def __init__(self, gauss_std=0.01,scale_range=(0.9, 1.1),uni_low=-0.01 ,uni_high=0.01): # Not setting these too high 
        self.gauss_std = gauss_std
        self.scale_range = scale_range
        self.uni_low = uni_low
        self.uni_high = uni_high

    def gaussian_noise(self, sample):
        noise = torch.randn_like(sample)*self.gauss_std
        return sample + noise

    def scaling(self, sample):
        scale = torch.empty(1, device=sample.device).uniform_(*self.scale_range).item()
        return sample * scale

    def random_uniform_noise(self, sample):
        noise = torch.empty_like(sample).uniform_(self.uni_low, self.uni_high)
        return sample + noise

    def __call__(self, sample):
        """pick gaussian, scaling, uniform with equal prob."""
        choice = np.random.randint(0, 3)
        if choice == 0:
            return self.gaussian_noise(sample) 
        elif choice == 1:
            return self.scaling(sample)
        else:
            return self.random_uniform_noise(sample)
        # maybe add another number for no augmentation, to keep some samples unchanged?

class AugmentedDataset(Dataset):
    def __init__(self, base_dataset, augment=None, use_augmentation=True):
        self.base = base_dataset
        self.augment = augment # Here call the GIKAugmentations
        self.use_augmentation = use_augmentation

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if self.use_augmentation and self.augment is not None:
            x = self.augment(x)
        return x, y

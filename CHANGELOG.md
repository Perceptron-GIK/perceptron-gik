# Changelog: soup/trainer-split-augmentation

## Purpose
Stratified/contiguous split, weighted sampler, on-the-fly synthetic augmentation.

## Files Changed

### `ml/models/gik_model.py`
- **split_strategy**: `"stratified_random"` or `"contiguous"` — stratified preserves class balance across splits
- **split_seed**: Random seed for reproducible stratified split
- **use_weighted_sampler**: Class-weighted sampling to balance imbalanced training data
- **synthetic_multiplier**, **precompute_synthetic**: Control synthetic augmentation (precompute vs on-the-fly)
- Augmentation/synthetic generation kept on CPU for GPU/MPS stability

### `src/pre_processing/augmentation.py`
- **On-the-fly synthetic expansion**: When `precompute_synthetic=false` and `synthetic_multiplier > 0`, generates augmented samples in `__getitem__` without storing a buffer
- `synthetic_multiplier`, `precompute_synthetic` as instance attributes

### `train_model.ipynb`
- Wire `split_strategy`, `split_seed`, `use_weighted_sampler`, `synthetic_multiplier`, `precompute_synthetic` from config into GIKTrainer

### `train_config.yaml`
- `split_strategy`, `split_seed`, `use_weighted_sampler`, `synthetic_multiplier`, `precompute_synthetic`

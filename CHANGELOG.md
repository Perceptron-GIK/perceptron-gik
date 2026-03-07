# Changelog: soup/reverse-time-augmentation

## Purpose
Time-reverse transition swap augmentation: flip sequences so `prev=i, next=j` becomes `prev=j, next=i`.

## Files Changed

### `src/pre_processing/augmentation.py`
- **reverse_time_prob**: Probability of applying reverse-time augmentation
- **_reverse_sequence_only()**: Flip temporal order of sequence (keep labels unchanged for precomputed path)
- **_reverse_with_transition_swap()**: Reverse time + swap labels (prev↔curr) for classification; update prev_char one-hot in features if present
- **_resolve_base_item_index()**: Map Subset index back to root dataset for label lookup
- Applied in both base-sample path and on-the-fly synthetic path

### `ml/models/gik_model.py`
- **reverse_time_aug_prob**: Passed to AugmentedDataset

### `train_model.ipynb`
- Wire `reverse_time_aug_prob` from config into GIKTrainer

### `train_config.yaml`
- `train.reverse_time_aug_prob`

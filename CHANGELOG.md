# Changelog: soup/preprocessing-leakage-fix

## Purpose
Fix data leakage in PCA and class-weight computation.

## Files Changed

### `src/pre_processing/reduce_dim.py`
- PCA fit-on-train-only: fit PCA on training split only, transform val/test
- Zero-padding preservation for variable-length sequences
- SVD modernization (use `torch.linalg.svd`)

### `pretraining.py`
- `get_class_weights`: class weights computed from train split only
- Split strategy and seed configurable for consistent train/val/test splits

## Config
- No new keys in `train_config.yaml`

## Notebook
- No changes (pretraining.py is used as-is; notebook already calls get_class_weights)

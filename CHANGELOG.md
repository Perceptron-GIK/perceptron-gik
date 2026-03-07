# Changelog: soup/alignment-context-inference-parity

## Purpose
Alignment context windows (prev/future), append_prev_char feature, inference parity with training.

## Files Changed

### `src/pre_processing/alignment.py`
- **context_prev_windows**, **context_future_windows**: Expand IMU window to include previous/future key events
- Time range for each sample: `[context_start_t, context_end_t]` based on event indices

### `pretraining.py`
- `alignment_prev_windows`, `alignment_future_windows` in `preprocess_multiple_sources`
- Passed to `preprocessor.align()`

### `inference_preprocessing.py`
- **append_prev_char**: Flag to add prev-char one-hot to input (default True)
- **prev_char null-check fix**: `if prev_char is None` instead of `if not prev_char` (handles empty string)

### `inference_receiver.py`
- `alignment_prev_windows`, `alignment_future_windows`, `append_prev_char` from config
- **Model load-once**: Global MODEL, load once on first inference
- **Decode buffering**: Context-aware event queue for lookahead when `context_future > 0`

### `train_model.ipynb`
- Wire `alignment_prev_windows`, `alignment_future_windows`, `append_prev_char_feature` into CONFIG, preprocess_multiple_sources, load_preprocessed_dataset

### `train_config.yaml`
- `experiment.alignment_context.prev_windows`, `future_windows`
- `experiment.append_prev_char_feature`

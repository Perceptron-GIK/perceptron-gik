# Changelog: soup/lm-fusion-decoding

## Purpose
N-gram LM, beam search, TTA (test-time augmentation), interpolated LM for sequence-level decoding.

## Files Changed

### `src/decoding/lm_fusion.py` (new)
- `build_char_ngram_lm`, `build_interpolated_char_lm`: Character n-gram LMs with additive smoothing
- `collect_logits_and_labels`, `collect_logits_and_labels_tta`: Gather model logits (with optional TTA)
- `beam_decode_with_lm`: Beam search decoding with LM fusion
- `tune_beta_on_validation`: Tune LM interpolation weight on val set
- `sequence_accuracy`: Per-sequence accuracy metric

### `ml/models/gik_model.py`
- **evaluate_with_lm_fusion()**: Evaluate with LM + beam search + optional TTA
- **transition_log_probs**: Bigram transition matrix from train data (for sequence auxiliary loss)
- **sequence_lm_beta**, **sequence_aux_weight**: In-training sequence LM auxiliary loss (when prev_char in input)

### `train_model.ipynb`
- LM fusion evaluation cell; wire `lm_order`, `lm_beam_width`, `lm_beta_values`, `lm_tta_passes`, etc.

### `train_config.yaml`
- `lm_order`, `lm_add_k`, `lm_beam_width`, `lm_beta_values`, `lm_use_interpolated`, `lm_tta_passes`, `lm_tta_noise_std`, `lm_tta_scale_jitter`, `sequence_lm_beta`, `sequence_aux_weight`

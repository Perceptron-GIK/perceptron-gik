# Test Suite for Perceptron GIK ML Components

This directory contains comprehensive unit and integration tests for the ML components of the Perceptron GIK project.

## Test Structure

```
tests/
├── __init__.py                          # Test package initialization
├── conftest.py                          # Shared pytest fixtures
├── ml/
│   ├── __init__.py
│   └── test_basic_nn.py                 # Tests for ML models and training
├── preprocessing/
│   ├── __init__.py
│   ├── test_alignment.py                # Tests for IMU-keyboard alignment
│   └── test_pretraining.py              # Tests for data preprocessing
└── test_integration.py                  # End-to-end integration tests
```

## Test Coverage

### ML Models (`tests/ml/test_basic_nn.py`)
- **Character Mappings**: Tests for CHAR_TO_INDEX, INDEX_TO_CHAR, and NUM_CLASSES
- **Model Architectures**: Tests for all 6 model types (LSTM, GRU, RNN, Transformer, Attention-LSTM, CNN)
- **GIKModelWrapper**: Tests for wrapper functionality, predictions, and probability outputs
- **FocalLoss**: Tests for focal loss implementation and gamma parameter effects
- **GIKTrainer**: Tests for training loop, validation, and early stopping
- **Model Creation**: Tests for factory functions and error handling
- **Prediction Decoding**: Tests for converting model outputs to characters

### Preprocessing (`tests/preprocessing/`)

#### Alignment Tests (`test_alignment.py`)
- **TrainingData**: CSV loading, feature extraction, sorting, and data concatenation
- **Preprocessing**: IMU-keyboard alignment, character mapping, sequence padding
- **Data Validation**: Invalid character handling, NaN handling, class distribution

#### Pretraining Tests (`test_pretraining.py`)
- **IMU Filtering**: Signal filtering, position computation, NaN/Inf handling
- **PreprocessedGIKDataset**: Dataset loading, normalization, class weights
- **CSV Export**: Dataset export functionality, special character handling
- **Data Consistency**: Leak detection, label distribution, data quality checks

### Integration Tests (`tests/test_integration.py`)
- **End-to-End Workflow**: Complete pipeline from raw data to trained model
- **Multiple Model Types**: Parametrized tests for all 6 model architectures
- **Model Persistence**: Save and load functionality
- **Training/Validation**: Full training loop with validation

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run specific test modules
```bash
pytest tests/ml/test_basic_nn.py
pytest tests/preprocessing/test_alignment.py
pytest tests/preprocessing/test_pretraining.py
pytest tests/test_integration.py
```

### Run with verbose output
```bash
pytest tests/ -v
```

### Run with coverage report
```bash
pytest tests/ --cov=ml --cov=pretraining --cov=src/pre_processing
```

### Run specific test classes or methods
```bash
pytest tests/ml/test_basic_nn.py::TestLSTMModel
pytest tests/ml/test_basic_nn.py::TestFocalLoss::test_focal_loss_computation
```

### Run parametrized tests for specific parameters
```bash
pytest tests/test_integration.py::TestMultipleModelTypes::test_model_type_trains[lstm]
```

## Test Statistics

- **Total Tests**: 88
- **Test Modules**: 4
- **Model Architecture Tests**: 36
- **Preprocessing Tests**: 40
- **Integration Tests**: 12

## Dependencies

The test suite requires:
- pytest >= 6.0
- pytest-cov (for coverage reports)
- torch
- numpy
- pandas
- matplotlib
- scipy

Install test dependencies:
```bash
pip install pytest pytest-cov torch numpy pandas matplotlib scipy
```

## Test Configuration

Tests are configured via `pytest.ini`:
- Minimum pytest version: 6.0
- Test discovery pattern: `test_*.py`
- Warnings are disabled for cleaner output
- Random seeds are fixed for reproducibility (see `conftest.py`)

## Writing New Tests

When adding new tests:

1. Place tests in the appropriate module:
   - ML model tests → `tests/ml/`
   - Preprocessing tests → `tests/preprocessing/`
   - Integration tests → `tests/test_integration.py`

2. Use descriptive test names following the pattern `test_<what_is_being_tested>`

3. Add docstrings explaining what each test validates

4. Use fixtures from `conftest.py` for common test data

5. Ensure tests are deterministic (use fixed random seeds)

## Example Test

```python
def test_lstm_forward_pass(sample_data):
    """Test that LSTM produces correct output shape."""
    model = LSTMModel(hidden_dim=16, num_layers=2)
    x = torch.randn(8, 10, 16)
    output = model(x)
    
    assert output.shape == (8, 16)
```

## Continuous Integration

These tests should be run:
- On every commit to ensure code quality
- Before merging pull requests
- As part of the CI/CD pipeline

## Known Issues

None currently. All 88 tests pass successfully.

## Contributing

When contributing new features:
1. Write tests for new functionality
2. Ensure all existing tests pass
3. Aim for >80% code coverage
4. Follow existing test patterns and naming conventions

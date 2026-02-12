"""
Shared pytest fixtures and configuration for all tests.
"""
import pytest
import torch
import numpy as np


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    yield


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    return 'cpu'  # Always use CPU for tests


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Ensure deterministic behavior (only if CUDA is available)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

import numpy as np
import pytest

@pytest.fixture
def music_labels():
    return ["croche", "noire", "blanche"]

@pytest.fixture
def train_data(music_labels):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((120, 8))
    y = [music_labels[i % 3] for i in range(120)]
    return X, y

@pytest.fixture
def test_data():
    rng = np.random.default_rng(99)
    return rng.standard_normal((30, 8))

@pytest.fixture
def gray_uniform():
    return np.random.default_rng(0).integers(0, 256, (64, 64), dtype=np.uint8)

@pytest.fixture
def gray_bimodal():
    rng = np.random.default_rng(1)
    return np.concatenate([
        rng.integers(20,  80, (32, 64), dtype=np.uint8),
        rng.integers(160, 230, (32, 64), dtype=np.uint8),
    ], axis=0)

@pytest.fixture
def gray_gradient():
    return np.tile(np.arange(256, dtype=np.uint8), (4, 1))
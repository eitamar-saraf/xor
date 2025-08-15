import numpy as np

from xor_demo.config import DataConfig
from xor_demo.data import make_xor


def test_make_xor_deterministic():
    cfg = DataConfig(n_samples=100, noise=0.05, seed=42)
    X1, y1 = make_xor(cfg)
    X2, y2 = make_xor(cfg)
    assert np.allclose(X1, X2)
    assert np.array_equal(y1, y2)


def test_make_xor_label_balance_reasonable():
    cfg = DataConfig(n_samples=2000, noise=0.0, seed=7)
    _, y = make_xor(cfg)
    p = y.mean()
    # XOR across a symmetric box should be ~50/50
    assert 0.45 < p < 0.55

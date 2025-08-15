from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .config import DataConfig


def make_xor(cfg: DataConfig) -> Tuple[np.ndarray, np.ndarray]:
	"""Generate a deterministic XOR dataset in [-1, 1]^2 with optional label noise.

	Returns X (n,2) and y (n,), where y=1 if x1*x2>0 else 0, with some flips.
	"""
	rng = np.random.RandomState(cfg.seed)
	X = rng.uniform(-1.0, 1.0, size=(cfg.n_samples, 2))
	y = (X[:, 0] * X[:, 1] > 0).astype(int)
	if cfg.noise > 0:
		flip_mask = rng.rand(cfg.n_samples) < cfg.noise
		y[flip_mask] = 1 - y[flip_mask]
	return X, y


@dataclass
class Dataset:
	X: np.ndarray
	y: np.ndarray


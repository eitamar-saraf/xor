from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
	"""Configuration for XOR dataset generation.

	n_samples: total number of samples to generate
	noise: probability of flipping a label (label noise)
	seed: RNG seed for determinism
	"""

	n_samples: int = 600
	noise: float = 0.08
	seed: int = 7


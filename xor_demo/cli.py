from __future__ import annotations

import argparse

from .config import DataConfig
from .plots import run_full_demo


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="XOR: Linear vs Non-Linear Models")
    p.add_argument("--out", default="plots", help="Output directory for plots")
    p.add_argument("--samples", type=int, default=600, help="Number of samples")
    p.add_argument("--noise", type=float, default=0.08, help="Label noise probability [0,1]")
    p.add_argument("--seed", type=int, default=7, help="Random seed")
    p.add_argument("--skip-3d", action="store_true", help="Skip 3D feature map plot")
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = DataConfig(n_samples=args.samples, noise=args.noise, seed=args.seed)
    out_map = run_full_demo(cfg, args.out, skip_3d=args.skip_3d)

    # Print a concise summary
    print(f"Saved {len(out_map)} figures to {args.out}")


__all__ = ["main", "build_parser"]

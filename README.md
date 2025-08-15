# XOR: Linear vs Non‑Linear Models

Demonstrates why linear models fail on XOR and how non‑linear methods succeed, with decision boundaries, learning curves, and a 3D feature map.

## Quick Start

Install requirements:

```bash
pip install -r requirements.txt
# Or as a package:
pip install -e .
```

Generate all plots to `plots/` using the CLI:

```bash
python -m xor_demo.cli --out plots --samples 600 --noise 0.08 --seed 7
# Or after install:
xor-demo --out plots
```

## Outputs

Plots are saved in `plots/` by default, along with `metadata.json` describing the run config and package versions.

## What to Look For

- Linear models (logistic regression, linear SVM) fail to separate XOR.
- Adding quadratic features or using non‑linear models (RBF SVM, MLP) succeeds.
- Learning curves highlight capacity/underfitting vs. non‑linear expressivity.

## License

MIT


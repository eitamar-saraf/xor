# XOR: Linear vs Non‑Linear Models

Demonstrates why linear models fail on XOR and how non‑linear methods succeed, with decision boundaries, learning curves, and a 3D feature map.

## Contents

## Quick start
```bash
Install requirements (or the package):

```
pip install -r requirements.txt
# Optional: pip install -e .
```

Generate plots to `plots/` using the CLI:

```
python -m xor_demo.cli --out plots --samples 600 --noise 0.08 --seed 7
# or after install: xor-demo --out plots
```

The legacy script still works:

```
python xor_demo.py --out plots
```
```

## Outputs
Plots are saved in `plots/` by default, along with `metadata.json` describing the run config and versions.

## What to look for
- Linear models (logistic regression, linear SVM) fail to separate XOR.
- Adding quadratic features or using non‑linear models (RBF SVM, MLP) succeeds.
- Learning curves highlight capacity/underfitting vs. non‑linear expressivity.

## License
MIT


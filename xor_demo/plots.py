from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import learning_curve, train_test_split

from .config import DataConfig
from .data import make_xor

plt.style.use("seaborn-v0_8-whitegrid")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def savefig(fig: plt.Figure, out_dir: str, filename: str) -> str:
    ensure_dir(out_dir)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    return path


def plot_data_scatter(X: np.ndarray, y: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(X[y == 0, 0], X[y == 0, 1], label="class 0", alpha=0.9)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], label="class 1", alpha=0.9, marker="x")
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_title(title)
    ax.legend(loc="best")
    return fig


def plot_decision_boundary(model, X_all: np.ndarray, y_all: np.ndarray, title: str, X_test: np.ndarray, y_test: np.ndarray, proba: bool = False) -> plt.Figure:
    x_min, x_max = X_all[:, 0].min() - 0.25, X_all[:, 0].max() + 0.25
    y_min, y_max = X_all[:, 1].min() - 0.25, X_all[:, 1].max() + 0.25
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict_proba(grid)[:, 1] if proba and hasattr(model, "predict_proba") else model.predict(grid)
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 6))
    if proba and hasattr(model, "predict_proba"):
        cs = ax.contourf(xx, yy, Z, levels=20, alpha=0.7)
        fig.colorbar(cs, ax=ax, label="P(class=1)")
    else:
        ax.contourf(xx, yy, Z, alpha=0.3, levels=[-1, 0, 1])
    ax.scatter(X_all[y_all == 0, 0], X_all[y_all == 0, 1], label="class 0", alpha=0.85)
    ax.scatter(X_all[y_all == 1, 0], X_all[y_all == 1, 1], label="class 1", alpha=0.85, marker="x")
    acc = (model.predict(X_test) == y_test).mean()
    loss_str = ""
    if hasattr(model, "predict_proba"):
        try:
            loss = log_loss(y_test, model.predict_proba(X_test))
            loss_str = f" | log-loss={loss:.3f}"
        except Exception:
            pass
    ax.set_title(f"{title}\nTest accuracy={acc:.3f}{loss_str}")
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.legend()
    return fig


def plot_learning_curve(estimator, X_train, y_train, title: str) -> plt.Figure:
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X_train, y_train, cv=5, scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 8), shuffle=True, random_state=42
    )
    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    test_mean, test_std = test_scores.mean(axis=1), test_scores.std(axis=1)
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.plot(train_sizes, train_mean, label="Training score")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    ax.plot(train_sizes, test_mean, label="CV score")
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2)
    ax.set_title(title); ax.set_xlabel("Training set size"); ax.set_ylabel("Accuracy")
    ax.legend(loc="best")
    return fig


def plot_feature_map_3d(X: np.ndarray, y: np.ndarray, title: str) -> plt.Figure:
    from sklearn.linear_model import LogisticRegression

    X_phi = np.c_[X[:, 0], X[:, 1], X[:, 0] * X[:, 1]]
    clf = LogisticRegression(max_iter=2000, random_state=42).fit(X_phi, y)
    x1 = np.linspace(X[:, 0].min() - 0.2, X[:, 0].max() + 0.2, 50)
    x2 = np.linspace(X[:, 1].min() - 0.2, X[:, 1].max() + 0.2, 50)
    X1, X2 = np.meshgrid(x1, x2); X3 = X1 * X2
    grid_phi = np.c_[X1.ravel(), X2.ravel(), X3.ravel()]
    proba = clf.predict_proba(grid_phi)[:, 1].reshape(X1.shape)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X_phi[y == 0, 0], X_phi[y == 0, 1], X_phi[y == 0, 2], label="class 0", alpha=0.8)
    ax.scatter(X_phi[y == 1, 0], X_phi[y == 1, 1], X_phi[y == 1, 2], label="class 1", alpha=0.8, marker="x")
    ax.plot_wireframe(X1, X2, X3, rstride=4, cstride=4, linewidth=0.3, alpha=0.3)
    mask = np.abs(proba - 0.5) < 0.02
    ax.scatter(X1[mask], X2[mask], X3[mask], alpha=0.4, s=5, label="~ decision boundary")
    ax.set_xlabel("phi1 = x1"); ax.set_ylabel("phi2 = x2"); ax.set_zlabel("phi3 = x1*x2")
    ax.set_title(title); ax.legend(loc="best")
    return fig


def run_full_demo(cfg: DataConfig, out_dir: str, skip_3d: bool = False) -> Dict[str, str]:
    """Reproduce all figures used in the blog/demo and write metadata.json.

    Returns a mapping of logical names to file paths of saved images.
    """
    from .models import get_models

    ensure_dir(out_dir)

    # Data
    X, y = make_xor(cfg)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=cfg.seed, stratify=y
    )

    # Metadata
    meta = {
        "data": asdict(cfg),
        "out_dir": out_dir,
        "versions": {
            "numpy": np.__version__,
            "matplotlib": plt.matplotlib.__version__,
        },
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    out: Dict[str, str] = {}

    # Plots: data
    out["train_scatter"] = savefig(
        plot_data_scatter(X_train, y_train, "XOR training set"), out_dir, "01_xor_training_scatter.png"
    )
    out["test_scatter"] = savefig(
        plot_data_scatter(X_test, y_test, "XOR test set"), out_dir, "02_xor_test_scatter.png"
    )

    # Models & decision boundaries
    models = get_models()
    for pipe in models.values():
        pipe.fit(X_train, y_train)
    out["boundary_linear_logreg"] = savefig(
        plot_decision_boundary(
            models["linear_logreg"], X, y, "Linear Logistic Regression (fails on XOR)", X_test, y_test, proba=True
        ),
        out_dir,
        "10_linear_logreg_boundary.png",
    )
    out["boundary_svm_linear"] = savefig(
        plot_decision_boundary(models["svm_linear"], X, y, "Linear SVM (fails on XOR)", X_test, y_test, proba=True),
        out_dir,
        "11_linear_svm_boundary.png",
    )
    out["boundary_poly2_logreg"] = savefig(
        plot_decision_boundary(
            models["poly2_logreg"], X, y, "Polynomial (degree 2) + Logistic Regression", X_test, y_test, proba=True
        ),
        out_dir,
        "12_poly2_logreg_boundary.png",
    )
    out["boundary_svm_rbf"] = savefig(
        plot_decision_boundary(models["svm_rbf"], X, y, "RBF Kernel SVM", X_test, y_test, proba=True),
        out_dir,
        "13_svm_rbf_boundary.png",
    )
    out["boundary_mlp"] = savefig(
        plot_decision_boundary(models["mlp_tanh_8x8"], X, y, "MLP (tanh, 8×8)", X_test, y_test, proba=True),
        out_dir,
        "14_mlp_boundary.png",
    )

    # Learning curves
    out["lc_linear_logreg"] = savefig(
        plot_learning_curve(
            models["linear_logreg"], X_train, y_train, "Learning Curve — Linear Logistic Regression (XOR)"
        ),
        out_dir,
        "20_lc_linear_logreg.png",
    )
    out["lc_mlp"] = savefig(
        plot_learning_curve(
            models["mlp_tanh_8x8"], X_train, y_train, "Learning Curve — MLP (tanh, 8×8) on XOR"
        ),
        out_dir,
        "21_lc_mlp.png",
    )

    # 3D feature map
    if not skip_3d:
        out["feature_map_3d"] = savefig(
            plot_feature_map_3d(
                X, y, "XOR becomes linearly separable in φ(x)=(x1,x2,x1·x2)"
            ),
            out_dir,
            "30_feature_map_3d.png",
        )

    return out

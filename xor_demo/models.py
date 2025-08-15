from __future__ import annotations

from typing import Dict

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC


def get_models() -> Dict[str, Pipeline]:
    """Return the dictionary of model pipelines used in the demo."""
    return {
        "linear_logreg": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(max_iter=2000, random_state=42)),
            ]
        ),
        "poly2_logreg": Pipeline(
            [
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("scaler", StandardScaler(with_mean=False)),
                ("logreg", LogisticRegression(max_iter=4000, random_state=42)),
            ]
        ),
        "svm_linear": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svc", SVC(kernel="linear", probability=True, random_state=42)),
            ]
        ),
        "svm_rbf": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svc", SVC(kernel="rbf", gamma="scale", C=1.0, probability=True, random_state=42)),
            ]
        ),
        "mlp_tanh_8x8": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(8, 8),
                        activation="tanh",
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

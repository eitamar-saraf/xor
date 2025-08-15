from xor_demo.models import get_models


def test_models_factory_contains_expected():
    models = get_models()
    for k in [
        "linear_logreg",
        "poly2_logreg",
        "svm_linear",
        "svm_rbf",
        "mlp_tanh_8x8",
    ]:
        assert k in models

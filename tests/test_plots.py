from xor_demo.config import DataConfig
from xor_demo.plots import run_full_demo


def test_run_full_demo_writes_outputs(tmp_path):
    out = run_full_demo(DataConfig(n_samples=120, noise=0.05, seed=7), out_dir=str(tmp_path), skip_3d=True)
    # basic subset
    expected = {
        "train_scatter": "01_xor_training_scatter.png",
        "test_scatter": "02_xor_test_scatter.png",
        "boundary_linear_logreg": "10_linear_logreg_boundary.png",
        "boundary_svm_linear": "11_linear_svm_boundary.png",
    }
    for key, fname in expected.items():
        assert key in out
        assert (tmp_path / fname).exists()

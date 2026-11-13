import torch

from model.Meta_Models.target_residual_calibrator import HorizonWiseResidualCalibrator


def test_zero_residual_keeps_base_forecast_unchanged():
    targets = torch.randn(40, 3, 36)
    residuals = torch.zeros_like(targets)
    calibrator = HorizonWiseResidualCalibrator()
    calibrator.fit_from_residuals(residuals, targets)
    base = torch.randn(5, 3, 36)
    corrected = base + calibrator(base)
    assert torch.allclose(corrected, base)
    assert torch.isfinite(corrected).all()


def test_correction_is_finite_bounded_and_shape_preserving():
    torch.manual_seed(7)
    targets = torch.randn(60, 4, 36) * 5 + 30
    residuals = torch.ones_like(targets) * 2.0
    residuals += 0.05 * torch.randn_like(residuals)
    calibrator = HorizonWiseResidualCalibrator(max_correction_ratio=0.25)
    summary = calibrator.fit_from_residuals(residuals, targets)
    base = torch.randn(8, 4, 36) * 4 + 30
    correction = calibrator(base)
    assert correction.shape == base.shape
    assert torch.isfinite(correction).all()
    assert correction.abs().max() <= 0.25 * calibrator.node_scale.max() + 1e-5
    assert len(summary["horizon_gates"]) == 3


def test_too_few_calibration_windows_falls_back_to_identity():
    targets = torch.randn(8, 2, 36)
    residuals = torch.ones_like(targets)
    calibrator = HorizonWiseResidualCalibrator(
        purge_windows=3, min_calibration_windows=4
    )
    calibrator.fit_from_residuals(residuals, targets)
    base = torch.randn(2, 2, 36)
    assert torch.equal(calibrator(base), torch.zeros_like(base))

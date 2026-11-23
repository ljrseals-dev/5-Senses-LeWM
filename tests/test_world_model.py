import pytest
import torch

from senses.world_model.adapter import WorldModelEye
from senses.world_model.jepa import LeWMStyle
from senses.world_model.planner import LatentCEMPlanner
from senses.world_model.sigreg import EppsPulley, SIGReg


def test_sigreg_is_finite_and_differentiable():
    z = torch.randn(12, 8, requires_grad=True)
    loss = SIGReg(n_projections=16)(z)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_sigreg_rejects_singleton_batch():
    with pytest.raises(ValueError, match="N >= 2"):
        SIGReg()(torch.randn(1, 8))


def test_epps_pulley_rejects_invalid_shape():
    with pytest.raises(ValueError):
        EppsPulley()(torch.empty(0, 2))


def test_cem_supports_single_elite_without_nan():
    model = LeWMStyle(latent_dim=8, action_dim=2, n_projections=4)
    planner = LatentCEMPlanner(model, horizon=2, n_candidates=4, n_elite=1, n_iters=2)
    actions, cost = planner.plan(torch.randn(8), torch.randn(8))
    assert actions.shape == (2, 2)
    assert torch.isfinite(actions).all()
    assert torch.isfinite(torch.tensor(cost))
    assert actions.min() >= -1.0
    assert actions.max() <= 1.0


def test_cem_rejects_mismatched_latent_shapes():
    model = LeWMStyle(latent_dim=8, action_dim=2, n_projections=4)
    planner = LatentCEMPlanner(model)
    with pytest.raises(ValueError, match="identical shapes"):
        planner.plan(torch.randn(8), torch.randn(7))


def test_eye_surprise_is_prediction_error_and_restores_mode():
    model = LeWMStyle(latent_dim=8, action_dim=2, n_projections=4)
    eye = WorldModelEye(model=model)
    eye.train_mode()
    obs_t = torch.randn(4, 3, 32, 32)
    obs_tp1 = torch.randn(4, 3, 32, 32)
    action = torch.randn(4, 2)
    with torch.no_grad():
        expected = model(obs_t, action, obs_tp1)[3].item()
    actual = eye.surprise(obs_t, action, obs_tp1)
    assert actual == pytest.approx(expected, rel=1e-5)
    assert model.training is True


def test_eye_rejects_wrong_observation_channels():
    eye = WorldModelEye()
    with pytest.raises(ValueError, match="3,H,W"):
        eye.encode(torch.randn(1, 1, 32, 32))


def test_model_rejects_bad_action_batch():
    model = LeWMStyle(latent_dim=8, action_dim=2, n_projections=4)
    with pytest.raises(ValueError, match="actions must have shape"):
        model(torch.randn(3, 3, 32, 32), torch.randn(2, 2))


def test_model_rejects_negative_regularization_weight():
    with pytest.raises(ValueError, match="non-negative"):
        LeWMStyle(lambda_reg=-0.1)

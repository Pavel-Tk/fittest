import numpy as np
import pytest

from trustswarm.config import WRONG_E, WRONG_F
from trustswarm.signals import (energy_spread, force_rmse, force_spread, frame_error, is_wrong,
                                max_abs_force, model_deviations, rel_e_err)


def test_force_spread_known_values() -> None:
    # atom 0: models disagree by ±1 along x -> s_0 = 1; atom 1: identical -> s_1 = 0
    forces = np.array([[[1.0, 0, 0], [0.3, 0.2, 0.1]],
                       [[-1.0, 0, 0], [0.3, 0.2, 0.1]]])
    mean, mx = force_spread(forces)
    assert mean == pytest.approx(0.5)
    assert mx == pytest.approx(1.0)


def test_force_spread_three_models_by_hand() -> None:
    forces = np.array([[[0.0, 0, 0]], [[3.0, 0, 0]], [[0.0, 3.0, 0]]])  # F̄ = (1, 1, 0)
    sq = [1 + 1, 4 + 1, 1 + 4]  # ||F_m - F̄||²
    expected = np.sqrt(np.mean(sq))
    assert force_spread(forces) == pytest.approx((expected, expected))


def test_force_spread_identical_models_is_zero_and_shape_checked() -> None:
    f = np.random.default_rng(0).normal(size=(4, 7, 3))
    assert force_spread(np.stack([f[0]] * 3)) == pytest.approx((0.0, 0.0), abs=1e-12)
    with pytest.raises(ValueError):
        force_spread(f[:1])
    with pytest.raises(ValueError):
        force_spread(f[..., :2])


def test_model_deviations_and_energy_spread() -> None:
    forces = np.array([[[2.0, 0, 0]], [[0.0, 0, 0]]])
    assert model_deviations(forces) == pytest.approx([1.0, 1.0])
    assert energy_spread([1.0, 3.0]) == pytest.approx(1.0)
    assert energy_spread([1.0]) == 0.0


def test_rel_energy_cancels_constant_model_offset() -> None:
    E, E_r = -1234.5, -1236.0
    for offset in (0.0, -17.3, 250.0):
        assert rel_e_err(E + offset, E_r + offset, E, E_r) == pytest.approx(0.0, abs=1e-9)
    assert rel_e_err(E + 0.4, E_r, E, E_r) == pytest.approx(0.4)


def test_force_rmse_and_max_force() -> None:
    F = np.zeros((2, 3))
    F_hat = np.full((2, 3), 0.5)
    assert force_rmse(F_hat, F) == pytest.approx(0.5)
    assert max_abs_force(np.array([[3.0, 4.0, 0], [0, 0, 1]])) == pytest.approx(5.0)
    with pytest.raises(ValueError):
        force_rmse(np.zeros((2, 3)), np.zeros((3, 3)))


def test_is_wrong_thresholds() -> None:
    assert not is_wrong(WRONG_F, WRONG_E)          # strictly greater-than
    assert is_wrong(WRONG_F + 1e-6, 0.0)
    assert is_wrong(0.0, WRONG_E + 1e-6)
    assert not is_wrong(0.1, None)                 # unknown relative energy: force only
    assert is_wrong(0.2, None, wrong_f=0.15)
    err = frame_error(np.full((1, 3), 0.1), np.zeros((1, 3)), rel_hat=1.0, rel_true=0.5)
    assert err.rel_e_err == pytest.approx(0.5) and err.wrong()
    assert frame_error(np.zeros((1, 3)), np.zeros((1, 3)), None, 0.5).rel_e_err is None

"""DisagreementHook tests.

CPU / numpy only (run anywhere): vectorised sigma == ``signals.force_spread``; the monitor's
flag ledger with fake committee members (thresholds, callable tau, failing member).
Torch twin: needs torch. Toolkit integration (real ``FusedStage`` + ``SizeAwareSampler`` +
FIRE2, graduation + refill): needs nvalchemi; skipped cleanly otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

from fittest.hook import DisagreementMonitor, per_system_force_spread
from fittest.signals import force_spread


def _ragged(rng, sizes, M=3):
    batch = np.concatenate([np.full(n, s) for s, n in enumerate(sizes)])
    F = rng.normal(size=(M, len(batch), 3))
    return F, batch


def test_vectorised_sigma_matches_signals_force_spread():
    rng = np.random.default_rng(0)
    sizes = [1, 2, 7, 23, 5]
    F, b = _ragged(rng, sizes)
    mean, mx = per_system_force_spread(F, b)
    for s in range(len(sizes)):
        ref_mean, ref_max = force_spread(F[:, b == s])
        assert mean[s] == pytest.approx(ref_mean, rel=1e-12, abs=1e-14)
        assert mx[s] == pytest.approx(ref_max, rel=1e-12, abs=1e-14)


def test_sigma_zero_for_identical_models():
    rng = np.random.default_rng(1)
    F, b = _ragged(rng, [4, 6], M=1)
    mean, mx = per_system_force_spread(np.repeat(F, 3, axis=0), b)
    assert np.allclose(mean, 0) and np.allclose(mx, 0)


def test_torch_twin_matches_numpy():
    torch = pytest.importorskip("torch")
    from fittest.hook import per_system_force_spread_torch

    rng = np.random.default_rng(2)
    sizes = [3, 9, 1, 12]
    F, b = _ragged(rng, sizes, M=4)
    m_np, x_np = per_system_force_spread(F, b)
    m_t, x_t = per_system_force_spread_torch(torch.as_tensor(F), torch.as_tensor(b), len(sizes))
    assert np.allclose(m_t.numpy(), m_np, rtol=1e-12)
    assert np.allclose(x_t.numpy(), x_np, rtol=1e-12)


# ---------------------------------------------------------------- fake committee
class FakeModel:
    """Numpy 'MLIP': F = base(R) + bias on systems listed in ``noisy`` (per-system index)."""

    def __init__(self, name, bias=0.0, noisy=(), fail=False, as_dict=True):
        self.model, self.bias, self.noisy, self.fail, self.as_dict = name, bias, set(noisy), fail, as_dict
        self.calls = 0

    def predict(self, Z, R, batch, charge=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("server down")
        F = -0.1 * np.asarray(R, dtype=float)
        mask = np.isin(batch, list(self.noisy))
        F[mask] += self.bias
        E = np.bincount(batch, weights=(R ** 2).sum(1))
        return {"energy": E, "forces": F} if self.as_dict else (E, F)


def _systems(sizes, rng):
    Z = np.concatenate([np.full(n, 6) for n in sizes])
    R = rng.normal(size=(len(Z), 3))
    b = np.concatenate([np.full(n, s) for s, n in enumerate(sizes)])
    return Z, R, b


def test_monitor_flags_exactly_the_disagreeing_systems():
    rng = np.random.default_rng(3)
    Z, R, b = _systems([4, 5, 6, 3], rng)
    # Two models agree; the third adds a constant (1,1,1) bias on systems 1 and 3.
    clients = [FakeModel("a"), FakeModel("b", as_dict=False), FakeModel("c", bias=1.0, noisy=(1, 3))]
    seen = []
    mon = DisagreementMonitor(clients, tau=0.5, on_flag=lambda r: seen.append(r["key"]))
    sig, newly = mon.check(["s0", "s1", "s2", "s3"], Z, R, b, step=0, phase="admission")
    # Deviations from the mean are (-1/3, -1/3, +2/3) * (1,1,1): s_i = sqrt(mean_m ||d_m||^2) = sqrt(2/3).
    expected = np.sqrt(np.mean([3 * (1 / 3) ** 2, 3 * (1 / 3) ** 2, 3 * (2 / 3) ** 2]))
    assert sig[[0, 2]] == pytest.approx([0, 0], abs=1e-12)
    assert sig[[1, 3]] == pytest.approx([expected, expected], rel=1e-12)
    assert newly.tolist() == [False, True, False, True]
    assert mon.flagged == ["s1", "s3"] and seen == ["s1", "s3"]
    rec = mon.records["s1"]
    assert rec["flagged"] and rec["flag_phase"] == "admission" and rec["sigma_admission"] == pytest.approx(expected)
    assert set(rec["at_flag"]["forces"]) == {"a", "b", "c"} and rec["at_flag"]["R"].shape == (5, 3)
    # A second check records sigma again but does not re-flag.
    _, newly2 = mon.check(["s0", "s1", "s2", "s3"], Z, R, b, step=10, phase="step")
    assert not newly2.any() and len(mon.records["s1"]["checks"]) == 2
    mon.close()


def test_monitor_callable_tau_and_failing_member():
    rng = np.random.default_rng(4)
    Z, R, b = _systems([3, 3], rng)
    tau = {"v": 10.0}
    clients = {"a": FakeModel("a"), "b": FakeModel("b", bias=1.0, noisy=(0,)),
               "dead": FakeModel("dead", fail=True)}
    mon = DisagreementMonitor(clients, tau=lambda: tau["v"], max_failures=2)
    _, n1 = mon.check([0, 1], Z, R, b, step=0, phase="admission")
    assert not n1.any() and "dead" in mon.failures and "dead" not in mon.dead
    tau["v"] = 0.1                                   # Calibrator lowered the threshold
    _, n2 = mon.check([0, 1], Z, R, b, step=10, phase="step")
    assert n2.tolist() == [True, False]
    assert mon.records[0]["tau_at_flag"] == 0.1 and "dead" in mon.dead
    calls = clients["dead"].calls
    mon.check([0, 1], Z, R, b, step=20, phase="step")
    assert clients["dead"].calls == calls          # dropped after max_failures
    assert mon.summary()["dead_models"] == ["dead"]
    mon.close()


def test_monitor_needs_two_answers():
    rng = np.random.default_rng(5)
    Z, R, b = _systems([3], rng)
    mon = DisagreementMonitor([FakeModel("a"), FakeModel("x", fail=True)], tau=0.0)
    sig, newly = mon.check(["k"], Z, R, b, step=0, phase="admission")
    assert np.isnan(sig).all() and not newly.any() and mon.records["k"]["checks"][0]["sigma"] is None
    mon.close()


# ---------------------------------------------------------------- toolkit integration
def _toolkit():
    pytest.importorskip("torch")
    return pytest.importorskip("nvalchemi")


class _FlagByMarker(FakeModel):
    """Disagrees (bias) on any system containing a nitrogen atom (Z == 7), position-independent."""

    def predict(self, Z, R, batch, charge=None):
        F = np.zeros((len(Z), 3))
        S = int(batch.max()) + 1
        has_n = np.zeros(S, bool)
        np.logical_or.at(has_n, batch, Z == 7)
        F[has_n[batch]] = self.bias
        return {"energy": np.zeros(S), "forces": F}


def test_hook_graduates_flagged_systems_and_sampler_refills():
    _toolkit()
    import torch
    from nvalchemi.dynamics import FIRE2, ConvergenceHook, FusedStage, SizeAwareSampler
    from nvalchemi.dynamics.base import DynamicsStage
    from nvalchemi.models.lj import LennardJonesModelWrapper

    from fittest.hook import DisagreementHook
    from fittest.models.toolkit_io import atomic_data

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(6)
    systems = []
    for i in range(12):
        n = 3 + i % 3
        Z = np.full(n, 6)
        if i % 4 == 1:
            Z[0] = 7                                  # marker: committee disagrees on these
        R = np.stack([np.arange(n) * 3.5, np.zeros(n), np.zeros(n)], 1) + 0.05 * rng.normal(size=(n, 3))
        systems.append((Z, R))
    flagged_truth = {i for i in range(12) if i % 4 == 1}

    class DS:
        def __init__(self):
            self.load_order = []

        def __len__(self):
            return len(systems)

        def get_metadata(self, i):
            return len(systems[i][0]), 0

        def __getitem__(self, i):
            self.load_order.append(i)
            Z, R = systems[i]
            return atomic_data(Z, R, device=dev, with_dynamics_fields=True), {}

    ds = DS()
    lj = LennardJonesModelWrapper(epsilon=0.01, sigma=3.0, cutoff=8.0)
    hook = DisagreementHook([_FlagByMarker("a"), _FlagByMarker("b", bias=1.0)], tau=0.3, every=5,
                            stage="both", graduate=True, id_map=lambda sid: ds.load_order[sid])
    relax = FIRE2(model=lj, dt=0.05, n_steps=15, convergence_hook=ConvergenceHook.from_fmax(1e-6))
    for h in lj.make_neighbor_hooks():
        relax.register_hook(h, stage=DynamicsStage.BEFORE_COMPUTE)
    sampler = SizeAwareSampler(ds, max_atoms=16, max_edges=None, max_batch_size=4, bin_width=1, shuffle=False)
    fused = FusedStage(sub_stages=[(0, relax)], sampler=sampler, sinks=[], refill_frequency=1, hooks=[hook])
    with fused:
        result = fused.run(batch=None, n_steps=2000)

    assert result is None                                   # sampler exhausted, everyone graduated
    assert set(hook.records) == set(range(12))              # every system admitted and scored
    assert set(hook.flagged) == flagged_truth
    for i in range(12):
        rec = hook.records[i]
        assert rec["sigma_admission"] is not None
        if i in flagged_truth:
            assert rec["flag_phase"] == "admission" and rec["reason"] == "disagreement" and rec["n_steps"] == 0
        else:
            assert rec["reason"] == "dynamics" and rec["n_steps"] >= 15 - 1
    assert sum(e["event"] == "admission" for e in hook.events) > 1   # refills happened
    hook.close()


def test_duplicate_member_names_rejected():
    from fittest.hook import DisagreementMonitor

    def predict(Z, R, batch):  # two different callables that share a __name__
        return np.zeros(int(np.max(batch)) + 1), np.zeros((len(Z), 3))

    other = lambda Z, R, batch: predict(Z, R, batch)  # noqa: E731
    other.__name__ = "predict"
    with pytest.raises(ValueError, match="duplicate"):
        DisagreementMonitor([predict, other], tau=0.1)

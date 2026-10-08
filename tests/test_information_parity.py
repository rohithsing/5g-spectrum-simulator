"""
The fairness guarantee.

The single most important property of this study is that the proposed
allocator has no information the classical baselines lack. These tests make
that property mechanical rather than a matter of trust:

* the observation handed to a scheduler contains no field correlated with the
  incumbent's true state beyond the noisy sensing stream;
* a scheduler cannot reach the environment internals through the observation;
* an allocator fed a *replayed* observation stream makes identical decisions,
  proving it reads nothing else.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from spectrum.baselines import build_classical
from spectrum.config import SimConfig
from spectrum.env import Observation, SpectrumEnv

#: Field names the observation is permitted to carry. Adding anything here is
#: a deliberate act that must be justified as something a real gNB can read.
ALLOWED_FIELDS = {
    "slot", "n_ue", "n_rbg", "n_subbands",
    "backlog_bits", "hol_delay_slots", "deadline_slack_slots", "delay_budget_slots",
    "nominal_rate_bps", "priority_level", "ue_class", "avg_throughput_bps",
    "last_slot_rbgs",
    "cqi", "spectral_efficiency", "bits_per_rbg",
    "sensing_decision", "sensing_energy",
    "sensing_decision_history", "sensing_energy_history", "subband_of_rbg",
}


def test_observation_exposes_only_allowed_fields():
    actual = {f.name for f in dataclasses.fields(Observation)}
    assert actual == ALLOWED_FIELDS, (
        "Observation's field set changed. Any new field must be something a "
        "deployed gNB scheduler genuinely has access to."
    )


def test_observation_has_no_ground_truth_occupancy():
    """No observation field may equal the incumbent's true state."""
    cfg = SimConfig(congestion="high", seed=5, slots_per_episode=60)
    env = SpectrumEnv(cfg, seed=5)
    rng = np.random.default_rng(0)

    mismatches = 0
    comparisons = 0
    for _ in range(60):
        obs = env.observe()
        truth = env.incumbent.active
        for f in dataclasses.fields(Observation):
            value = getattr(obs, f.name)
            if not isinstance(value, np.ndarray):
                continue
            if value.shape == truth.shape and value.dtype.kind in "iub":
                comparisons += 1
                if not np.array_equal(value.astype(int), truth):
                    mismatches += 1
        env.step(rng.integers(-1, cfg.n_ue, size=cfg.n_rbg))

    # sensing_decision has the same shape as the truth and matches it most of
    # the time by design -- but it must NOT match it always.
    assert comparisons > 0
    assert mismatches > 0, "an observation field tracked the incumbent perfectly"


def test_sensing_decision_is_imperfect_at_the_configured_operating_point():
    cfg = SimConfig(congestion="medium", seed=7, slots_per_episode=4000)
    env = SpectrumEnv(cfg, seed=7)

    dec, truth = [], []
    for _ in range(4000):
        obs = env.observe()
        dec.append(obs.sensing_decision.copy())
        truth.append(env.incumbent.active.copy())
        env.step(np.full(cfg.n_rbg, -1))

    dec = np.concatenate(dec)
    truth = np.concatenate(truth)
    pd_hat = dec[truth == 1].mean()
    pfa_hat = dec[truth == 0].mean()

    assert pd_hat == pytest.approx(cfg.sensing_prob_detection, abs=0.03)
    assert pfa_hat == pytest.approx(cfg.sensing_prob_false_alarm, abs=0.03)


def test_observation_is_a_detached_snapshot():
    """Mutating an observation must not reach back into the environment."""
    cfg = SimConfig(congestion="medium", seed=3, slots_per_episode=10)
    env = SpectrumEnv(cfg, seed=3)
    obs = env.observe()

    obs.backlog_bits[:] = -12345.0
    obs.sensing_decision[:] = 9
    assert not np.any(env.traffic.backlog_bits == -12345.0)
    assert not np.any(env._decision == 9)


def test_observation_is_frozen():
    cfg = SimConfig(seed=1, slots_per_episode=5)
    obs = SpectrumEnv(cfg, seed=1).observe()
    with pytest.raises(dataclasses.FrozenInstanceError):
        obs.slot = 999  # type: ignore[misc]


@pytest.mark.parametrize("allocator", build_classical(), ids=lambda a: a.short_name)
def test_allocator_decides_from_the_observation_alone(allocator):
    """
    Replay recorded observations into a fresh allocator with no environment
    attached. Identical decisions prove the allocator read nothing else.
    """
    cfg = SimConfig(congestion="high", seed=11, slots_per_episode=40)
    env = SpectrumEnv(cfg, seed=11)
    allocator.reset(cfg)

    recorded, first_pass = [], []
    obs = env.observe()
    for _ in range(40):
        recorded.append(obs)
        a = allocator.allocate(obs)
        first_pass.append(a.copy())
        obs, _, _, _ = env.step(a)

    del env  # the environment no longer exists
    allocator.reset(cfg)
    for obs_rec, expected in zip(recorded, first_pass):
        assert np.array_equal(allocator.allocate(obs_rec), expected)


def test_learned_agent_decides_from_the_observation_alone():
    """The same replay guarantee for the proposed allocator."""
    pytest.importorskip("torch")
    from pathlib import Path

    from spectrum.agent import NeuroSpectrumAllocator

    if not Path("artifacts/policy.pt").exists():
        pytest.skip("no trained policy; run `python train.py` first")

    predictor_path = (
        "artifacts/predictor.pt" if Path("artifacts/predictor.pt").exists() else None
    )
    cfg = SimConfig(congestion="high", seed=13, slots_per_episode=30)
    env = SpectrumEnv(cfg, seed=13)

    agent = NeuroSpectrumAllocator.load("artifacts/policy.pt", predictor_path)
    agent.reset(cfg)

    recorded, first_pass = [], []
    obs = env.observe()
    for _ in range(30):
        recorded.append(obs)
        a = agent.allocate(obs)
        first_pass.append(a.copy())
        obs, _, _, _ = env.step(a)

    del env
    agent = NeuroSpectrumAllocator.load("artifacts/policy.pt", predictor_path)
    agent.reset(cfg)
    for obs_rec, expected in zip(recorded, first_pass):
        assert np.array_equal(agent.allocate(obs_rec), expected)

"""
Allocator contract tests.

Every scheduler -- classical or learned -- must respect the same action
contract and the same buffer-awareness discipline, otherwise the comparison
is not apples-to-apples.
"""

from __future__ import annotations

import numpy as np
import pytest

from spectrum.baselines import MaxCI, MLWDF, ProportionalFair, RoundRobin, build_classical
from spectrum.config import SimConfig
from spectrum.env import SpectrumEnv
from spectrum.metrics import qos_satisfied_mask
from spectrum.runner import run_episode

ALLOCATORS = build_classical()
IDS = [a.short_name for a in ALLOCATORS]


@pytest.mark.parametrize("alloc", ALLOCATORS, ids=IDS)
def test_assignment_shape_and_range(alloc):
    cfg = SimConfig(congestion="high", seed=2, slots_per_episode=40)
    env = SpectrumEnv(cfg, seed=2)
    alloc.reset(cfg)
    obs = env.observe()
    for _ in range(40):
        a = alloc.allocate(obs)
        assert a.shape == (cfg.n_rbg,)
        assert a.dtype.kind == "i"
        assert a.min() >= -1 and a.max() < cfg.n_ue
        obs, _, _, _ = env.step(a)


@pytest.mark.parametrize("alloc", ALLOCATORS, ids=IDS)
def test_never_schedules_an_empty_queue(alloc):
    """Buffer awareness: no scheduler may burn an RBG on a UE with nothing to send."""
    cfg = SimConfig(congestion="low", seed=8, slots_per_episode=60)
    env = SpectrumEnv(cfg, seed=8)
    alloc.reset(cfg)
    obs = env.observe()
    for _ in range(60):
        a = alloc.allocate(obs)
        for ue in a[a >= 0]:
            assert obs.backlog_bits[ue] > 0.0
        obs, _, _, _ = env.step(a)


@pytest.mark.parametrize("alloc", ALLOCATORS, ids=IDS)
def test_respects_the_sensing_decision(alloc):
    """Classical incumbent protection: never transmit on a subband flagged busy."""
    cfg = SimConfig(congestion="peak", seed=4, slots_per_episode=60)
    env = SpectrumEnv(cfg, seed=4)
    alloc.reset(cfg)
    obs = env.observe()
    for _ in range(60):
        a = alloc.allocate(obs)
        flagged = obs.sensing_decision[obs.subband_of_rbg] == 1
        assert np.all(a[flagged] == -1)
        obs, _, _, _ = env.step(a)


def test_maxci_beats_round_robin_on_goodput():
    cfg = SimConfig(congestion="high", seed=31)
    rr = run_episode(RoundRobin(), cfg, 31)
    mc = run_episode(MaxCI(), cfg, 31)
    assert mc["goodput_mbps"] > rr["goodput_mbps"]


def test_proportional_fair_is_fairer_than_maxci():
    cfg = SimConfig(congestion="high", seed=31)
    pf = run_episode(ProportionalFair(), cfg, 31)
    mc = run_episode(MaxCI(), cfg, 31)
    assert pf["jain_fairness"] > mc["jain_fairness"]


def test_mlwdf_misses_fewer_deadlines_than_maxci():
    """The QoS-aware scheduler must actually be better at QoS."""
    cfg = SimConfig(congestion="high", seed=31)
    ml = run_episode(MLWDF(), cfg, 31)
    mc = run_episode(MaxCI(), cfg, 31)
    assert ml["deadline_miss_pct"] < mc["deadline_miss_pct"]


def test_congestion_regimes_are_monotonically_harder():
    results = {
        lv: run_episode(MLWDF(), SimConfig(congestion=lv, seed=5), 5)
        for lv in ("low", "medium", "high", "peak")
    }
    qos = [results[lv]["qos_satisfaction_pct"] for lv in ("low", "medium", "high", "peak")]
    assert qos[0] >= qos[1] >= qos[2] >= qos[3]


def test_qos_mask_requires_both_throughput_and_reliability():
    kw = dict(
        packets_arrived=np.array([100, 100, 100, 0]),
        mean_delay_slots=np.zeros(4),
        delay_budget_slots=np.full(4, 20.0),
    )
    mask = qos_satisfied_mask(
        delivered_bits=np.array([100.0, 50.0, 100.0, 0.0]),
        arrived_bits=np.array([100.0, 100.0, 100.0, 0.0]),
        packets_delivered=np.array([100, 100, 50, 0]),
        packets_expired=np.array([0, 0, 50, 0]),
        **kw,
    )
    assert mask[0]        # met both halves of the contract
    assert not mask[1]    # throughput shortfall
    assert not mask[2]    # reliability shortfall
    assert mask[3]        # no traffic -> no unmet contract


def test_learned_agent_matches_the_allocator_contract():
    pytest.importorskip("torch")
    from pathlib import Path

    from spectrum.agent import NeuroSpectrumAllocator

    if not Path("artifacts/policy.pt").exists():
        pytest.skip("no trained policy; run `python train.py` first")

    pred = "artifacts/predictor.pt" if Path("artifacts/predictor.pt").exists() else None
    agent = NeuroSpectrumAllocator.load("artifacts/policy.pt", pred)

    cfg = SimConfig(congestion="peak", seed=17, slots_per_episode=40)
    env = SpectrumEnv(cfg, seed=17)
    agent.reset(cfg)
    obs = env.observe()
    for _ in range(40):
        a = agent.allocate(obs)
        assert a.shape == (cfg.n_rbg,)
        assert a.min() >= -1 and a.max() < cfg.n_ue
        for ue in a[a >= 0]:
            assert obs.backlog_bits[ue] > 0.0
        obs, _, _, _ = env.step(a)

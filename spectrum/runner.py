"""Episode runner shared by training, evaluation and the dashboard."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .baselines import Allocator
from .config import SimConfig
from .env import SpectrumEnv


def run_episode(
    allocator: Allocator,
    cfg: SimConfig,
    seed: int,
    *,
    collect_sensing_trace: bool = False,
    collect_timeline: bool = False,
) -> Dict[str, Any]:
    """
    Run one full episode and return its KPI summary.

    ``collect_sensing_trace`` additionally returns the per-slot sensing reports
    paired with the ground-truth occupancy revealed *after* each decision. That
    pairing is what supervises the incumbent predictor, and it is exactly the
    data a real operator accumulates from incumbent-detection reports and
    post-hoc interference measurements.
    """
    env = SpectrumEnv(cfg, seed=seed)
    allocator.reset(cfg)

    obs = env.observe()
    decisions: List[np.ndarray] = []
    energies: List[np.ndarray] = []
    truths: List[np.ndarray] = []
    timeline: List[Dict[str, float]] = []

    done = False
    total_reward = 0.0
    while not done:
        if collect_sensing_trace:
            decisions.append(obs.sensing_decision.copy())
            energies.append(obs.sensing_energy.copy())

        assignment = allocator.allocate(obs)
        obs, reward, done, info = env.step(assignment)
        total_reward += reward

        if collect_sensing_trace:
            truths.append(info["_truth_incumbent"])
        if collect_timeline:
            timeline.append(
                {
                    "slot": env.slot,
                    "useful_mbps": info["useful_bits"] / cfg.slot_duration_s / 1e6,
                    "collisions": info["collisions"],
                    "qos_fraction": info["qos_fraction"],
                    "jain": info["jain"],
                    "reward": reward,
                }
            )

    result: Dict[str, Any] = env.episode_summary()
    result["total_reward"] = total_reward
    result["allocator"] = allocator.name
    result["congestion"] = cfg.congestion
    result["seed"] = seed

    if collect_sensing_trace:
        result["_sensing"] = {
            "decision": np.array(decisions),
            "energy": np.array(energies),
            "truth": np.array(truths),
        }
    if collect_timeline:
        result["_timeline"] = timeline
    return result


def run_suite(
    allocator: Allocator,
    congestion_levels: List[str],
    seeds: List[int],
    *,
    base_cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Run an allocator across every (congestion, seed) pair."""
    base = dict(base_cfg or {})
    out: List[Dict[str, Any]] = []
    for level in congestion_levels:
        for seed in seeds:
            cfg = SimConfig(congestion=level, seed=seed, **base)
            out.append(run_episode(allocator, cfg, seed))
    return out

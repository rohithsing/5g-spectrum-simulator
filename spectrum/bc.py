"""
Behaviour cloning from a classical scheduler.

PPO starting from random weights spends a long time rediscovering things
M-LWDF already knows (serve the urgent bearer, prefer the UE with the better
subband). Initialising the policy by imitating the strongest classical
scheduler removes that phase and, just as importantly, means the learned agent
starts from parity with its comparator rather than from far below it -- so any
reported gain is unambiguously what reinforcement learning *added*.

This is initialisation only. The teacher sees exactly the same
:class:`~spectrum.env.Observation` as everything else, so no privileged
information leaks in through the clone.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .baselines import Allocator, MLWDF
from .config import SimConfig
from .env import SpectrumEnv
from .policy import MAX_UE, MUTE, SchedulerPolicy, extract_features, rbg_order
from .agent import PredictorAugmentedBaseline
from .predictor import GRUPredictor, OccupancyEstimator


def replay_masks(
    assignment: np.ndarray, feat: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recover the action/mask sequence a teacher's assignment corresponds to.

    The candidate set shrinks as queues are covered, exactly as in
    :func:`spectrum.policy.decode`, so replaying the teacher's choices through
    the same residual bookkeeping reproduces the masks the student would have
    faced.
    """
    n_rbg = assignment.size
    bits = feat["bits"]
    residual = feat["backlog"].copy()
    valid = feat["valid"]

    actions = np.full(n_rbg, MUTE, dtype=np.int64)
    masks = np.zeros((n_rbg, MAX_UE + 1), dtype=bool)
    masks[:, MUTE] = True

    for k in rbg_order(feat["slot"], n_rbg):
        eligible = valid & (residual > 1e-9)
        masks[k, :MAX_UE] = eligible
        ue = int(assignment[k])
        if ue >= 0 and eligible[ue]:
            actions[k] = ue
            residual[ue] -= bits[ue, k]
        else:
            actions[k] = MUTE
    return actions, masks


def collect_demonstrations(
    teacher: Allocator,
    predictor: Optional[GRUPredictor],
    *,
    n_slots: int = 6000,
    congestion_mix: Sequence[str] = ("medium", "high", "peak"),
    episode_slots: int = 500,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Roll the teacher through the simulator and record (features, action) pairs."""
    rng = np.random.default_rng(seed)
    store: Dict[str, List[np.ndarray]] = {
        "ue": [], "rbg": [], "glob": [], "z": [], "valid": [], "actions": [], "masks": []
    }

    # The teacher may itself be driven by the predictor. Sharing one slot-cached
    # estimator keeps the filter advancing exactly once per slot no matter how
    # many components ask for the estimate.
    estimator = OccupancyEstimator(predictor)
    if isinstance(teacher, PredictorAugmentedBaseline):
        teacher.estimator = estimator

    collected = 0
    episode = 0
    while collected < n_slots:
        level = congestion_mix[episode % len(congestion_mix)]
        ep_seed = int(rng.integers(0, 2**31 - 1))
        cfg = SimConfig(congestion=level, seed=ep_seed, slots_per_episode=episode_slots)
        env = SpectrumEnv(cfg, seed=ep_seed)
        teacher.reset(cfg)
        estimator.reset(cfg.n_subbands)
        if isinstance(teacher, PredictorAugmentedBaseline):
            teacher.estimator = estimator

        obs = env.observe()
        done = False
        while not done and collected < n_slots:
            feat = extract_features(obs, estimator(obs))
            assignment = teacher.allocate(obs)
            actions, masks = replay_masks(assignment, feat)

            store["ue"].append(feat["ue"])
            store["rbg"].append(feat["rbg"])
            store["glob"].append(feat["global"])
            store["z"].append(feat["z"])
            store["valid"].append(feat["valid"])
            store["actions"].append(actions)
            store["masks"].append(masks)
            collected += 1

            obs, _, done, _ = env.step(assignment)
        episode += 1

    return {k: np.stack(v) for k, v in store.items()}


def behaviour_clone(
    policy: SchedulerPolicy,
    demos: Dict[str, np.ndarray],
    *,
    epochs: int = 6,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
    verbose: bool = True,
) -> List[float]:
    """
    Fit the policy's assignment distribution to the teacher's choices.

    The learning rate is cosine-annealed to zero over the run. Imitation
    fidelity is the binding constraint on the whole pipeline: PPO is anchored to
    this clone, so whatever the clone fails to reproduce is a ceiling RL starts
    below and cannot climb past. A flat learning rate left the clone at 66 %
    teacher-match -- which is underfitting, not a capacity limit, because the
    factored logit ``score_u + gain_u * z_uk`` represents M-LWDF's
    ``log(urgency_u / R_u) + log(bits_uk)`` exactly when ``gain_u`` is constant.
    """
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    data = {
        "ue": torch.from_numpy(demos["ue"]),
        "rbg": torch.from_numpy(demos["rbg"]),
        "glob": torch.from_numpy(demos["glob"]),
        "z": torch.from_numpy(demos["z"]),
        "valid": torch.from_numpy(demos["valid"]),
        "actions": torch.from_numpy(demos["actions"]),
        "masks": torch.from_numpy(demos["masks"]),
    }
    n = data["ue"].shape[0]
    losses: List[float] = []

    for epoch in range(epochs):
        perm = torch.randperm(n, generator=gen)
        total, seen, correct = 0.0, 0, 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            logits, _ = policy(
                data["ue"][idx], data["rbg"][idx], data["glob"][idx],
                data["z"][idx], data["valid"][idx],
            )
            masked = logits.masked_fill(~data["masks"][idx], -1e9)
            tgt = data["actions"][idx]
            loss = F.cross_entropy(masked.reshape(-1, MAX_UE + 1), tgt.reshape(-1))

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

            total += float(loss.detach()) * idx.numel()
            seen += idx.numel()
            correct += int((masked.argmax(-1) == tgt).float().sum())

        sched.step()
        mean_loss = total / max(seen, 1)
        acc = correct / max(seen * data["actions"].shape[1], 1)
        losses.append(mean_loss)
        if verbose:
            print(f"  BC epoch {epoch + 1}/{epochs}  loss {mean_loss:.4f}  "
                  f"teacher-match {acc:.1%}  lr {opt.param_groups[0]['lr']:.2e}",
                  flush=True)

    return losses


def build_teacher(predictor: Optional[GRUPredictor]) -> Allocator:
    """
    The strongest non-RL scheduler available, used as the cloning target.

    When a predictor exists the teacher is M-LWDF driven by the *learned*
    occupancy estimate rather than the raw detector bit. This matters: M-LWDF
    mutes exactly when ``sensing_decision == 1``, so cloning it teaches the
    student that the hard bit perfectly predicts the teacher and that ``p_busy``
    can be ignored -- and the student then inherits the classical detector's
    collision rate despite having a far better estimate in its input. Measured
    on plain-M-LWDF demonstrations, the clone reproduced the teacher's 11.5 %
    peak collision rate; the predictor-augmented teacher sits at 5.2 %.
    """
    if predictor is None:
        return MLWDF()
    return PredictorAugmentedBaseline(MLWDF(), predictor)


def warm_start(
    policy: SchedulerPolicy,
    predictor: Optional[GRUPredictor],
    *,
    n_slots: int = 6000,
    epochs: int = 6,
    seed: int = 0,
    verbose: bool = True,
) -> List[float]:
    """Collect demonstrations from the strongest classical teacher and clone them."""
    teacher = build_teacher(predictor)
    if verbose:
        print(f"  collecting {n_slots} demonstration slots from {teacher.name}...", flush=True)
    demos = collect_demonstrations(teacher, predictor, n_slots=n_slots, seed=seed)
    return behaviour_clone(policy, demos, epochs=epochs, seed=seed, verbose=verbose)

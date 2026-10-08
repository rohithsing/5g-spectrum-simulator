"""
NeuroSpectrum: the proposed allocator.

It is the composition of two learned components sitting behind the same
observation interface every classical baseline uses:

1. :class:`~spectrum.predictor.GRUPredictor` turns the raw sensing stream into a
   calibrated incumbent-occupancy probability.
2. :class:`~spectrum.policy.SchedulerPolicy` turns bearer state, CQI reports and
   that probability into an RBG assignment.

Nothing else is added. In particular the agent never reads the incumbent's true
state -- it only ever sees :class:`~spectrum.env.Observation`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .baselines import Allocator
from .config import SimConfig
from .env import Observation
from .policy import SchedulerPolicy, decode, extract_features
from .predictor import GRUPredictor, OccupancyEstimator


class NeuroSpectrumAllocator(Allocator):
    """Learned occupancy estimation + learned attention scheduler."""

    name = "NeuroSpectrum (proposed)"
    short_name = "Ours"

    def __init__(
        self,
        policy: SchedulerPolicy,
        predictor: Optional[GRUPredictor] = None,
        *,
        greedy: bool = True,
        seed: int = 0,
        name: Optional[str] = None,
    ):
        self.policy = policy
        self.predictor = predictor
        self.estimator = OccupancyEstimator(predictor)
        self.greedy = greedy
        self.rng = np.random.default_rng(seed)
        if name:
            self.name = name

    def reset(self, cfg: SimConfig) -> None:
        super().reset(cfg)
        self.estimator.reset(cfg.n_subbands)

    # ------------------------------------------------------------------ infer
    def _p_busy(self, obs: Observation) -> np.ndarray:
        return self.estimator(obs)

    @torch.no_grad()
    def allocate(self, obs: Observation) -> np.ndarray:
        feat = extract_features(obs, self._p_busy(obs))
        logits, _ = self.policy(
            torch.from_numpy(feat["ue"]).unsqueeze(0),
            torch.from_numpy(feat["rbg"]).unsqueeze(0),
            torch.from_numpy(feat["global"]).unsqueeze(0),
            torch.from_numpy(feat["z"]).unsqueeze(0),
            torch.from_numpy(feat["valid"]).unsqueeze(0),
        )
        assignment, _ = decode(
            logits[0].numpy().astype(np.float64), feat, self.rng, greedy=self.greedy
        )
        return assignment

    # ------------------------------------------------------------------- i/o
    @classmethod
    def load(
        cls,
        policy_path: str | Path,
        predictor_path: Optional[str | Path] = None,
        **kwargs,
    ) -> "NeuroSpectrumAllocator":
        policy = SchedulerPolicy()
        policy.load_state_dict(torch.load(policy_path, map_location="cpu"))
        policy.eval()

        predictor = None
        if predictor_path is not None:
            predictor = GRUPredictor()
            predictor.load_state_dict(torch.load(predictor_path, map_location="cpu"))
            predictor.eval()

        return cls(policy, predictor, **kwargs)


class PredictorAugmentedBaseline(Allocator):
    """
    Ablation: a classical scheduler whose sensing front-end is the learned
    predictor instead of the hard energy decision.

    Running this alongside the full agent separates the two contributions --
    how much of the gain comes from better incumbent estimation, and how much
    from better scheduling.
    """

    def __init__(self, base: Allocator, predictor: GRUPredictor, threshold: float = 0.5):
        self.base = base
        self.predictor = predictor
        self.estimator = OccupancyEstimator(predictor)
        self.threshold = threshold
        self.name = f"{base.name} + learned sensing"
        self.short_name = f"{base.short_name}+LS"

    def reset(self, cfg: SimConfig) -> None:
        super().reset(cfg)
        self.base.reset(cfg)
        self.estimator.reset(cfg.n_subbands)

    def allocate(self, obs: Observation) -> np.ndarray:
        p = self.estimator(obs)
        # Replace the hard decision with the predictor's thresholded belief and
        # hand the baseline an otherwise identical observation.
        from dataclasses import replace

        patched = replace(obs, sensing_decision=(p >= self.threshold).astype(np.int64))
        return self.base.allocate(patched)

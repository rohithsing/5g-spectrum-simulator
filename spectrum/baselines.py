"""
Classical schedulers: the "existing framework" side of the comparison.

These are the rules actually deployed in LTE/NR base stations today, not
strawmen. Every one of them is:

* **PRB-granular** -- it assigns individual resource-block groups, so it has
  exactly the same action space as the learned policy;
* **buffer-aware** -- it stops piling resources onto a UE once that UE's queue
  is covered, which removes the most obvious source of artificial waste;
* **sensing-driven** -- it mutes an RBG whose subband the energy detector flags
  as occupied, which is the standard DSA/CBRS incumbent-protection rule.

The only thing they lack is the ability to *learn*: their weights are fixed,
and they treat the instantaneous sensing decision as ground truth.

References
----------
[1] P. Viswanath, D. Tse, R. Laroia, "Opportunistic beamforming using dumb
    antennas", IEEE Trans. Inf. Theory, 2002.  (Proportional Fair)
[2] M. Andrews et al., "Providing quality of service over a shared wireless
    link", IEEE Commun. Mag., vol. 39, no. 2, 2001.  (M-LWDF)
[3] J.-H. Rhee, J. M. Holtzman, D.-K. Kim, "Scheduling of real/non-real time
    services: adaptive EXP/PF algorithm", IEEE VTC, 2003.  (EXP/PF)
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .config import SimConfig
from .env import Observation

_EPS = 1e-9


class Allocator:
    """Interface shared by every scheduler in the study."""

    name: str = "abstract"
    short_name: str = "abstract"

    def reset(self, cfg: SimConfig) -> None:  # noqa: D102 - trivial hook
        self.cfg = cfg

    def allocate(self, obs: Observation) -> np.ndarray:
        """Return an ``(n_rbg,)`` int array of UE indices, or ``-1`` to mute."""
        raise NotImplementedError


def usable_rbgs(obs: Observation) -> np.ndarray:
    """
    Boolean mask of RBGs the incumbent-protection rule permits using.

    This is the classical behaviour: trust the detector. It both collides
    (whenever the detector misses, 15 % of the time) and throws away spectrum
    (whenever it false-alarms, 12 % of the time).
    """
    return obs.sensing_decision[obs.subband_of_rbg] == 0


def greedy_assign(
    obs: Observation,
    metric: Callable[[np.ndarray, int, np.ndarray], np.ndarray],
    *,
    permitted: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Sequentially assign each permitted RBG to the UE maximising ``metric``.

    Parameters
    ----------
    metric:
        Called as ``metric(residual_bits, rbg, eligible)`` and must return a
        per-UE score vector. ``residual_bits`` is the backlog still outstanding
        after the assignments made so far this slot, which is what makes every
        baseline buffer-aware.

    RBGs are visited in a random order drawn from the observation's slot index,
    so no scheduler gains an advantage from the fixed frequency ordering.
    """
    n_rbg = obs.n_rbg
    assignment = np.full(n_rbg, -1, dtype=np.int64)
    if permitted is None:
        permitted = usable_rbgs(obs)

    residual = obs.backlog_bits.astype(np.float64).copy()
    order = np.argsort(np.sin(np.arange(n_rbg) * (obs.slot + 1) * 0.7))

    for rbg in order:
        if not permitted[rbg]:
            continue
        eligible = residual > _EPS
        if not np.any(eligible):
            break
        scores = metric(residual, int(rbg), eligible)
        scores = np.where(eligible, scores, -np.inf)
        ue = int(np.argmax(scores))
        if not np.isfinite(scores[ue]):
            continue
        assignment[rbg] = ue
        residual[ue] -= obs.bits_per_rbg[ue, rbg]

    return assignment


# ---------------------------------------------------------------------------
# 1. Round Robin -- the "static / semi-static" allocator in the project brief.
# ---------------------------------------------------------------------------
class RoundRobin(Allocator):
    """Cyclic assignment, blind to channel quality and to QoS class."""

    name = "Round-Robin (static)"
    short_name = "RR"

    def reset(self, cfg: SimConfig) -> None:
        super().reset(cfg)
        self._pointer = 0

    def allocate(self, obs: Observation) -> np.ndarray:
        permitted = usable_rbgs(obs)
        assignment = np.full(obs.n_rbg, -1, dtype=np.int64)
        backlogged = np.flatnonzero(obs.backlog_bits > _EPS)
        if backlogged.size == 0:
            return assignment
        for rbg in np.flatnonzero(permitted):
            assignment[rbg] = int(backlogged[self._pointer % backlogged.size])
            self._pointer += 1
        return assignment


# ---------------------------------------------------------------------------
# 2. Max-C/I -- throughput-optimal, maximally unfair.
# ---------------------------------------------------------------------------
class MaxCI(Allocator):
    """Always serve the UE with the best instantaneous rate on each RBG."""

    name = "Max-C/I"
    short_name = "MaxCI"

    def allocate(self, obs: Observation) -> np.ndarray:
        def metric(residual, rbg, eligible):
            return obs.bits_per_rbg[:, rbg]

        return greedy_assign(obs, metric)


# ---------------------------------------------------------------------------
# 3. Proportional Fair -- the 3GPP workhorse.
# ---------------------------------------------------------------------------
class ProportionalFair(Allocator):
    """Maximise ``r_ik(t) / R_i(t)^alpha``."""

    name = "Proportional Fair"
    short_name = "PF"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def allocate(self, obs: Observation) -> np.ndarray:
        denom = np.maximum(obs.avg_throughput_bps, 1e3) ** self.alpha

        def metric(residual, rbg, eligible):
            return obs.bits_per_rbg[:, rbg] / denom

        return greedy_assign(obs, metric)


# ---------------------------------------------------------------------------
# 4. M-LWDF -- the deployed QoS-aware scheduler.
# ---------------------------------------------------------------------------
class MLWDF(Allocator):
    """
    Modified Largest Weighted Delay First (Andrews et al., 2001).

    Metric: ``-log(delta_i)/tau_i * W_i(t) * r_ik(t)/R_i(t)`` where ``delta_i``
    is the tolerable packet-drop rate, ``tau_i`` the delay budget and ``W_i``
    the head-of-line delay. This is the strongest classical comparator here: it
    already balances throughput, fairness and deadlines.
    """

    name = "M-LWDF"
    short_name = "MLWDF"

    def reset(self, cfg: SimConfig) -> None:
        super().reset(cfg)
        self._weights: Optional[np.ndarray] = None

    def allocate(self, obs: Observation) -> np.ndarray:
        if self._weights is None or self._weights.size != obs.n_ue:
            from .config import TRAFFIC_CLASSES

            delta = np.array([TRAFFIC_CLASSES[c].error_rate for c in obs.ue_class])
            self._weights = -np.log(delta) / np.maximum(obs.delay_budget_slots, 1.0)

        urgency = self._weights * np.maximum(obs.hol_delay_slots, 0.0)
        denom = np.maximum(obs.avg_throughput_bps, 1e3)

        def metric(residual, rbg, eligible):
            return urgency * obs.bits_per_rbg[:, rbg] / denom

        return greedy_assign(obs, metric)


# ---------------------------------------------------------------------------
# 5. EXP/PF -- exponential delay weighting for mixed real-time traffic.
# ---------------------------------------------------------------------------
class ExpPF(Allocator):
    """Rhee et al. (2003): exponentially amplify UEs whose delay exceeds the mean."""

    name = "EXP/PF"
    short_name = "EXP/PF"

    def reset(self, cfg: SimConfig) -> None:
        super().reset(cfg)
        self._weights: Optional[np.ndarray] = None

    def allocate(self, obs: Observation) -> np.ndarray:
        if self._weights is None or self._weights.size != obs.n_ue:
            from .config import TRAFFIC_CLASSES

            delta = np.array([TRAFFIC_CLASSES[c].error_rate for c in obs.ue_class])
            self._weights = -np.log(delta) / np.maximum(obs.delay_budget_slots, 1.0)

        aw = self._weights * np.maximum(obs.hol_delay_slots, 0.0)
        mean_aw = float(np.mean(aw))
        exponent = (aw - mean_aw) / (1.0 + np.sqrt(max(mean_aw, 0.0)))
        boost = np.exp(np.clip(exponent, -30.0, 30.0))
        denom = np.maximum(obs.avg_throughput_bps, 1e3)

        def metric(residual, rbg, eligible):
            return boost * obs.bits_per_rbg[:, rbg] / denom

        return greedy_assign(obs, metric)


CLASSICAL_ALLOCATORS = (RoundRobin, ProportionalFair, MaxCI, MLWDF, ExpPF)


def build_classical() -> list:
    """Fresh instances of every classical scheduler, in presentation order."""
    return [RoundRobin(), ProportionalFair(), MaxCI(), MLWDF(), ExpPF()]

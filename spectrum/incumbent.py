"""
Incumbent (primary user) occupancy and the imperfect energy-detection front-end.

This module is the single source of truth for spectrum-sharing uncertainty, and
it is deliberately the narrowest interface in the testbed:

* :attr:`IncumbentProcess.active` is the **ground truth**. It is consumed only by
  the physics in :mod:`spectrum.channel` and by the collision accounting in
  :mod:`spectrum.env`. No allocator ever receives it.
* :meth:`IncumbentProcess.sense` returns what a real cognitive radio actually
  gets: a hard binary decision from an energy detector operating at a fixed
  (P_d, P_fa) point, plus the underlying normalised test statistic.

Because the detector is memoryless but the occupancy process is not, there is
genuine, recoverable structure in the *history* of sensing reports. Exploiting
it is the learning problem that :mod:`spectrum.predictor` solves; the classical
baselines simply trust the instantaneous decision, which is exactly what
deployed CBRS/DSA equipment does.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

from .config import SimConfig


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"probability out of range: {p}")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > p_high:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


class IncumbentProcess:
    """
    Per-subband two-state semi-Markov occupancy with an energy-detection sensor.

    Sojourn times are geometric, which makes occupancy a first-order Markov
    chain whose transition probabilities are set by the configured mean ON/OFF
    durations and rescaled so that the steady-state duty cycle matches the
    congestion regime.
    """

    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.n_subbands = cfg.n_subbands

        duty = cfg.regime.incumbent_duty_cycle
        mean_on = float(cfg.incumbent_mean_on_slots)
        # Hold the mean burst length fixed and stretch the idle gaps so that
        # mean_on / (mean_on + mean_off) reproduces the regime's duty cycle.
        mean_off = mean_on * (1.0 - duty) / max(duty, 1e-6)

        #: P(idle -> busy) and P(busy -> idle) for the embedded Markov chain.
        self.p_on = 1.0 / mean_off
        self.p_off = 1.0 / mean_on
        self.duty_cycle = duty
        self.mean_on_slots = mean_on
        self.mean_off_slots = mean_off

        #: Ground-truth occupancy. Never exposed through the environment's
        #: observation; see ``spectrum.env.SpectrumEnv.observe``.
        self.active = (rng.random(self.n_subbands) < duty).astype(np.int64)

        #: Dwell-time distribution. ``geometric`` makes occupancy a first-order
        #: Markov chain, which is the family a two-state HMM filter assumes and
        #: can therefore fit exactly. The concentrated shapes break that
        #: assumption while holding the mean burst length and duty cycle fixed,
        #: which is how the benefit of a learned filter is isolated from the
        #: benefit of merely having memory.
        self.sojourn = getattr(cfg, "incumbent_sojourn", "geometric")
        self._remaining = np.ones(self.n_subbands, dtype=np.int64)
        if self.sojourn != "geometric":
            # Start mid-dwell so the first burst is not biased long.
            self._refresh_remaining(np.ones(self.n_subbands, dtype=bool))
            self._remaining = np.maximum(
                (rng.random(self.n_subbands) * self._remaining).astype(np.int64), 1
            )

        # Energy-detector calibration. Under H0 the normalised statistic is
        # N(0, 1); under H1 it is N(mu, 1). The threshold is fixed by the
        # required false-alarm rate and mu by the required detection rate.
        self.threshold = _norm_ppf(1.0 - cfg.sensing_prob_false_alarm)
        self.mu_h1 = self.threshold - _norm_ppf(1.0 - cfg.sensing_prob_detection)

    # ---------------------------------------------------------------- dynamics
    def _draw_sojourn(self, mean: float, n: int) -> np.ndarray:
        """
        Draw ``n`` dwell times with the configured shape and the given mean.

        ``geometric`` reproduces a memoryless chain exactly. The others keep the
        same mean but concentrate the distribution, which is what makes the
        process genuinely semi-Markov: the hazard of switching then depends on
        how long the state has already been held, and no two-state Markov chain
        can represent that.
        """
        shape = self.sojourn
        if shape == "geometric":
            return self.rng.geometric(1.0 / max(mean, 1.0), size=n).astype(np.int64)
        if shape == "deterministic":
            # +-10 % jitter around the mean: a radar-like near-periodic dwell.
            lo, hi = 0.9 * mean, 1.1 * mean
            return np.maximum(np.rint(self.rng.uniform(lo, hi, size=n)), 1).astype(np.int64)
        if shape == "erlang":
            # Sum of 4 exponentials: same mean, variance cut 4x vs geometric.
            k = 4
            draws = self.rng.gamma(shape=k, scale=mean / k, size=n)
            return np.maximum(np.rint(draws), 1).astype(np.int64)
        raise ValueError(
            f"unknown sojourn shape {shape!r}; expected 'geometric', "
            "'deterministic' or 'erlang'"
        )

    def _refresh_remaining(self, which: np.ndarray) -> None:
        """Re-draw dwell times for the subbands flagged in ``which``."""
        if not np.any(which):
            return
        idx = np.flatnonzero(which)
        busy = self.active[idx] == 1
        out = np.empty(idx.size, dtype=np.int64)
        if np.any(busy):
            out[busy] = self._draw_sojourn(self.mean_on_slots, int(np.sum(busy)))
        if np.any(~busy):
            out[~busy] = self._draw_sojourn(self.mean_off_slots, int(np.sum(~busy)))
        self._remaining[idx] = out

    def advance(self) -> None:
        """Evolve occupancy by one slot."""
        if self.sojourn == "geometric":
            # Memoryless: the explicit countdown is unnecessary and this keeps
            # the original code path bit-for-bit identical.
            u = self.rng.random(self.n_subbands)
            idle = self.active == 0
            turn_on = idle & (u < self.p_on)
            turn_off = (~idle) & (u < self.p_off)
            self.active = np.where(
                turn_on, 1, np.where(turn_off, 0, self.active)
            ).astype(np.int64)
            return

        self._remaining -= 1
        expired = self._remaining <= 0
        if np.any(expired):
            self.active = np.where(expired, 1 - self.active, self.active).astype(np.int64)
            self._refresh_remaining(expired)

    # ----------------------------------------------------------------- sensing
    def sense(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run one sensing sweep.

        Returns
        -------
        decision:
            Hard binary per-subband decision, with the configured P_d and P_fa.
        energy:
            The normalised energy statistic the decision was thresholded from.
            Supplying it costs the detector nothing and is strictly more
            informative than the hard bit, so it is given to *every* allocator.
        """
        noise = self.rng.standard_normal(self.n_subbands)
        energy = noise + self.mu_h1 * self.active
        decision = (energy > self.threshold).astype(np.int64)
        return decision, energy

    # -------------------------------------------------------------- diagnostics
    def transition_matrix(self) -> np.ndarray:
        """The 2x2 chain ``[[P(0->0), P(0->1)], [P(1->0), P(1->1)]]``."""
        return np.array(
            [[1.0 - self.p_on, self.p_on], [self.p_off, 1.0 - self.p_off]], dtype=np.float64
        )

    def describe(self) -> Dict[str, float]:
        return {
            "duty_cycle": self.duty_cycle,
            "mean_on_slots": self.mean_on_slots,
            "mean_off_slots": self.mean_off_slots,
            "p_idle_to_busy": self.p_on,
            "p_busy_to_idle": self.p_off,
            "detector_threshold": self.threshold,
            "detector_mu_h1": self.mu_h1,
            "prob_detection": self.cfg.sensing_prob_detection,
            "prob_false_alarm": self.cfg.sensing_prob_false_alarm,
        }

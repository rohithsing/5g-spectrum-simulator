"""
Learned incumbent-occupancy estimation.

A single energy-detection sweep is a memoryless measurement of a *correlated*
process: incumbent bursts last tens of slots, so the detector looks at the same
hidden state many times in a row. A classical DSA radio throws that away and
acts on the latest hard bit, which is why it both collides (one miss in seven)
and wastes spectrum (one false alarm in eight).

This module recovers the discarded information three ways:

``EnergyDetector``
    The classical reference: threshold the current statistic. No memory.
``BayesFilter``
    The exact forward algorithm for the two-state Markov chain driving the
    incumbent, given the true transition probabilities and detector operating
    point. This is the information-theoretic ceiling for any estimator that
    sees only the sensing stream.
``GRUPredictor``
    A recurrent network trained purely on sensing traces. It is given neither
    the transition matrix nor the detector calibration and has to infer both
    from data, which is the realistic operating assumption; ``BayesFilter``
    exists so we can report how much of the achievable gain it captures.

Every estimator consumes exactly the same input -- the sensing stream in
``Observation`` -- so using one instead of another cannot break information
parity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Classical reference
# ---------------------------------------------------------------------------
class EnergyDetector:
    """Hard-decision energy detection -- what deployed equipment does."""

    name = "Energy detector (classical)"

    def __init__(self, threshold: float):
        self.threshold = threshold

    def reset(self, n_subbands: int) -> None:
        pass

    def probability(self, decision: np.ndarray, energy: np.ndarray) -> np.ndarray:
        # The hard decision *is* the estimate; it carries no confidence.
        return decision.astype(np.float64)


# ---------------------------------------------------------------------------
# Bayes-optimal reference
# ---------------------------------------------------------------------------
class BayesFilter:
    """
    Exact HMM forward filter over the incumbent's two-state chain.

    Requires the ground-truth model parameters, so it is an upper bound rather
    than a deployable estimator.
    """

    name = "Bayes-optimal filter"

    def __init__(self, p_on: float, p_off: float, mu_h1: float, duty: float):
        self.transition = np.array([[1.0 - p_on, p_on], [p_off, 1.0 - p_off]])
        self.mu_h1 = mu_h1
        self.duty = duty
        self.belief = np.zeros(0)

    def reset(self, n_subbands: int) -> None:
        self.belief = np.full(n_subbands, self.duty)

    def probability(self, decision: np.ndarray, energy: np.ndarray) -> np.ndarray:
        if self.belief.size != energy.size:
            self.reset(energy.size)
        # Predict: propagate the prior through the chain.
        prior = self.belief * self.transition[1, 1] + (1.0 - self.belief) * self.transition[0, 1]
        # Update: Gaussian likelihoods N(0,1) under H0 and N(mu,1) under H1.
        llr = self.mu_h1 * energy - 0.5 * self.mu_h1 ** 2
        odds = (prior / np.maximum(1.0 - prior, 1e-12)) * np.exp(np.clip(llr, -30.0, 30.0))
        self.belief = np.clip(odds / (1.0 + odds), 1e-6, 1.0 - 1e-6)
        return self.belief.copy()


# ---------------------------------------------------------------------------
# Fitted parametric references -- the baselines a classical engineer builds
# ---------------------------------------------------------------------------
class GaussianHMMFilter:
    """
    Two-state Gaussian-emission HMM forward filter with **estimated** parameters.

    This is the baseline that makes the learned filter's result meaningful.
    :class:`EnergyDetector` is memoryless and the occupancy process is not, so a
    recurrent model beating it is close to a foregone conclusion and says little.
    :class:`BayesFilter` is handed the true parameters, so it bounds performance
    but is not deployable. The honest comparator sits between them: the *same*
    model structure, with every parameter fitted from data, which is what an
    engineer without a neural network would actually deploy.

    Two fitting routines are provided, and which one is fair depends on what the
    learned model is given:

    ``fit_hmm_supervised``
        Counts transitions and fits per-class Gaussians from labelled traces.
        This has **exactly** the information the GRU has -- same traces, same
        labels -- so it is the strict apples-to-apples comparison.
    ``fit_hmm_em``
        Baum-Welch on the sensing stream alone, with no labels at all. Strictly
        less information than the GRU gets, so beating it is a weaker claim, but
        it is what you deploy when occupancy was never ground-truthed.

    Both are evaluated on held-out traces, exactly like the GRU.
    """

    def __init__(
        self,
        p_on: float,
        p_off: float,
        mu0: float,
        sd0: float,
        mu1: float,
        sd1: float,
        duty: float,
        name: str = "Fitted HMM",
    ):
        self.transition = np.array([[1.0 - p_on, p_on], [p_off, 1.0 - p_off]])
        self.mu0, self.sd0 = float(mu0), max(float(sd0), 1e-3)
        self.mu1, self.sd1 = float(mu1), max(float(sd1), 1e-3)
        self.duty = float(np.clip(duty, 1e-4, 1.0 - 1e-4))
        self.name = name
        self.belief = np.zeros(0)

    # -- parameters as a plain dict, so a fit can be serialised and reused ----
    def params(self) -> Dict[str, float]:
        return {
            "p_on": float(self.transition[0, 1]),
            "p_off": float(self.transition[1, 0]),
            "mu0": self.mu0, "sd0": self.sd0,
            "mu1": self.mu1, "sd1": self.sd1,
            "duty": self.duty,
        }

    @classmethod
    def from_params(cls, p: Dict[str, float], name: str = "Fitted HMM") -> "GaussianHMMFilter":
        return cls(p["p_on"], p["p_off"], p["mu0"], p["sd0"],
                   p["mu1"], p["sd1"], p["duty"], name=name)

    def reset(self, n_subbands: int) -> None:
        self.belief = np.full(n_subbands, self.duty)

    def probability(self, decision: np.ndarray, energy: np.ndarray) -> np.ndarray:
        """Advance the filter one slot; identical interface to every estimator."""
        if self.belief.size != energy.size:
            self.reset(energy.size)
        prior = (
            self.belief * self.transition[1, 1]
            + (1.0 - self.belief) * self.transition[0, 1]
        )
        # Log-likelihood ratio of the two fitted Gaussians (unequal variances
        # allowed, since EM has no reason to return equal ones).
        ll1 = -0.5 * ((energy - self.mu1) / self.sd1) ** 2 - np.log(self.sd1)
        ll0 = -0.5 * ((energy - self.mu0) / self.sd0) ** 2 - np.log(self.sd0)
        odds = (prior / np.maximum(1.0 - prior, 1e-12)) * np.exp(
            np.clip(ll1 - ll0, -30.0, 30.0)
        )
        self.belief = np.clip(odds / (1.0 + odds), 1e-6, 1.0 - 1e-6)
        return self.belief.copy()


def fit_hmm_supervised(energy: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    """
    Fit the HMM by counting, from labelled sequences.

    ``energy`` and ``truth`` are both ``(N, T)``. Transitions are counted within
    sequences only. This estimator sees precisely what the GRU's supervised loss
    sees, which is what makes the resulting comparison fair.
    """
    energy = np.asarray(energy, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.int64)

    a, b = truth[:, :-1], truth[:, 1:]
    n01 = int(np.sum((a == 0) & (b == 1)))
    n00 = int(np.sum((a == 0) & (b == 0)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n11 = int(np.sum((a == 1) & (b == 1)))

    idle, busy = truth == 0, truth == 1
    return {
        "p_on": n01 / max(n01 + n00, 1),
        "p_off": n10 / max(n10 + n11, 1),
        "mu0": float(energy[idle].mean()) if idle.any() else 0.0,
        "sd0": float(energy[idle].std()) if idle.any() else 1.0,
        "mu1": float(energy[busy].mean()) if busy.any() else 1.0,
        "sd1": float(energy[busy].std()) if busy.any() else 1.0,
        "duty": float(busy.mean()),
    }


def fit_hmm_em(
    energy: np.ndarray,
    *,
    n_iter: int = 60,
    tol: float = 1e-6,
    seed: int = 0,
    verbose: bool = False,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Baum-Welch (EM) fit of the two-state Gaussian HMM, from the sensing stream only.

    No occupancy labels are used anywhere in here. Returns the fitted parameters
    and a small diagnostic record (final log-likelihood, iterations, whether it
    converged).

    Initialisation is deliberately uninformed -- a median split of the energies
    for the means, unit variances, and a generic persistent transition matrix --
    so the fit cannot be accused of being handed the answer. The state with the
    larger fitted mean is identified as "busy", which is the only labelling
    convention the model needs.
    """
    x = np.asarray(energy, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    n_seq, T = x.shape

    med = np.median(x)
    mu = np.array([x[x <= med].mean(), x[x > med].mean()], dtype=np.float64)
    sd = np.array([1.0, 1.0], dtype=np.float64)
    A = np.array([[0.95, 0.05], [0.10, 0.90]], dtype=np.float64)
    pi = np.array([0.8, 0.2], dtype=np.float64)

    prev_ll = -np.inf
    converged = False
    it = 0
    for it in range(1, n_iter + 1):
        # -- E step: scaled forward-backward over every sequence -------------
        g_sum = np.zeros(2)            # sum of gamma
        g_sum_t0 = np.zeros(2)         # gamma at t=0, for pi
        xi_sum = np.zeros((2, 2))
        w_sum = np.zeros(2)            # sum of gamma, for the emission updates
        wx_sum = np.zeros(2)
        wxx_sum = np.zeros(2)
        ll = 0.0

        for s in range(n_seq):
            obs = x[s]
            # Emission matrix B[k, t]
            z = (obs[None, :] - mu[:, None]) / sd[:, None]
            B = np.exp(-0.5 * z ** 2) / (sd[:, None] * np.sqrt(2.0 * np.pi))
            B = np.maximum(B, 1e-300)

            alpha = np.zeros((T, 2))
            c = np.zeros(T)
            alpha[0] = pi * B[:, 0]
            c[0] = alpha[0].sum()
            alpha[0] /= c[0]
            for t in range(1, T):
                alpha[t] = (alpha[t - 1] @ A) * B[:, t]
                c[t] = alpha[t].sum()
                alpha[t] /= c[t]
            ll += float(np.sum(np.log(c)))

            beta = np.zeros((T, 2))
            beta[T - 1] = 1.0
            for t in range(T - 2, -1, -1):
                beta[t] = (A @ (B[:, t + 1] * beta[t + 1])) / c[t + 1]

            gamma = alpha * beta
            gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

            for t in range(T - 1):
                num = (
                    alpha[t][:, None]
                    * A
                    * (B[:, t + 1] * beta[t + 1])[None, :]
                    / c[t + 1]
                )
                xi_sum += num

            g_sum += gamma.sum(axis=0)
            g_sum_t0 += gamma[0]
            w_sum += gamma.sum(axis=0)
            wx_sum += gamma.T @ obs
            wxx_sum += gamma.T @ (obs ** 2)

        # -- M step -----------------------------------------------------------
        pi = g_sum_t0 / max(n_seq, 1)
        pi = np.clip(pi, 1e-6, 1.0)
        pi /= pi.sum()

        A = xi_sum / np.maximum(xi_sum.sum(axis=1, keepdims=True), 1e-300)
        A = np.clip(A, 1e-6, 1.0 - 1e-6)
        A /= A.sum(axis=1, keepdims=True)

        mu = wx_sum / np.maximum(w_sum, 1e-300)
        var = wxx_sum / np.maximum(w_sum, 1e-300) - mu ** 2
        sd = np.sqrt(np.maximum(var, 1e-6))

        if verbose:
            print(f"    EM iter {it:3d}  logL {ll:.2f}")
        if ll - prev_ll < tol * max(abs(prev_ll), 1.0):
            converged = True
            prev_ll = ll
            break
        prev_ll = ll

    # Identify the busy state as the one with the larger mean.
    busy = int(np.argmax(mu))
    idle = 1 - busy
    p_on = float(A[idle, busy])
    p_off = float(A[busy, idle])
    stationary = p_on / max(p_on + p_off, 1e-12)

    params = {
        "p_on": p_on, "p_off": p_off,
        "mu0": float(mu[idle]), "sd0": float(sd[idle]),
        "mu1": float(mu[busy]), "sd1": float(sd[busy]),
        "duty": float(stationary),
    }
    info = {"log_likelihood": float(prev_ll), "iterations": it,
            "converged": float(converged)}
    return params, info


class MultiStateHMMFilter:
    """
    EM-fitted Gaussian HMM with ``K > 2`` hidden states, filtered online.

    This is the baseline that closes the obvious objection to the two-state
    comparison. A two-state Markov chain can only produce geometric dwell times,
    so once the incumbent's bursts are concentrated the two-state filter is
    misspecified *by construction*. The textbook classical remedy is not a
    neural network, it is more states: a chain of ``k`` sub-states with
    exponential dwell yields an Erlang-``k`` sojourn, and a free ``K``-state
    transition matrix can approximate a wide family of duration distributions.

    Emissions stay Gaussian per state, and a state is labelled *busy* when its
    fitted mean falls above the midpoint of the fitted means -- the two
    hypotheses are far apart in energy (roughly 0 versus 2.2), so this grouping
    is unambiguous. ``P(busy)`` is the posterior mass on the busy group.
    """

    def __init__(self, A, mu, sd, pi, busy_mask, name="Fitted HMM (K states)"):
        self.A = np.asarray(A, dtype=np.float64)
        self.mu = np.asarray(mu, dtype=np.float64)
        self.sd = np.maximum(np.asarray(sd, dtype=np.float64), 1e-3)
        self.pi = np.asarray(pi, dtype=np.float64)
        self.busy = np.asarray(busy_mask, dtype=bool)
        self.name = name
        self._b = None

    def n_states(self) -> int:
        return int(self.mu.size)

    def params(self) -> Dict[str, object]:
        return {"A": self.A.tolist(), "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                "pi": self.pi.tolist(), "busy_mask": self.busy.tolist()}

    @classmethod
    def from_params(cls, p: Dict[str, object], name="Fitted HMM (K states)"):
        return cls(p["A"], p["mu"], p["sd"], p["pi"], p["busy_mask"], name=name)

    def reset(self, n_subbands: int) -> None:
        self._b = np.tile(self.pi, (n_subbands, 1))

    def probability(self, decision: np.ndarray, energy: np.ndarray) -> np.ndarray:
        energy = np.asarray(energy, dtype=np.float64)
        if self._b is None or self._b.shape[0] != energy.size:
            self.reset(energy.size)
        # Predict, then update with the per-state Gaussian likelihood.
        b = self._b @ self.A
        z = (energy[:, None] - self.mu[None, :]) / self.sd[None, :]
        lik = np.exp(-0.5 * z ** 2) / self.sd[None, :]
        b = b * np.maximum(lik, 1e-300)
        b /= np.maximum(b.sum(axis=1, keepdims=True), 1e-300)
        self._b = b
        return b[:, self.busy].sum(axis=1)


def fit_hmm_em_k(
    energy: np.ndarray,
    n_states: int,
    *,
    n_iter: int = 80,
    tol: float = 1e-7,
    seed: int = 0,
    verbose: bool = False,
) -> Tuple[MultiStateHMMFilter, Dict[str, float]]:
    """
    Baum-Welch for a general ``n_states`` Gaussian HMM, from energies alone.

    Means are initialised at quantiles of the data so the states start spread
    out; the transition matrix starts strongly diagonal, which is the right
    prior for a process with long dwell times and keeps EM away from the
    degenerate all-states-identical fixed point.
    """
    x = np.asarray(energy, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    n_seq, T = x.shape
    K = int(n_states)

    qs = np.linspace(5.0, 95.0, K)
    mu = np.percentile(x, qs)
    mu = mu + 1e-3 * np.arange(K)  # break exact ties
    sd = np.full(K, max(float(x.std()) * 0.5, 1e-2))
    A = np.full((K, K), 0.02 / max(K - 1, 1)) + np.eye(K) * 0.98
    A /= A.sum(axis=1, keepdims=True)
    pi = np.full(K, 1.0 / K)

    prev_ll, converged, it = -np.inf, False, 0
    for it in range(1, n_iter + 1):
        g_t0 = np.zeros(K)
        xi_sum = np.zeros((K, K))
        w = np.zeros(K)
        wx = np.zeros(K)
        wxx = np.zeros(K)
        ll = 0.0

        for s in range(n_seq):
            obs = x[s]
            z = (obs[None, :] - mu[:, None]) / sd[:, None]
            B = np.exp(-0.5 * z ** 2) / (sd[:, None] * np.sqrt(2.0 * np.pi))
            B = np.maximum(B, 1e-300)

            alpha = np.zeros((T, K)); c = np.zeros(T)
            alpha[0] = pi * B[:, 0]
            c[0] = alpha[0].sum(); alpha[0] /= c[0]
            for t in range(1, T):
                alpha[t] = (alpha[t - 1] @ A) * B[:, t]
                c[t] = alpha[t].sum(); alpha[t] /= c[t]
            ll += float(np.sum(np.log(c)))

            beta = np.zeros((T, K)); beta[T - 1] = 1.0
            for t in range(T - 2, -1, -1):
                beta[t] = (A @ (B[:, t + 1] * beta[t + 1])) / c[t + 1]

            gamma = alpha * beta
            gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

            # Vectorised xi accumulation over t.
            Bb = (B[:, 1:] * beta[1:].T) / c[1:]           # (K, T-1)
            xi_sum += A * (alpha[:-1].T @ Bb.T)

            g_t0 += gamma[0]
            w += gamma.sum(axis=0)
            wx += gamma.T @ obs
            wxx += gamma.T @ (obs ** 2)

        pi = np.clip(g_t0 / max(n_seq, 1), 1e-8, 1.0); pi /= pi.sum()
        A = xi_sum / np.maximum(xi_sum.sum(axis=1, keepdims=True), 1e-300)
        A = np.clip(A, 1e-8, 1.0); A /= A.sum(axis=1, keepdims=True)
        mu = wx / np.maximum(w, 1e-300)
        sd = np.sqrt(np.maximum(wxx / np.maximum(w, 1e-300) - mu ** 2, 1e-6))

        if verbose:
            print(f"    EM(K={K}) iter {it:3d}  logL {ll:.2f}")
        if ll - prev_ll < tol * max(abs(prev_ll), 1.0):
            converged = True; prev_ll = ll; break
        prev_ll = ll

    midpoint = 0.5 * (mu.min() + mu.max())
    busy = mu > midpoint
    if not busy.any() or busy.all():       # degenerate fit; fall back to a split
        busy = mu >= np.median(mu)
    filt = MultiStateHMMFilter(A, mu, sd, pi, busy,
                               name=f"HMM, EM-fitted, {K} states")
    # With K > 2 on a two-hypothesis process, EM splits each emission into
    # several components, and grouping them by a midpoint on the fitted means
    # can put a component of the idle distribution in the busy group. Callers
    # with labels should re-group the states with assign_busy_states(), which
    # makes this baseline as strong as it can be rather than handicapping it.
    info = {"log_likelihood": float(prev_ll), "iterations": it,
            "converged": float(converged), "n_states": float(K),
            "n_busy_states": float(busy.sum())}
    return filt, info


def assign_busy_states(
    filt: MultiStateHMMFilter, energy: np.ndarray, truth: np.ndarray
) -> MultiStateHMMFilter:
    """
    Re-group a ``K``-state filter's states into idle/busy using labels.

    Only the *grouping* uses supervision; the dynamics and emissions stay as EM
    fitted them. The GRU is trained on these same labels, so letting the
    parametric baseline use them for this one discrete choice keeps the
    comparison honest in the baseline's favour. For each state, the filtered
    posterior over the training traces is used as a weight on the occupancy
    label, and a state joins the busy group when that weighted mean exceeds 0.5.
    """
    x = np.asarray(energy, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    if x.ndim == 1:
        x, y = x[None, :], y[None, :]

    K = filt.n_states()
    num = np.zeros(K)
    den = np.zeros(K)
    for s in range(x.shape[0]):
        filt.reset(1)
        for t in range(x.shape[1]):
            e = x[s, t: t + 1]
            filt.probability(np.zeros(1), e)
            b = filt._b[0]
            num += b * y[s, t]
            den += b

    frac = num / np.maximum(den, 1e-12)
    busy = frac > 0.5
    if not busy.any() or busy.all():
        busy = frac >= np.median(frac)
    filt.busy = busy
    filt._b = None
    return filt


def fit_hmm_baselines(
    dataset: "SensingDataset", *, em_iters: int = 60, verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Fit both parametric baselines on the same traces the GRU trains on.

    ``dataset.features[..., 1]`` is the normalised energy statistic and
    ``dataset.labels`` the occupancy. The supervised fit uses both; the EM fit
    uses only the energies.
    """
    energy = np.asarray(dataset.features[..., 1], dtype=np.float64)
    truth = np.asarray(dataset.labels, dtype=np.int64)

    sup = fit_hmm_supervised(energy, truth)
    if verbose:
        print(f"  supervised HMM: p_on {sup['p_on']:.4f} p_off {sup['p_off']:.4f} "
              f"mu0 {sup['mu0']:+.3f} mu1 {sup['mu1']:+.3f}")

    em, info = fit_hmm_em(energy, n_iter=em_iters)
    if verbose:
        print(f"  EM (Baum-Welch) HMM: p_on {em['p_on']:.4f} p_off {em['p_off']:.4f} "
              f"mu0 {em['mu0']:+.3f} mu1 {em['mu1']:+.3f} "
              f"(logL {info['log_likelihood']:.0f}, {info['iterations']} iters, "
              f"converged {bool(info['converged'])})")

    return {"supervised": sup, "em": em, "em_info": info}


# ---------------------------------------------------------------------------
# Learned estimator
# ---------------------------------------------------------------------------
class GRUPredictor(nn.Module):
    """
    Recurrent occupancy estimator shared across subbands.

    Each subband is an independent realisation of the same process, so one set
    of weights is applied to every subband's stream in parallel. Input per step
    is the pair (hard decision, normalised energy statistic).
    """

    name = "Learned GRU predictor"

    def __init__(self, hidden: int = 32, layers: int = 1):
        super().__init__()
        self.hidden_size = hidden
        self.gru = nn.GRU(input_size=2, hidden_size=hidden, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
        self._state: Optional[torch.Tensor] = None

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """``seq`` is ``(B, T, 2)``; returns per-step logits ``(B, T)``."""
        out, _ = self.gru(seq)
        return self.head(out).squeeze(-1)

    # ---------------------------------------------------------- online filter
    def reset(self, n_subbands: int) -> None:
        self._state = torch.zeros(
            self.gru.num_layers, n_subbands, self.hidden_size, dtype=torch.float32
        )

    @torch.no_grad()
    def probability(self, decision: np.ndarray, energy: np.ndarray) -> np.ndarray:
        """Advance the filter one slot and return P(incumbent active) per subband."""
        n = energy.size
        if self._state is None or self._state.shape[1] != n:
            self.reset(n)
        step = torch.tensor(
            np.stack([decision, energy], axis=-1)[:, None, :], dtype=torch.float32
        )
        out, self._state = self.gru(step, self._state)
        logit = self.head(out).squeeze(-1).squeeze(-1)
        return torch.sigmoid(logit).numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Online use
# ---------------------------------------------------------------------------
class OccupancyEstimator:
    """
    Slot-cached front-end for an occupancy estimator.

    :class:`GRUPredictor` is a *filter*: every call to ``probability`` advances
    its hidden state by one step. Several components may want the estimate for
    the same slot -- the feature extractor and the scheduler itself, say -- and
    calling the predictor twice would advance the filter twice and feed it each
    observation more than once. This wrapper makes repeated queries within a
    slot free and exact.

    With ``predictor=None`` it degrades to the classical behaviour of trusting
    the hard sensing decision, which is what the ablations need.
    """

    def __init__(self, predictor: Optional[GRUPredictor]):
        self.predictor = predictor
        self._slot: Optional[int] = None
        self._value: Optional[np.ndarray] = None

    def reset(self, n_subbands: int) -> None:
        if self.predictor is not None:
            self.predictor.reset(n_subbands)
        self._slot = None
        self._value = None

    def __call__(self, obs) -> np.ndarray:
        if self._slot == obs.slot and self._value is not None:
            return self._value
        if self.predictor is None:
            value = obs.sensing_decision.astype(np.float64)
        else:
            value = self.predictor.probability(obs.sensing_decision, obs.sensing_energy)
        self._slot = obs.slot
        self._value = value
        return value


# ---------------------------------------------------------------------------
# Trace collection and training
# ---------------------------------------------------------------------------
@dataclass
class SensingDataset:
    """Sensing streams paired with the occupancy that was revealed afterwards."""

    features: np.ndarray  # (N, T, 2)
    labels: np.ndarray    # (N, T)

    def __len__(self) -> int:
        return self.features.shape[0]


def collect_sensing_traces(
    congestion_levels, seeds, slots: int = 2000, seq_len: int = 128
) -> SensingDataset:
    """
    Generate supervision for the predictor by running the incumbent process.

    No scheduler is involved: occupancy evolves independently of what the cell
    transmits, so the traces can be produced far faster than full episodes.
    """
    from .config import SimConfig
    from .incumbent import IncumbentProcess

    feats, labels = [], []
    for level in congestion_levels:
        for seed in seeds:
            cfg = SimConfig(congestion=level, seed=seed)
            rng = np.random.default_rng(seed * 7919 + 13)
            inc = IncumbentProcess(cfg, rng)

            dec_t, en_t, truth_t = [], [], []
            for _ in range(slots):
                d, e = inc.sense()
                dec_t.append(d)
                en_t.append(e)
                truth_t.append(inc.active.copy())
                inc.advance()

            dec = np.array(dec_t)      # (slots, n_subbands)
            en = np.array(en_t)
            truth = np.array(truth_t)

            # Split each subband's stream into fixed-length training sequences.
            n_seq = slots // seq_len
            for sb in range(cfg.n_subbands):
                for i in range(n_seq):
                    sl = slice(i * seq_len, (i + 1) * seq_len)
                    feats.append(np.stack([dec[sl, sb], en[sl, sb]], axis=-1))
                    labels.append(truth[sl, sb])

    return SensingDataset(
        features=np.asarray(feats, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.float32),
    )


def train_predictor(
    dataset: SensingDataset,
    *,
    hidden: int = 32,
    epochs: int = 12,
    batch_size: int = 256,
    lr: float = 3e-3,
    burn_in: int = 16,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[GRUPredictor, Dict[str, float]]:
    """
    Fit the GRU with binary cross-entropy on the occupancy label.

    The first ``burn_in`` steps of every sequence are excluded from the loss:
    the filter cannot be expected to know the state before it has observed
    anything, and including them would reward memorising the duty cycle.
    """
    torch.manual_seed(seed)
    model = GRUPredictor(hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()

    x = torch.tensor(dataset.features)
    y = torch.tensor(dataset.labels)
    n = x.shape[0]
    n_val = max(int(0.2 * n), 1)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    history: Dict[str, float] = {}
    for epoch in range(epochs):
        model.train()
        order = train_idx[torch.randperm(train_idx.numel(), generator=g)]
        total = 0.0
        for i in range(0, order.numel(), batch_size):
            batch = order[i : i + batch_size]
            logits = model(x[batch])
            loss = lossf(logits[:, burn_in:], y[batch][:, burn_in:])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach()) * batch.numel()
        train_loss = total / max(order.numel(), 1)

        model.eval()
        with torch.no_grad():
            vl = float(lossf(model(x[val_idx])[:, burn_in:], y[val_idx][:, burn_in:]))
        history[f"epoch_{epoch}"] = vl
        if verbose:
            print(f"  epoch {epoch + 1:2d}/{epochs}  train BCE {train_loss:.4f}  val BCE {vl:.4f}")

    return model, history


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def classification_report(
    truth: np.ndarray, prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Accuracy / precision / recall / F1 / AUC / Brier for a binary estimator."""
    truth = np.asarray(truth).ravel().astype(int)
    prob = np.asarray(prob).ravel().astype(float)
    pred = (prob >= threshold).astype(int)

    tp = int(np.sum((pred == 1) & (truth == 1)))
    fp = int(np.sum((pred == 1) & (truth == 0)))
    fn = int(np.sum((pred == 0) & (truth == 1)))
    tn = int(np.sum((pred == 0) & (truth == 0)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "accuracy": (tp + tn) / max(truth.size, 1),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "auc": _roc_auc(truth, prob),
        "brier": float(np.mean((prob - truth) ** 2)),
        "false_alarm_rate": fp / max(fp + tn, 1),
        "miss_rate": fn / max(fn + tp, 1),
        "n_samples": int(truth.size),
    }


def _roc_auc(truth: np.ndarray, score: np.ndarray) -> float:
    """AUC via the rank-sum identity (handles ties)."""
    pos, neg = int(np.sum(truth == 1)), int(np.sum(truth == 0))
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(score.size, dtype=np.float64)
    ranks[order] = np.arange(1, score.size + 1)
    # Average ranks within tied groups.
    s_sorted = score[order]
    i = 0
    while i < s_sorted.size:
        j = i
        while j + 1 < s_sorted.size and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((np.sum(ranks[truth == 1]) - pos * (pos + 1) / 2.0) / (pos * neg))

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

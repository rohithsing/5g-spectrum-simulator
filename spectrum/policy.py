"""
The learned scheduler: feature extraction, policy network and action decoder.

Design notes
------------
*Permutation equivariance.* UEs carry no meaningful index, so the network
encodes each UE independently and then mixes them with self-attention. The same
weights therefore work for a 20-UE cell and an 80-UE cell, which is what lets a
single policy be evaluated across every congestion regime.

*Factorised logits.* Scoring all ``n_ue x n_rbg`` pairs with a joint network
would be expensive. Instead each UE emits a priority ``score_u`` and a channel
sensitivity ``gain_u``, and the logit for putting UE ``u`` on RBG ``k`` is
``score_u + gain_u * z_uk`` where ``z_uk`` is its normalised log transport-block
size. This is rank-2 in the pair space but still lets different UEs prefer
different parts of the band, which is the part that matters.

*Sequential decode with a residual mask.* RBGs are assigned one at a time; once
a UE's queue is covered it leaves the candidate set. Over-allocation is
therefore structurally impossible -- exactly as in the classical baselines'
``greedy_assign``, so this is not an advantage over them.

*Explicit mute action.* Each RBG may be left silent. The mute logit is driven by
the predicted incumbent probability, which is where the learned occupancy
estimator enters the scheduling decision.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import Observation

#: Largest cell the padded tensors must accommodate (the "peak" regime).
MAX_UE = 80

#: Number of per-UE input features produced by :func:`extract_features`.
N_UE_FEATURES = 18
#: Number of per-RBG input features.
N_RBG_FEATURES = 6
#: Number of cell-level scalars.
N_GLOBAL_FEATURES = 6


def rbg_order(slot: int, n_rbg: int) -> np.ndarray:
    """
    Visit order for sequential assignment, shared with the classical baselines.

    Deterministic given the slot, but decorrelated from frequency index so that
    no scheduler benefits from always starting at RBG 0.
    """
    return np.argsort(np.sin(np.arange(n_rbg) * (slot + 1) * 0.7))


def extract_features(
    obs: Observation, p_busy: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Turn an :class:`Observation` plus the predictor's output into padded tensors.

    ``p_busy`` is the per-subband incumbent probability from
    :mod:`spectrum.predictor`. It is derived purely from the sensing stream that
    is already inside ``obs``, so no new information enters here.
    """
    n_ue, n_rbg = obs.n_ue, obs.n_rbg

    # -- normalised per-(UE, RBG) channel quality --------------------------
    log_bits = np.log1p(obs.bits_per_rbg)
    z = (log_bits - log_bits.mean()) / (log_bits.std() + 1e-6)

    budget = np.maximum(obs.delay_budget_slots, 1.0)
    urgency = obs.hol_delay_slots / budget
    slack = obs.deadline_slack_slots / budget
    has_backlog = (obs.backlog_bits > 0.0).astype(np.float64)

    best_bits = obs.bits_per_rbg.max(axis=1) + 1e-9
    rbgs_needed = np.clip(obs.backlog_bits / best_bits, 0.0, n_rbg) / n_rbg

    onehot = np.zeros((n_ue, 4))
    onehot[np.arange(n_ue), np.clip(obs.ue_class, 0, 3)] = 1.0

    ue = np.column_stack(
        [
            np.log1p(obs.backlog_bits) / 15.0,
            np.clip(urgency, 0.0, 3.0),
            np.clip(slack, -2.0, 2.0),
            np.log1p(budget) / 7.0,
            np.log1p(obs.nominal_rate_bps) / 20.0,
            obs.priority_level / 100.0,
            onehot,
            np.log1p(obs.avg_throughput_bps) / 20.0,
            obs.last_slot_rbgs / n_rbg,
            z.mean(axis=1),
            z.max(axis=1),
            z.std(axis=1),
            rbgs_needed,
            has_backlog,
            np.clip(urgency, 0.0, 3.0) * has_backlog,
        ]
    )
    assert ue.shape[1] == N_UE_FEATURES, f"expected {N_UE_FEATURES} UE features, got {ue.shape[1]}"

    # -- per-RBG features ---------------------------------------------------
    sb = obs.subband_of_rbg
    rbg = np.column_stack(
        [
            p_busy[sb],
            obs.sensing_decision[sb].astype(np.float64),
            np.clip(obs.sensing_energy[sb] / 3.0, -3.0, 3.0),
            z[:n_ue].mean(axis=0),
            z[:n_ue].max(axis=0),
            np.full(n_rbg, n_ue / MAX_UE),
        ]
    )
    assert rbg.shape[1] == N_RBG_FEATURES

    # -- cell-level scalars -------------------------------------------------
    glob = np.array(
        [
            n_ue / MAX_UE,
            float(np.mean(has_backlog)),
            float(np.mean(urgency[has_backlog > 0])) if np.any(has_backlog > 0) else 0.0,
            float(np.mean(p_busy)),
            float(np.log1p(obs.backlog_bits.sum()) / 20.0),
            float(np.mean(obs.last_slot_rbgs) * n_ue / n_rbg),
        ]
    )
    assert glob.size == N_GLOBAL_FEATURES

    # -- pad to MAX_UE ------------------------------------------------------
    ue_pad = np.zeros((MAX_UE, N_UE_FEATURES))
    ue_pad[:n_ue] = ue
    z_pad = np.zeros((MAX_UE, n_rbg))
    z_pad[:n_ue] = z
    bits_pad = np.zeros((MAX_UE, n_rbg))
    bits_pad[:n_ue] = obs.bits_per_rbg
    backlog_pad = np.zeros(MAX_UE)
    backlog_pad[:n_ue] = obs.backlog_bits
    valid = np.zeros(MAX_UE, dtype=bool)
    valid[:n_ue] = True

    return {
        "ue": ue_pad.astype(np.float32),
        "rbg": rbg.astype(np.float32),
        "global": glob.astype(np.float32),
        "z": z_pad.astype(np.float32),
        "bits": bits_pad,
        "backlog": backlog_pad,
        "valid": valid,
        "n_ue": n_ue,
        "slot": obs.slot,
    }


class SchedulerPolicy(nn.Module):
    """Attention encoder over UEs with factorised assignment logits and a value head."""

    def __init__(self, hidden: int = 64, n_heads: int = 4):
        super().__init__()
        self.hidden = hidden

        self.ue_encoder = nn.Sequential(
            nn.Linear(N_UE_FEATURES, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.ReLU(), nn.Linear(hidden * 2, hidden))
        self.norm2 = nn.LayerNorm(hidden)

        self.score_head = nn.Linear(hidden, 1)
        self.gain_head = nn.Linear(hidden, 1)

        # The quantity a good scheduler ranks on (urgency x rate / mean rate)
        # spans ten-plus nats, but the heads read LayerNorm'd features and so
        # emit roughly unit-scale values. Without a learned temperature the
        # softmax can never become peaked enough to express a decisive ranking.
        self.log_scale = nn.Parameter(torch.tensor(1.6))

        self.mute_head = nn.Sequential(
            nn.Linear(N_RBG_FEATURES + N_GLOBAL_FEATURES, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden + N_GLOBAL_FEATURES, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        ue: torch.Tensor,      # (B, MAX_UE, N_UE_FEATURES)
        rbg: torch.Tensor,     # (B, n_rbg, N_RBG_FEATURES)
        glob: torch.Tensor,    # (B, N_GLOBAL_FEATURES)
        z: torch.Tensor,       # (B, MAX_UE, n_rbg)
        valid: torch.Tensor,   # (B, MAX_UE) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return assignment logits ``(B, n_rbg, MAX_UE + 1)`` and values ``(B,)``."""
        h = self.ue_encoder(ue)
        pad_mask = ~valid
        attended, _ = self.attn(h, h, h, key_padding_mask=pad_mask, need_weights=False)
        h = self.norm1(h + attended)
        h = self.norm2(h + self.ff(h))
        h = torch.nan_to_num(h)

        score = self.score_head(h).squeeze(-1)              # (B, MAX_UE)
        gain = F.softplus(self.gain_head(h)).squeeze(-1)    # (B, MAX_UE)

        # logits[b, k, u] = scale * (score[b, u] + gain[b, u] * z[b, u, k])
        scale = F.softplus(self.log_scale)
        pair = scale * (score.unsqueeze(-1) + gain.unsqueeze(-1) * z)  # (B, MAX_UE, n_rbg)
        pair = pair.transpose(1, 2)                                     # (B, n_rbg, MAX_UE)

        n_rbg = rbg.shape[1]
        glob_rep = glob.unsqueeze(1).expand(-1, n_rbg, -1)
        mute = scale * self.mute_head(torch.cat([rbg, glob_rep], dim=-1))  # (B, n_rbg, 1)

        logits = torch.cat([pair, mute], dim=-1)             # (B, n_rbg, MAX_UE + 1)

        # Mean-pool only over real UEs for the value estimate.
        vmask = valid.unsqueeze(-1).float()
        pooled = (h * vmask).sum(1) / vmask.sum(1).clamp(min=1.0)
        value = self.value_head(torch.cat([pooled, glob], dim=-1)).squeeze(-1)
        return logits, value


MUTE = MAX_UE  # index of the "leave this RBG silent" action


def decode(
    logits: np.ndarray,
    feat: Dict[str, np.ndarray],
    rng: np.random.Generator,
    *,
    greedy: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sequentially turn logits into an RBG assignment.

    Returns the assignment (``-1`` means muted) and the per-RBG action mask that
    PPO needs in order to recompute log-probabilities during the update.
    """
    n_rbg = logits.shape[0]
    bits = feat["bits"]
    residual = feat["backlog"].copy()
    valid = feat["valid"]

    assignment = np.full(n_rbg, -1, dtype=np.int64)
    actions = np.full(n_rbg, MUTE, dtype=np.int64)
    masks = np.zeros((n_rbg, MAX_UE + 1), dtype=bool)
    masks[:, MUTE] = True  # muting is always permitted

    for k in rbg_order(feat["slot"], n_rbg):
        eligible = valid & (residual > 1e-9)
        masks[k, :MAX_UE] = eligible

        row = np.where(masks[k], logits[k], -np.inf)
        if greedy:
            choice = int(np.argmax(row))
        else:
            # Gumbel-max sampling is equivalent to sampling from softmax(row).
            g = rng.gumbel(size=row.shape)
            choice = int(np.argmax(row + g))

        actions[k] = choice
        if choice != MUTE:
            assignment[k] = choice
            residual[choice] -= bits[choice, k]

    return assignment, masks

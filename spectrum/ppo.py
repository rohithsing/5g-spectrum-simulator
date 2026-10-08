"""
PPO trainer for the scheduling policy.

Each environment step is one slot, and one slot contains ``n_rbg`` sequential
sub-decisions. The policy is therefore treated as autoregressive within a slot:
the joint log-probability of a slot's action is the sum of the ``n_rbg``
categorical log-probabilities, each under the candidate mask that was in force
when that RBG was decided. Those masks are recorded during rollout so the
update can reconstruct the exact distributions in one vectorised pass.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SimConfig
from .env import SpectrumEnv
from .policy import MAX_UE, SchedulerPolicy, decode, extract_features
from .predictor import GRUPredictor, OccupancyEstimator


@dataclass
class PPOConfig:
    #: Chosen as a whole multiple of ``episode_slots`` x len(congestion_mix) so
    #: every rollout contains exactly one episode of each regime. Otherwise the
    #: reported mean reward swings with whichever regimes a rollout happened to
    #: sample, and genuine progress is invisible.
    rollout_slots: int = 1500
    epochs: int = 4
    minibatch: int = 256
    gamma: float = 0.97
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coef: float = 0.5
    #: The KL anchor supplies the regularisation, so the entropy bonus is
    #: small; a large one fights the anchor and pushes the policy to uniform.
    entropy_coef: float = 0.0005
    entropy_final: float = 0.0001
    lr: float = 1e-4
    max_grad_norm: float = 0.5
    #: Stop the update early once the mean per-decision KL from the rollout
    #: policy exceeds this. Protects the behaviour-cloned initialisation.
    target_kl: float = 0.015
    #: Weight on KL(pi || pi_reference) against the frozen behaviour-cloned
    #: policy. One scalar advantage per slot has to shape 18 decisions over ~80
    #: candidates each, so the gradient is noisy; without an anchor the policy
    #: drifts away from a good initialisation faster than it finds genuine
    #: improvements. Measured without it: goodput fell from 184 to 28 Mb/s over
    #: 12 iterations while the agent rediscovered "mute everything".
    bc_kl_coef: float = 1.0
    #: The anchor is annealed: strong early, when the value function is
    #: still poor and the advantage is mostly noise, then relaxed so RL can
    #: actually depart from the teacher once the critic is informative.
    bc_kl_final: float = 0.1
    iterations: int = 60
    #: Rewards are divided by this before bootstrapping. It leaves the optimal
    #: policy unchanged (advantages are normalised anyway) but keeps the value
    #: targets near unit scale, which the value head needs to fit quickly.
    reward_scale: float = 0.1
    congestion_mix: Sequence[str] = ("medium", "high", "peak")
    episode_slots: int = 500
    seed: int = 0


@dataclass
class Rollout:
    ue: List[np.ndarray] = field(default_factory=list)
    rbg: List[np.ndarray] = field(default_factory=list)
    glob: List[np.ndarray] = field(default_factory=list)
    z: List[np.ndarray] = field(default_factory=list)
    valid: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)
    #: Per-RBG log-probabilities, one ``(n_rbg,)`` array per slot.
    logprob: List[np.ndarray] = field(default_factory=list)
    value: List[float] = field(default_factory=list)
    reward: List[float] = field(default_factory=list)
    done: List[bool] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.reward)

    def stack(self) -> Dict[str, torch.Tensor]:
        return {
            "ue": torch.from_numpy(np.stack(self.ue)),
            "rbg": torch.from_numpy(np.stack(self.rbg)),
            "glob": torch.from_numpy(np.stack(self.glob)),
            "z": torch.from_numpy(np.stack(self.z)),
            "valid": torch.from_numpy(np.stack(self.valid)),
            "actions": torch.from_numpy(np.stack(self.actions)),
            "masks": torch.from_numpy(np.stack(self.masks)),
            "logprob": torch.from_numpy(np.stack(self.logprob).astype(np.float32)),
            "value": torch.tensor(self.value, dtype=torch.float32),
        }


def masked_log_probs(
    logits: torch.Tensor, masks: torch.Tensor, actions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-sub-action log-probability and entropy for a slot.

    ``logits`` is ``(B, n_rbg, MAX_UE+1)``, ``masks`` the same shape in bool and
    ``actions`` is ``(B, n_rbg)``. Both returns are ``(B, n_rbg)`` -- kept
    **unreduced over RBGs** on purpose.

    A slot's action is 18 sequential categorical choices. Summing their
    log-probabilities gives the correct joint, but feeding that joint to PPO's
    ratio is a trap: the ratio becomes a product of 18 per-decision ratios, so a
    0.05 shift in each inflates it to e^0.9 = 2.5 and the clip zeroes the
    gradient. Measured on this problem, the joint form held clip_frac at 0.5-0.64
    while the true per-decision KL was only 0.004, and learning stalled.
    The trainer therefore clips each sub-action separately against the shared
    slot advantage, which is the standard credit assignment for a factored
    policy.
    """
    masked = logits.masked_fill(~masks, -1e9)
    logp = F.log_softmax(masked, dim=-1)
    chosen = logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)  # (B, n_rbg)

    probs = logp.exp()
    entropy = -(probs * logp).masked_fill(~masks, 0.0).sum(-1)   # (B, n_rbg)
    return chosen, entropy


def joint_log_prob(logits: torch.Tensor, masks: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Summed log-probability of a slot's whole action, for diagnostics."""
    return masked_log_probs(logits, masks, actions)[0].sum(-1)


def masked_kl(logits: torch.Tensor, ref_logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Per-sub-action KL(pi || pi_ref) over the masked candidate set, ``(B, n_rbg)``."""
    logp = F.log_softmax(logits.masked_fill(~masks, -1e9), dim=-1)
    logq = F.log_softmax(ref_logits.masked_fill(~masks, -1e9), dim=-1)
    return (logp.exp() * (logp - logq)).masked_fill(~masks, 0.0).sum(-1)


class PPOTrainer:
    """Collects rollouts across congestion regimes and optimises the policy."""

    def __init__(
        self,
        policy: SchedulerPolicy,
        predictor: Optional[GRUPredictor],
        cfg: PPOConfig,
        *,
        base_cfg: Optional[Dict] = None,
    ):
        self.policy = policy
        self.predictor = predictor
        self.cfg = cfg
        # Frozen snapshot of the behaviour-cloned policy, used as the KL anchor.
        self.reference = copy.deepcopy(policy).eval()
        for p in self.reference.parameters():
            p.requires_grad_(False)
        self.base_cfg = dict(base_cfg or {})
        self.estimator = OccupancyEstimator(predictor)
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
        self.rng = np.random.default_rng(cfg.seed)
        self.torch_gen = torch.Generator().manual_seed(cfg.seed)

        self._env: Optional[SpectrumEnv] = None
        self._obs = None
        self._episode_index = 0
        self.history: List[Dict[str, float]] = []

    # ------------------------------------------------------------- env plumbing
    def _new_episode(self) -> None:
        level = self.cfg.congestion_mix[self._episode_index % len(self.cfg.congestion_mix)]
        seed = int(self.rng.integers(0, 2**31 - 1))
        cfg = SimConfig(
            congestion=level,
            seed=seed,
            slots_per_episode=self.cfg.episode_slots,
            **self.base_cfg,
        )
        self._env = SpectrumEnv(cfg, seed=seed)
        self._obs = self._env.observe()
        self.estimator.reset(cfg.n_subbands)
        self._episode_index += 1

    def _p_busy(self, obs) -> np.ndarray:
        return self.estimator(obs)

    # ------------------------------------------------------------------ rollout
    @torch.no_grad()
    def collect(self) -> Rollout:
        roll = Rollout()
        if self._env is None:
            self._new_episode()

        while len(roll) < self.cfg.rollout_slots:
            feat = extract_features(self._obs, self._p_busy(self._obs))
            logits_t, value_t = self.policy(
                torch.from_numpy(feat["ue"]).unsqueeze(0),
                torch.from_numpy(feat["rbg"]).unsqueeze(0),
                torch.from_numpy(feat["global"]).unsqueeze(0),
                torch.from_numpy(feat["z"]).unsqueeze(0),
                torch.from_numpy(feat["valid"]).unsqueeze(0),
            )
            logits = logits_t[0].numpy().astype(np.float64)
            assignment, masks = decode(logits, feat, self.rng, greedy=False)

            action_vec = np.where(assignment >= 0, assignment, MAX_UE)
            lp, _ = masked_log_probs(
                logits_t,
                torch.from_numpy(masks).unsqueeze(0),
                torch.from_numpy(action_vec).unsqueeze(0),
            )

            self._obs, reward, done, _ = self._env.step(assignment)

            roll.ue.append(feat["ue"])
            roll.rbg.append(feat["rbg"])
            roll.glob.append(feat["global"])
            roll.z.append(feat["z"])
            roll.valid.append(feat["valid"])
            roll.actions.append(action_vec)
            roll.masks.append(masks)
            roll.logprob.append(lp[0].numpy().copy())  # per-RBG, shape (n_rbg,)
            roll.value.append(float(value_t.item()))
            roll.reward.append(float(reward) * self.cfg.reward_scale)
            roll.done.append(bool(done))

            if done:
                self._new_episode()

        return roll

    # ------------------------------------------------------------------- update
    def _advantages(self, roll: Rollout, last_value: float) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(roll)
        adv = np.zeros(n, dtype=np.float32)
        gae = 0.0
        values = np.asarray(roll.value + [last_value], dtype=np.float32)
        for t in reversed(range(n)):
            nonterminal = 0.0 if roll.done[t] else 1.0
            delta = roll.reward[t] + self.cfg.gamma * values[t + 1] * nonterminal - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * nonterminal * gae
            adv[t] = gae
        returns = adv + np.asarray(roll.value, dtype=np.float32)
        return torch.from_numpy(adv), torch.from_numpy(returns)

    def update(
        self, roll: Rollout, last_value: float, entropy_coef: float, bc_coef: float
    ) -> Dict[str, float]:
        data = roll.stack()
        adv, returns = self._advantages(roll, last_value)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = len(roll)
        stats = {
            "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
            "clip_frac": 0.0, "approx_kl": 0.0, "bc_kl": 0.0,
        }
        n_batches = 0
        stopped_early = False

        for _ in range(self.cfg.epochs):
            if stopped_early:
                break
            perm = torch.randperm(n, generator=self.torch_gen)
            for i in range(0, n, self.cfg.minibatch):
                idx = perm[i : i + self.cfg.minibatch]
                logits, value = self.policy(
                    data["ue"][idx],
                    data["rbg"][idx],
                    data["glob"][idx],
                    data["z"][idx],
                    data["valid"][idx],
                )
                # Both are (B, n_rbg): one entry per sub-action, not reduced.
                logp, entropy = masked_log_probs(logits, data["masks"][idx], data["actions"][idx])

                log_ratio = torch.clamp(logp - data["logprob"][idx], -20.0, 20.0)
                ratio = torch.exp(log_ratio)
                # Schulman's low-variance KL estimator, per sub-action.
                with torch.no_grad():
                    approx_kl = float(((ratio - 1.0) - log_ratio).mean())

                # The slot's advantage is shared by every sub-action taken in it,
                # and each is clipped on its own ratio.
                a = adv[idx].unsqueeze(-1)
                unclipped = ratio * a
                clipped = torch.clamp(ratio, 1 - self.cfg.clip, 1 + self.cfg.clip) * a
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(value, returns[idx])
                ent = entropy.mean()

                with torch.no_grad():
                    ref_logits, _ = self.reference(
                        data["ue"][idx], data["rbg"][idx], data["glob"][idx],
                        data["z"][idx], data["valid"][idx],
                    )
                bc_kl = masked_kl(logits, ref_logits, data["masks"][idx]).mean()

                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    - entropy_coef * ent
                    + bc_coef * bc_kl
                )

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.opt.step()

                stats["policy_loss"] += float(policy_loss.detach())
                stats["value_loss"] += float(value_loss.detach())
                stats["entropy"] += float(ent.detach())
                stats["clip_frac"] += float(
                    ((ratio - 1.0).abs() > self.cfg.clip).float().mean().detach()
                )
                stats["approx_kl"] += approx_kl
                stats["bc_kl"] += float(bc_kl.detach())
                n_batches += 1

                # Abandon the rest of the epochs once the policy has moved too
                # far from the one that generated this rollout. Without this a
                # warm-started policy can destroy its initialisation in a single
                # over-long update.
                if approx_kl > self.cfg.target_kl:
                    stopped_early = True
                    break

        for k in stats:
            stats[k] /= max(n_batches, 1)
        stats["stopped_early"] = float(stopped_early)
        return stats

    # --------------------------------------------------------------------- loop
    def train(self, *, verbose: bool = True, callback=None) -> List[Dict[str, float]]:
        for it in range(self.cfg.iterations):
            roll = self.collect()

            with torch.no_grad():
                feat = extract_features(self._obs, self._p_busy(self._obs))
                _, last_v = self.policy(
                    torch.from_numpy(feat["ue"]).unsqueeze(0),
                    torch.from_numpy(feat["rbg"]).unsqueeze(0),
                    torch.from_numpy(feat["global"]).unsqueeze(0),
                    torch.from_numpy(feat["z"]).unsqueeze(0),
                    torch.from_numpy(feat["valid"]).unsqueeze(0),
                )

            frac = it / max(self.cfg.iterations - 1, 1)
            ent_coef = self.cfg.entropy_coef + frac * (self.cfg.entropy_final - self.cfg.entropy_coef)
            bc_coef = self.cfg.bc_kl_coef + frac * (self.cfg.bc_kl_final - self.cfg.bc_kl_coef)
            stats = self.update(roll, float(last_v.item()), ent_coef, bc_coef)
            stats["iteration"] = it
            stats["mean_reward"] = float(np.mean(roll.reward)) / self.cfg.reward_scale
            stats["entropy_coef"] = ent_coef
            stats["bc_coef"] = bc_coef
            self.history.append(stats)

            if verbose:
                print(
                    f"  iter {it + 1:3d}/{self.cfg.iterations}  "
                    f"reward/slot {stats['mean_reward']:+.4f}  "
                    f"pi {stats['policy_loss']:+.4f}  v {stats['value_loss']:.4f}  "
                    f"H {stats['entropy']:.3f}  clip {stats['clip_frac']:.3f}  "
                    f"KL {stats['approx_kl']:.4f}  bcKL {stats['bc_kl']:.4f}",
                    flush=True,
                )
            if callback is not None:
                callback(it, stats)

        return self.history

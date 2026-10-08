"""
The spectrum-allocation environment.

One :class:`SpectrumEnv` instance is one 5G NR cell sharing a 100 MHz carrier
with a stochastic incumbent. Every slot the environment hands an allocator an
:class:`Observation`, receives an RBG-to-UE assignment, and applies it to the
physical layer.

Information parity
------------------
:class:`Observation` is a frozen dataclass containing **only** quantities a real
gNB scheduler can read: buffer status reports, head-of-line delays, 5QI
parameters, aged and quantised CQI reports, and the output of the energy
detector. The incumbent's true occupancy is never a field of it. This is
asserted mechanically by ``tests/test_information_parity.py``, so the headline
comparison cannot silently regress into an unfair one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .channel import RadioChannel
from .config import CQI_SPECTRAL_EFFICIENCY, SimConfig
from .incumbent import IncumbentProcess
from .metrics import EpisodeMetrics, jain_fairness, qos_satisfied_mask
from .traffic import TrafficModel

#: How many past sensing sweeps are exposed to the allocator.
SENSING_HISTORY_LEN = 16


@dataclass(frozen=True)
class Observation:
    """
    Everything -- and only what -- a scheduler is allowed to see.

    Adding a field to this class is the one place where the fairness of the
    whole study can be compromised, so any new field must be justified as
    something a deployed gNB genuinely has access to.
    """

    slot: int
    n_ue: int
    n_rbg: int
    n_subbands: int

    # -- bearer state (from Buffer Status Reports and the local scheduler) ---
    backlog_bits: np.ndarray            # (n_ue,)
    hol_delay_slots: np.ndarray         # (n_ue,)
    deadline_slack_slots: np.ndarray    # (n_ue,)
    delay_budget_slots: np.ndarray      # (n_ue,)
    nominal_rate_bps: np.ndarray        # (n_ue,)
    priority_level: np.ndarray          # (n_ue,)
    ue_class: np.ndarray                # (n_ue,)
    avg_throughput_bps: np.ndarray      # (n_ue,) EWMA, used by PF-family rules
    last_slot_rbgs: np.ndarray          # (n_ue,)

    # -- channel state information (aged + quantised, as reported by the UE) -
    cqi: np.ndarray                     # (n_ue, n_rbg) int, 0-based CQI index
    spectral_efficiency: np.ndarray     # (n_ue, n_rbg) bit/s/Hz implied by cqi
    bits_per_rbg: np.ndarray            # (n_ue, n_rbg) transport-block size

    # -- spectrum sensing (identical front-end for every allocator) ----------
    sensing_decision: np.ndarray        # (n_subbands,) hard binary call
    sensing_energy: np.ndarray          # (n_subbands,) normalised statistic
    sensing_decision_history: np.ndarray  # (H, n_subbands)
    sensing_energy_history: np.ndarray    # (H, n_subbands)
    subband_of_rbg: np.ndarray          # (n_rbg,)


class SpectrumEnv:
    """Slot-level 5G NR downlink scheduler environment with incumbent sharing."""

    def __init__(self, cfg: SimConfig, seed: Optional[int] = None):
        self.cfg = cfg
        self._base_seed = cfg.seed if seed is None else seed
        self.reset(self._base_seed)

    # --------------------------------------------------------------- lifecycle
    def reset(self, seed: Optional[int] = None) -> Observation:
        cfg = self.cfg
        if seed is not None:
            self._base_seed = seed
        self.rng = np.random.default_rng(self._base_seed)

        self.channel = RadioChannel(cfg, self.rng)
        self.traffic = TrafficModel(cfg, self.rng)
        self.incumbent = IncumbentProcess(cfg, self.rng)
        self.metrics = EpisodeMetrics(cfg)

        self.slot = 0
        #: Expected packet arrivals per slot across the whole cell. Used to put
        #: the completed/dropped reward terms on a load-independent scale.
        self._arrival_rate = float(np.sum(self.traffic.arrival_prob))
        self.avg_throughput_bps = np.full(cfg.n_ue, 1e5)
        self.last_slot_rbgs = np.zeros(cfg.n_ue, dtype=np.int64)
        self.harq_retx = np.zeros(cfg.n_ue, dtype=np.int64)

        h = (SENSING_HISTORY_LEN, cfg.n_subbands)
        self._decision_hist = np.zeros(h, dtype=np.float64)
        self._energy_hist = np.zeros(h, dtype=np.float64)

        self._sense()
        self.traffic.arrive(self.slot)
        return self.observe()

    # ------------------------------------------------------------------ sensing
    def _sense(self) -> None:
        decision, energy = self.incumbent.sense()
        self._decision = decision
        self._energy = energy
        self._decision_hist = np.roll(self._decision_hist, -1, axis=0)
        self._energy_hist = np.roll(self._energy_hist, -1, axis=0)
        self._decision_hist[-1] = decision
        self._energy_hist[-1] = energy

    # -------------------------------------------------------------- observation
    def observe(self) -> Observation:
        cfg = self.cfg
        cqi = self.channel.reported_cqi()
        se = CQI_SPECTRAL_EFFICIENCY[cqi]
        return Observation(
            slot=self.slot,
            n_ue=cfg.n_ue,
            n_rbg=cfg.n_rbg,
            n_subbands=cfg.n_subbands,
            backlog_bits=self.traffic.backlog_bits.copy(),
            hol_delay_slots=self.traffic.head_of_line_delay_slots(self.slot),
            deadline_slack_slots=self.traffic.deadline_slack_slots(self.slot),
            delay_budget_slots=self.traffic.delay_budget_slots.copy(),
            nominal_rate_bps=self.traffic.nominal_rate_bps.copy(),
            priority_level=self.traffic.priority_level.copy(),
            ue_class=self.traffic.ue_class.copy(),
            avg_throughput_bps=self.avg_throughput_bps.copy(),
            last_slot_rbgs=self.last_slot_rbgs.copy(),
            cqi=cqi.copy(),
            spectral_efficiency=se,
            bits_per_rbg=np.asarray(cfg.bits_per_rbg(se)),
            sensing_decision=self._decision.copy(),
            sensing_energy=self._energy.copy(),
            sensing_decision_history=self._decision_hist.copy(),
            sensing_energy_history=self._energy_hist.copy(),
            subband_of_rbg=cfg.subband_of_rbg(),
        )

    # -------------------------------------------------------------------- step
    def step(self, assignment: np.ndarray) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Apply one slot's RBG assignment.

        ``assignment[k]`` is the UE scheduled on RBG ``k``, or ``-1`` to mute it.
        """
        cfg = self.cfg
        assignment = np.asarray(assignment, dtype=np.int64).reshape(cfg.n_rbg)
        if np.any(assignment >= cfg.n_ue):
            raise ValueError("assignment references a UE index outside the cell")

        truth = self.incumbent.active
        delivered, acked, collided = self.channel.transmit(assignment, truth, self.harq_retx)

        # -- turn delivered bits into useful bits, UE by UE ------------------
        active = assignment >= 0
        useful_bits = 0.0
        total_delivered = float(np.sum(delivered))
        per_ue_bits = np.zeros(cfg.n_ue)
        for rbg in np.flatnonzero(active):
            ue = int(assignment[rbg])
            per_ue_bits[ue] += delivered[rbg]
        completed_before = int(np.sum(self.traffic.packets_delivered))
        for ue in np.flatnonzero(per_ue_bits > 0.0):
            useful_bits += self.traffic.serve(int(ue), float(per_ue_bits[ue]), self.slot)
        completed = int(np.sum(self.traffic.packets_delivered)) - completed_before

        # -- HARQ bookkeeping -------------------------------------------------
        nacks = int(np.sum(active & ~acked))
        scheduled_ues = np.unique(assignment[active]) if np.any(active) else np.array([], dtype=np.int64)
        for ue in scheduled_ues:
            ue_rbgs = assignment == ue
            if np.any(acked[ue_rbgs]):
                self.harq_retx[ue] = 0
            else:
                self.harq_retx[ue] = min(self.harq_retx[ue] + 1, cfg.max_harq_retx)

        expired = self.traffic.expire(self.slot)

        # -- throughput tracking (EWMA over a 100-slot window) ----------------
        inst_bps = per_ue_bits / cfg.slot_duration_s
        alpha = 0.01
        self.avg_throughput_bps = (1.0 - alpha) * self.avg_throughput_bps + alpha * inst_bps
        self.last_slot_rbgs = np.bincount(
            assignment[active], minlength=cfg.n_ue
        ) if np.any(active) else np.zeros(cfg.n_ue, dtype=np.int64)

        # -- spectrum-sharing accounting --------------------------------------
        n_collisions = int(np.sum(collided & active))
        subband = cfg.subband_of_rbg()
        idle_rbgs = truth[subband] == 0
        # An RBG we muted although the incumbent was genuinely idle is spectrum
        # thrown away by over-conservative protection.
        wasted_vacancies = int(np.sum(idle_rbgs & ~active))
        idle_rbg_count = int(np.sum(idle_rbgs))

        jain = jain_fairness(self.avg_throughput_bps)
        qos_now = self._instantaneous_qos()

        self.metrics.record_slot(
            useful_bits=useful_bits,
            delivered_bits=total_delivered,
            rbgs_used=int(np.sum(active)),
            rbgs_muted=int(np.sum(~active)),
            collisions=n_collisions,
            harq_nacks=nacks,
            wasted_vacancies=wasted_vacancies,
            idle_subband_rbgs=idle_rbg_count,
            jain=jain,
            qos_fraction=qos_now,
        )

        reward, reward_terms = self._reward(
            useful_bits=useful_bits,
            delivered_bits=total_delivered,
            n_collisions=n_collisions,
            completed=completed,
            expired=int(np.sum(expired)),
            jain=jain,
            rbgs_used=int(np.sum(active)),
            wasted_vacancies=wasted_vacancies,
            idle_rbg_count=idle_rbg_count,
        )

        # -- advance the world one slot ---------------------------------------
        self.slot += 1
        self.channel.advance()
        self.channel.update_cqi_reports()
        self.incumbent.advance()
        self._sense()
        self.traffic.arrive(self.slot)

        done = self.slot >= cfg.slots_per_episode
        info: Dict[str, Any] = {
            "useful_bits": useful_bits,
            "delivered_bits": total_delivered,
            "collisions": n_collisions,
            "completed": completed,
            "expired": int(np.sum(expired)),
            "jain": jain,
            "qos_fraction": qos_now,
            "reward_terms": reward_terms,
            # Ground truth, for *evaluation and predictor training only*. It is
            # returned from step(), i.e. strictly after the decision was made,
            # and is never part of Observation.
            "_truth_incumbent": truth.copy(),
        }
        return self.observe(), reward, done, info

    # ------------------------------------------------------------------ reward
    def _reward(
        self,
        *,
        useful_bits: float,
        delivered_bits: float,
        n_collisions: int,
        completed: int,
        expired: int,
        jain: float,
        rbgs_used: int,
        wasted_vacancies: int,
        idle_rbg_count: int,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Multi-objective slot reward, every term normalised to roughly [-1, 1].

        QoS is scored by *packet outcomes* -- how many packets completed inside
        their budget and how many expired -- rather than by any queue-state
        proxy. Both are normalised by the cell's known mean arrival rate, so the
        two terms are directly comparable and dropping a packet always costs
        more than delivering one gains. This is what makes "mute everything"
        strictly dominated: it drives ``completed`` to zero and ``expired`` to
        the full arrival rate.
        """
        cfg = self.cfg
        max_bits = cfg.bits_per_rbg(CQI_SPECTRAL_EFFICIENCY[-1]) * cfg.n_rbg
        lam = max(self._arrival_rate, 1e-6)  # expected packet arrivals per slot

        terms = {
            "goodput": float(useful_bits / max_bits),
            "completed": float(completed) / lam,
            "dropped": -float(expired) / lam,
            "fairness": float(jain),
            "collision": -float(n_collisions) / max(cfg.n_rbg, 1),
            "waste": -float(max(delivered_bits - useful_bits, 0.0)) / max_bits,
            "vacancy": -float(wasted_vacancies) / max(idle_rbg_count, 1),
        }
        weights = {
            "goodput": 1.5,
            "completed": 1.5,
            "dropped": 3.0,
            "fairness": 0.4,
            "collision": 2.0,
            "waste": 0.5,
            "vacancy": 0.3,
        }
        reward = sum(weights[k] * v for k, v in terms.items())
        return float(reward), terms

    # -------------------------------------------------------------------- QoS
    def _instantaneous_qos(self) -> float:
        """
        Fraction of UEs currently inside their delay budget.

        Reported for monitoring only -- deliberately **not** used in the reward.
        Expiry removes a packet from the queue, so head-of-line delay resets when
        a deadline is missed: an allocator that transmits nothing keeps this
        statistic near 1.0 while dropping every packet. Rewarding it produced
        exactly that degenerate policy. The reward counts completed and expired
        packets instead, which cannot be gamed this way.
        """
        t = self.traffic
        hol = t.head_of_line_delay_slots(self.slot)
        return float(np.mean(hol <= t.delay_budget_slots))

    def episode_summary(self) -> Dict[str, float]:
        """Final KPI record for the episode."""
        t = self.traffic
        mean_delay_slots = np.where(
            t.packets_delivered > 0, t.delay_sum / np.maximum(t.packets_delivered, 1), 0.0
        )
        satisfied = qos_satisfied_mask(
            delivered_bits=t.delivered_bits,
            arrived_bits=t.arrived_bits,
            packets_arrived=t.packets_arrived,
            packets_delivered=t.packets_delivered,
            packets_expired=t.packets_expired,
            mean_delay_slots=mean_delay_slots,
            delay_budget_slots=t.delay_budget_slots,
        )
        return self.metrics.summarise(
            ue_class=t.ue_class,
            delivered_per_ue=t.delivered_bits,
            arrived_per_ue=t.arrived_bits,
            packets_arrived=t.packets_arrived,
            packets_delivered=t.packets_delivered,
            packets_expired=t.packets_expired,
            delays_ms=t.delay_samples,
            qos_satisfied=satisfied,
        )

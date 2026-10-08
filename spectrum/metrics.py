"""
KPI definitions and episode-level aggregation.

Keeping every metric definition in one module means the reward function, the
benchmark tables and the dashboard are all computed from identical formulas --
there is no way for the training objective and the reported result to drift
apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .config import CLASS_NAMES, SimConfig


def jain_fairness(x: np.ndarray) -> float:
    """Jain's fairness index of a non-negative allocation vector."""
    x = np.asarray(x, dtype=np.float64)
    denom = x.size * np.sum(x ** 2)
    if denom <= 1e-12:
        return 1.0
    return float(np.sum(x) ** 2 / denom)


@dataclass
class EpisodeMetrics:
    """Accumulates one episode's KPIs and renders them as a flat record."""

    cfg: SimConfig

    # -- running totals ------------------------------------------------------
    slots: int = 0
    useful_bits: float = 0.0
    delivered_bits: float = 0.0
    wasted_bits: float = 0.0
    arrived_bits: float = 0.0
    rbgs_used: int = 0
    rbgs_muted: int = 0
    rbg_slots: int = 0
    collisions: int = 0
    transmissions: int = 0
    harq_nacks: int = 0
    #: Transmit opportunities given up because sensing said "busy" while the
    #: incumbent was in fact idle -- the cost of over-conservative protection.
    wasted_vacancies: int = 0
    idle_subband_rbg_slots: int = 0

    per_slot_jain: List[float] = field(default_factory=list)
    per_slot_qos: List[float] = field(default_factory=list)
    delays_ms: List[float] = field(default_factory=list)

    def record_slot(
        self,
        *,
        useful_bits: float,
        delivered_bits: float,
        rbgs_used: int,
        rbgs_muted: int,
        collisions: int,
        harq_nacks: int,
        wasted_vacancies: int,
        idle_subband_rbgs: int,
        jain: float,
        qos_fraction: float,
    ) -> None:
        self.slots += 1
        self.useful_bits += useful_bits
        self.delivered_bits += delivered_bits
        self.wasted_bits += max(delivered_bits - useful_bits, 0.0)
        self.rbgs_used += rbgs_used
        self.rbgs_muted += rbgs_muted
        self.rbg_slots += self.cfg.n_rbg
        self.collisions += collisions
        self.transmissions += rbgs_used
        self.harq_nacks += harq_nacks
        self.wasted_vacancies += wasted_vacancies
        self.idle_subband_rbg_slots += idle_subband_rbgs
        self.per_slot_jain.append(jain)
        self.per_slot_qos.append(qos_fraction)

    # ------------------------------------------------------------------ output
    def summarise(
        self,
        *,
        ue_class: np.ndarray,
        delivered_per_ue: np.ndarray,
        arrived_per_ue: np.ndarray,
        packets_arrived: np.ndarray,
        packets_delivered: np.ndarray,
        packets_expired: np.ndarray,
        delays_ms: Sequence[float],
        qos_satisfied: np.ndarray,
    ) -> Dict[str, float]:
        cfg = self.cfg
        duration_s = max(self.slots, 1) * cfg.slot_duration_s
        total_hz = cfg.rbg_bandwidth_hz * cfg.n_rbg

        delays = np.asarray(delays_ms, dtype=np.float64)
        pkts_arrived = float(np.sum(packets_arrived))
        pkts_expired = float(np.sum(packets_expired))

        out: Dict[str, float] = {
            # -- primary system KPIs -------------------------------------
            "goodput_mbps": self.useful_bits / duration_s / 1e6,
            "spectral_efficiency_bps_hz": self.useful_bits / duration_s / total_hz,
            "qos_satisfaction_pct": 100.0 * float(np.mean(qos_satisfied)),
            "mean_delay_ms": float(np.mean(delays)) if delays.size else 0.0,
            "p99_delay_ms": float(np.percentile(delays, 99)) if delays.size else 0.0,
            "deadline_miss_pct": 100.0 * pkts_expired / max(pkts_arrived, 1.0),
            "jain_fairness": float(np.mean(self.per_slot_jain)) if self.per_slot_jain else 0.0,
            # -- spectrum-sharing behaviour -------------------------------
            "collision_pct": 100.0 * self.collisions / max(self.transmissions, 1),
            "prb_utilisation_pct": 100.0 * self.rbgs_used / max(self.rbg_slots, 1),
            "wasted_vacancy_pct": 100.0 * self.wasted_vacancies
            / max(self.idle_subband_rbg_slots, 1),
            # -- efficiency of the allocation itself ----------------------
            "allocation_waste_pct": 100.0 * self.wasted_bits / max(self.delivered_bits, 1.0),
            "harq_nack_pct": 100.0 * self.harq_nacks / max(self.transmissions, 1),
            "demand_served_pct": 100.0 * float(np.sum(delivered_per_ue))
            / max(float(np.sum(arrived_per_ue)), 1.0),
        }

        # Per-class QoS satisfaction: the headline the PPT's objectives target.
        for ci, name in enumerate(CLASS_NAMES):
            mask = ue_class == ci
            key = f"qos_{name}_pct"
            out[key] = 100.0 * float(np.mean(qos_satisfied[mask])) if np.any(mask) else float("nan")

        return out


def qos_satisfied_mask(
    *,
    delivered_bits: np.ndarray,
    arrived_bits: np.ndarray,
    packets_arrived: np.ndarray,
    packets_delivered: np.ndarray,
    packets_expired: np.ndarray,
    mean_delay_slots: np.ndarray,
    delay_budget_slots: np.ndarray,
    throughput_sla_ratio: float = 0.90,
    reliability_target: float = 0.95,
) -> np.ndarray:
    """
    A UE is QoS-satisfied when it meets *both* halves of its 5QI contract.

    1. **Throughput**: at least ``throughput_sla_ratio`` of its offered traffic
       was delivered rather than dropped.
    2. **Latency / reliability**: at least ``reliability_target`` of its packets
       arrived inside the packet delay budget.

    UEs that never generated traffic are counted as satisfied -- they have no
    unmet contract.
    """
    idle = packets_arrived == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        served_ratio = np.where(arrived_bits > 0, delivered_bits / np.maximum(arrived_bits, 1e-9), 1.0)
        on_time_ratio = np.where(
            packets_arrived > 0,
            packets_delivered / np.maximum(packets_arrived, 1),
            1.0,
        )
    ok = (served_ratio >= throughput_sla_ratio) & (on_time_ratio >= reliability_target)
    ok[idle] = True
    return ok

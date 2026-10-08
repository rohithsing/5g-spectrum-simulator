"""
Per-UE traffic generation and radio-bearer queue state.

Each UE runs one 5QI-derived bearer (TS 23.501 Table 5.7.4-1) with a bursty
arrival process and a per-packet delay budget. Queues are maintained at packet
granularity so that head-of-line delay and deadline expiry are exact rather
than approximated by an aggregate backlog.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List

import numpy as np

from .config import TRAFFIC_CLASSES, SimConfig


class TrafficModel:
    """Bursty packet arrivals plus deadline-aware queues for every UE."""

    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.n_ue = cfg.n_ue

        shares = np.array([tc.population_share for tc in TRAFFIC_CLASSES], dtype=np.float64)
        shares = shares / shares.sum()
        self.ue_class = rng.choice(len(TRAFFIC_CLASSES), size=self.n_ue, p=shares)

        load = cfg.regime.load_multiplier
        slot_ms = cfg.slot_duration_s * 1e3

        self.delay_budget_slots = np.array(
            [TRAFFIC_CLASSES[c].delay_budget_ms / slot_ms for c in self.ue_class]
        )
        self.nominal_rate_bps = np.array(
            [TRAFFIC_CLASSES[c].nominal_rate_mbps * 1e6 for c in self.ue_class]
        )
        self.priority_level = np.array(
            [TRAFFIC_CLASSES[c].priority_level for c in self.ue_class], dtype=np.float64
        )
        self.mean_payload_bits = np.array(
            [TRAFFIC_CLASSES[c].mean_payload_bits for c in self.ue_class]
        )
        self.payload_cv = np.array([TRAFFIC_CLASSES[c].payload_cv for c in self.ue_class])
        #: Probability that a packet arrives in a given slot, scaled by offered load.
        self.arrival_prob = np.clip(
            slot_ms / np.array([TRAFFIC_CLASSES[c].mean_interarrival_ms for c in self.ue_class]) * load,
            0.0,
            1.0,
        )

        #: queues[u] holds (remaining_bits, arrival_slot) tuples, oldest first.
        self.queues: List[Deque[List[float]]] = [deque() for _ in range(self.n_ue)]
        self.backlog_bits = np.zeros(self.n_ue)

        # Cumulative counters used for QoS accounting.
        self.delivered_bits = np.zeros(self.n_ue)
        self.arrived_bits = np.zeros(self.n_ue)
        self.packets_arrived = np.zeros(self.n_ue, dtype=np.int64)
        self.packets_delivered = np.zeros(self.n_ue, dtype=np.int64)
        self.packets_expired = np.zeros(self.n_ue, dtype=np.int64)
        self.delay_samples: List[float] = []
        self.delay_sum = np.zeros(self.n_ue)

    # ------------------------------------------------------------------ step
    def arrive(self, slot: int) -> None:
        """Generate this slot's packet arrivals."""
        fires = self.rng.random(self.n_ue) < self.arrival_prob
        idx = np.flatnonzero(fires)
        if idx.size == 0:
            return

        mean = self.mean_payload_bits[idx]
        cv = self.payload_cv[idx]
        # Log-normal payloads matched to the requested mean and CV.
        sigma = np.sqrt(np.log1p(cv ** 2))
        mu = np.log(mean) - 0.5 * sigma ** 2
        sizes = np.maximum(self.rng.lognormal(mu, sigma), 64.0)

        for u, bits in zip(idx, sizes):
            self.queues[u].append([float(bits), float(slot)])
            self.backlog_bits[u] += bits
            self.arrived_bits[u] += bits
            self.packets_arrived[u] += 1

    def serve(self, ue: int, bits: float, slot: int) -> float:
        """
        Drain up to ``bits`` from a UE's queue, oldest packet first.

        Returns the number of bits actually consumed (less than ``bits`` when
        the queue empties first, which is how over-allocation waste is measured).
        """
        remaining = bits
        q = self.queues[ue]
        while remaining > 0.0 and q:
            head = q[0]
            take = min(head[0], remaining)
            head[0] -= take
            remaining -= take
            self.backlog_bits[ue] -= take
            self.delivered_bits[ue] += take
            if head[0] <= 1e-9:
                latency_slots = slot - head[1]
                self.delay_samples.append(latency_slots * self.cfg.slot_duration_s * 1e3)
                self.delay_sum[ue] += latency_slots
                self.packets_delivered[ue] += 1
                q.popleft()
        return bits - remaining

    def expire(self, slot: int) -> np.ndarray:
        """
        Drop packets that have exceeded their delay budget.

        Returns a per-UE count of newly expired packets. Dropping is what makes
        the deadline constraint bite: an allocator that starves a URLLC bearer
        does not merely delay it, it loses the data.
        """
        dropped = np.zeros(self.n_ue, dtype=np.int64)
        for u in range(self.n_ue):
            budget = self.delay_budget_slots[u]
            q = self.queues[u]
            while q and (slot - q[0][1]) > budget:
                head = q.popleft()
                self.backlog_bits[u] -= head[0]
                self.packets_expired[u] += 1
                dropped[u] += 1
        np.maximum(self.backlog_bits, 0.0, out=self.backlog_bits)
        return dropped

    # ------------------------------------------------------------- observables
    def head_of_line_delay_slots(self, slot: int) -> np.ndarray:
        """Age of the oldest queued packet, per UE, in slots."""
        out = np.zeros(self.n_ue)
        for u in range(self.n_ue):
            q = self.queues[u]
            if q:
                out[u] = slot - q[0][1]
        return out

    def deadline_slack_slots(self, slot: int) -> np.ndarray:
        """
        Slots remaining before the head-of-line packet expires.

        Large positive values are reported for empty queues so that idle UEs
        never look urgent.
        """
        hol = self.head_of_line_delay_slots(slot)
        slack = self.delay_budget_slots - hol
        empty = self.backlog_bits <= 0.0
        slack[empty] = self.delay_budget_slots[empty]
        return slack

    def class_counts(self) -> Dict[str, int]:
        return {
            tc.name: int(np.sum(self.ue_class == i)) for i, tc in enumerate(TRAFFIC_CLASSES)
        }

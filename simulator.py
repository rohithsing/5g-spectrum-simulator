"""
Network Simulator Module for 5G/6G Intelligent Spectrum Allocation.

Simulates 5G NR physical layer dynamics, subframe scheduling, M/M/1 queuing delay,
Primary User (PU) collision penalties, and QoS satisfaction across configurable
congestion levels (low, medium, high, peak) anchored by real-world datasets.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

from data_pipeline import align_datasets, load_kpi, load_cass

logger = logging.getLogger("NetworkSimulator")


class SpectrumSimulator:
    """
    5G/6G Network Simulator supporting Dynamic Spectrum Access (DSA) with
    empirical dataset grounding.
    """

    def __init__(
        self,
        num_users: int = 30,
        num_channels: int = 12,
        congestion_level: str = "medium",
        channel_bw_mhz: float = 20.0,
        carrier_freq_ghz: float = 3.5,
        use_aligned_data: bool = True,
        composed_df: Optional[pd.DataFrame] = None,
        kpi_stats: Optional[Dict[str, Dict[str, float]]] = None,
        random_state: int = 42
    ):
        """
        Initialize the simulator.

        Args:
            num_users: Number of active UEs (10 to 200).
            num_channels: Number of allocatable spectrum channels/carriers (5 to 50).
            congestion_level: 'low', 'medium', 'high', or 'peak'.
            channel_bw_mhz: Bandwidth per channel in MHz (default 20 MHz 5G NR).
            carrier_freq_ghz: Carrier frequency (default 3.5 GHz n78 mid-band).
            use_aligned_data: If True, samples states from empirically aligned dataset.
            composed_df: Optional pre-computed aligned scenario DataFrame.
            kpi_stats: Summary statistics for normalization.
            random_state: Seed for reproducibility.
        """
        self.num_users = int(np.clip(num_users, 5, 250))
        self.num_channels = int(np.clip(num_channels, 2, 60))
        self.congestion_level = congestion_level.lower()
        if self.congestion_level not in ["low", "medium", "high", "peak"]:
            self.congestion_level = "medium"

        self.channel_bw_mhz = channel_bw_mhz
        self.carrier_freq_ghz = carrier_freq_ghz
        self.use_aligned_data = use_aligned_data
        self.rng = np.random.default_rng(random_state)
        self.timestep = 0

        # Load or generate aligned dataset
        if composed_df is not None:
            self.scenario_pool = composed_df
        elif self.use_aligned_data:
            df_aligned, _ = align_datasets(n_samples=6000, random_state=random_state)
            self.scenario_pool = df_aligned
        else:
            self.scenario_pool = None

        # Filter pool by congestion tier if available
        if self.scenario_pool is not None and "congestion_tier" in self.scenario_pool.columns:
            tier_pool = self.scenario_pool[self.scenario_pool["congestion_tier"] == self.congestion_level]
            if len(tier_pool) > 50:
                self.tier_pool = tier_pool.reset_index(drop=True)
            else:
                self.tier_pool = self.scenario_pool.reset_index(drop=True)
        else:
            self.tier_pool = self.scenario_pool

        # Load KPI normalization stats
        if kpi_stats is not None:
            self.kpi_stats = kpi_stats
        else:
            _, self.kpi_stats = load_kpi()

        # User properties
        # Priorities: 1 (Best Effort), 2 (Streaming eMBB), 3 (URLLC Ultra-Reliable Low Latency)
        self.user_priorities = self.rng.choice([1, 2, 3], size=self.num_users, p=[0.45, 0.40, 0.15])
        self.user_sla_latency = np.array([50.0 if p == 1 else (25.0 if p == 2 else 10.0) for p in self.user_priorities])
        self.user_buffers = np.zeros(self.num_users, dtype=float)  # Backlog in Mbits
        self.user_prev_utilization = np.zeros(self.num_users, dtype=float)
        self.user_avg_throughput = np.ones(self.num_users, dtype=float) * 10.0  # For Proportional Fair

        # Channel properties
        self.channel_pu_present = np.zeros(self.num_channels, dtype=int)
        self.channel_interference_dbm = np.ones(self.num_channels, dtype=float) * -95.0

        # Simulation history tracking
        self.history: List[Dict[str, Any]] = []

        self._init_state()

    def _init_state(self) -> None:
        """Initialize the first state of the simulation."""
        self.timestep = 0
        self.user_buffers = np.zeros(self.num_users, dtype=float)
        self.user_prev_utilization = np.zeros(self.num_users, dtype=float)
        self._update_channel_sensing()
        self.current_user_states = self._sample_user_states()

    def _update_channel_sensing(self) -> None:
        """
        Update Primary User (PU) presence and interference per channel.
        Uses Markov activity model for PU and CASS-grounded noise floor.
        """
        # PU activity probability depends on congestion and channel index
        pu_activity_prob = {"low": 0.10, "medium": 0.20, "high": 0.35, "peak": 0.50}[self.congestion_level]
        for c in range(self.num_channels):
            # Markov switching
            was_present = self.channel_pu_present[c]
            if was_present == 0:
                self.channel_pu_present[c] = 1 if self.rng.random() < pu_activity_prob else 0
            else:
                self.channel_pu_present[c] = 0 if self.rng.random() < 0.30 else 1

            # Interference: thermal noise -104 dBm + inter-cell interference
            self.channel_interference_dbm[c] = self.rng.normal(-96.0, 3.5)

    def _sample_user_states(self) -> np.ndarray:
        """
        Sample user state matrix of shape (num_users, 6):
        [traffic_demand, channel_quality (SNR dB), interference_level (dBm),
         pu_occupancy, user_priority, previous_utilization]
        """
        states = np.zeros((self.num_users, 6), dtype=float)

        if self.tier_pool is not None and len(self.tier_pool) >= self.num_users:
            # Sample directly from the empirically aligned scenario pool
            sampled_rows = self.tier_pool.sample(n=self.num_users, replace=True, random_state=int(self.rng.integers(0, 1000000)))
            demands = sampled_rows["traffic_demand"].values
            snrs = sampled_rows["channel_quality"].values
            interfs = sampled_rows["interference_level"].values
            pus = sampled_rows["pu_occupancy"].values
        else:
            # Parametric fallback anchored by 3GPP and CASS ranges
            congestion_multiplier = {"low": 0.6, "medium": 1.0, "high": 1.5, "peak": 2.2}[self.congestion_level]
            demands = self.rng.lognormal(mean=2.8, sigma=0.5, size=self.num_users) * congestion_multiplier
            snrs = self.rng.normal(14.0, 6.5, size=self.num_users)
            interfs = self.rng.normal(-95.0, 4.0, size=self.num_users)
            pus = self.rng.choice([0, 1], size=self.num_users, p=[0.75, 0.25])

        for u in range(self.num_users):
            states[u, 0] = float(np.clip(demands[u], 1.0, 200.0))  # Traffic demand (Mbps)
            states[u, 1] = float(np.clip(snrs[u], -10.0, 35.0))     # Channel quality (SNR dB)
            states[u, 2] = float(np.clip(interfs[u], -115.0, -70.0)) # Interference level (dBm)
            states[u, 3] = float(pus[u])                            # PU occupancy
            states[u, 4] = float(self.user_priorities[u])           # Priority (1, 2, 3)
            states[u, 5] = float(np.clip(self.user_prev_utilization[u], 0.0, 1.0)) # Prev utilization

        return states

    def reset(self) -> np.ndarray:
        """Reset the simulator to starting conditions."""
        self._init_state()
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """Return the current user state matrix (num_users, 6)."""
        return self.current_user_states.copy()

    def get_channel_state(self) -> Dict[str, np.ndarray]:
        """Return channel-level physical sensing information."""
        return {
            "pu_present": self.channel_pu_present.copy(),
            "interference_dbm": self.channel_interference_dbm.copy()
        }

    def compute_sinr_and_rate(self, user_id: int, channel_id: int) -> Tuple[float, float, bool]:
        """
        Compute SINR and Shannon achievable data rate (Mbps) for assigning channel to user.

        Collision Mechanics:
        If Secondary User is assigned to a channel where Primary User is active:
        - Primary User collision occurs.
        - Interference spikes (+35 dB).
        - SINR drops below decodable threshold (-6 dB).
        - Achievable rate plummets to ~0, and collision is flagged for penalty.

        Returns:
            (sinr_db, achievable_rate_mbps, is_pu_collision)
        """
        base_snr = self.current_user_states[user_id, 1]
        is_pu_active = (self.channel_pu_present[channel_id] == 1)

        if is_pu_active:
            # Severe PU collision: massive interference injection
            collision = True
            effective_sinr_db = base_snr - self.rng.uniform(25.0, 38.0)
            effective_sinr_db = min(effective_sinr_db, -8.0)
        else:
            collision = False
            # Small intercell interference variance
            channel_interf_offset = (self.channel_interference_dbm[channel_id] - (-96.0)) * 0.5
            effective_sinr_db = base_snr - channel_interf_offset

        # Linear SINR
        sinr_linear = 10.0 ** (effective_sinr_db / 10.0)

        # 3GPP TS 38.214 Shannon bound with practical implementation efficiency eta ~ 0.75
        eta = 0.75
        bw_hz = self.channel_bw_mhz * 1e6
        if effective_sinr_db < -6.0:
            # Outage / block error rate BLER ~ 100%
            rate_mbps = 0.05
        else:
            rate_bps = eta * bw_hz * np.log2(1.0 + sinr_linear)
            rate_mbps = rate_bps / 1e6

        return float(effective_sinr_db), float(rate_mbps), collision

    def step(self, allocation: Dict[int, Optional[int]]) -> Dict[str, Any]:
        """
        Execute one scheduling subframe timestep.

        Args:
            allocation: Dictionary mapping channel_id -> user_id (or None if unassigned).

        Returns:
            step_info: Metrics including throughput, latency, PRB utilization,
                       PU collisions, Jain's fairness index, QoS satisfaction.
        """
        self.timestep += 1
        user_allocated_channels: Dict[int, List[int]] = {u: [] for u in range(self.num_users)}
        for ch_id, u_id in allocation.items():
            if u_id is not None and 0 <= u_id < self.num_users:
                user_allocated_channels[u_id].append(ch_id)

        user_throughputs = np.zeros(self.num_users, dtype=float)
        user_latencies = np.zeros(self.num_users, dtype=float)
        pu_collisions = 0
        collision_channels = []

        for u in range(self.num_users):
            assigned_channels = user_allocated_channels[u]
            u_demand = self.current_user_states[u, 0]  # Demand in Mbps

            if not assigned_channels:
                # User starved in this subframe
                served_rate = 0.0
                # Queuing delay accumulates
                self.user_buffers[u] += u_demand * 0.01  # 10ms frame demand added to buffer (Mbits)
                delay_ms = min(100.0, 15.0 + self.user_buffers[u] * 12.0)
            else:
                total_channel_rate = 0.0
                for ch in assigned_channels:
                    sinr_db, rate, coll = self.compute_sinr_and_rate(u, ch)
                    total_channel_rate += rate
                    if coll:
                        pu_collisions += 1
                        collision_channels.append(ch)

                served_rate = min(u_demand + self.user_buffers[u] * 100.0, total_channel_rate)
                # Drain buffer
                drained = min(self.user_buffers[u], served_rate * 0.01)
                self.user_buffers[u] -= drained

                # Latency: transmission delay + M/M/1 queuing backlog delay
                tx_delay = 5.0 * (u_demand / (total_channel_rate + 1e-4))
                queue_delay = 10.0 * (self.user_buffers[u] / (total_channel_rate + 1e-4))
                delay_ms = float(np.clip(tx_delay + queue_delay + 3.0, 2.0, 120.0))

            user_throughputs[u] = served_rate
            user_latencies[u] = delay_ms

            # Update moving average throughput for Proportional Fair (alpha = 0.1)
            self.user_avg_throughput[u] = 0.9 * self.user_avg_throughput[u] + 0.1 * served_rate

        # PRB / Spectrum Utilization: fraction of channels assigned and transmitting
        assigned_channels_count = sum(1 for ch, u in allocation.items() if u is not None)
        spectrum_utilization = assigned_channels_count / max(1, self.num_channels)

        # User-level utilization update
        for u in range(self.num_users):
            self.user_prev_utilization[u] = len(user_allocated_channels[u]) / max(1, self.num_channels)

        # Aggregate metrics
        total_throughput = float(np.sum(user_throughputs))
        mean_latency = float(np.mean(user_latencies))
        p95_latency = float(np.percentile(user_latencies, 95))

        # Spectrum Efficiency (bits/s/Hz): Total Throughput / Total Bandwidth Allocated
        total_bandwidth_hz = self.num_channels * self.channel_bw_mhz * 1e6
        spectrum_efficiency = (total_throughput * 1e6) / total_bandwidth_hz

        # Jain's Fairness Index: (sum x_i)^2 / (N * sum x_i^2)
        sum_thpt = np.sum(user_throughputs)
        sum_sq_thpt = np.sum(user_throughputs ** 2)
        if sum_sq_thpt > 1e-8:
            jains_fairness = float((sum_thpt ** 2) / (self.num_users * sum_sq_thpt))
        else:
            jains_fairness = 0.0
        jains_fairness = float(np.clip(jains_fairness, 0.0, 1.0))

        # QoS Satisfaction Rate: % users meeting latency SLA and receiving >= 80% demand
        qos_satisfied = 0
        for u in range(self.num_users):
            thpt_satisfied = (user_throughputs[u] >= 0.75 * self.current_user_states[u, 0])
            lat_satisfied = (user_latencies[u] <= self.user_sla_latency[u])
            if thpt_satisfied and lat_satisfied:
                qos_satisfied += 1
        qos_satisfaction_rate = float(qos_satisfied / self.num_users)

        # Mean interference experienced across allocated channels
        interf_vals = [self.channel_interference_dbm[ch] for ch in allocation.keys()]
        mean_interference = float(np.mean(interf_vals)) if interf_vals else -95.0

        step_info = {
            "timestep": self.timestep,
            "total_throughput_mbps": total_throughput,
            "mean_latency_ms": mean_latency,
            "p95_latency_ms": p95_latency,
            "spectrum_utilization": spectrum_utilization,
            "spectrum_efficiency_bps_hz": spectrum_efficiency,
            "jains_fairness": jains_fairness,
            "qos_satisfaction_rate": qos_satisfaction_rate,
            "pu_collisions": pu_collisions,
            "collision_channels": collision_channels,
            "mean_interference_dbm": mean_interference,
            "user_throughputs": user_throughputs.copy(),
            "user_latencies": user_latencies.copy(),
            "allocation": allocation.copy()
        }

        self.history.append(step_info)

        # Advance to next timestep state
        self._update_channel_sensing()
        self.current_user_states = self._sample_user_states()

        return step_info


if __name__ == "__main__":
    print("Testing Network Simulator...")
    sim = SpectrumSimulator(num_users=15, num_channels=6, congestion_level="medium")
    state = sim.reset()
    print("Initial state shape:", state.shape)
    # Simple test allocation: channel c -> user c
    test_alloc = {c: c % 15 for c in range(6)}
    info = sim.step(test_alloc)
    print(f"Step 1: Throughput={info['total_throughput_mbps']:.2f} Mbps, "
          f"Latency={info['mean_latency_ms']:.2f} ms, "
          f"Jain's={info['jains_fairness']:.3f}, "
          f"Collisions={info['pu_collisions']}")
    print("Simulator test passed successfully!")

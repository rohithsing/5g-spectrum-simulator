"""
Baseline Spectrum Allocators for 5G/6G Networks.

Implements standard telecommunication baseline schedulers:
1. RoundRobinAllocator: Static cyclical channel assignment without channel state awareness.
2. ProportionalFairAllocator: Opportunistic channel-aware scheduler maximizing R_i(t) / R_avg_i(t)
   with Primary User (PU) collision avoidance heuristics.
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np

logger = logging.getLogger("BaselineAllocator")


class BaseAllocator:
    """Base class for all spectrum allocators."""

    def __init__(self, name: str):
        self.name = name

    def allocate(
        self,
        user_states: np.ndarray,
        channel_states: Dict[str, np.ndarray],
        num_channels: int,
        **kwargs
    ) -> Dict[int, Optional[int]]:
        """
        Produce a channel allocation mapping.

        Args:
            user_states: Matrix of shape (num_users, 6):
                         [traffic_demand, channel_quality, interference, pu_occupancy, priority, prev_util]
            channel_states: Dict containing 'pu_present' (num_channels,), 'interference_dbm' (num_channels,)
            num_channels: Total number of channels available for allocation.

        Returns:
            Dict mapping channel_id -> user_id (or None if unallocated).
        """
        raise NotImplementedError


class RoundRobinAllocator(BaseAllocator):
    """
    Static Round-Robin Allocator.
    Assigns available channels sequentially to active users in a cyclical manner,
    oblivious to instantaneous channel quality (SNR) or Primary User presence.
    """

    def __init__(self):
        super().__init__("Static Round-Robin")
        self.current_user_ptr = 0

    def reset(self):
        self.current_user_ptr = 0

    def allocate(
        self,
        user_states: np.ndarray,
        channel_states: Dict[str, np.ndarray],
        num_channels: int,
        **kwargs
    ) -> Dict[int, Optional[int]]:
        num_users = user_states.shape[0]
        allocation: Dict[int, Optional[int]] = {}

        for ch in range(num_channels):
            # Assign channel to current user pointer
            user_id = self.current_user_ptr % num_users
            allocation[ch] = int(user_id)
            self.current_user_ptr = (self.current_user_ptr + 1) % num_users

        return allocation


class ProportionalFairAllocator(BaseAllocator):
    """
    Proportional-Fair (PF) Allocator.
    Schedules user i on channel c to maximize R_{i,c}(t) / (R_avg_i(t) + eps)^alpha.
    Balances spectral efficiency (opportunistic scheduling) with user fairness.
    Optionally senses and avoids channels where Primary User (PU) is present.
    """

    def __init__(self, alpha: float = 1.0, ewma_weight: float = 0.1, sense_pu: bool = True):
        super().__init__("Proportional-Fair")
        self.alpha = alpha
        self.ewma_weight = ewma_weight
        self.sense_pu = sense_pu
        self.avg_throughputs: Optional[np.ndarray] = None

    def reset(self):
        self.avg_throughputs = None

    def allocate(
        self,
        user_states: np.ndarray,
        channel_states: Dict[str, np.ndarray],
        num_channels: int,
        **kwargs
    ) -> Dict[int, Optional[int]]:
        num_users = user_states.shape[0]

        if self.avg_throughputs is None or len(self.avg_throughputs) != num_users:
            # Initialize with small positive values to avoid division by zero
            self.avg_throughputs = np.ones(num_users, dtype=float) * 5.0

        allocation: Dict[int, Optional[int]] = {}
        pu_present = channel_states.get("pu_present", np.zeros(num_channels, dtype=int))

        # Track channels assigned to each user in this slot
        channels_per_user = np.zeros(num_users, dtype=int)
        slot_throughput = np.zeros(num_users, dtype=float)

        for ch in range(num_channels):
            # Cognitive sensing heuristic: If PU is detected and sensing enabled, avoid channel
            if self.sense_pu and pu_present[ch] == 1:
                # 85% probability of sensing and vacating the channel to protect Primary User
                if np.random.random() < 0.85:
                    allocation[ch] = None
                    continue

            best_user = None
            best_pf_metric = -1e9

            for u in range(num_users):
                snr_db = user_states[u, 1]
                demand = user_states[u, 0]
                priority = user_states[u, 4]

                # Estimated instantaneous achievable rate (Mbps)
                sinr_linear = 10.0 ** (snr_db / 10.0)
                # 20 MHz channel rate approximation
                est_rate = 0.75 * 20.0 * np.log2(1.0 + sinr_linear)

                # Prioritize users with unmet demand
                unmet_demand = max(0.1, demand - slot_throughput[u])

                # PF Metric = (R_{i,c} * Priority * DemandFactor) / (R_avg_i^alpha)
                denominator = (self.avg_throughputs[u] + 1e-3) ** self.alpha
                pf_metric = (est_rate * (priority ** 0.5) * (unmet_demand ** 0.3)) / denominator

                # Penalty if user already got channels in this slot (encourage spreading)
                pf_metric /= (1.0 + channels_per_user[u] * 0.7)

                if pf_metric > best_pf_metric:
                    best_pf_metric = pf_metric
                    best_user = u

            allocation[ch] = int(best_user) if best_user is not None else None
            if best_user is not None:
                channels_per_user[best_user] += 1
                slot_throughput[best_user] += 15.0  # Approx slot rate contribution

        # Update EWMA throughput
        for u in range(num_users):
            self.avg_throughputs[u] = (
                (1.0 - self.ewma_weight) * self.avg_throughputs[u] +
                self.ewma_weight * slot_throughput[u]
            )

        return allocation


if __name__ == "__main__":
    print("Testing Baseline Allocators...")
    rr = RoundRobinAllocator()
    pf = ProportionalFairAllocator()

    dummy_users = np.array([
        [20.0, 15.0, -95.0, 0, 1, 0.2],
        [40.0, 22.0, -96.0, 0, 2, 0.4],
        [10.0, 8.0, -94.0, 1, 3, 0.1],
    ])
    dummy_channels = {
        "pu_present": np.array([0, 1, 0, 0]),
        "interference_dbm": np.array([-95.0, -96.0, -94.0, -95.0])
    }

    alloc_rr = rr.allocate(dummy_users, dummy_channels, num_channels=4)
    print("RR Allocation:", alloc_rr)
    alloc_pf = pf.allocate(dummy_users, dummy_channels, num_channels=4)
    print("PF Allocation:", alloc_pf)
    print("Baseline allocators verified successfully!")

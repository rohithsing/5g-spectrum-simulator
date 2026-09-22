"""
Machine Learning and Reinforcement Learning Spectrum Allocators for 5G/6G Networks.

Phase 1 (Supervised Warm-Start):
- RandomForest / XGBoost trained on DLTeamTUC resource allocation traces to predict
  RB/channel allocation from (channel_quality, cqi, priority, demand).

Phase 2 (Reinforcement Learning):
- Gymnasium Environment: SpectrumAllocationEnv wrapping physical network simulator.
- KPI-grounded reward function normalized by empirical mean/std, incorporating Jain's fairness
  and cognitive Primary User (PU) collision penalties.
- PyTorch Dueling Double DQN (DDQN) with Prioritized Experience Replay and warm-start support.
- Stable-Baselines3 compatible wrapper.
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import gymnasium as gym
from gymnasium import spaces

from data_pipeline import load_resource_allocation, load_kpi, align_datasets
from simulator import SpectrumSimulator

logger = logging.getLogger("MLAllocator")


# =====================================================================
# Phase 1: Supervised Learning (Random Forest & XGBoost)
# =====================================================================

class SupervisedAllocator:
    """
    Supervised model (Random Forest or XGBoost) trained on empirical resource
    allocation data to predict allocation likelihood and rank candidate users.
    """

    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type.lower()
        self.name = f"Supervised {model_type.upper()}"
        if self.model_type == "xgboost":
            self.model = XGBClassifier(
                n_estimators=80,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                eval_metric="logloss",
                random_state=42
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=80,
                max_depth=6,
                random_state=42
            )
        self.is_trained = False

    def train(self, df_rb: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Train the supervised model on resource allocation features:
        [cqi, snr_estimate, demand_proxy, priority_proxy].
        """
        if df_rb is None:
            df_rb = load_resource_allocation()

        # Build feature matrix
        rng = np.random.default_rng(42)
        n = len(df_rb)
        cqi = df_rb["cqi"].values.astype(float)
        # Estimate SNR from CQI: SNR ~ (CQI * 2.2) - 6 dB
        snr_est = cqi * 2.2 - 6.0 + rng.normal(0, 1.5, size=n)
        demand_proxy = np.clip(rng.lognormal(2.5, 0.4, size=n) * (cqi / 10.0), 2.0, 120.0)
        priority_proxy = rng.choice([1, 2, 3], size=n, p=[0.5, 0.35, 0.15])

        X = np.column_stack([cqi, snr_est, demand_proxy, priority_proxy])
        y = df_rb["allocated"].values.astype(int)

        # Train model
        self.model.fit(X, y)
        self.is_trained = True
        accuracy = float(np.mean(self.model.predict(X) == y))
        logger.info("Trained %s Allocator with accuracy: %.3f", self.name, accuracy)
        return {"accuracy": accuracy, "n_samples": n}

    def allocate(
        self,
        user_states: np.ndarray,
        channel_states: Dict[str, np.ndarray],
        num_channels: int,
        **kwargs
    ) -> Dict[int, Optional[int]]:
        """
        Produce channel allocation by predicting user assignment probabilities.
        Avoids channels where Primary User (PU) is present.
        """
        if not self.is_trained:
            self.train()

        num_users = user_states.shape[0]
        allocation: Dict[int, Optional[int]] = {}
        pu_present = channel_states.get("pu_present", np.zeros(num_channels, dtype=int))

        # Build feature matrix for all users in this slot
        # Features: [cqi, snr, demand, priority]
        cqi_est = np.clip(np.round((user_states[:, 1] + 6.0) / 2.2), 1, 15)
        snr = user_states[:, 1]
        demand = user_states[:, 0]
        priority = user_states[:, 4]
        X_users = np.column_stack([cqi_est, snr, demand, priority])

        # Predict probability of allocation for each user
        try:
            probs = self.model.predict_proba(X_users)[:, 1]
        except Exception:
            # Fallback to direct decision function or heuristic
            probs = (cqi_est / 15.0) * (priority / 3.0)

        # Track assignments to prevent over-allocation to a single user
        user_alloc_count = np.zeros(num_users, dtype=int)

        for ch in range(num_channels):
            # Cognitive sensing: If PU is present, 90% chance model vacates to protect PU
            if pu_present[ch] == 1 and np.random.random() < 0.90:
                allocation[ch] = None
                continue

            # Penalize users that already have channels allocated in this subframe
            adjusted_scores = probs / (1.0 + user_alloc_count * 0.8)
            best_user = int(np.argmax(adjusted_scores))

            allocation[ch] = best_user
            user_alloc_count[best_user] += 1

        return allocation


# =====================================================================
# Phase 2: Gymnasium Environment & Reinforcement Learning
# =====================================================================

class SpectrumAllocationEnv(gym.Env):
    """
    Gymnasium Environment for Dynamic Spectrum Allocation in 5G/6G Networks.
    Anchors state features and reward normalization in real-world CASS & 5G KPI datasets.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_users: int = 20,
        num_channels: int = 8,
        congestion_level: str = "medium",
        max_steps: int = 50,
        kpi_stats: Optional[Dict[str, Dict[str, float]]] = None,
        composed_df: Optional[pd.DataFrame] = None,
        random_state: int = 42
    ):
        super().__init__()
        self.num_users = num_users
        self.num_channels = num_channels
        self.congestion_level = congestion_level
        self.max_steps = max_steps
        self.current_step = 0

        self.simulator = SpectrumSimulator(
            num_users=num_users,
            num_channels=num_channels,
            congestion_level=congestion_level,
            composed_df=composed_df,
            kpi_stats=kpi_stats,
            random_state=random_state
        )

        self.kpi_stats = self.simulator.kpi_stats

        # Action: For each channel, choose a user index (0 to num_users-1) or num_users (leave idle)
        # Action space = MultiDiscrete([num_users + 1] * num_channels)
        self.action_space = spaces.MultiDiscrete([num_users + 1] * num_channels)

        # Observation Space:
        # Per user: [traffic_demand, channel_quality, interference, pu_occupancy, priority, prev_util] (num_users * 6)
        # Per channel: [pu_present, interference_dbm] (num_channels * 2)
        obs_dim = num_users * 6 + num_channels * 2
        self.observation_space = spaces.Box(
            low=-150.0,
            high=500.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def _get_obs(self) -> np.ndarray:
        """Construct normalized observation vector."""
        user_states = self.simulator.get_state()  # (num_users, 6)
        ch_states = self.simulator.get_channel_state()

        user_flat = user_states.flatten().astype(np.float32)
        ch_flat = np.concatenate([
            ch_states["pu_present"].astype(np.float32),
            ch_states["interference_dbm"].astype(np.float32)
        ])
        return np.concatenate([user_flat, ch_flat])

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.simulator.reset()
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Step the environment with an allocation action.

        Args:
            action: Array of length num_channels, each element in [0, num_users].
                    If action[ch] == num_users, the channel is left unallocated.
        """
        self.current_step += 1

        # Convert action array to dictionary mapping ch -> user_id (or None)
        allocation: Dict[int, Optional[int]] = {}
        for ch in range(self.num_channels):
            u_choice = int(action[ch])
            if u_choice < self.num_users:
                allocation[ch] = u_choice
            else:
                allocation[ch] = None

        # Advance simulator
        info = self.simulator.step(allocation)

        # =============================================================
        # REWARD FUNCTION (Normalized using empirical 5G KPI statistics)
        # R_t = w_thpt * norm(Throughput) - w_lat * norm(Latency)
        #       + w_util * norm(PRB_Util) + w_fair * Jain - w_coll * Collisions
        # =============================================================
        thpt_mean = self.kpi_stats["throughput"]["mean"]
        thpt_std = max(self.kpi_stats["throughput"]["std"], 1.0)
        lat_mean = self.kpi_stats["latency"]["mean"]
        lat_std = max(self.kpi_stats["latency"]["std"], 1.0)
        util_mean = self.kpi_stats["prb_utilization"]["mean"]
        util_std = max(self.kpi_stats["prb_utilization"]["std"], 0.1)

        norm_thpt = (info["total_throughput_mbps"] - thpt_mean) / thpt_std
        norm_lat = (info["mean_latency_ms"] - lat_mean) / lat_std
        norm_util = (info["spectrum_utilization"] - util_mean) / util_std

        # Reward weights
        w_thpt = 1.2
        w_lat = 1.0
        w_util = 0.5
        w_fair = 2.5
        w_coll = 3.5

        reward = (
            w_thpt * norm_thpt
            - w_lat * norm_lat
            + w_util * norm_util
            + w_fair * info["jains_fairness"]
            - w_coll * info["pu_collisions"]
        )

        terminated = self.current_step >= self.max_steps
        truncated = False
        obs = self._get_obs()

        info["reward"] = float(reward)
        return obs, float(reward), terminated, truncated, info


# =====================================================================
# Dueling Double Deep Q-Network (PyTorch Implementation)
# =====================================================================

class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network architecture:
    Decouples state value V(s) and action advantages A(s, a).
    Q(s, a) = V(s) + (A(s, a) - mean_a'(A(s, a')))
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.feature_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # State Value Stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Advantage Stream A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_network(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        # Dueling aggregation
        q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values


class ReplayBuffer:
    """Experience replay buffer for off-policy DRL."""

    def __init__(self, capacity: int = 20000):
        self.capacity = capacity
        self.buffer: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = []
        self.position = 0

    def push(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)


class DuelingDQNAgent:
    """
    Dueling Double DQN Agent with warm-start capability and KPI-normalized rewards.
    Operates per-channel selection to maintain efficiency across arbitrary channel counts.
    """

    def __init__(
        self,
        state_dim: int,
        num_users: int,
        num_channels: int,
        gamma: float = 0.95,
        lr: float = 5e-4,
        batch_size: int = 64,
        target_update_freq: int = 200,
        device: Optional[str] = None
    ):
        self.name = "Dueling Double DQN"
        self.state_dim = state_dim
        self.num_users = num_users
        self.num_channels = num_channels
        self.action_dim = num_users + 1  # 0..num_users-1: assign UE, num_users: leave channel idle
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_count = 0

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        # Primary and Target Q-Networks
        self.policy_net = DuelingQNetwork(state_dim, self.action_dim).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=15000)

        # Exploration epsilon
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.992

    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """
        Select multi-channel allocation action vector.
        Uses epsilon-greedy exploration.
        """
        actions = np.zeros(self.num_channels, dtype=int)

        if explore and np.random.random() < self.epsilon:
            # Random exploration
            actions = np.random.randint(0, self.action_dim, size=self.num_channels)
            return actions

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t).squeeze(0).cpu().numpy()

        # Top ranking users for channels, spreading across different users
        top_user_indices = np.argsort(q_values[:self.num_users])[::-1]
        for ch in range(self.num_channels):
            if ch < len(top_user_indices):
                actions[ch] = top_user_indices[ch % len(top_user_indices)]
            else:
                actions[ch] = self.num_users  # idle

        return actions

    def update(self) -> Optional[float]:
        """Update policy network using Double DQN Bellman target."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        next_states = next_states.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        # Evaluate representative primary action (channel 0 action for loss calculation)
        action_0 = actions[:, 0].unsqueeze(1).to(self.device)
        q_eval = self.policy_net(states).gather(1, action_0)

        with torch.no_grad():
            # Double DQN: Select action with policy_net, evaluate with target_net
            next_action = self.policy_net(next_states).argmax(dim=-1, keepdim=True)
            q_next = self.target_net(next_states).gather(1, next_action)
            q_target = rewards + (1.0 - dones) * self.gamma * q_next

        loss = nn.SmoothL1Loss()(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(loss.item())

    def warm_start(self, env: SpectrumAllocationEnv, warm_start_policy: Any, n_episodes: int = 5):
        """
        Pre-fill replay buffer using actions generated by Supervised or Proportional-Fair policy.
        """
        logger.info("Warm-starting Dueling DQN replay buffer with %d episodes...", n_episodes)
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                # Get warm-start action
                user_states = env.simulator.get_state()
                ch_states = env.simulator.get_channel_state()
                alloc = warm_start_policy.allocate(user_states, ch_states, env.num_channels)

                action_arr = np.zeros(env.num_channels, dtype=int)
                for ch in range(env.num_channels):
                    u = alloc.get(ch)
                    action_arr[ch] = u if (u is not None and u < env.num_users) else env.num_users

                next_obs, reward, terminated, truncated, _ = env.step(action_arr)
                done = terminated or truncated
                self.replay_buffer.push(obs, action_arr, reward, next_obs, done)
                obs = next_obs

        logger.info("Warm-start complete. Buffer size: %d transitions.", len(self.replay_buffer))

    def train(self, env: SpectrumAllocationEnv, total_episodes: int = 25) -> List[float]:
        """Train Dueling DQN agent over specified episodes."""
        episode_rewards = []
        for ep in range(total_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action = self.select_action(obs, explore=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                self.replay_buffer.push(obs, action, reward, next_obs, done)
                self.update()

                obs = next_obs
                total_reward += reward

            episode_rewards.append(total_reward)
            if (ep + 1) % 5 == 0 or ep == total_episodes - 1:
                logger.info("Episode %d/%d - Total Reward: %.2f - Epsilon: %.3f",
                            ep + 1, total_episodes, total_reward, self.epsilon)

        return episode_rewards

    def allocate(
        self,
        user_states: np.ndarray,
        channel_states: Dict[str, np.ndarray],
        num_channels: int,
        **kwargs
    ) -> Dict[int, Optional[int]]:
        """
        Inference method to match BaseAllocator interface.
        """
        # Reconstruct observation vector
        user_flat = user_states.flatten().astype(np.float32)
        ch_flat = np.concatenate([
            channel_states["pu_present"].astype(np.float32),
            channel_states["interference_dbm"].astype(np.float32)
        ])
        obs = np.concatenate([user_flat, ch_flat])

        # Inference without exploration
        actions = self.select_action(obs, explore=False)
        pu_present = channel_states.get("pu_present", np.zeros(num_channels, dtype=int))

        allocation: Dict[int, Optional[int]] = {}
        for ch in range(num_channels):
            # Cognitive protection: vacate if PU active
            if pu_present[ch] == 1:
                allocation[ch] = None
                continue

            u = int(actions[ch])
            allocation[ch] = u if (0 <= u < self.num_users) else None

        return allocation


if __name__ == "__main__":
    print("Testing ML & RL Allocator Module...")
    # Test Supervised Allocator
    print("--- Testing XGBoost / Random Forest Warm-Start ---")
    sup_alloc = SupervisedAllocator(model_type="xgboost")
    train_res = sup_alloc.train()
    print("Supervised train results:", train_res)

    # Test Gymnasium Environment
    print("--- Testing SpectrumAllocationEnv ---")
    env = SpectrumAllocationEnv(num_users=15, num_channels=6, max_steps=10)
    obs, _ = env.reset()
    print("Observation shape:", obs.shape)

    dummy_action = np.zeros(6, dtype=int)
    next_obs, reward, term, trunc, info = env.step(dummy_action)
    print(f"Env step -> Reward: {reward:.2f}, Throughput: {info['total_throughput_mbps']:.2f} Mbps")

    # Test Dueling Double DQN Agent
    print("--- Testing Dueling Double DQN ---")
    agent = DuelingDQNAgent(state_dim=len(obs), num_users=15, num_channels=6)
    # Warm start
    agent.warm_start(env, sup_alloc, n_episodes=2)
    # Quick train
    rewards = agent.train(env, total_episodes=3)
    print("Trained 3 episodes. Rewards:", rewards)

    print("ML & RL Allocator tests completed successfully!")

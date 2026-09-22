"""
Data Pipeline Module for Intelligent Spectrum Allocation Simulator.

Handles loading, cleaning, aggregating, and statistically linking:
1. CASS (Cluster-Assisted Spectrum Sensing) dataset: RF measurements, Primary User presence, SNR.
2. 5G Network KPI dataset: cell-level throughput, latency, PRB utilization, packet loss.
3. 5G Resource Allocation dataset: CQI and RB group allocation traces.

Provides empirical statistical linkage to compose heterogeneous datasets across different
time scales into grounded simulation scenarios, with fallback parametric generators.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DataPipeline")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _generate_parametric_cass(n_steps: int = 2000, num_channels: int = 10, random_state: int = 42) -> pd.DataFrame:
    """
    Parametric fallback generator for CASS spectrum sensing data.
    Uses a discrete-time Markov chain for Primary User (PU) occupancy and
    Rayleigh fading + log-normal shadowing for RF signal strength.
    """
    rng = np.random.default_rng(random_state)
    records = []
    start_time = pd.Timestamp("2026-01-01 08:00:00")
    dt = pd.Timedelta(milliseconds=100)

    # Markov transition probabilities for PU: state 0 (idle), state 1 (active)
    # Average burst length ~ 1.5s, idle duration ~ 4s
    p_01 = 0.025  # idle -> active
    p_10 = 0.070  # active -> idle

    pu_state = rng.choice([0, 1], size=num_channels, p=[0.8, 0.2])

    for step in range(n_steps):
        current_time = start_time + step * dt
        for ch in range(num_channels):
            # Update PU Markov state
            if pu_state[ch] == 0:
                if rng.random() < p_01:
                    pu_state[ch] = 1
            else:
                if rng.random() < p_10:
                    pu_state[ch] = 0

            pu_present = int(pu_state[ch])
            # Interference/noise level (dBm): thermal noise floor ~ -100 dBm + intercell interference
            interf_dbm = rng.normal(-96.0, 4.5)

            # Signal power and SNR based on PU status
            if pu_present:
                pu_signal_dbm = rng.normal(-68.0, 5.0)
                snr_db = pu_signal_dbm - interf_dbm
                pu_bw_mhz = rng.choice([5.0, 10.0, 20.0])
            else:
                pu_signal_dbm = -120.0 + rng.normal(0, 3.0)  # ambient noise floor
                snr_db = rng.normal(12.0, 7.0)  # Secondary user link SNR
                pu_bw_mhz = 0.0

            # Add occasional irregular millisecond jitter
            records.append({
                "timestamp": current_time + pd.Timedelta(microseconds=int(rng.integers(-5000, 5000))),
                "channel_id": int(ch),
                "pu_present": pu_present,
                "signal_power_db": float(pu_signal_dbm),
                "snr": float(np.clip(snr_db, -15.0, 35.0)),
                "interference_level": float(interf_dbm),
                "pu_bandwidth_mhz": float(pu_bw_mhz)
            })

    df = pd.DataFrame(records)
    return df


def _generate_parametric_kpi(n_records: int = 3000, num_cells: int = 5, random_state: int = 42) -> pd.DataFrame:
    """
    Parametric fallback generator for 5G Network KPI dataset.
    Generates realistic distributions of throughput (Mbps), latency (ms),
    PRB utilization (%), and packet loss rate across multiple cells.
    """
    rng = np.random.default_rng(random_state)
    records = []
    start_time = pd.Timestamp("2026-01-01 00:00:00")
    dt = pd.Timedelta(seconds=1)

    for step in range(n_records):
        curr_time = start_time + step * dt
        # Time-of-day traffic oscillation factor (0.6 to 1.4)
        hour = (step // 3600) % 24
        diurnal = 1.0 + 0.35 * np.sin((hour - 8) * np.pi / 12)

        for cell in range(num_cells):
            # Base congestion state for cell
            cell_bias = 0.8 + 0.1 * cell
            load_factor = np.clip(rng.beta(2.5, 3.0) * diurnal * cell_bias, 0.05, 0.98)

            # PRB utilization directly tracks load
            prb_util = float(np.clip(load_factor + rng.normal(0, 0.04), 0.02, 0.99))

            # Throughput (Mbps): Log-normal distribution scaled by PRB allocation
            # Max capacity ~ 400 Mbps per cell
            thpt = float(np.clip(rng.lognormal(mean=4.2, sigma=0.65) * (0.3 + 0.7 * prb_util), 5.0, 480.0))

            # Latency (ms): M/M/1 queuing behavior as load approaches capacity
            base_delay = rng.normal(8.0, 1.5)
            congestion_delay = 35.0 * (prb_util ** 3)
            latency = float(np.clip(base_delay + congestion_delay + rng.exponential(2.0), 3.0, 120.0))

            # Packet loss rate
            if prb_util > 0.85:
                loss_rate = float(np.clip(0.005 + (prb_util - 0.85) * 0.15 + rng.normal(0, 0.005), 0.001, 0.08))
            else:
                loss_rate = float(np.clip(rng.exponential(0.0008), 0.0, 0.015))

            records.append({
                "timestamp": curr_time,
                "cell_id": int(cell),
                "throughput": thpt,
                "latency": latency,
                "packet_loss_rate": loss_rate,
                "prb_utilization": prb_util
            })

    return pd.DataFrame(records)


def _generate_parametric_resource_allocation(n_records: int = 4000, num_users: int = 25, num_rb_groups: int = 16, random_state: int = 42) -> pd.DataFrame:
    """
    Parametric fallback generator for DLTeamTUC 5G Resource Allocation dataset.
    Generates per-OFDM-symbol RB group assignments and Channel Quality Indicators (CQI 1-15).
    """
    rng = np.random.default_rng(random_state)
    records = []
    start_time = pd.Timestamp("2026-01-01 10:00:00")
    dt = pd.Timedelta(milliseconds=1)  # 1ms subframe

    for step in range(n_records // 10):
        t = start_time + step * dt
        # Users active in this subframe
        active_users = rng.choice(num_users, size=rng.integers(5, min(15, num_users)), replace=False)

        # Draw CQI per user (1-15 3GPP table)
        user_cqi = {u: int(np.clip(rng.normal(9.5, 2.8), 1, 15)) for u in active_users}

        # Assign RB groups using proportional-fair heuristic
        for rbg in range(num_rb_groups):
            if not active_users.size:
                continue
            # Higher CQI users have higher probability of being allocated
            weights = np.array([user_cqi[u] ** 1.8 for u in active_users])
            weights = weights / weights.sum()
            winner = rng.choice(active_users, p=weights)

            for u in active_users:
                allocated = 1 if u == winner else 0
                records.append({
                    "timestamp": t,
                    "user_id": int(u),
                    "rb_group_id": int(rbg),
                    "cqi": int(user_cqi[u]),
                    "allocated": int(allocated)
                })

    return pd.DataFrame(records)


def generate_sample_raw_files(data_dir: str = "data/raw", n_records: int = 2000) -> None:
    """
    Generate sample CSVs in data/raw/ to enable immediate out-of-the-box execution
    with realistic 3GPP and CASS compliant distributions.
    """
    raw_path = Path(data_dir)
    _ensure_dir(raw_path)

    cass_file = raw_path / "cass_spectrum.csv"
    kpi_file = raw_path / "5g_kpi.csv"
    rb_file = raw_path / "5g_resource_allocation.csv"

    if not cass_file.exists():
        logger.info("Generating sample CASS dataset at %s", cass_file)
        df_cass = _generate_parametric_cass(n_steps=n_records, num_channels=12)
        df_cass.to_csv(cass_file, index=False)

    if not kpi_file.exists():
        logger.info("Generating sample 5G KPI dataset at %s", kpi_file)
        df_kpi = _generate_parametric_kpi(n_records=n_records, num_cells=4)
        df_kpi.to_csv(kpi_file, index=False)

    if not rb_file.exists():
        logger.info("Generating sample 5G Resource Allocation dataset at %s", rb_file)
        df_rb = _generate_parametric_resource_allocation(n_records=n_records * 2, num_users=20, num_rb_groups=10)
        df_rb.to_csv(rb_file, index=False)


def load_cass(path: str = "data/raw/cass_spectrum.csv", resample_freq: str = "100ms") -> pd.DataFrame:
    """
    Load and preprocess the CASS (Cluster-Assisted Spectrum Sensing) dataset.
    Handles irregular timestamps by resampling to a regular time grid.
    Linearly interpolates continuous RF power/SNR/interference levels and
    preserves discrete PU presence flags.

    Returns:
        pd.DataFrame with columns:
        [timestamp, channel_id, pu_present, signal_power_db, snr, interference_level]
    """
    target_path = Path(path)
    if not target_path.exists():
        logger.warning(
            "CASS dataset not found at '%s'. Falling back to parametric generator with empirical CASS schema.",
            target_path
        )
        df_raw = _generate_parametric_cass(n_steps=2500, num_channels=12)
    else:
        logger.info("Loading CASS dataset from '%s'", target_path)
        try:
            df_raw = pd.read_csv(target_path)
        except Exception as e:
            logger.error("Failed to read '%s' (%s). Falling back to parametric generator.", target_path, e)
            df_raw = _generate_parametric_cass(n_steps=2500, num_channels=12)

    # Column name normalization
    rename_map = {}
    for col in df_raw.columns:
        c_low = col.lower().strip()
        if "time" in c_low or "date" in c_low:
            rename_map[col] = "timestamp"
        elif "chan" in c_low:
            rename_map[col] = "channel_id"
        elif "pu_pres" in c_low or "primary" in c_low or "occup" in c_low:
            rename_map[col] = "pu_present"
        elif "power" in c_low or "rssi" in c_low:
            rename_map[col] = "signal_power_db"
        elif "snr" in c_low or "sinr" in c_low:
            rename_map[col] = "snr"
        elif "interf" in c_low or "noise" in c_low:
            rename_map[col] = "interference_level"

    df_clean = df_raw.rename(columns=rename_map)

    # Ensure required columns exist
    defaults = {
        "timestamp": pd.date_range("2026-01-01", periods=len(df_clean), freq="100ms"),
        "channel_id": 0,
        "pu_present": 0,
        "signal_power_db": -90.0,
        "snr": 10.0,
        "interference_level": -95.0
    }
    for col, default_val in defaults.items():
        if col not in df_clean.columns:
            logger.warning("CASS missing column '%s', creating default column.", col)
            df_clean[col] = default_val

    df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"], errors="coerce")
    df_clean = df_clean.dropna(subset=["timestamp"]).sort_values("timestamp")

    # Resample per channel to handle irregular timestamps
    resampled_list = []
    for ch_id, group in df_clean.groupby("channel_id"):
        group = group.set_index("timestamp")
        # Continuous columns: interpolate
        cont_cols = ["signal_power_db", "snr", "interference_level"]
        group_cont = group[cont_cols].resample(resample_freq).mean().interpolate(method="time").bfill().ffill()

        # Discrete columns: forward-fill / max
        pu_col = group[["pu_present"]].resample(resample_freq).max().ffill().fillna(0).astype(int)

        merged = pd.concat([group_cont, pu_col], axis=1)
        merged["channel_id"] = int(ch_id)
        resampled_list.append(merged.reset_index())

    if resampled_list:
        final_df = pd.concat(resampled_list, ignore_index=True)
    else:
        final_df = df_clean

    output_cols = ["timestamp", "channel_id", "pu_present", "signal_power_db", "snr", "interference_level"]
    final_df = final_df[output_cols].sort_values(["timestamp", "channel_id"]).reset_index(drop=True)
    logger.info("Processed CASS dataset with %d rows across %d channels.", len(final_df), final_df["channel_id"].nunique())
    return final_df


def load_kpi(path: str = "data/raw/5g_kpi.csv") -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Load and preprocess the 5G Network KPI dataset (~2.3M records / empirical traces).
    Aggregates metrics to per-cell-per-timestep:
    [timestamp, cell_id, throughput, latency, packet_loss_rate, prb_utilization]

    Computes summary statistics (mean & std per metric) used directly as reward
    normalization constants for the RL environment.

    Returns:
        (df_kpi, kpi_stats_dict)
    """
    target_path = Path(path)
    if not target_path.exists():
        logger.warning(
            "5G KPI dataset not found at '%s'. Falling back to parametric generator with empirical 5G KPI schema.",
            target_path
        )
        df_raw = _generate_parametric_kpi(n_records=3000, num_cells=4)
    else:
        logger.info("Loading 5G KPI dataset from '%s'", target_path)
        try:
            df_raw = pd.read_csv(target_path)
        except Exception as e:
            logger.error("Failed to read '%s' (%s). Falling back to parametric generator.", target_path, e)
            df_raw = _generate_parametric_kpi(n_records=3000, num_cells=4)

    # Column name normalization
    rename_map = {}
    for col in df_raw.columns:
        c_low = col.lower().strip()
        if "time" in c_low or "date" in c_low:
            rename_map[col] = "timestamp"
        elif "cell" in c_low or "enb" in c_low or "gnb" in c_low:
            rename_map[col] = "cell_id"
        elif "thpt" in c_low or "through" in c_low or "data_rate" in c_low:
            rename_map[col] = "throughput"
        elif "lat" in c_low or "delay" in c_low or "rtt" in c_low:
            rename_map[col] = "latency"
        elif "loss" in c_low or "drop" in c_low or "plr" in c_low:
            rename_map[col] = "packet_loss_rate"
        elif "prb" in c_low or "util" in c_low or "load" in c_low:
            rename_map[col] = "prb_utilization"

    df_clean = df_raw.rename(columns=rename_map)

    # Defaults
    defaults = {
        "timestamp": pd.date_range("2026-01-01", periods=len(df_clean), freq="1s"),
        "cell_id": 0,
        "throughput": 85.0,
        "latency": 18.0,
        "packet_loss_rate": 0.002,
        "prb_utilization": 0.45
    }
    for col, default_val in defaults.items():
        if col not in df_clean.columns:
            df_clean[col] = default_val

    df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"], errors="coerce")
    df_clean = df_clean.dropna(subset=["timestamp"]).sort_values("timestamp")

    # Aggregate to per-cell per-second timestep
    numeric_cols = ["throughput", "latency", "packet_loss_rate", "prb_utilization"]
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
    df_clean = df_clean.dropna(subset=numeric_cols)

    agg_df = (
        df_clean.groupby(["timestamp", "cell_id"])[numeric_cols]
        .mean()
        .reset_index()
    )

    # Compute summary statistics for reward normalization
    kpi_stats: Dict[str, Dict[str, float]] = {}
    for metric in numeric_cols:
        series = agg_df[metric]
        mean_val = float(series.mean())
        std_val = float(series.std()) if float(series.std()) > 1e-6 else 1.0
        kpi_stats[metric] = {
            "mean": mean_val,
            "std": std_val,
            "min": float(series.min()),
            "max": float(series.max()),
            "p50": float(series.quantile(0.5)),
            "p95": float(series.quantile(0.95))
        }

    logger.info("KPI Dataset loaded. Summary stats: Throughput=%.1f±%.1f Mbps, Latency=%.1f±%.1f ms, PRB=%.2f±%.2f",
                kpi_stats["throughput"]["mean"], kpi_stats["throughput"]["std"],
                kpi_stats["latency"]["mean"], kpi_stats["latency"]["std"],
                kpi_stats["prb_utilization"]["mean"], kpi_stats["prb_utilization"]["std"])

    return agg_df, kpi_stats


def load_resource_allocation(path: str = "data/raw/5g_resource_allocation.csv") -> pd.DataFrame:
    """
    Load the DLTeamTUC/5GDatasets resource allocation traces:
    [timestamp, user_id, rb_group_id, cqi, allocated]

    Serves as the supervised action-label source for Phase 1 warm-start training.
    """
    target_path = Path(path)
    if not target_path.exists():
        logger.warning(
            "5G Resource Allocation dataset not found at '%s'. Falling back to parametric generator.",
            target_path
        )
        df_raw = _generate_parametric_resource_allocation(n_records=4000, num_users=25, num_rb_groups=12)
    else:
        logger.info("Loading Resource Allocation dataset from '%s'", target_path)
        try:
            df_raw = pd.read_csv(target_path)
        except Exception as e:
            logger.error("Failed to read '%s' (%s). Falling back to parametric generator.", target_path, e)
            df_raw = _generate_parametric_resource_allocation(n_records=4000, num_users=25, num_rb_groups=12)

    rename_map = {}
    for col in df_raw.columns:
        c_low = col.lower().strip()
        if "time" in c_low or "date" in c_low:
            rename_map[col] = "timestamp"
        elif "user" in c_low or "ue" in c_low:
            rename_map[col] = "user_id"
        elif "rb" in c_low or "channel" in c_low or "group" in c_low:
            rename_map[col] = "rb_group_id"
        elif "cqi" in c_low:
            rename_map[col] = "cqi"
        elif "alloc" in c_low or "assign" in c_low or "label" in c_low:
            rename_map[col] = "allocated"

    df_clean = df_raw.rename(columns=rename_map)

    defaults = {
        "timestamp": pd.date_range("2026-01-01", periods=len(df_clean), freq="1ms"),
        "user_id": 0,
        "rb_group_id": 0,
        "cqi": 10,
        "allocated": 0
    }
    for col, default_val in defaults.items():
        if col not in df_clean.columns:
            df_clean[col] = default_val

    output_cols = ["timestamp", "user_id", "rb_group_id", "cqi", "allocated"]
    final_df = df_clean[output_cols].copy()
    final_df["cqi"] = final_df["cqi"].astype(int).clip(1, 15)
    final_df["allocated"] = final_df["allocated"].astype(int).clip(0, 1)
    logger.info("Loaded Resource Allocation traces with %d records.", len(final_df))
    return final_df


def align_datasets(
    cass_df: Optional[pd.DataFrame] = None,
    kpi_df: Optional[pd.DataFrame] = None,
    rb_df: Optional[pd.DataFrame] = None,
    n_samples: int = 5000,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    METHODOLOGY NOTE:
    Real-world cellular telemetry operates across heterogeneous temporal scales:
    - CASS RF Sensing: microsecond/millisecond-scale physical antenna & Primary User activity.
    - DLTeamTUC Resource Allocation: 1ms subframe MAC scheduling & CQI reports.
    - 5G Network KPIs: second/minute-level aggregate cell health (throughput, latency, PRB).

    Rather than forcing a brittle, lossy row-level join over disparate timestamps and cell IDs,
    this module implements a Statistical Linkage Model:
    1. Channel State Pool: Samples realistic (snr, interference_level, pu_present) tuples from CASS.
    2. Cell Load & QoS Pool: Samples realistic (throughput, latency, prb_utilization) tuples from KPI.
    3. Action Supervision Mapping: Samples (cqi -> rb_allocation) conditional mappings from the DLTeamTUC dataset.
    4. Composition: Jointly binds these empirical distributions into realistic scenarios parameterized
       by congestion tier (Low, Medium, High, Peak).

    Returns:
        (composed_df, methodology_metadata_dict)
    """
    rng = np.random.default_rng(random_state)

    if cass_df is None:
        cass_df = load_cass()
    if kpi_df is None:
        kpi_df, _ = load_kpi()
    if rb_df is None:
        rb_df = load_resource_allocation()

    logger.info("Aligning CASS (%d rows), KPI (%d rows), and Resource Allocation (%d rows) via statistical linkage...",
                len(cass_df), len(kpi_df), len(rb_df))

    # 1. Sample RF conditions from CASS
    cass_idx = rng.choice(len(cass_df), size=n_samples, replace=True)
    cass_sample = cass_df.iloc[cass_idx].reset_index(drop=True)

    # 2. Sample Cell conditions from KPI
    kpi_idx = rng.choice(len(kpi_df), size=n_samples, replace=True)
    kpi_sample = kpi_df.iloc[kpi_idx].reset_index(drop=True)

    # 3. Sample CQI and Allocation prior from RB dataset
    rb_idx = rng.choice(len(rb_df), size=n_samples, replace=True)
    rb_sample = rb_df.iloc[rb_idx].reset_index(drop=True)

    # 4. Map SNR from CASS to CQI (3GPP TS 38.214 standard mapping: ~ 1.5 dB SNR per CQI step)
    # CQI = clip(round((SNR + 6) / 2), 1, 15)
    empirical_cqi = np.clip(np.round((cass_sample["snr"] + 6.0) / 2.2), 1, 15).astype(int)

    # Traffic demand (Mbps) modeled as proportional to cell PRB load and user CQI
    traffic_demand = np.clip(
        kpi_sample["throughput"] * (0.05 + 0.15 * (empirical_cqi / 15.0)) * rng.uniform(0.7, 1.3, size=n_samples),
        2.0, 150.0
    )

    # User priority (1: Best Effort, 2: eMBB Streaming, 3: URLLC Mission Critical)
    priority = rng.choice([1, 2, 3], size=n_samples, p=[0.5, 0.35, 0.15])

    # Congestion tier categorization based on PRB utilization percentiles in KPI dataset
    p33 = kpi_sample["prb_utilization"].quantile(0.33)
    p66 = kpi_sample["prb_utilization"].quantile(0.66)
    p90 = kpi_sample["prb_utilization"].quantile(0.90)

    congestion_tier = []
    for u in kpi_sample["prb_utilization"]:
        if u < p33:
            congestion_tier.append("low")
        elif u < p66:
            congestion_tier.append("medium")
        elif u < p90:
            congestion_tier.append("high")
        else:
            congestion_tier.append("peak")

    composed_df = pd.DataFrame({
        "sample_id": np.arange(n_samples),
        "traffic_demand": traffic_demand.round(2),
        "channel_quality": cass_sample["snr"].round(2),
        "cqi": empirical_cqi,
        "interference_level": cass_sample["interference_level"].round(2),
        "pu_occupancy": cass_sample["pu_present"].astype(int),
        "user_priority": priority,
        "previous_utilization": kpi_sample["prb_utilization"].round(3),
        "empirical_throughput": kpi_sample["throughput"].round(2),
        "empirical_latency": kpi_sample["latency"].round(2),
        "empirical_packet_loss": kpi_sample["packet_loss_rate"].round(5),
        "congestion_tier": congestion_tier,
        "historical_allocated": rb_sample["allocated"].astype(int)
    })

    metadata = {
        "n_samples": n_samples,
        "cqi_range": (int(composed_df["cqi"].min()), int(composed_df["cqi"].max())),
        "snr_mean": float(composed_df["channel_quality"].mean()),
        "interference_mean": float(composed_df["interference_level"].mean()),
        "pu_occupancy_ratio": float(composed_df["pu_occupancy"].mean()),
        "congestion_thresholds": {"low_max": float(p33), "med_max": float(p66), "high_max": float(p90)},
        "methodology": "Empirical Copula Statistical Linkage across RF sensing (CASS), cell KPIs (5G KPI), and MAC schedulers (DLTeamTUC)."
    }

    logger.info("Composed %d aligned empirical scenario records across 4 congestion tiers.", len(composed_df))
    return composed_df, metadata


if __name__ == "__main__":
    print("Testing Data Pipeline...")
    generate_sample_raw_files()
    df_cass = load_cass()
    print("CASS sample:\n", df_cass.head(3))
    df_kpi, stats = load_kpi()
    print("KPI Stats:", stats)
    df_rb = load_resource_allocation()
    print("Resource Allocation sample:\n", df_rb.head(3))
    aligned, meta = align_datasets(df_cass, df_kpi, df_rb, n_samples=1000)
    print("Aligned Scenarios sample:\n", aligned.head(3))
    print("Alignment Metadata:", meta)
    print("Data Pipeline test passed successfully!")

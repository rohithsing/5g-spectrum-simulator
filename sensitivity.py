#!/usr/bin/env python
"""
Sensitivity analysis: does the advantage survive a different world?

Two sweeps, both answering objections a reviewer will raise:

1. **Detector quality.** The headline result uses one energy-detector operating
   point (P_d = 0.85, P_fa = 0.12). If the learned system only wins because that
   particular detector is bad, the gap should vanish as the detector improves.
2. **Incumbent predictability.** The learned occupancy filter exploits temporal
   correlation in the incumbent's bursts. If bursts are short, there is less to
   exploit and the gap should shrink. This sweep quantifies how much structure
   the method actually needs.

Neither sweep retrains anything -- the same frozen policy and predictor from
``artifacts/`` are evaluated in every world, which is the harder test.

    python sensitivity.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from spectrum.agent import NeuroSpectrumAllocator
from spectrum.baselines import MLWDF, ProportionalFair
from spectrum.config import SimConfig
from spectrum.runner import run_episode

ARTIFACTS = Path("artifacts")
RESULTS = Path("results")

METRICS = ["qos_satisfaction_pct", "goodput_mbps", "collision_pct",
           "deadline_miss_pct", "wasted_vacancy_pct"]


def evaluate(allocators, seeds: List[int], **cfg_kwargs) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        cfg = SimConfig(seed=seed, **cfg_kwargs)
        for alloc in allocators:
            r = run_episode(alloc, cfg, seed)
            rows.append({"allocator": alloc.name, "seed": seed,
                         **{m: r[m] for m in METRICS}})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--congestion", default="high")
    ap.add_argument("--slots", type=int, default=800)
    args = ap.parse_args()

    if not (ARTIFACTS / "policy.pt").exists():
        raise SystemExit("artifacts/policy.pt not found -- run `python train.py` first.")

    RESULTS.mkdir(exist_ok=True)
    seeds = [500 + 31 * i for i in range(args.seeds)]
    pred = ARTIFACTS / "predictor.pt"
    base_kwargs = dict(congestion=args.congestion, slots_per_episode=args.slots)

    def allocators():
        return [
            ProportionalFair(),
            MLWDF(),
            NeuroSpectrumAllocator.load(
                ARTIFACTS / "policy.pt", pred if pred.exists() else None
            ),
        ]

    # ---------------------------------------------------------- sweep 1
    print("\nSweep 1: energy-detector quality")
    print("-" * 68)
    detector_points = [
        (0.70, 0.25, "poor"),
        (0.85, 0.12, "baseline"),
        (0.92, 0.06, "good"),
        (0.97, 0.02, "excellent"),
        (0.995, 0.005, "near-perfect"),
    ]
    rows = []
    for pd_, pfa, label in detector_points:
        df = evaluate(
            allocators(), seeds,
            sensing_prob_detection=pd_, sensing_prob_false_alarm=pfa, **base_kwargs,
        )
        agg = df.groupby("allocator", sort=False)[METRICS].mean()
        ours = agg.loc[NeuroSpectrumAllocator.name]
        best_qos = agg.drop(index=NeuroSpectrumAllocator.name)["qos_satisfaction_pct"].max()
        best_coll = agg.drop(index=NeuroSpectrumAllocator.name)["collision_pct"].min()
        rows.append({
            "detector": label, "P_d": pd_, "P_fa": pfa,
            "classical_qos_pct": best_qos,
            "proposed_qos_pct": ours["qos_satisfaction_pct"],
            "qos_gain_pts": ours["qos_satisfaction_pct"] - best_qos,
            "classical_collision_pct": best_coll,
            "proposed_collision_pct": ours["collision_pct"],
            "collision_reduction_pts": best_coll - ours["collision_pct"],
        })
        print(f"  P_d={pd_:.3f} P_fa={pfa:.3f} ({label:12s})  "
              f"QoS {best_qos:5.1f} -> {ours['qos_satisfaction_pct']:5.1f}  "
              f"collisions {best_coll:5.2f} -> {ours['collision_pct']:5.2f}")

    df_det = pd.DataFrame(rows)
    df_det.to_csv(RESULTS / "sensitivity_detector.csv", index=False)

    # ---------------------------------------------------------- sweep 2
    print("\nSweep 2: incumbent burst length (how much temporal structure exists)")
    print("-" * 68)
    rows = []
    for mean_on in [4.0, 12.0, 42.0, 120.0, 300.0]:
        df = evaluate(allocators(), seeds, incumbent_mean_on_slots=mean_on, **base_kwargs)
        agg = df.groupby("allocator", sort=False)[METRICS].mean()
        ours = agg.loc[NeuroSpectrumAllocator.name]
        best_qos = agg.drop(index=NeuroSpectrumAllocator.name)["qos_satisfaction_pct"].max()
        best_coll = agg.drop(index=NeuroSpectrumAllocator.name)["collision_pct"].min()
        rows.append({
            "mean_burst_slots": mean_on,
            "mean_burst_ms": mean_on * 0.5,
            "classical_qos_pct": best_qos,
            "proposed_qos_pct": ours["qos_satisfaction_pct"],
            "qos_gain_pts": ours["qos_satisfaction_pct"] - best_qos,
            "classical_collision_pct": best_coll,
            "proposed_collision_pct": ours["collision_pct"],
            "collision_reduction_pts": best_coll - ours["collision_pct"],
        })
        print(f"  burst {mean_on:6.1f} slots ({mean_on * 0.5:6.1f} ms)   "
              f"QoS {best_qos:5.1f} -> {ours['qos_satisfaction_pct']:5.1f}  "
              f"collisions {best_coll:5.2f} -> {ours['collision_pct']:5.2f}")

    df_burst = pd.DataFrame(rows)
    df_burst.to_csv(RESULTS / "sensitivity_burst.csv", index=False)

    _plot(df_det, df_burst)
    print("\nWrote results/sensitivity_detector.csv, results/sensitivity_burst.csv, "
          "results/plots/sensitivity.png")


def _plot(df_det: pd.DataFrame, df_burst: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = RESULTS / "plots"
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))

    x = np.arange(len(df_det))
    axes[0, 0].plot(x, df_det["classical_qos_pct"], "o-", color="#8a8f98", label="best classical")
    axes[0, 0].plot(x, df_det["proposed_qos_pct"], "o-", color="#2f6fdb", label="proposed")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(df_det["detector"], rotation=20)
    axes[0, 0].set_ylabel("QoS satisfaction (%)")
    axes[0, 0].set_title("vs. detector quality")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(x, df_det["classical_collision_pct"], "o-", color="#8a8f98")
    axes[0, 1].plot(x, df_det["proposed_collision_pct"], "o-", color="#2f6fdb")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(df_det["detector"], rotation=20)
    axes[0, 1].set_ylabel("Incumbent collisions (%)")
    axes[0, 1].set_title("vs. detector quality")

    b = df_burst["mean_burst_ms"]
    axes[1, 0].semilogx(b, df_burst["classical_qos_pct"], "o-", color="#8a8f98")
    axes[1, 0].semilogx(b, df_burst["proposed_qos_pct"], "o-", color="#2f6fdb")
    axes[1, 0].set_xlabel("mean incumbent burst (ms)")
    axes[1, 0].set_ylabel("QoS satisfaction (%)")
    axes[1, 0].set_title("vs. incumbent predictability")

    axes[1, 1].semilogx(b, df_burst["classical_collision_pct"], "o-", color="#8a8f98")
    axes[1, 1].semilogx(b, df_burst["proposed_collision_pct"], "o-", color="#2f6fdb")
    axes[1, 1].set_xlabel("mean incumbent burst (ms)")
    axes[1, 1].set_ylabel("Incumbent collisions (%)")
    axes[1, 1].set_title("vs. incumbent predictability")

    for ax in axes.ravel():
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
    fig.suptitle("Sensitivity of the advantage to the operating environment", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "sensitivity.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

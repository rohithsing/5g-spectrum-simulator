#!/usr/bin/env python
"""
Benchmark the proposed allocator against the classical schedulers.

Produces, in ``results/``:

* ``kpi_by_regime.csv``      -- every KPI, every allocator, every congestion tier
* ``headline.csv/.md``       -- proposed vs. best classical, with bootstrap CIs
* ``predictor_accuracy.csv`` -- occupancy-estimation accuracy / precision / recall
* ``ablation.csv``           -- which component contributes what
* ``significance.csv``       -- paired tests against the strongest baseline
* ``plots/*.png``            -- figures for the report

Every allocator is run on an identical set of seeds, so each comparison is
paired: the same UE drop, the same traffic realisation and the same incumbent
trajectory are presented to all of them.

    python evaluate.py              # full benchmark
    python evaluate.py --seeds 3    # quicker
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from spectrum.agent import NeuroSpectrumAllocator, PredictorAugmentedBaseline
from spectrum.baselines import MLWDF, build_classical
from spectrum.config import PARAMETER_PROVENANCE, SimConfig
from spectrum.incumbent import IncumbentProcess
from spectrum.policy import SchedulerPolicy
from spectrum.predictor import (
    BayesFilter,
    EnergyDetector,
    GRUPredictor,
    classification_report,
)
from spectrum.runner import run_episode

ARTIFACTS = Path("artifacts")
RESULTS = Path("results")
LEVELS = ["low", "medium", "high", "peak"]

#: Metrics where a larger value is better.
HIGHER_IS_BETTER = {
    "goodput_mbps", "spectral_efficiency_bps_hz", "qos_satisfaction_pct",
    "jain_fairness", "prb_utilisation_pct", "demand_served_pct",
    "qos_URLLC_pct", "qos_VoIP_pct", "qos_eMBB-Video_pct", "qos_BestEffort_pct",
}

HEADLINE_METRICS = [
    ("qos_satisfaction_pct", "QoS satisfaction (%)"),
    ("goodput_mbps", "Goodput (Mb/s)"),
    ("spectral_efficiency_bps_hz", "Spectral efficiency (b/s/Hz)"),
    ("deadline_miss_pct", "Deadline misses (%)"),
    ("p99_delay_ms", "p99 delay (ms)"),
    ("collision_pct", "Incumbent collisions (%)"),
    ("wasted_vacancy_pct", "Wasted vacancy (%)"),
    ("jain_fairness", "Jain fairness"),
    ("allocation_waste_pct", "Allocation waste (%)"),
]


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def bootstrap_ci(x: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap confidence interval of the mean."""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return float(np.mean(x)), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    return (
        float(np.mean(x)),
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """
    Exact-ish paired Wilcoxon signed-rank test with a normal approximation.

    Used instead of a t-test because KPI distributions across seeds are skewed
    and the sample is small.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    n_total = d.size
    d = d[d != 0]
    n = d.size
    if n == 0:
        # The two allocators were identical on every seed: no evidence of a
        # difference, rather than an undefined test.
        return {"n": n_total, "statistic": 0.0, "z": 0.0, "p_value": 1.0}
    if n < 3:
        return {"n": n, "statistic": float("nan"), "z": float("nan"), "p_value": float("nan")}

    order = np.argsort(np.abs(d), kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    absd = np.abs(d)[order]
    i = 0
    while i < n:  # average tied ranks
        j = i
        while j + 1 < n and absd[j + 1] == absd[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1

    w_plus = float(np.sum(ranks[d > 0]))
    w_minus = float(np.sum(ranks[d < 0]))
    w = min(w_plus, w_minus)
    mean_w = n * (n + 1) / 4.0
    std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (w - mean_w) / max(std_w, 1e-12)
    # Two-sided normal approximation to the signed-rank null distribution.
    p = _erfc(abs(z) / np.sqrt(2.0))
    return {"n": n, "statistic": w, "z": float(z), "p_value": float(min(p, 1.0))}


def _erfc(x: float) -> float:
    import math

    return math.erfc(x)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Non-parametric effect size in [-1, 1]."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = np.sum(a[:, None] > b[None, :])
    lt = np.sum(a[:, None] < b[None, :])
    return float((gt - lt) / (a.size * b.size))


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------
def build_allocators(args) -> List:
    allocators = build_classical()

    policy_path = ARTIFACTS / "policy.pt"
    predictor_path = ARTIFACTS / "predictor.pt"
    if not policy_path.exists():
        raise SystemExit("artifacts/policy.pt not found -- run `python train.py` first.")

    predictor = None
    if predictor_path.exists():
        predictor = GRUPredictor()
        predictor.load_state_dict(torch.load(predictor_path, map_location="cpu"))
        predictor.eval()

    if args.ablations and predictor is not None:
        # (a) classical scheduler, learned sensing front-end
        allocators.append(PredictorAugmentedBaseline(MLWDF(), predictor))
        # (b) learned scheduler, classical sensing front-end
        pol_no_pred = SchedulerPolicy()
        pol_no_pred.load_state_dict(torch.load(policy_path, map_location="cpu"))
        pol_no_pred.eval()
        allocators.append(
            NeuroSpectrumAllocator(
                pol_no_pred, predictor=None, name="NeuroSpectrum, hard sensing (ablation)"
            )
        )
        # (c) behaviour-cloned policy before any RL
        bc_path = ARTIFACTS / "policy_bc.pt"
        if bc_path.exists():
            pol_bc = SchedulerPolicy()
            pol_bc.load_state_dict(torch.load(bc_path, map_location="cpu"))
            pol_bc.eval()
            allocators.append(
                NeuroSpectrumAllocator(
                    pol_bc, predictor, name="NeuroSpectrum, BC only / no RL (ablation)"
                )
            )

    allocators.append(NeuroSpectrumAllocator.load(policy_path, predictor_path))
    return allocators


def run_benchmark(allocators, seeds: List[int], levels: List[str], slots: int) -> pd.DataFrame:
    records = []
    total = len(allocators) * len(levels) * len(seeds)
    done = 0
    for level in levels:
        for seed in seeds:
            cfg = SimConfig(congestion=level, seed=seed, slots_per_episode=slots)
            for alloc in allocators:
                res = run_episode(alloc, cfg, seed)
                res = {k: v for k, v in res.items() if not k.startswith("_")}
                res["allocator"] = alloc.name
                res["short"] = getattr(alloc, "short_name", alloc.name)
                res["congestion"] = level
                res["seed"] = seed
                records.append(res)
                done += 1
                print(f"\r  {done}/{total} episodes", end="", flush=True)
    print()
    return pd.DataFrame(records)


def predictor_accuracy(slots: int = 6000) -> pd.DataFrame:
    """Score the three occupancy estimators on held-out incumbent traces."""
    path = ARTIFACTS / "predictor.pt"
    if not path.exists():
        return pd.DataFrame()
    model = GRUPredictor()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()

    rows = []
    for level in LEVELS:
        cfg = SimConfig(congestion=level)
        rng = np.random.default_rng(7_000_000 + LEVELS.index(level))
        inc = IncumbentProcess(cfg, rng)
        det = EnergyDetector(inc.threshold)
        bayes = BayesFilter(inc.p_on, inc.p_off, inc.mu_h1, inc.duty_cycle)
        bayes.reset(cfg.n_subbands)
        model.reset(cfg.n_subbands)

        truth, p_ed, p_bf, p_gru = [], [], [], []
        for _ in range(slots):
            d, e = inc.sense()
            truth.append(inc.active.copy())
            p_ed.append(det.probability(d, e))
            p_bf.append(bayes.probability(d, e))
            p_gru.append(model.probability(d, e))
            inc.advance()

        burn = 200
        t = np.array(truth)[burn:]
        for name, p in (
            ("Energy detector (classical)", p_ed),
            ("Learned GRU (proposed)", p_gru),
            ("Bayes-optimal (upper bound)", p_bf),
        ):
            r = classification_report(t, np.array(p)[burn:])
            r.update({"congestion": level, "estimator": name})
            rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def headline_table(df: pd.DataFrame, proposed: str, seeds: List[int]) -> pd.DataFrame:
    """Proposed vs. the best classical scheduler per metric, with CIs and tests."""
    classical = [
        n for n in df["allocator"].unique()
        if n != proposed and "ablation" not in n and "learned sensing" not in n
    ]
    rows = []
    for level in df["congestion"].unique():
        sub = df[df["congestion"] == level]
        ours = sub[sub["allocator"] == proposed].sort_values("seed")

        for key, label in HEADLINE_METRICS:
            higher = key in HIGHER_IS_BETTER
            # Pick the strongest classical comparator for this metric.
            best_name, best_val = None, None
            for name in classical:
                v = sub[sub["allocator"] == name][key].mean()
                if best_val is None or (v > best_val if higher else v < best_val):
                    best_name, best_val = name, v

            base = sub[sub["allocator"] == best_name].sort_values("seed")
            a = ours[key].to_numpy()
            b = base[key].to_numpy()
            mean_a, lo_a, hi_a = bootstrap_ci(a)
            mean_b, lo_b, hi_b = bootstrap_ci(b)

            if higher:
                rel = 100.0 * (mean_a - mean_b) / abs(mean_b) if mean_b else float("nan")
            else:
                rel = 100.0 * (mean_b - mean_a) / abs(mean_b) if mean_b else float("nan")

            test = wilcoxon_signed_rank(a, b)
            rows.append({
                "congestion": level,
                "metric": label,
                "best_classical": best_name,
                "classical_mean": mean_b,
                "classical_ci": f"[{lo_b:.3g}, {hi_b:.3g}]",
                "proposed_mean": mean_a,
                "proposed_ci": f"[{lo_a:.3g}, {hi_a:.3g}]",
                "improvement_pct": rel,
                "p_value": test["p_value"],
                "cliffs_delta": cliffs_delta(a, b),
            })
    return pd.DataFrame(rows)


def write_markdown(path: Path, df_kpi: pd.DataFrame, head: pd.DataFrame,
                   pred: pd.DataFrame, n_seeds: int, slots: int) -> None:
    lines: List[str] = []
    ap = lines.append
    ap("# Intelligent Spectrum Allocation for 5G/6G -- Benchmark Report\n")
    ap(f"Paired evaluation over **{n_seeds} seeds** x **{len(LEVELS)} congestion regimes**, "
       f"{slots} slots ({slots * 0.5:.0f} ms) per episode.\n")
    ap("Every allocator sees the identical observation -- buffer status, aged CQI "
       "reports and the output of one shared energy detector. The incumbent's true "
       "occupancy is never exposed to any of them.\n")

    ap("\n## 1. Incumbent-occupancy estimation\n")
    if not pred.empty:
        agg = (pred.groupby("estimator", sort=False)
               [["accuracy", "precision", "recall", "f1", "auc", "miss_rate", "false_alarm_rate"]]
               .mean().round(4))
        ap(agg.to_markdown())
        ap("\nAveraged over all four congestion regimes on held-out traces. "
           "The Bayes-optimal filter is given the true Markov transition matrix and "
           "detector calibration, so it bounds what any estimator can achieve from "
           "the sensing stream alone.\n")

    ap("\n## 2. Headline system KPIs\n")
    for level in LEVELS:
        sub = head[head["congestion"] == level]
        if sub.empty:
            continue
        ap(f"\n### {level.capitalize()} congestion\n")
        show = sub[["metric", "best_classical", "classical_mean", "proposed_mean",
                    "improvement_pct", "p_value"]].copy()
        show.columns = ["Metric", "Best classical", "Classical", "Proposed", "Improvement %", "p"]
        show["Classical"] = show["Classical"].map(lambda v: f"{v:,.3g}")
        show["Proposed"] = show["Proposed"].map(lambda v: f"{v:,.3g}")
        show["Improvement %"] = show["Improvement %"].map(lambda v: f"{v:+.1f}")
        show["p"] = show["p"].map(lambda v: f"{v:.3f}" if v == v else "n/a")
        ap(show.to_markdown(index=False))

    ap("\n## 3. All allocators, all regimes\n")
    pivot = (df_kpi.groupby(["congestion", "allocator"], sort=False)
             [["qos_satisfaction_pct", "goodput_mbps", "deadline_miss_pct",
               "collision_pct", "jain_fairness", "p99_delay_ms"]]
             .mean().round(2))
    ap(pivot.to_markdown())

    ap("\n## 4. Parameter provenance\n")
    for k, v in PARAMETER_PROVENANCE.items():
        ap(f"- `{k}` -- {v}")

    path.write_text("\n".join(lines), encoding="utf-8")


def make_plots(df: pd.DataFrame, pred: pd.DataFrame, proposed: str, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    order = [n for n in df["allocator"].unique() if "ablation" not in n]

    # -- 1. KPI bars per regime -----------------------------------------
    panels = [
        ("qos_satisfaction_pct", "QoS satisfaction (%)"),
        ("goodput_mbps", "Goodput (Mb/s)"),
        ("deadline_miss_pct", "Deadline misses (%)"),
        ("collision_pct", "Incumbent collisions (%)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    x = np.arange(len(LEVELS))
    width = 0.8 / len(order)
    for ax, (key, label) in zip(axes.ravel(), panels):
        for i, name in enumerate(order):
            vals = [df[(df["allocator"] == name) & (df["congestion"] == lv)][key].mean()
                    for lv in LEVELS]
            ax.bar(x + i * width, vals, width,
                   label=name if ax is axes[0, 0] else None,
                   color="#2f6fdb" if name == proposed else None,
                   edgecolor="white", linewidth=0.5)
        ax.set_xticks(x + 0.4 - width / 2)
        ax.set_xticklabels([lv.capitalize() for lv in LEVELS])
        ax.set_title(label, fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("Existing schedulers vs. proposed RL allocator", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "kpi_comparison.png", dpi=150)
    plt.close(fig)

    # -- 2. Predictor accuracy ------------------------------------------
    if not pred.empty:
        agg = pred.groupby("estimator", sort=False)[
            ["accuracy", "precision", "recall", "f1", "auc"]].mean()
        fig, ax = plt.subplots(figsize=(9, 4.5))
        m = np.arange(5)
        w = 0.26
        for i, (name, row) in enumerate(agg.iterrows()):
            ax.bar(m + i * w, row.values, w, label=name, edgecolor="white", linewidth=0.5)
        ax.set_xticks(m + w)
        ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "AUC"])
        ax.set_ylim(0.5, 1.02)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9, loc="lower right")
        ax.set_title("Incumbent-occupancy estimation quality")
        fig.tight_layout()
        fig.savefig(out / "predictor_accuracy.png", dpi=150)
        plt.close(fig)

    # -- 3. QoS by traffic class ----------------------------------------
    class_keys = [k for k in df.columns if k.startswith("qos_") and k.endswith("_pct")
                  and k != "qos_satisfaction_pct"]
    if class_keys:
        fig, ax = plt.subplots(figsize=(11, 4.8))
        labels = [k[4:-4] for k in class_keys]
        m = np.arange(len(class_keys))
        w = 0.8 / len(order)
        for i, name in enumerate(order):
            sub = df[df["allocator"] == name]
            ax.bar(m + i * w, [sub[k].mean() for k in class_keys], w, label=name,
                   color="#2f6fdb" if name == proposed else None,
                   edgecolor="white", linewidth=0.5)
        ax.set_xticks(m + 0.4 - w / 2)
        ax.set_xticklabels(labels)
        ax.set_ylabel("QoS satisfaction (%)")
        ax.set_title("QoS satisfaction by 5QI traffic class (all regimes pooled)")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(out / "qos_by_class.png", dpi=150)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--slots", type=int, default=1000)
    ap.add_argument("--no-ablations", dest="ablations", action="store_false")
    ap.add_argument("--levels", nargs="*", default=LEVELS)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "plots").mkdir(exist_ok=True)
    seeds = [1000 + 97 * i for i in range(args.seeds)]

    print(f"Benchmarking over {args.seeds} seeds x {len(args.levels)} regimes...")
    allocators = build_allocators(args)
    proposed = NeuroSpectrumAllocator.name

    df = run_benchmark(allocators, seeds, args.levels, args.slots)
    df.to_csv(RESULTS / "kpi_by_regime.csv", index=False)

    pred = predictor_accuracy()
    if not pred.empty:
        pred.to_csv(RESULTS / "predictor_accuracy.csv", index=False)

    head = headline_table(df, proposed, seeds)
    head.to_csv(RESULTS / "headline.csv", index=False)

    # Plain M-LWDF is included as the reference point the decomposition builds on:
    # M-LWDF -> + learned sensing -> + learned scheduler (BC) -> + RL.
    abl = (df[df["allocator"].str.contains(
                "^M-LWDF$|ablation|learned sensing|NeuroSpectrum", regex=True)]
           .groupby("allocator", sort=False)
           [["qos_satisfaction_pct", "goodput_mbps", "deadline_miss_pct",
             "collision_pct", "wasted_vacancy_pct", "jain_fairness"]]
           .mean().round(3))
    abl.to_csv(RESULTS / "ablation.csv")

    write_markdown(RESULTS / "benchmark_report.md", df, head, pred, args.seeds, args.slots)
    make_plots(df, pred, proposed, RESULTS / "plots")

    # -- console summary --------------------------------------------------
    print("\n" + "=" * 78)
    print("HEADLINE: proposed vs. best classical scheduler")
    print("=" * 78)
    for level in args.levels:
        sub = head[head["congestion"] == level]
        print(f"\n{level.upper()}")
        for _, r in sub.iterrows():
            flag = "*" if r["p_value"] == r["p_value"] and r["p_value"] < 0.05 else " "
            print(f"  {flag} {r['metric']:<32s} {r['classical_mean']:>9.3f} -> "
                  f"{r['proposed_mean']:>9.3f}  ({r['improvement_pct']:+6.1f}%)  "
                  f"vs {r['best_classical']}")
    print("\n  (* = paired Wilcoxon p < 0.05)")
    print("\nWrote results/ -- benchmark_report.md, kpi_by_regime.csv, headline.csv, plots/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Interactive dashboard: existing spectrum allocation vs. the proposed RL system.

    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from spectrum.baselines import build_classical
from spectrum.config import CONGESTION_REGIMES, PARAMETER_PROVENANCE, SimConfig
from spectrum.env import SpectrumEnv

ARTIFACTS = Path("artifacts")
RESULTS = Path("results")

st.set_page_config(
    page_title="Intelligent Spectrum Allocation",
    page_icon="📡",
    layout="wide",
)

ACCENT = "#2f6fdb"
MUTED = "#8a8f98"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_agent():
    """Load the trained allocator, or return None if training has not run."""
    if not (ARTIFACTS / "policy.pt").exists():
        return None
    from spectrum.agent import NeuroSpectrumAllocator

    pred = ARTIFACTS / "predictor.pt"
    return NeuroSpectrumAllocator.load(
        ARTIFACTS / "policy.pt", pred if pred.exists() else None
    )


@st.cache_resource(show_spinner=False)
def load_predictor():
    if not (ARTIFACTS / "predictor.pt").exists():
        return None
    import torch

    from spectrum.predictor import GRUPredictor

    m = GRUPredictor()
    m.load_state_dict(torch.load(ARTIFACTS / "predictor.pt", map_location="cpu"))
    m.eval()
    return m


@st.cache_data(show_spinner=False)
def load_results() -> Dict[str, Optional[pd.DataFrame]]:
    def read(name):
        p = RESULTS / name
        return pd.read_csv(p) if p.exists() else None

    return {
        "kpi": read("kpi_by_regime.csv"),
        "headline": read("headline.csv"),
        "predictor": read("predictor_accuracy.csv"),
        "ablation": read("ablation.csv"),
    }


def allocator_registry() -> Dict[str, object]:
    reg = {a.name: a for a in build_classical()}
    agent = load_agent()
    if agent is not None:
        reg[agent.name] = agent
    return reg


# ---------------------------------------------------------------------------
# live simulation
# ---------------------------------------------------------------------------
def simulate(allocator, cfg: SimConfig, seed: int) -> Dict[str, object]:
    """Run one episode and keep the per-slot trace for plotting."""
    env = SpectrumEnv(cfg, seed=seed)
    allocator.reset(cfg)

    obs = env.observe()
    grid = np.full((cfg.slots_per_episode, cfg.n_rbg), -1, dtype=np.int64)
    truth = np.zeros((cfg.slots_per_episode, cfg.n_subbands), dtype=np.int64)
    series: List[Dict[str, float]] = []

    for t in range(cfg.slots_per_episode):
        a = allocator.allocate(obs)
        grid[t] = a
        obs, reward, done, info = env.step(a)
        truth[t] = info["_truth_incumbent"]
        series.append({
            "slot": t,
            "Goodput (Mb/s)": info["useful_bits"] / cfg.slot_duration_s / 1e6,
            "Collisions": info["collisions"],
            "In-budget UEs (%)": 100.0 * info["qos_fraction"],
            "Jain fairness": info["jain"],
        })
        if done:
            break

    return {
        "summary": env.episode_summary(),
        "series": pd.DataFrame(series),
        "grid": grid,
        "truth": truth,
        "ue_class": env.traffic.ue_class.copy(),
    }


def occupancy_figure(res_a, res_b, cfg: SimConfig, name_a: str, name_b: str):
    """Spectrum waterfall: who used which RBG, and where the incumbent really was."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    window = min(220, cfg.slots_per_episode)

    for ax, res, title in ((axes[0], res_a, name_a), (axes[1], res_b, name_b)):
        grid = res["grid"][:window]
        truth = res["truth"][:window]
        # 0 = muted, 1 = transmitted cleanly, 2 = transmitted into an incumbent
        occupied = np.repeat(truth, cfg.rbgs_per_subband, axis=1)
        img = np.where(grid < 0, 0, np.where(occupied == 1, 2, 1))
        ax.imshow(
            img.T, aspect="auto", origin="lower", interpolation="nearest",
            cmap=matplotlib.colors.ListedColormap(["#e9ecf1", ACCENT, "#d1495b"]),
            vmin=0, vmax=2,
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("slot")
    axes[0].set_ylabel("resource block group")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#e9ecf1"),
        plt.Rectangle((0, 0), 1, 1, color=ACCENT),
        plt.Rectangle((0, 0), 1, 1, color="#d1495b"),
    ]
    fig.legend(handles, ["muted", "clean transmission", "collided with incumbent"],
               loc="upper center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------
def page_overview(data):
    st.subheader("Existing framework vs. proposed RL framework")
    st.markdown(
        "Both sides schedule the **same 100 MHz 5G NR carrier**, shared with a "
        "stochastic incumbent, and both see the **same observation**: buffer status "
        "reports, aged CQI reports and one shared energy detector. "
        "The incumbent's true occupancy is never shown to either."
    )

    head = data["headline"]
    if head is None:
        st.info("Run `python evaluate.py` to populate the benchmark tables.")
        return

    levels = [l for l in ["low", "medium", "high", "peak"] if l in set(head["congestion"])]
    level = st.radio("Congestion regime", levels,
                     index=min(2, len(levels) - 1), horizontal=True)
    sub = head[head["congestion"] == level]

    featured = ["QoS satisfaction (%)", "Goodput (Mb/s)",
                "Incumbent collisions (%)", "Deadline misses (%)"]
    cols = st.columns(len(featured))
    for col, metric in zip(cols, featured):
        row = sub[sub["metric"] == metric]
        if row.empty:
            continue
        row = row.iloc[0]
        col.metric(
            metric,
            f"{row['proposed_mean']:.3g}",
            f"{row['improvement_pct']:+.1f}% vs {row['best_classical'].split('(')[0].strip()}",
        )

    st.markdown("##### Full comparison against the strongest classical scheduler per metric")
    show = sub[["metric", "best_classical", "classical_mean", "classical_ci",
                "proposed_mean", "proposed_ci", "improvement_pct", "p_value"]].copy()
    show.columns = ["Metric", "Best classical", "Classical", "95% CI",
                    "Proposed", "95% CI ", "Improvement %", "p"]
    st.dataframe(
        show.style.format({
            "Classical": "{:,.3f}", "Proposed": "{:,.3f}",
            "Improvement %": "{:+.1f}", "p": "{:.4f}",
        }),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "Improvement is sign-corrected so positive always means better. "
        "p is a paired Wilcoxon signed-rank test across seeds; each allocator "
        "faced an identical UE drop, traffic realisation and incumbent trajectory."
    )

    for name, caption in (
        ("kpi_comparison.png", "KPIs across congestion regimes"),
        ("qos_by_class.png", "QoS satisfaction by 5QI traffic class"),
    ):
        p = RESULTS / "plots" / name
        if p.exists():
            st.image(str(p), caption=caption, use_container_width=True)


def page_live(data):
    st.subheader("Run the simulator")
    registry = allocator_registry()
    agent = load_agent()

    c1, c2, c3, c4 = st.columns(4)
    names = list(registry)
    default_b = names.index(agent.name) if agent is not None else len(names) - 1
    name_a = c1.selectbox("Existing system", names, index=0)
    name_b = c2.selectbox("Proposed system", names, index=default_b)
    level = c3.selectbox("Congestion", list(CONGESTION_REGIMES), index=2)
    slots = c4.slider("Slots (0.5 ms each)", 200, 2000, 600, step=100)
    seed = st.number_input("Seed", min_value=0, max_value=10_000, value=42, step=1)

    regime = CONGESTION_REGIMES[level]
    st.caption(
        f"{regime.n_ue} UEs - offered load x{regime.load_multiplier:.2f} - "
        f"incumbent duty cycle {regime.incumbent_duty_cycle:.0%} - "
        f"{slots * 0.5:.0f} ms of simulated time"
    )

    if not st.button("Run both", type="primary"):
        return

    cfg = SimConfig(congestion=level, seed=int(seed), slots_per_episode=int(slots))
    with st.spinner("Simulating..."):
        res_a = simulate(registry[name_a], cfg, int(seed))
        res_b = simulate(registry[name_b], cfg, int(seed))

    sa, sb = res_a["summary"], res_b["summary"]
    rows = [
        ("QoS satisfaction (%)", "qos_satisfaction_pct", True),
        ("Goodput (Mb/s)", "goodput_mbps", True),
        ("Spectral efficiency (b/s/Hz)", "spectral_efficiency_bps_hz", True),
        ("Deadline misses (%)", "deadline_miss_pct", False),
        ("p99 delay (ms)", "p99_delay_ms", False),
        ("Incumbent collisions (%)", "collision_pct", False),
        ("Wasted vacancy (%)", "wasted_vacancy_pct", False),
        ("Jain fairness", "jain_fairness", True),
        ("Allocation waste (%)", "allocation_waste_pct", False),
    ]
    table = []
    for label, key, higher in rows:
        va, vb = sa[key], sb[key]
        delta = (vb - va) if higher else (va - vb)
        rel = 100.0 * delta / abs(va) if va else float("nan")
        table.append({"Metric": label, name_a: va, name_b: vb, "Improvement %": rel})
    st.dataframe(
        pd.DataFrame(table).style.format(
            {name_a: "{:,.3f}", name_b: "{:,.3f}", "Improvement %": "{:+.1f}"}
        ),
        use_container_width=True, hide_index=True,
    )

    st.markdown("##### Spectrum occupancy")
    st.pyplot(occupancy_figure(res_a, res_b, cfg, name_a, name_b))
    st.caption(
        "Red cells are transmissions that landed on top of an active incumbent. "
        "Light cells are resources left unused -- some necessary, some thrown away "
        "by a false alarm."
    )

    st.markdown("##### Per-slot behaviour")
    metric = st.selectbox(
        "Signal", ["Goodput (Mb/s)", "In-budget UEs (%)", "Jain fairness", "Collisions"]
    )
    merged = pd.DataFrame({
        "slot": res_a["series"]["slot"],
        name_a: res_a["series"][metric].rolling(20, min_periods=1).mean(),
        name_b: res_b["series"][metric].rolling(20, min_periods=1).mean(),
    }).set_index("slot")
    st.line_chart(merged, color=[MUTED, ACCENT])
    st.caption("20-slot rolling mean.")


def page_predictor(data):
    st.subheader("Incumbent-occupancy estimation")
    st.markdown(
        "The energy detector is memoryless, but incumbent bursts last tens of "
        "slots. The learned filter accumulates evidence across that burst; the "
        "Bayes-optimal filter is given the true Markov parameters and bounds "
        "what any estimator can extract from the same sensing stream."
    )

    pred = data["predictor"]
    if pred is None:
        st.info("Run `python evaluate.py` to populate the predictor comparison.")
        return

    cols = ["accuracy", "precision", "recall", "f1", "auc", "miss_rate", "false_alarm_rate"]
    agg = pred.groupby("estimator", sort=False)[cols].mean()
    st.dataframe(agg.style.format("{:.4f}"), use_container_width=True)

    ed = agg[agg.index.str.contains("Energy")]
    gru = agg[agg.index.str.contains("Learned")]
    if not ed.empty and not gru.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{gru['accuracy'].iloc[0]:.1%}",
                  f"{100 * (gru['accuracy'].iloc[0] - ed['accuracy'].iloc[0]):+.1f} pts")
        c2.metric("Precision", f"{gru['precision'].iloc[0]:.1%}",
                  f"{100 * (gru['precision'].iloc[0] - ed['precision'].iloc[0]):+.1f} pts")
        c3.metric("Missed detections", f"{gru['miss_rate'].iloc[0]:.1%}",
                  f"{100 * (gru['miss_rate'].iloc[0] - ed['miss_rate'].iloc[0]):+.1f} pts",
                  delta_color="inverse")

    p = RESULTS / "plots" / "predictor_accuracy.png"
    if p.exists():
        st.image(str(p), use_container_width=True)

    st.markdown("##### Per regime")
    st.dataframe(
        pred.pivot_table(index="congestion", columns="estimator",
                         values=["accuracy", "f1"], sort=False).round(4),
        use_container_width=True,
    )


def page_ablation(data):
    st.subheader("Where the improvement comes from")
    st.markdown(
        "The proposed system has two learned parts. Running each in isolation "
        "separates better incumbent estimation from better scheduling."
    )
    abl = data["ablation"]
    if abl is None:
        st.info("Run `python evaluate.py --ablations` to populate this view.")
        return
    st.dataframe(abl, use_container_width=True)
    st.caption(
        "`+ learned sensing` is M-LWDF with the GRU front-end. "
        "`hard sensing` is the learned scheduler trusting the raw detector. "
        "`BC only / no RL` is the policy after imitating M-LWDF but before any "
        "reinforcement learning -- the difference from the full system is what "
        "RL contributed."
    )


def page_method():
    st.subheader("How the testbed is built")
    st.markdown(
        """
**Physical layer.** 100 MHz carrier at 3.5 GHz (band n78), 30 kHz subcarrier
spacing, 18 resource block groups, 0.5 ms slots. Propagation uses the 3GPP
TR 38.901 Urban Macro NLOS path loss with log-normal shadowing. Small-scale
fading is correlated in time (first-order Gauss-Markov with a Jakes
coefficient from each UE's Doppler) and in frequency (exponential decay across
RBGs), so different UEs genuinely peak on different parts of the band.

**Link adaptation.** SINR maps to CQI through 3GPP TS 38.214 table 2 at a
10 % BLER target; the scheduler picks its modulation from a *delayed,
quantised* CQI report, so an optimistic report causes a real decode failure.
HARQ retransmissions accumulate chase-combining gain.

**Traffic.** Four 5QI bearers from TS 23.501 table 5.7.4-1 -- URLLC, VoIP,
live video and best effort -- with bursty arrivals, per-packet delay budgets
and packets that are *dropped* when their budget expires.

**Incumbent sharing.** A two-state semi-Markov process per subband, sensed by
an energy detector at a fixed operating point. Transmitting on an occupied
subband is a harmful-interference event.
        """
    )
    st.markdown("##### Information parity")
    st.markdown(
        "Every allocator receives the identical `Observation` object. The "
        "incumbent's true state is not a field of it, and `tests/"
        "test_information_parity.py` asserts this mechanically -- including a "
        "replay test that proves each allocator reproduces its decisions from "
        "recorded observations alone, with no environment attached."
    )
    st.markdown("##### Parameter provenance")
    st.dataframe(
        pd.DataFrame(
            [{"Parameter": k, "Source": v} for k, v in PARAMETER_PROVENANCE.items()]
        ),
        use_container_width=True, hide_index=True,
    )


# ---------------------------------------------------------------------------
def main() -> None:
    st.title("Intelligent Spectrum Allocation for 5G/6G Networks")
    st.caption(
        "Reinforcement-learning dynamic spectrum allocation, benchmarked against "
        "the schedulers deployed today under strict information parity."
    )

    data = load_results()
    if load_agent() is None:
        st.warning("No trained policy found. Run `python train.py`, then `python evaluate.py`.")

    tabs = st.tabs(
        ["Overview", "Live simulation", "Occupancy prediction", "Ablation", "Methodology"]
    )
    with tabs[0]:
        page_overview(data)
    with tabs[1]:
        page_live(data)
    with tabs[2]:
        page_predictor(data)
    with tabs[3]:
        page_ablation(data)
    with tabs[4]:
        page_method()


if __name__ == "__main__":
    main()

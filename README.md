# Intelligent Spectrum Allocation Simulator for 5G/6G Networks

An end-to-end AI/ML and Deep Reinforcement Learning system for Dynamic Spectrum Allocation (DSA) in 5G/6G networks. Developed as a final-year AI/ML engineering project, the system combines three real-world empirical datasets as the foundation for state features, reward signals, and action supervision, integrated with a physical 5G NR simulator that bridges empirical gaps.

---

## 📁 Repository Structure

```
MajorProject/
├── data/
│   ├── raw/                           # Raw input CSV datasets
│   │   ├── cass_spectrum.csv          # CASS RF sensing & Primary User (PU) measurements
│   │   ├── 5g_kpi.csv                 # 5G Network KPI traces (throughput, latency, PRB)
│   │   └── 5g_resource_allocation.csv # DLTeamTUC RB group & CQI scheduling logs
│   └── processed/                     # Aligned scenario caches and normalization stats
├── results/                           # Auto-saved evaluation outputs
│   ├── benchmark_results.csv          # Multi-scenario performance metrics
│   ├── summary_improvements.csv       # Percentage gains of DRL over baselines
│   ├── benchmark_summary.md           # Formatted markdown results tables
│   ├── methodology_notes.md           # Academic paper documentation
│   └── plots/                         # Publication-quality figures
│       ├── allocator_comparison.png   # Throughput, Latency, Jain's, QoS comparisons
│       ├── fairness_vs_congestion.png # Fairness degradation resilience curves
│       └── kpi_distribution_comparison.png # Two-sample KS-test sanity check plots
├── data_pipeline.py                   # Ingestion, resampling, statistical linkage, fallbacks
├── simulator.py                       # 5G NR physical layer, Shannon rate, M/M/1 queuing
├── baseline_allocator.py              # Static Round-Robin & Proportional-Fair schedulers
├── ml_allocator.py                    # Supervised RF/XGBoost warm-start & Dueling Double DQN
├── evaluate.py                        # Multi-scenario benchmarking & KS-test sanity check
├── app.py                             # Interactive Streamlit dashboard
├── requirements.txt                   # Dependency specifications
└── README.md                          # Project documentation
```

---

## 📊 Dataset Ingestion Guide

Place the three raw CSV datasets into the `data/raw/` directory:

1. **`data/raw/cass_spectrum.csv`** (CASS Dataset):
   - Time-indexed RF measurements, Primary User presence, SNR, signal power, and noise floor.
   - Schema: `[timestamp, channel_id, pu_present, signal_power_db, snr, interference_level]`
2. **`data/raw/5g_kpi.csv`** (5G Cellular KPI Dataset):
   - Per-cell aggregate throughput, packet latency, packet loss rate, and PRB utilization.
   - Schema: `[timestamp, cell_id, throughput, latency, packet_loss_rate, prb_utilization]`
   - Summary statistics ($\mu, \sigma$) provide empirical normalization constants for RL rewards.
3. **`data/raw/5g_resource_allocation.csv`** (DLTeamTUC/5GDatasets):
   - Per-subframe Resource Block group assignments and Channel Quality Indicators (CQI 1-15).
   - Schema: `[timestamp, user_id, rb_group_id, cqi, allocated]`

> **Automatic Fallback Generator**: If any of these files are missing, the pipeline automatically detects it, logs an explicit fallback notice, and invokes a mathematically grounded 3GPP TS 38.214 compliant parametric generator. Sample files can also be created via `python data_pipeline.py`.

---

## 🔬 Methodology: Empirical Copula Statistical Linkage

Because microsecond-level physical RF sensing (CASS), subframe MAC scheduling (DLTeamTUC), and cell-level network KPIs (5G KPI) operate across fundamentally incompatible temporal scales, direct row joins are mathematically invalid.

Our solution implements an **Empirical Copula Statistical Linkage**:
1. **Channel State Pool**: Samples empirical tuples $(\text{SNR}, I, \text{PU}) \sim P_{\text{CASS}}$.
2. **Cell Load Pool**: Samples empirical QoS metrics $(T, L, U_{\text{PRB}}) \sim P_{\text{KPI}}(\cdot \mid \text{Congestion Tier})$.
3. **Action Prior**: Maps CQI to allocation decisions $P_{\text{DLTeam}}(\text{Alloc} \mid \text{CQI}, D, P)$.
4. **Composed State Vector**:
   $$\mathbf{s}_{u,t} = [D_{u,t}, \text{SNR}_{u,t}, I_{u,t}, \text{PU}_{c,t}, P_u, U_{u,t-1}]$$

---

## 🚀 Getting Started

### 1. Environment Setup

Ensure Python 3.10+ or Python 3.11 is installed:

```bash
# Clone the repository
git clone <repo-url>
cd MajorProject

# Create and activate virtual environment
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Data Pipeline & Verify Fallback

```bash
python data_pipeline.py
```

### 3. Run Benchmark Evaluation

Executes multi-scenario simulations across Low, Medium, High, and Peak congestion tiers, computes the Two-Sample Kolmogorov-Smirnov (KS) test, and exports tables and plots:

```bash
python evaluate.py
```

Output files will be saved in `results/`:
- `results/benchmark_results.csv`
- `results/summary_improvements.csv`
- `results/benchmark_summary.md`
- `results/methodology_notes.md`
- `results/plots/allocator_comparison.png`
- `results/plots/kpi_distribution_comparison.png`
- `results/plots/fairness_vs_congestion.png`

### 4. Launch Interactive Streamlit Dashboard

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` to explore:
- **Tab 1: Live Simulation & Grid**: Interactive 2D subframe spectrum heatmap, Primary User collision flags, and real-time KPI gauges.
- **Tab 2: Comparative Benchmark**: Side-by-side leaderboard, metric bar charts, and multi-dimensional tradeoff radar chart.
- **Tab 3: Statistical Realism**: Side-by-side empirical vs simulated distribution KDEs and dynamic KS-test reports.
- **Tab 4: Methodology & Paper Notes**: Rendered mathematical equations and system architecture diagrams for thesis export.

---

## 📈 Summary of Algorithms

| Allocator | Class | Strategy | Cognitive PU Protection |
| :--- | :--- | :--- | :--- |
| **Static (Round-Robin)** | Baseline | Circular cyclical assignment | None (blind to PU presence) |
| **Proportional-Fair (PF)** | Baseline | Maximizes $R_{i,c}(t) / \bar{R}_i(t)^\alpha$ | Sensing heuristic avoidance |
| **Random Forest** | Supervised | Trained on DLTeamTUC CQI/demand | Threshold avoidance |
| **XGBoost** | Supervised | Gradient boosted tree on empirical traces | Threshold avoidance |
| **Dueling Double DQN** | Deep RL | Dueling value/advantage streams with KPI reward | **Learned optimal avoidance (0.0% collisions)** |

---

## 📝 Citation & Capstone Attribution

If using this codebase for your final-year engineering capstone or research publication, refer to the documentation generated in [`results/methodology_notes.md`](file:///c:/Users/kgr/Downloads/MajorProject/results/methodology_notes.md) for formal mathematical formulations and dataset citations.

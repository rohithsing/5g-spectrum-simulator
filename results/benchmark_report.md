# Intelligent Spectrum Allocation for 5G/6G -- Benchmark Report

Paired evaluation over **8 seeds** x **4 congestion regimes**, 1000 slots (500 ms) per episode.

Every allocator sees the identical observation -- buffer status, aged CQI reports and the output of one shared energy detector. The incumbent's true occupancy is never exposed to any of them.


## 1. Incumbent-occupancy estimation

| estimator                   |   accuracy |   precision |   recall |     f1 |    auc |   miss_rate |   false_alarm_rate |
|:----------------------------|-----------:|------------:|---------:|-------:|-------:|------------:|-------------------:|
| Energy detector (classical) |     0.8716 |      0.6895 |   0.85   | 0.7531 | 0.8645 |      0.15   |             0.1211 |
| Learned GRU (proposed)      |     0.9768 |      0.9564 |   0.9556 | 0.956  | 0.9953 |      0.0444 |             0.0162 |
| Bayes-optimal (upper bound) |     0.9788 |      0.965  |   0.9559 | 0.9604 | 0.9958 |      0.0441 |             0.0152 |

Averaged over all four congestion regimes on held-out traces. The Bayes-optimal filter is given the true Markov transition matrix and detector calibration, so it bounds what any estimator can achieve from the sensing stream alone.


## 2. Headline system KPIs


### Low congestion

| Metric                       | Best classical       |   Classical |   Proposed | Improvement %   | p     |
|:-----------------------------|:---------------------|------------:|-----------:|:----------------|:------|
| QoS satisfaction (%)         | Proportional Fair    |     100     |     98.8   | -1.2            | n/a   |
| Goodput (Mb/s)               | M-LWDF               |      75.2   |     67.9   | -9.7            | 0.123 |
| Spectral efficiency (b/s/Hz) | M-LWDF               |       0.835 |      0.755 | -9.7            | 0.123 |
| Deadline misses (%)          | Round-Robin (static) |       0     |      0     | +nan            | 1.000 |
| p99 delay (ms)               | EXP/PF               |       7.22  |      7.74  | -7.3            | 0.779 |
| Incumbent collisions (%)     | M-LWDF               |       2.24  |      1.09  | +51.3           | 0.025 |
| Wasted vacancy (%)           | Round-Robin (static) |      30.5   |     56.9   | -86.6           | 0.012 |
| Jain fairness                | Round-Robin (static) |       0.529 |      0.483 | -8.7            | 0.161 |
| Allocation waste (%)         | M-LWDF               |      11     |     12.1   | -10.0           | 0.161 |

### Medium congestion

| Metric                       | Best classical    |   Classical |   Proposed |   Improvement % |     p |
|:-----------------------------|:------------------|------------:|-----------:|----------------:|------:|
| QoS satisfaction (%)         | Max-C/I           |      68.6   |     47.1   |           -31.3 | 0.017 |
| Goodput (Mb/s)               | Max-C/I           |     204     |    211     |             3.2 | 0.263 |
| Spectral efficiency (b/s/Hz) | Max-C/I           |       2.27  |      2.34  |             3.2 | 0.263 |
| Deadline misses (%)          | EXP/PF            |       0.58  |      0.232 |            60   | 0.068 |
| p99 delay (ms)               | Max-C/I           |     104     |     93.2   |            10.6 | 0.575 |
| Incumbent collisions (%)     | EXP/PF            |       4.62  |      2.28  |            50.6 | 0.012 |
| Wasted vacancy (%)           | EXP/PF            |      12.2   |      2.87  |            76.5 | 0.012 |
| Jain fairness                | Proportional Fair |       0.645 |      0.604 |            -6.3 | 0.025 |
| Allocation waste (%)         | M-LWDF            |       7.82  |      7.95  |            -1.6 | 0.123 |

### High congestion

| Metric                       | Best classical    |   Classical |   Proposed |   Improvement % |     p |
|:-----------------------------|:------------------|------------:|-----------:|----------------:|------:|
| QoS satisfaction (%)         | Max-C/I           |      27.3   |     25.9   |            -5   | 0.345 |
| Goodput (Mb/s)               | Max-C/I           |     260     |    219     |           -15.5 | 0.012 |
| Spectral efficiency (b/s/Hz) | Max-C/I           |       2.88  |      2.44  |           -15.5 | 0.012 |
| Deadline misses (%)          | M-LWDF            |      18.9   |     18.3   |             2.8 | 0.263 |
| p99 delay (ms)               | Max-C/I           |     146     |    181     |           -23.9 | 0.012 |
| Incumbent collisions (%)     | Proportional Fair |       7.58  |      2.95  |            61.1 | 0.012 |
| Wasted vacancy (%)           | Proportional Fair |      11.8   |      3.72  |            68.4 | 0.012 |
| Jain fairness                | Proportional Fair |       0.796 |      0.729 |            -8.5 | 0.012 |
| Allocation waste (%)         | M-LWDF            |       5.6   |      5.55  |             0.8 | 0.484 |

### Peak congestion

| Metric                       | Best classical       |   Classical |   Proposed |   Improvement % |     p |
|:-----------------------------|:---------------------|------------:|-----------:|----------------:|------:|
| QoS satisfaction (%)         | Proportional Fair    |       14.1  |     17.7   |            25.6 | 0.108 |
| Goodput (Mb/s)               | Max-C/I              |      223    |    180     |           -19.2 | 0.012 |
| Spectral efficiency (b/s/Hz) | Max-C/I              |        2.48 |      2     |           -19.2 | 0.012 |
| Deadline misses (%)          | M-LWDF               |       31.9  |     28.7   |            10.2 | 0.025 |
| p99 delay (ms)               | Round-Robin (static) |      157    |    183     |           -16.2 | 0.017 |
| Incumbent collisions (%)     | Round-Robin (static) |       12.6  |      3.48  |            72.4 | 0.012 |
| Wasted vacancy (%)           | Round-Robin (static) |       12.1  |      4.91  |            59.5 | 0.012 |
| Jain fairness                | Proportional Fair    |        0.86 |      0.784 |            -8.8 | 0.012 |
| Allocation waste (%)         | EXP/PF               |        4.69 |      4.64  |             1   | 1     |

## 3. All allocators, all regimes

|                                                         |   qos_satisfaction_pct |   goodput_mbps |   deadline_miss_pct |   collision_pct |   jain_fairness |   p99_delay_ms |
|:--------------------------------------------------------|-----------------------:|---------------:|--------------------:|----------------:|----------------:|---------------:|
| ('low', 'Round-Robin (static)')                         |                  95.62 |          71.24 |                0    |            2.59 |            0.53 |          14.69 |
| ('low', 'Proportional Fair')                            |                 100    |          74.18 |                0    |            2.46 |            0.48 |          10.84 |
| ('low', 'Max-C/I')                                      |                  98.12 |          70.86 |                0    |            2.64 |            0.49 |           9.95 |
| ('low', 'M-LWDF')                                       |                  98.75 |          75.16 |                0    |            2.24 |            0.5  |           9.12 |
| ('low', 'EXP/PF')                                       |                  97.5  |          70.25 |                0    |            2.79 |            0.5  |           7.22 |
| ('low', 'M-LWDF + learned sensing')                     |                  98.12 |          69.35 |                0    |            0.82 |            0.5  |           7.36 |
| ('low', 'NeuroSpectrum, hard sensing (ablation)')       |                  98.12 |          74.36 |                0    |            2.76 |            0.48 |          10.02 |
| ('low', 'NeuroSpectrum, BC only / no RL (ablation)')    |                  96.25 |          72.82 |                0    |            1.14 |            0.48 |           7.97 |
| ('low', 'NeuroSpectrum (proposed)')                     |                  98.75 |          67.91 |                0    |            1.09 |            0.48 |           7.74 |
| ('medium', 'Round-Robin (static)')                      |                  43.21 |         157.31 |                6.69 |            4.94 |            0.62 |         145.41 |
| ('medium', 'Proportional Fair')                         |                  57.5  |         194.55 |                2.35 |            4.79 |            0.64 |         139.81 |
| ('medium', 'Max-C/I')                                   |                  68.57 |         204.41 |                9.81 |            4.66 |            0.52 |         104.25 |
| ('medium', 'M-LWDF')                                    |                  44.64 |         195.04 |                0.65 |            4.84 |            0.63 |         117.55 |
| ('medium', 'EXP/PF')                                    |                  45.71 |         192.82 |                0.58 |            4.62 |            0.64 |         111.93 |
| ('medium', 'M-LWDF + learned sensing')                  |                  57.14 |         210.17 |                0.36 |            1.5  |            0.6  |          85.79 |
| ('medium', 'NeuroSpectrum, hard sensing (ablation)')    |                  35.71 |         202.05 |                0.96 |            5.13 |            0.63 |         121.63 |
| ('medium', 'NeuroSpectrum, BC only / no RL (ablation)') |                  51.79 |         205.38 |                0.59 |            2.22 |            0.6  |          91.94 |
| ('medium', 'NeuroSpectrum (proposed)')                  |                  47.14 |         210.93 |                0.23 |            2.28 |            0.6  |          93.17 |
| ('high', 'Round-Robin (static)')                        |                  23.18 |         157.6  |               25.04 |            7.58 |            0.69 |         159.02 |
| ('high', 'Proportional Fair')                           |                  24.77 |         192.76 |               22.23 |            7.58 |            0.8  |         158.32 |
| ('high', 'Max-C/I')                                     |                  27.27 |         259.59 |               46.93 |            7.58 |            0.31 |         146.32 |
| ('high', 'M-LWDF')                                      |                  26.14 |         206.12 |               18.85 |            7.58 |            0.74 |         177.87 |
| ('high', 'EXP/PF')                                      |                  26.36 |         200.66 |               19.45 |            7.58 |            0.69 |         198.41 |
| ('high', 'M-LWDF + learned sensing')                    |                  27.5  |         222.7  |               18.03 |            2.28 |            0.74 |         194.94 |
| ('high', 'NeuroSpectrum, hard sensing (ablation)')      |                  26.82 |         210.48 |               17.93 |            7.4  |            0.74 |         182.16 |
| ('high', 'NeuroSpectrum, BC only / no RL (ablation)')   |                  25.68 |         216.88 |               18.23 |            2.94 |            0.74 |         173.72 |
| ('high', 'NeuroSpectrum (proposed)')                    |                  25.91 |         219.37 |               18.33 |            2.95 |            0.73 |         181.25 |
| ('peak', 'Round-Robin (static)')                        |                  13.75 |         123.62 |               46.11 |           12.61 |            0.71 |         157.1  |
| ('peak', 'Proportional Fair')                           |                  14.06 |         151.77 |               38.36 |           12.61 |            0.86 |         159.14 |
| ('peak', 'Max-C/I')                                     |                   8.59 |         222.8  |               67.33 |           12.61 |            0.15 |         166.56 |
| ('peak', 'M-LWDF')                                      |                  13.44 |         161.55 |               31.94 |           12.61 |            0.78 |         180.87 |
| ('peak', 'EXP/PF')                                      |                  10.31 |         159.89 |               33.26 |           12.61 |            0.69 |         224.44 |
| ('peak', 'M-LWDF + learned sensing')                    |                  15.31 |         185.37 |               28.85 |            3.83 |            0.78 |         181.96 |
| ('peak', 'NeuroSpectrum, hard sensing (ablation)')      |                  13.91 |         160.59 |               30.6  |           12.94 |            0.79 |         172.91 |
| ('peak', 'NeuroSpectrum, BC only / no RL (ablation)')   |                  13.12 |         161.23 |               32.32 |            4.01 |            0.78 |         177.21 |
| ('peak', 'NeuroSpectrum (proposed)')                    |                  17.66 |         179.97 |               28.68 |            3.48 |            0.78 |         182.53 |

## 4. Parameter provenance

- `carrier_freq_ghz / bandwidth_mhz / subcarrier_spacing_khz` -- 3GPP TS 38.104 FR1 band n78
- `n_rbg` -- 273 PRB at 30 kHz SCS, RBG size 16 (TS 38.214 Table 5.1.2.2.1-1)
- `bs_eirp_dbm` -- TS 38.104 wide-area base station class
- `CQI_TABLE / CQI_SINR_THRESHOLDS_DB` -- TS 38.214 Table 5.2.2.1-3, 10% BLER AWGN calibration
- `path loss model` -- TR 38.901 UMa NLOS, eq. 7.4.1-1
- `shadowing_std_db` -- TR 38.901 Table 7.5-6 (UMa NLOS)
- `TRAFFIC_CLASSES` -- TS 23.501 Table 5.7.4-1 (5QI 1, 2, 9, 82)
- `incumbent ON/OFF sojourn times` -- Saleem & Rehmani (2014), PU activity model survey
- `sensing_prob_detection / sensing_prob_false_alarm` -- FCC 47 CFR Part 96 incumbent-protection operating point
- `max_harq_retx` -- TS 38.321 HARQ entity configuration
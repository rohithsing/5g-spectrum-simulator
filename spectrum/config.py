"""
System configuration for the 5G/6G intelligent spectrum allocation testbed.

Every numeric constant in this module is anchored to a published specification or
measurement study; the provenance is recorded in ``PARAMETER_PROVENANCE`` so the
simulator's realism claims can be audited line by line.

References
----------
[1] 3GPP TR 38.901 v17.0.0, "Study on channel model for frequencies from 0.5 to 100 GHz".
[2] 3GPP TS 38.214 v17.5.0, "NR; Physical layer procedures for data", Table 5.2.2.1-3.
[3] 3GPP TS 38.104 v17.0.0, "NR; Base Station radio transmission and reception".
[4] 3GPP TS 23.501 v17.0.0, Table 5.7.4-1, "Standardized 5QI to QoS characteristics mapping".
[5] 3GPP TR 38.913 v16.0.0, "Study on scenarios and requirements for next generation access".
[6] FCC 47 CFR Part 96 (Citizens Broadband Radio Service), incumbent protection criteria.
[7] Y. Saleem and M. H. Rehmani, "Primary radio user activity models for cognitive radio
    networks: A survey", J. Netw. Comput. Appl., vol. 43, pp. 1-16, 2014.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 3GPP TS 38.214 Table 5.2.2.1-3 -- CQI table 2 (up to 256QAM).
# Columns: (CQI index, modulation order, code rate x 1024, spectral efficiency).
# ---------------------------------------------------------------------------
CQI_TABLE: List[Tuple[int, int, int, float]] = [
    (1, 2, 78, 0.1523),
    (2, 2, 193, 0.3770),
    (3, 2, 449, 0.8770),
    (4, 4, 378, 1.4766),
    (5, 4, 490, 1.9141),
    (6, 4, 616, 2.4063),
    (7, 6, 466, 2.7305),
    (8, 6, 567, 3.3223),
    (9, 6, 666, 3.9023),
    (10, 6, 772, 4.5234),
    (11, 6, 873, 5.1152),
    (12, 8, 711, 5.5547),
    (13, 8, 797, 6.2266),
    (14, 8, 885, 6.9141),
    (15, 8, 948, 7.4063),
]

#: Post-equalisation SINR (dB) required by each CQI for a 10 % transport-block
#: error rate on an AWGN link. Standard link-level calibration points.
CQI_SINR_THRESHOLDS_DB: np.ndarray = np.array(
    [-6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 8.1, 10.3, 11.7, 14.1, 16.3, 18.7, 21.0, 22.7],
    dtype=np.float64,
)

CQI_SPECTRAL_EFFICIENCY: np.ndarray = np.array([row[3] for row in CQI_TABLE], dtype=np.float64)


# ---------------------------------------------------------------------------
# Traffic classes -- 3GPP TS 23.501 Table 5.7.4-1 (5QI characteristics).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrafficClass:
    """A 5QI-derived service class."""

    name: str
    five_qi: int
    #: Packet Delay Budget in milliseconds (TS 23.501 Table 5.7.4-1).
    delay_budget_ms: float
    #: Packet Error Rate target (TS 23.501 Table 5.7.4-1).
    error_rate: float
    #: Guaranteed / nominal bit rate in Mbit/s used as the throughput SLA.
    nominal_rate_mbps: float
    #: Mean application payload per arrival, in bits.
    mean_payload_bits: float
    #: Mean inter-arrival time in milliseconds.
    mean_interarrival_ms: float
    #: Scheduling priority level (lower is more important), TS 23.501.
    priority_level: int
    #: Relative population share when a cell is instantiated.
    population_share: float
    #: Burstiness of the arrival process (coefficient of variation of payload).
    payload_cv: float


#: 5QI 1 (conversational voice), 5QI 2 (live video), 5QI 9 (best effort),
#: 5QI 82 (discrete automation / URLLC), per TS 23.501 Table 5.7.4-1.
TRAFFIC_CLASSES: Tuple[TrafficClass, ...] = (
    TrafficClass(
        name="URLLC",
        five_qi=82,
        delay_budget_ms=10.0,
        error_rate=1e-4,
        nominal_rate_mbps=2.0,
        mean_payload_bits=2048.0,
        mean_interarrival_ms=2.0,
        priority_level=19,
        population_share=0.15,
        payload_cv=0.25,
    ),
    TrafficClass(
        name="VoIP",
        five_qi=1,
        delay_budget_ms=100.0,
        error_rate=1e-2,
        nominal_rate_mbps=0.1,
        mean_payload_bits=640.0,
        mean_interarrival_ms=20.0,
        priority_level=20,
        population_share=0.15,
        payload_cv=0.10,
    ),
    TrafficClass(
        name="eMBB-Video",
        five_qi=2,
        delay_budget_ms=150.0,
        error_rate=1e-3,
        nominal_rate_mbps=12.0,
        mean_payload_bits=120_000.0,
        mean_interarrival_ms=10.0,
        priority_level=40,
        population_share=0.40,
        payload_cv=0.65,
    ),
    TrafficClass(
        name="BestEffort",
        five_qi=9,
        delay_budget_ms=300.0,
        error_rate=1e-6,
        nominal_rate_mbps=5.0,
        mean_payload_bits=400_000.0,
        mean_interarrival_ms=60.0,
        priority_level=90,
        population_share=0.30,
        payload_cv=1.10,
    ),
)

CLASS_INDEX: Dict[str, int] = {tc.name: i for i, tc in enumerate(TRAFFIC_CLASSES)}
CLASS_NAMES: Tuple[str, ...] = tuple(tc.name for tc in TRAFFIC_CLASSES)


# ---------------------------------------------------------------------------
# Congestion regimes. These scale offered load, not the scheduler's resources,
# so every allocator is stressed identically.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CongestionRegime:
    name: str
    #: Multiplier on per-UE arrival rate.
    load_multiplier: float
    #: Number of UEs attached to the cell.
    n_ue: int
    #: Steady-state probability that an incumbent occupies a given subband.
    incumbent_duty_cycle: float


CONGESTION_REGIMES: Dict[str, CongestionRegime] = {
    "low": CongestionRegime("low", 0.55, 20, 0.12),
    "medium": CongestionRegime("medium", 1.00, 35, 0.22),
    "high": CongestionRegime("high", 1.60, 55, 0.34),
    "peak": CongestionRegime("peak", 2.30, 80, 0.46),
}


@dataclass
class SimConfig:
    """Complete parameterisation of one cell-level simulation."""

    # -- Radio numerology (3GPP TS 38.104, FR1 band n78) --------------------
    carrier_freq_ghz: float = 3.5
    bandwidth_mhz: float = 100.0
    subcarrier_spacing_khz: float = 30.0
    #: 273 PRBs fit in 100 MHz at 30 kHz SCS; RBG size 16 -> 18 resource groups.
    n_rbg: int = 18
    #: Guard band / control overhead fraction removed from the nominal bandwidth.
    usable_bandwidth_fraction: float = 0.90
    #: Fraction of symbols in a slot carrying PDSCH (rest is PDCCH/DMRS/CSI-RS).
    pdsch_symbol_fraction: float = 0.86

    # -- Transmitter / receiver (3GPP TS 38.104 wide-area BS class) ---------
    bs_eirp_dbm: float = 46.0
    ue_noise_figure_db: float = 7.0
    thermal_noise_dbm_per_hz: float = -174.0

    # -- Deployment geometry (3GPP TR 38.901 UMa) ---------------------------
    cell_radius_m: float = 350.0
    min_ue_distance_m: float = 35.0
    ue_height_m: float = 1.5
    shadowing_std_db: float = 5.0
    #: Inter-cell interference floor seen by a UE, relative to thermal noise.
    intercell_interference_db_over_noise: float = 4.0

    # -- Mobility / Doppler --------------------------------------------------
    ue_speed_min_mps: float = 0.5
    ue_speed_max_mps: float = 16.7  # 60 km/h
    #: Coherence bandwidth as a fraction of the carrier bandwidth; controls how
    #: correlated small-scale fading is across RBGs (frequency selectivity).
    coherence_bandwidth_frac: float = 0.11

    # -- Incumbent / primary user (FCC Part 96 CBRS-style sharing) ----------
    n_subbands: int = 6
    #: Mean ON and OFF sojourn times in slots, from PU activity survey [7].
    incumbent_mean_on_slots: float = 42.0
    incumbent_mean_off_slots: float = 128.0
    #: Interference injected into a subband while an incumbent transmits.
    incumbent_interference_dbm: float = -62.0
    #: Energy-detector operating point. Both probabilities are fixed for all
    #: allocators so that information parity is structurally guaranteed.
    sensing_prob_detection: float = 0.85
    sensing_prob_false_alarm: float = 0.12

    # -- CSI reporting imperfection (equal for every allocator) -------------
    #: Age of the CQI report in slots when the scheduler consumes it.
    cqi_report_delay_slots: int = 2
    #: Period, in slots, between CQI reports (wideband + subband).
    cqi_report_period_slots: int = 4
    #: Standard deviation of CQI estimation error in dB.
    cqi_estimation_error_db: float = 1.0

    # -- HARQ ----------------------------------------------------------------
    max_harq_retx: int = 3
    #: Each retransmission adds this much effective SINR via chase combining.
    harq_combining_gain_db: float = 2.5

    # -- Scenario ------------------------------------------------------------
    congestion: str = "medium"
    n_ue: int = 0  # 0 -> take from the congestion regime
    slots_per_episode: int = 1000
    seed: int = 0

    # -- Derived (filled in __post_init__) ----------------------------------
    slot_duration_s: float = field(init=False)
    rbg_bandwidth_hz: float = field(init=False)
    rbg_noise_dbm: float = field(init=False)
    rbg_tx_power_dbm: float = field(init=False)
    rbgs_per_subband: int = field(init=False)

    def __post_init__(self) -> None:
        if self.congestion not in CONGESTION_REGIMES:
            raise ValueError(
                f"unknown congestion regime {self.congestion!r}; "
                f"expected one of {sorted(CONGESTION_REGIMES)}"
            )
        regime = CONGESTION_REGIMES[self.congestion]
        if self.n_ue <= 0:
            self.n_ue = regime.n_ue

        # Slot length follows the NR numerology: 1 ms / 2^mu with mu = log2(SCS/15).
        mu = int(round(np.log2(self.subcarrier_spacing_khz / 15.0)))
        self.slot_duration_s = 1e-3 / (2 ** mu)

        usable_hz = self.bandwidth_mhz * 1e6 * self.usable_bandwidth_fraction
        self.rbg_bandwidth_hz = usable_hz / self.n_rbg

        self.rbg_noise_dbm = float(
            self.thermal_noise_dbm_per_hz
            + 10.0 * np.log10(self.rbg_bandwidth_hz)
            + self.ue_noise_figure_db
        )
        self.rbg_tx_power_dbm = float(self.bs_eirp_dbm - 10.0 * np.log10(self.n_rbg))

        if self.n_rbg % self.n_subbands != 0:
            raise ValueError(
                f"n_rbg ({self.n_rbg}) must be divisible by n_subbands ({self.n_subbands}) "
                "so that incumbent subbands map onto whole resource groups"
            )
        self.rbgs_per_subband = self.n_rbg // self.n_subbands

    # -- helpers -------------------------------------------------------------
    @property
    def regime(self) -> CongestionRegime:
        return CONGESTION_REGIMES[self.congestion]

    def subband_of_rbg(self) -> np.ndarray:
        """Vector of length ``n_rbg`` giving each RBG's subband index."""
        return np.arange(self.n_rbg, dtype=np.int64) // self.rbgs_per_subband

    def bits_per_rbg(self, spectral_efficiency):
        """Transport-block size, in bits, carried by one RBG in one slot."""
        return (
            spectral_efficiency
            * self.rbg_bandwidth_hz
            * self.slot_duration_s
            * self.pdsch_symbol_fraction
        )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


PARAMETER_PROVENANCE: Dict[str, str] = {
    "carrier_freq_ghz / bandwidth_mhz / subcarrier_spacing_khz": "3GPP TS 38.104 FR1 band n78",
    "n_rbg": "273 PRB at 30 kHz SCS, RBG size 16 (TS 38.214 Table 5.1.2.2.1-1)",
    "bs_eirp_dbm": "TS 38.104 wide-area base station class",
    "CQI_TABLE / CQI_SINR_THRESHOLDS_DB": "TS 38.214 Table 5.2.2.1-3, 10% BLER AWGN calibration",
    "path loss model": "TR 38.901 UMa NLOS, eq. 7.4.1-1",
    "shadowing_std_db": "TR 38.901 Table 7.5-6 (UMa NLOS)",
    "TRAFFIC_CLASSES": "TS 23.501 Table 5.7.4-1 (5QI 1, 2, 9, 82)",
    "incumbent ON/OFF sojourn times": "Saleem & Rehmani (2014), PU activity model survey",
    "sensing_prob_detection / sensing_prob_false_alarm": "FCC 47 CFR Part 96 incumbent-protection operating point",
    "max_harq_retx": "TS 38.321 HARQ entity configuration",
}

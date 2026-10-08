"""
Physical-layer model: propagation, frequency-selective fading, SINR, link adaptation.

The channel is correlated in **both** time and frequency, which is what makes
scheduling a non-trivial decision problem:

* Time correlation (first-order Gauss-Markov with a Jakes-derived coefficient)
  means a UE's channel is predictable over a few slots, so a scheduler that
  reasons about the near future can out-perform a purely instantaneous one.
* Frequency correlation (exponential decay across RBGs with a configurable
  coherence bandwidth) means different UEs peak on different parts of the band,
  so *which* RBG goes to *which* UE matters.

All SINR-to-rate mapping follows 3GPP TS 38.214 CQI table 2 with a 10 % BLER
target and chase-combining HARQ.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .config import CQI_SINR_THRESHOLDS_DB, CQI_SPECTRAL_EFFICIENCY, SimConfig


def uma_nlos_pathloss_db(distance_m: np.ndarray, carrier_ghz: float, ue_height_m: float) -> np.ndarray:
    """3GPP TR 38.901 Urban Macro NLOS path loss (eq. 7.4.1-1)."""
    d3d = np.maximum(distance_m, 10.0)
    return (
        13.54
        + 39.08 * np.log10(d3d)
        + 20.0 * np.log10(carrier_ghz)
        - 0.6 * (ue_height_m - 1.5)
    )


class RadioChannel:
    """Per-UE, per-RBG wideband + small-scale channel with HARQ-aware link adaptation."""

    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.n_ue = cfg.n_ue
        self.n_rbg = cfg.n_rbg

        # -- Static geometry -------------------------------------------------
        # Uniform-in-area UE drop between the minimum distance and the cell edge.
        u = rng.random(self.n_ue)
        r_min2, r_max2 = cfg.min_ue_distance_m ** 2, cfg.cell_radius_m ** 2
        self.distance_m = np.sqrt(r_min2 + u * (r_max2 - r_min2))

        self.pathloss_db = uma_nlos_pathloss_db(
            self.distance_m, cfg.carrier_freq_ghz, cfg.ue_height_m
        )
        self.shadowing_db = rng.normal(0.0, cfg.shadowing_std_db, size=self.n_ue)

        # -- Mobility -> Doppler -> time-correlation coefficient -------------
        self.speed_mps = rng.uniform(cfg.ue_speed_min_mps, cfg.ue_speed_max_mps, size=self.n_ue)
        wavelength_m = 3e8 / (cfg.carrier_freq_ghz * 1e9)
        doppler_hz = self.speed_mps / wavelength_m
        # Jakes' model autocorrelation over one slot: rho = J0(2*pi*fd*T).
        jakes_arg = 2.0 * np.pi * doppler_hz * cfg.slot_duration_s
        self.rho_time = np.clip(_bessel_j0(jakes_arg), 0.0, 0.9995)

        # -- Frequency correlation across RBGs -------------------------------
        coherence_rbgs = max(cfg.coherence_bandwidth_frac * cfg.n_rbg, 1e-6)
        lag = np.abs(np.arange(self.n_rbg)[:, None] - np.arange(self.n_rbg)[None, :])
        corr = np.exp(-lag / coherence_rbgs)
        # Cholesky factor used to colour i.i.d. noise into a frequency-correlated profile.
        self._freq_chol = np.linalg.cholesky(corr + 1e-9 * np.eye(self.n_rbg))

        # -- State -----------------------------------------------------------
        #: Small-scale fading in dB, shape (n_ue, n_rbg), zero-mean.
        self.fading_db = self._draw_fading()
        self.noise_plus_interference_dbm = self._compute_noise_floor()

        # CQI reporting pipeline: ring buffer of past reported CQI grids.
        depth = max(cfg.cqi_report_delay_slots, 1) + 1
        self._cqi_history = [self._measure_cqi() for _ in range(depth)]
        self._last_report = self._cqi_history[-1].copy()
        self._slot = 0

    # ------------------------------------------------------------------ core
    def _draw_fading(self) -> np.ndarray:
        """Draw a frequency-correlated, zero-mean log-domain fading profile."""
        white = self.rng.standard_normal((self.n_ue, self.n_rbg))
        coloured = white @ self._freq_chol.T
        # ~5 dB standard deviation reproduces the depth of Rayleigh-like fades
        # once converted back to the linear domain.
        return 5.0 * coloured

    def _compute_noise_floor(self) -> np.ndarray:
        """Thermal noise plus inter-cell interference, in dBm, per RBG."""
        noise_mw = 10.0 ** (self.cfg.rbg_noise_dbm / 10.0)
        icic_mw = noise_mw * 10.0 ** (self.cfg.intercell_interference_db_over_noise / 10.0)
        total = noise_mw + icic_mw
        return np.full(self.n_rbg, 10.0 * np.log10(total))

    def advance(self) -> None:
        """Evolve small-scale fading one slot under the Gauss-Markov model."""
        rho = self.rho_time[:, None]
        innovation = self._draw_fading()
        self.fading_db = rho * self.fading_db + np.sqrt(1.0 - rho ** 2) * innovation
        self._slot += 1

    # ------------------------------------------------------- SINR and rates
    def true_sinr_db(self, incumbent_active: np.ndarray) -> np.ndarray:
        """
        Instantaneous SINR in dB for every (UE, RBG) pair.

        ``incumbent_active`` is the *ground-truth* per-subband occupancy vector.
        It is consumed here, inside the physics, and is never surfaced to a
        scheduler -- see :mod:`spectrum.env` for the observation contract.
        """
        cfg = self.cfg
        rx_dbm = (
            cfg.rbg_tx_power_dbm
            - self.pathloss_db[:, None]
            - self.shadowing_db[:, None]
            + self.fading_db
        )

        noise_mw = 10.0 ** (self.noise_plus_interference_dbm / 10.0)
        occupied = incumbent_active[cfg.subband_of_rbg()].astype(bool)
        incumbent_mw = np.where(occupied, 10.0 ** (cfg.incumbent_interference_dbm / 10.0), 0.0)
        total_noise_dbm = 10.0 * np.log10(noise_mw + incumbent_mw)

        return rx_dbm - total_noise_dbm[None, :]

    def _measure_cqi(self) -> np.ndarray:
        """
        UE-side CQI measurement on a *clean* channel.

        Real CSI-RS measurements are taken on reference symbols that do not
        collide with an incumbent burst, so the reported CQI reflects the
        propagation channel only. Discovering incumbent activity is the job of
        the sensing subsystem, not of CSI feedback -- which is precisely why
        spectrum sharing is hard.
        """
        clean = np.zeros(self.cfg.n_subbands, dtype=np.int64)
        sinr = self.true_sinr_db(clean)
        noisy = sinr + self.rng.normal(0.0, self.cfg.cqi_estimation_error_db, size=sinr.shape)
        return sinr_to_cqi(noisy)

    def update_cqi_reports(self) -> None:
        """Push a fresh measurement into the reporting pipeline."""
        self._cqi_history.append(self._measure_cqi())
        if len(self._cqi_history) > self.cfg.cqi_report_delay_slots + 1:
            self._cqi_history.pop(0)
        if self._slot % self.cfg.cqi_report_period_slots == 0:
            self._last_report = self._cqi_history[0].copy()

    def reported_cqi(self) -> np.ndarray:
        """The aged, quantised CQI grid a scheduler is allowed to use."""
        return self._last_report

    # ------------------------------------------------------- link adaptation
    def transmit(
        self,
        assignment: np.ndarray,
        incumbent_active: np.ndarray,
        harq_retx_count: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate one slot of downlink transmission.

        Parameters
        ----------
        assignment:
            Length ``n_rbg`` array; entry ``k`` is the UE index scheduled on RBG
            ``k``, or ``-1`` if the RBG is muted.
        incumbent_active:
            Ground-truth per-subband incumbent occupancy.
        harq_retx_count:
            Per-UE number of consecutive HARQ retransmissions, used for the
            chase-combining gain.

        Returns
        -------
        delivered_bits:
            Per-RBG successfully delivered bits (0 on a HARQ NACK).
        acked:
            Per-RBG boolean, ``True`` when the transport block decoded.
        collided:
            Per-RBG boolean, ``True`` when we transmitted into a subband that an
            incumbent was actually using (a harmful-interference event).
        """
        cfg = self.cfg
        n_rbg = cfg.n_rbg
        delivered = np.zeros(n_rbg, dtype=np.float64)
        acked = np.zeros(n_rbg, dtype=bool)
        collided = np.zeros(n_rbg, dtype=bool)

        active = assignment >= 0
        if not np.any(active):
            return delivered, acked, collided

        sinr = self.true_sinr_db(incumbent_active)
        subband = cfg.subband_of_rbg()
        occupied = incumbent_active[subband].astype(bool)

        rbgs = np.flatnonzero(active)
        ues = assignment[rbgs]

        collided[rbgs] = occupied[rbgs]

        # The scheduler picks its MCS from the *reported* CQI, not from the
        # instantaneous channel: an optimistic report causes a decode failure.
        reported = self._last_report[ues, rbgs]
        eff_sinr = sinr[ues, rbgs] + cfg.harq_combining_gain_db * harq_retx_count[ues]

        required = CQI_SINR_THRESHOLDS_DB[reported]
        margin = eff_sinr - required
        bler = _bler_from_margin(margin)
        success = self.rng.random(rbgs.size) >= bler

        se = CQI_SPECTRAL_EFFICIENCY[reported]
        tb_bits = cfg.bits_per_rbg(se)

        delivered[rbgs] = np.where(success, tb_bits, 0.0)
        acked[rbgs] = success
        return delivered, acked, collided


# --------------------------------------------------------------------------- utils
def sinr_to_cqi(sinr_db: np.ndarray) -> np.ndarray:
    """Highest CQI index whose 10 % BLER threshold is met (0-based)."""
    idx = np.searchsorted(CQI_SINR_THRESHOLDS_DB, sinr_db, side="right") - 1
    return np.clip(idx, 0, len(CQI_SINR_THRESHOLDS_DB) - 1).astype(np.int64)


def _bler_from_margin(margin_db: np.ndarray) -> np.ndarray:
    """
    Block error rate as a function of SINR margin above the CQI threshold.

    A logistic fit to NR link-level curves: BLER = 0.1 at zero margin (the CQI
    selection target), falling below 1e-3 roughly 4 dB above it and saturating
    towards 1.0 several dB below.
    """
    return np.clip(1.0 / (1.0 + np.exp(1.35 * (margin_db + 1.63))), 1e-6, 1.0)


def _bessel_j0(x: np.ndarray) -> np.ndarray:
    """Zeroth-order Bessel function of the first kind (SciPy-free)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)

    small = np.abs(x) < 8.0
    xs = x[small]
    y = xs * xs
    num = 57568490574.0 + y * (
        -13362590354.0 + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * -184.9052456)))
    )
    den = 57568490411.0 + y * (
        1029532985.0 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y)))
    )
    out[small] = num / den

    big = ~small
    xb = np.abs(x[big])
    z = 8.0 / xb
    y = z * z
    xx = xb - 0.785398164
    p = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)))
    q = -0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5 + y * (0.7621095161e-6 + y * -0.934935152e-7)))
    out[big] = np.sqrt(0.636619772 / xb) * (np.cos(xx) * p - z * np.sin(xx) * q)
    return out

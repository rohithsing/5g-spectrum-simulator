"""
Physical-layer and protocol correctness.

These check the simulator against the specifications it claims to follow, so a
reviewer can see the realism claims are testable rather than asserted.
"""

from __future__ import annotations

import numpy as np
import pytest

from spectrum.channel import (
    RadioChannel,
    _bessel_j0,
    _bler_from_margin,
    sinr_to_cqi,
    uma_nlos_pathloss_db,
)
from spectrum.config import (
    CQI_SINR_THRESHOLDS_DB,
    CQI_SPECTRAL_EFFICIENCY,
    SimConfig,
)
from spectrum.env import SpectrumEnv
from spectrum.incumbent import IncumbentProcess, _norm_ppf
from spectrum.metrics import jain_fairness


# --------------------------------------------------------------------- numerics
def test_bessel_j0_matches_known_values():
    x = np.array([0.0, 1.0, 2.404825557695773, 5.0, 10.0])
    expected = np.array([1.0, 0.7651976866, 0.0, -0.1775967713, -0.2459357645])
    assert np.allclose(_bessel_j0(x), expected, atol=1e-6)


def test_norm_ppf_matches_known_quantiles():
    assert _norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-4)
    assert _norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
    assert _norm_ppf(0.01) == pytest.approx(-2.326348, abs=1e-4)


# ------------------------------------------------------------------- numerology
def test_nr_numerology_is_consistent():
    cfg = SimConfig(subcarrier_spacing_khz=30.0, bandwidth_mhz=100.0)
    assert cfg.slot_duration_s == pytest.approx(0.5e-3)
    # 90 % of 100 MHz spread over 18 RBGs
    assert cfg.rbg_bandwidth_hz == pytest.approx(5e6)
    # kTB + noise figure
    assert cfg.rbg_noise_dbm == pytest.approx(-174 + 10 * np.log10(5e6) + 7, abs=1e-6)
    # EIRP split evenly across RBGs
    assert cfg.rbg_tx_power_dbm == pytest.approx(46 - 10 * np.log10(18), abs=1e-6)


def test_subbands_tile_the_carrier_exactly():
    cfg = SimConfig()
    sb = cfg.subband_of_rbg()
    assert sb.size == cfg.n_rbg
    assert sorted(np.bincount(sb)) == [cfg.rbgs_per_subband] * cfg.n_subbands


def test_peak_rate_is_physically_plausible():
    """100 MHz NR carrier, 256QAM, single layer: a few hundred Mb/s."""
    cfg = SimConfig()
    peak_bps = cfg.bits_per_rbg(CQI_SPECTRAL_EFFICIENCY[-1]) * cfg.n_rbg / cfg.slot_duration_s
    assert 400e6 < peak_bps < 800e6


# ----------------------------------------------------------------- propagation
def test_uma_pathloss_increases_with_distance_and_frequency():
    d = np.array([50.0, 200.0, 500.0])
    pl = uma_nlos_pathloss_db(d, 3.5, 1.5)
    assert np.all(np.diff(pl) > 0)
    assert uma_nlos_pathloss_db(np.array([200.0]), 28.0, 1.5)[0] > pl[1]
    # TR 38.901 UMa NLOS at 200 m / 3.5 GHz sits near 115 dB.
    assert 105.0 < pl[1] < 125.0


def test_link_adaptation_targets_ten_percent_bler():
    """CQI selection is calibrated so that zero SINR margin gives 10 % BLER."""
    assert _bler_from_margin(np.array([0.0]))[0] == pytest.approx(0.1, abs=0.005)
    assert _bler_from_margin(np.array([6.0]))[0] < 1e-3
    assert _bler_from_margin(np.array([-6.0]))[0] > 0.9


def test_sinr_to_cqi_is_monotonic_and_bounded():
    sinr = np.linspace(-20, 40, 400)
    cqi = sinr_to_cqi(sinr)
    assert np.all(np.diff(cqi) >= 0)
    assert cqi.min() == 0 and cqi.max() == len(CQI_SINR_THRESHOLDS_DB) - 1
    for i, thr in enumerate(CQI_SINR_THRESHOLDS_DB):
        assert sinr_to_cqi(np.array([thr + 0.01]))[0] >= i


def test_incumbent_activity_destroys_sinr():
    cfg = SimConfig(seed=2, n_ue=40)
    ch = RadioChannel(cfg, np.random.default_rng(2))
    clean = ch.true_sinr_db(np.zeros(cfg.n_subbands, dtype=np.int64))
    jammed = ch.true_sinr_db(np.ones(cfg.n_subbands, dtype=np.int64))
    assert np.median(clean) - np.median(jammed) > 25.0


def test_fading_is_correlated_in_time_and_frequency():
    cfg = SimConfig(seed=4, n_ue=200)
    ch = RadioChannel(cfg, np.random.default_rng(4))

    before = ch.fading_db.copy()
    ch.advance()
    after = ch.fading_db
    # Slot-to-slot correlation must be high but not perfect.
    r_time = np.corrcoef(before.ravel(), after.ravel())[0, 1]
    assert 0.3 < r_time < 0.999

    # Adjacent RBGs correlate more strongly than distant ones.
    adj = np.corrcoef(after[:, 0], after[:, 1])[0, 1]
    far = np.corrcoef(after[:, 0], after[:, -1])[0, 1]
    assert adj > far


# -------------------------------------------------------------------- incumbent
@pytest.mark.parametrize("level", ["low", "medium", "high", "peak"])
def test_incumbent_duty_cycle_matches_the_regime(level):
    cfg = SimConfig(congestion=level)
    inc = IncumbentProcess(cfg, np.random.default_rng(0))
    occ = []
    for _ in range(40_000):
        inc.advance()
        occ.append(inc.active.mean())
    assert np.mean(occ) == pytest.approx(cfg.regime.incumbent_duty_cycle, abs=0.03)


def test_incumbent_bursts_are_temporally_correlated():
    """Occupancy must be predictable -- that is the structure the agent exploits."""
    cfg = SimConfig(congestion="medium")
    inc = IncumbentProcess(cfg, np.random.default_rng(0))
    states = []
    for _ in range(20_000):
        states.append(inc.active[0])
        inc.advance()
    s = np.array(states, dtype=float)
    assert np.corrcoef(s[:-1], s[1:])[0, 1] > 0.9


# ----------------------------------------------------------------------- metrics
def test_jain_fairness_endpoints():
    assert jain_fairness(np.ones(10)) == pytest.approx(1.0)
    one_hot = np.zeros(10)
    one_hot[0] = 1.0
    assert jain_fairness(one_hot) == pytest.approx(0.1)
    assert jain_fairness(np.zeros(5)) == pytest.approx(1.0)


# ------------------------------------------------------------------- environment
def test_muting_every_rbg_delivers_nothing_and_never_collides():
    cfg = SimConfig(congestion="peak", seed=9, slots_per_episode=50)
    env = SpectrumEnv(cfg, seed=9)
    for _ in range(50):
        _, _, _, info = env.step(np.full(cfg.n_rbg, -1))
        assert info["useful_bits"] == 0.0
        assert info["collisions"] == 0
    assert env.episode_summary()["collision_pct"] == 0.0


def test_useful_bits_never_exceed_delivered_bits():
    cfg = SimConfig(congestion="low", seed=6, slots_per_episode=120)
    env = SpectrumEnv(cfg, seed=6)
    rng = np.random.default_rng(6)
    for _ in range(120):
        _, _, _, info = env.step(rng.integers(-1, cfg.n_ue, size=cfg.n_rbg))
        assert info["useful_bits"] <= info["delivered_bits"] + 1e-6


def test_assignment_outside_the_cell_is_rejected():
    cfg = SimConfig(seed=1, slots_per_episode=5)
    env = SpectrumEnv(cfg, seed=1)
    bad = np.full(cfg.n_rbg, cfg.n_ue)  # one past the last valid UE
    with pytest.raises(ValueError):
        env.step(bad)


def test_episode_is_reproducible_under_a_fixed_seed():
    cfg = SimConfig(congestion="medium", seed=21, slots_per_episode=80)

    def rollout():
        env = SpectrumEnv(cfg, seed=21)
        rng = np.random.default_rng(0)
        total = 0.0
        for _ in range(80):
            _, r, _, _ = env.step(rng.integers(-1, cfg.n_ue, size=cfg.n_rbg))
            total += r
        return total, env.episode_summary()

    a_reward, a_sum = rollout()
    b_reward, b_sum = rollout()
    assert a_reward == pytest.approx(b_reward)
    assert a_sum == pytest.approx(b_sum, nan_ok=True)

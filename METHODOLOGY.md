# Methodology

This document states precisely what is simulated, what each allocator is allowed
to know, how the learned components are trained, and how the comparison is scored.
It is written so a reviewer can find the specific line of code behind every claim.

---

## 1. Problem statement

A single 5G NR cell shares a 100 MHz carrier at 3.5 GHz with a stochastic
incumbent. Every 0.5 ms slot the scheduler must decide, for each of 18 resource
block groups, either which attached UE to serve on it or to leave it silent.

The decision is hard for four reasons that are all represented in the simulator:

1. **Frequency selectivity.** Different UEs peak on different parts of the band,
   so *which* RBG goes to *which* UE matters.
2. **Heterogeneous QoS.** URLLC, VoIP, live video and best-effort bearers have
   delay budgets spanning 10 ms to 300 ms, and packets are *dropped* when the
   budget expires.
3. **Imperfect CSI.** The scheduler chooses its modulation from a CQI report that
   is two slots old and quantised, so an optimistic report produces a real decode
   failure.
4. **Incumbent uncertainty.** Transmitting on an occupied subband is harmful
   interference, but the only evidence available is a noisy energy detector.

---

## 2. Simulator

### 2.1 Radio numerology (`spectrum/config.py`)

| Parameter | Value | Source |
|---|---|---|
| Carrier / bandwidth / SCS | 3.5 GHz, 100 MHz, 30 kHz | TS 38.104 band n78 |
| Resource block groups | 18 (273 PRB, RBG size 16) | TS 38.214 Table 5.1.2.2.1-1 |
| Slot duration | 0.5 ms (μ = 1) | TS 38.211 |
| BS EIRP | 46 dBm | TS 38.104 wide-area class |
| UE noise figure | 7 dB | TS 38.101 |
| Per-RBG noise floor | −100.0 dBm | kTB + NF over 5 MHz |

`tests/test_physics.py::test_nr_numerology_is_consistent` recomputes each derived
quantity from first principles.

### 2.2 Propagation and fading (`spectrum/channel.py`)

Path loss is TR 38.901 UMa NLOS (eq. 7.4.1-1) over a 350 m cell with UEs dropped
uniformly in area, plus 5 dB log-normal shadowing. Small-scale fading is a
zero-mean log-domain process that is:

* **correlated in time** — first-order Gauss-Markov with ρ = J₀(2π f_D T_slot),
  the Jakes autocorrelation at each UE's own Doppler (speeds 0.5–16.7 m/s);
* **correlated in frequency** — an exponential kernel across RBGs with a coherence
  bandwidth of 11 % of the carrier, applied through a Cholesky factor.

The resulting SINR distribution is approximately −3 dB (5th pct) / 11.6 dB
(median) / 32 dB (95th pct).

### 2.3 Link adaptation

SINR maps to CQI through TS 38.214 Table 5.2.2.1-3 (CQI table 2, up to 256QAM) at
a 10 % BLER target. Block error rate is a logistic fit to NR link-level curves,
calibrated so that zero SINR margin above the selected CQI's threshold gives
exactly 10 % BLER (asserted in `test_link_adaptation_targets_ten_percent_bler`).
HARQ retransmissions accumulate 2.5 dB of chase-combining gain each, up to three.

**CQI is measured on a clean channel.** CSI-RS measurements are not taken during
an incumbent burst, so the report reflects propagation only. Discovering incumbent
activity is the sensing subsystem's job — which is exactly why spectrum sharing is
hard, and why the occupancy predictor has something to contribute.

### 2.4 Traffic (`spectrum/traffic.py`)

Four 5QI bearers from TS 23.501 Table 5.7.4-1:

| Class | 5QI | Delay budget | Nominal rate | Population |
|---|---|---|---|---|
| URLLC | 82 | 10 ms | 2 Mb/s | 15 % |
| VoIP | 1 | 100 ms | 0.1 Mb/s | 15 % |
| eMBB video | 2 | 150 ms | 12 Mb/s | 40 % |
| Best effort | 9 | 300 ms | 5 Mb/s | 30 % |

Arrivals are Bernoulli per slot with log-normal payloads matched to a per-class
mean and coefficient of variation. Queues are maintained **per packet**, so
head-of-line delay and deadline expiry are exact. A packet whose age exceeds its
budget is dropped, not merely delayed.

### 2.5 Incumbent and sensing (`spectrum/incumbent.py`)

Each of 6 subbands follows an independent two-state Markov chain with geometric
sojourn times. Mean burst length is held at 42 slots (21 ms) and the idle gap is
stretched so the steady-state duty cycle matches the congestion regime
(12 % → 46 %). Burst-to-burst autocorrelation exceeds 0.9, so occupancy is
genuinely predictable — this is the structure the learned filter exploits.

Sensing is energy detection with a normalised statistic that is N(0,1) under H₀
and N(μ,1) under H₁. The threshold is fixed by the required false-alarm rate and
μ by the required detection rate, so the configured operating point
(P_d = 0.85, P_fa = 0.12) is reproduced exactly — verified empirically in
`test_sensing_decision_is_imperfect_at_the_configured_operating_point`.

---

## 3. Information parity

This is the load-bearing assumption of the whole study.

Every allocator receives exactly one object, `spectrum.env.Observation`, a frozen
dataclass whose fields are:

* **Bearer state** the gNB already has: backlog (buffer status reports),
  head-of-line delay, deadline slack, delay budget, nominal rate, priority level,
  5QI class, EWMA throughput, RBGs held last slot.
* **Channel state** as actually reported: a CQI grid that is two slots old and
  quantised to the 15-entry table, plus the implied spectral efficiency and
  transport-block size.
* **Sensing**: the current hard decision, the current raw energy statistic, and a
  16-slot history of both.

The incumbent's true occupancy is **not** a field. It is consumed inside
`RadioChannel.true_sinr_db` and the collision accounting in `SpectrumEnv.step`,
and it is returned from `step()` under the key `_truth_incumbent` — i.e. strictly
*after* the decision was made — solely so evaluation and predictor training can
score against it.

Four tests enforce this (`tests/test_information_parity.py`):

| Test | What it rules out |
|---|---|
| `test_observation_exposes_only_allowed_fields` | silently adding a privileged field |
| `test_observation_has_no_ground_truth_occupancy` | any field tracking the incumbent perfectly |
| `test_observation_is_a_detached_snapshot` | reaching into env state through the observation |
| `test_allocator_decides_from_the_observation_alone` | reading anything *other* than the observation |

The last is a replay test: recorded observations are fed to a fresh allocator with
**no environment object in existence**, and it must reproduce its original
decisions exactly. It runs for every classical scheduler and for the learned agent.

---

## 4. Baselines

All five are PRB-granular, buffer-aware and sensing-driven, so none is a strawman.

| Scheduler | Metric | Reference |
|---|---|---|
| Round-Robin | cyclic, channel-blind | the "static allocation" of the problem statement |
| Proportional Fair | r_ik(t) / R̄_i(t) | Viswanath, Tse & Laroia (2002) |
| Max-C/I | argmax r_ik(t) | throughput-optimal, fairness-blind |
| M-LWDF | −log(δ_i)/τ_i · W_i(t) · r_ik(t)/R̄_i(t) | Andrews et al. (2001) |
| EXP/PF | exponential delay weighting · r_ik/R̄_i | Rhee, Holtzman & Kim (2003) |

Shared discipline, enforced by `tests/test_allocators.py`:

* **Buffer awareness** — a UE leaves the candidate set once its queue is covered,
  so no RBG is ever spent on an empty queue (`test_never_schedules_an_empty_queue`).
* **Incumbent protection** — an RBG whose subband the detector flags as busy is
  muted (`test_respects_the_sensing_decision`).
* **Identical visit order** — RBGs are assigned in the same slot-dependent
  scrambled order as the learned policy uses, so nobody benefits from always
  starting at RBG 0.

Sanity checks assert the baselines behave as the literature says they should:
Max-C/I beats Round-Robin on goodput, PF is fairer than Max-C/I, and M-LWDF misses
fewer deadlines than Max-C/I.

---

## 5. The proposed system

### 5.1 Incumbent-occupancy predictor (`spectrum/predictor.py`)

A GRU (32 hidden units, ~7k parameters) consuming the pair (hard decision,
normalised energy) per subband per slot, with weights shared across subbands and
hidden state carried across slots — i.e. an online filter. Trained by binary
cross-entropy against the occupancy revealed after the fact, with the first 16
steps of each sequence excluded so it cannot score by memorising the duty cycle.

Two references are scored alongside it:

* `EnergyDetector` — threshold the current statistic. What deployed equipment does.
* `BayesFilter` — the exact HMM forward algorithm, **given** the true transition
  matrix and detector calibration. Not deployable; it bounds what any estimator can
  extract from the sensing stream, so it shows how much of the achievable gain the
  learned filter captures.

### 5.2 Scheduling policy (`spectrum/policy.py`)

Per-UE features (18 of them) → shared MLP encoder → one multi-head self-attention
block → two heads. Attention makes the network **permutation-equivariant** over
UEs and independent of cell size, so a single set of weights serves 20-UE and
80-UE cells alike.

Assignment logits are factorised:

```
logit[k, u] = scale · (score_u + gain_u · z_uk)
```

where `z_uk` is the normalised log transport-block size for UE *u* on RBG *k*.
This is rank-2 in the (UE, RBG) pair space but still lets different UEs prefer
different parts of the band. The learned `scale` is necessary rather than
cosmetic: the quantity a good scheduler ranks on spans ten-plus nats, while heads
reading LayerNorm'd features emit roughly unit-scale values — without a
temperature the softmax can never become decisive. (Omitting it was measured: the
cloned policy's logits were nearly constant across RBGs and goodput collapsed from
228 to 29 Mb/s.)

Each RBG also has a **mute** action whose logit is driven by the predictor's
output; this is the only place the occupancy estimate enters the scheduling
decision.

Decoding is sequential: RBGs are visited in the shared scrambled order, a UE is
sampled from the masked softmax, and that UE's residual backlog is decremented.
A UE drops out once its queue is covered, so over-allocation is structurally
impossible — exactly as in the baselines' `greedy_assign`, so this is not an
advantage over them.

### 5.3 Training (`spectrum/bc.py`, `spectrum/ppo.py`)

**Stage 1** — supervised predictor on sensing traces only (no scheduler involved).

**Stage 2a, behaviour cloning.** The policy is fitted by cross-entropy to the
teacher's choices, replayed through the same residual bookkeeping so the student
faces the masks the teacher faced. This is initialisation only, and it matters
for the integrity of the result: PPO begins at parity with the strongest
non-RL scheduler rather than far below it, so the reported gain is unambiguously
what RL added. The pre-RL checkpoint is kept and reported as an ablation.

**The teacher is M-LWDF driven by the learned occupancy estimate, not plain
M-LWDF.** This correction was necessary: plain M-LWDF mutes exactly when
`sensing_decision == 1`, so the hard bit perfectly predicts its behaviour and a
student cloning it learns to ignore `p_busy` entirely — inheriting the classical
detector's collision rate despite having a far better estimate in its own input.
Measured: a clone of plain M-LWDF reproduced the teacher's 11.5 % peak collision
rate, while the predictor-augmented teacher sits at 5.2 %.

**Stage 2b, PPO.** Each slot is one environment step containing 18 sequential
sub-decisions. Masks are recorded during rollout so the update reconstructs the
exact categorical distributions in one vectorised pass.

Three details were necessary to make this converge, each fixing a failure that was
observed before it was fixed:

* **Clip per sub-action, not on the joint.** The joint log-probability is the sum
  of 18 categorical log-probabilities, so the PPO ratio is a *product* of 18
  per-decision ratios: a 0.05 shift in each inflates it to e^0.9 ≈ 2.5 and the
  clip zeroes the gradient. Measured: `clip_frac` sat at 0.5–0.64 while the true
  per-decision KL was 0.004, and reward did not improve over 6 iterations. Each
  sub-action is now clipped on its own ratio against the slot's shared advantage
  — the standard credit assignment for a factored policy. `clip_frac` fell to
  0.09–0.22 and entropy began decreasing instead of rising.
* **Average entropy over RBGs.** Summing it scales the exploration bonus by 18×
  and swamps the policy gradient.
* **KL early stop** (`target_kl = 0.015`, on the per-decision KL) protects the
  cloned initialisation from being destroyed by one over-long update.
* **A KL anchor to the frozen cloned policy.** Even with correct clipping, one
  scalar advantage per slot cannot reliably shape 18 decisions over ~80
  candidates each, and the policy drifted off its initialisation faster than it
  found improvements — goodput fell from 184 to 28 Mb/s over 12 iterations as
  the agent rediscovered "mute everything". Adding
  `bc_kl_coef · KL(π ‖ π_ref)` against a frozen copy of the cloned policy makes
  training stable. The coefficient is annealed from 1.0 to 0.1: strong early,
  when the critic is poor and the advantage is mostly noise, then relaxed so RL
  can actually depart from the teacher.

That PPO itself is correctly implemented was verified separately by repeatedly
optimising a single fixed batch and confirming the surrogate falls back toward
zero rather than diverging — the failure was signal quality, not a sign error.

Rollouts cycle through the medium, high and peak regimes so one policy must handle
all of them, and `rollout_slots` is a whole multiple of
`episode_slots × len(congestion_mix)` so every rollout contains exactly one
episode of each regime. Without that alignment the reported mean reward swings
with whichever regimes a rollout happened to sample and real progress is
invisible.

### 5.4 Reward

Seven terms, each normalised to roughly [−1, 1]:

| Term | Weight | Meaning |
|---|---|---|
| goodput | 1.5 | useful bits / peak bits |
| completed | 1.5 | packets finished this slot / mean arrival rate |
| dropped | 3.0 | − packets expired this slot / mean arrival rate |
| fairness | 0.4 | Jain index of EWMA throughput |
| collision | 2.0 | − RBGs transmitted into an active incumbent |
| waste | 0.5 | − bits delivered beyond what the queue needed |
| vacancy | 0.3 | − idle RBGs left unused |

`collision` and `vacancy` pull against each other through the mute action, which
is the central trade-off: mute too eagerly and spectrum is wasted, mute too
little and the incumbent is harmed.

**QoS is scored by packet outcomes, not by queue state.** An earlier version of
this reward used "fraction of UEs whose head-of-line delay is inside the budget"
as a QoS proxy, and PPO converged to a policy that muted every RBG — 18 Mb/s
goodput, 0 % collisions, 44 % of packets dropped. The proxy was gameable: expiry
*removes* a packet from the queue, so missing every deadline keeps head-of-line
delay permanently low and the proxy near 1.0. Counting completions and expiries
instead, both normalised by the cell's known mean arrival rate, makes the
degenerate policy strictly dominated — it drives `completed` to zero and
`dropped` to the full arrival rate.

The reward ordering was verified directly against reference policies before
retraining (medium / peak congestion, mean reward per slot):

| Policy | medium | peak |
|---|---|---|
| mute everything | −2.34 | −1.97 |
| fill every RBG, ignore sensing | +0.77 | −2.05 |
| Round-Robin | +1.92 | +0.12 |
| Proportional Fair | +2.07 | +0.82 |
| M-LWDF | +1.93 | +1.35 |

`_instantaneous_qos` is retained in the `info` dictionary for monitoring, with a
comment recording why it must not re-enter the reward.

---

## 6. Scoring

**Paired design.** Every allocator is run on an identical set of seeds; a seed
fixes the UE drop, the traffic realisation and the incumbent trajectory, so all
allocators face the same world.

**Uncertainty.** Each KPI is reported with a percentile bootstrap 95 % confidence
interval over seeds (10 000 resamples).

**Significance.** A paired Wilcoxon signed-rank test against the strongest
classical comparator *for that metric* — not a fixed comparator, which would let a
favourable choice inflate the result. Reported alongside Cliff's delta as a
non-parametric effect size. A rank test is used instead of a t-test because KPI
distributions across seeds are skewed and the sample is small.

**Sign correction.** Improvements are sign-corrected so positive always means
better, for latency and collisions as well as throughput.

**Ablations** (`evaluate.py --ablations`, the default) separate the contributions:

| Variant | Isolates |
|---|---|
| M-LWDF + learned sensing | the predictor's contribution to a classical scheduler |
| learned scheduler + hard sensing | the scheduler's contribution without the predictor |
| BC only, no RL | what reinforcement learning added over imitation |
| full system | both |

**Sensitivity** (`sensitivity.py`) re-evaluates the *frozen* policy in worlds it
was never trained in: detector operating points from poor to near-perfect, and
incumbent burst lengths from 2 ms to 150 ms at constant duty cycle. If the
advantage were an artifact of one operating point it would disappear here.

---

## 7. Limitations

* **Single cell.** Inter-cell interference is a static floor, not a second
  scheduler reacting to this one.
* **Single antenna port.** No MIMO, beamforming or spatial multiplexing.
* **One training seed.** Evaluation spans many seeds, but the policy itself was
  trained once; training-seed variance is not characterised.
* **Markov incumbent.** Real radar and satellite incumbents have structured duty
  cycles that would be *easier* to predict, so the predictor result is likely
  conservative. The Bayes-optimal reference is optimal only for this model.
* **Simulator-internal.** These results show the learned policy dominates classical
  schedulers under these dynamics. They are not a measurement of a deployed network,
  and no claim is made about absolute throughput in any real deployment.

#!/usr/bin/env python
"""
Train both learned components of the proposed system.

Stage 1  Incumbent predictor -- supervised, on sensing traces only.
Stage 2  Scheduling policy   -- PPO, in the simulator, with the stage-1
         predictor frozen and used as the sensing front-end.

Artifacts land in ``artifacts/`` and are consumed by ``evaluate.py`` and
``app.py``.

    python train.py                 # full run (~55 min on CPU)
    python train.py --quick         # ~4 min smoke run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from spectrum.bc import warm_start
from spectrum.config import SimConfig
from spectrum.incumbent import IncumbentProcess
from spectrum.policy import SchedulerPolicy
from spectrum.ppo import PPOConfig, PPOTrainer
from spectrum.predictor import (
    BayesFilter,
    EnergyDetector,
    GRUPredictor,
    classification_report,
    collect_sensing_traces,
    train_predictor,
)

ARTIFACTS = Path("artifacts")
ALL_LEVELS = ["low", "medium", "high", "peak"]


def stage1_predictor(args) -> GRUPredictor:
    print("\n" + "=" * 72)
    print("STAGE 1  Incumbent-occupancy predictor (supervised)")
    print("=" * 72)

    t0 = time.perf_counter()
    seeds = [1, 2, 3] if not args.quick else [1]
    slots = 4096 if not args.quick else 2048
    ds = collect_sensing_traces(ALL_LEVELS, seeds=seeds, slots=slots, seq_len=128)
    print(f"  sensing traces: {ds.features.shape[0]} sequences x {ds.features.shape[1]} slots "
          f"({ds.labels.mean():.1%} occupied)")

    model, _ = train_predictor(
        ds, epochs=4 if args.quick else 12, seed=args.seed, verbose=True
    )
    model.eval()
    torch.save(model.state_dict(), ARTIFACTS / "predictor.pt")
    print(f"  saved artifacts/predictor.pt   ({time.perf_counter() - t0:.0f}s)")

    report = evaluate_predictor(model, args)
    with open(ARTIFACTS / "predictor_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return model


def evaluate_predictor(model: GRUPredictor, args) -> dict:
    """Score the three estimators on held-out traces, in online filtering mode."""
    print("\n  Held-out occupancy-estimation accuracy (unseen seeds):")
    out: dict = {}
    eval_slots = 6000 if not args.quick else 2000
    burn_in = 200

    for level in ALL_LEVELS:
        cfg = SimConfig(congestion=level)
        rng = np.random.default_rng(900_000 + hash(level) % 1000)
        inc = IncumbentProcess(cfg, rng)

        detector = EnergyDetector(inc.threshold)
        bayes = BayesFilter(inc.p_on, inc.p_off, inc.mu_h1, inc.duty_cycle)
        bayes.reset(cfg.n_subbands)
        model.reset(cfg.n_subbands)

        truth, p_ed, p_bf, p_gru = [], [], [], []
        for _ in range(eval_slots):
            d, e = inc.sense()
            truth.append(inc.active.copy())
            p_ed.append(detector.probability(d, e))
            p_bf.append(bayes.probability(d, e))
            p_gru.append(model.probability(d, e))
            inc.advance()

        truth_a = np.array(truth)[burn_in:]
        level_out = {}
        for name, probs in (
            ("energy_detector", p_ed),
            ("learned_gru", p_gru),
            ("bayes_optimal", p_bf),
        ):
            level_out[name] = classification_report(truth_a, np.array(probs)[burn_in:])
        out[level] = level_out

        ed, gru = level_out["energy_detector"], level_out["learned_gru"]
        print(
            f"    {level:6s}  energy-det acc {ed['accuracy']:.3f} F1 {ed['f1']:.3f}"
            f"   |  learned acc {gru['accuracy']:.3f} F1 {gru['f1']:.3f}"
        )
    return out


def stage2_policy(predictor: GRUPredictor, args) -> SchedulerPolicy:
    print("\n" + "=" * 72)
    print("STAGE 2  Scheduling policy (PPO)")
    print("=" * 72)

    t0 = time.perf_counter()
    policy = SchedulerPolicy()
    print(f"  policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # Warm start from the strongest non-RL scheduler (M-LWDF driven by the
    # learned occupancy estimate) so PPO begins at parity with it rather than
    # far below. Everything reported as a gain is therefore what RL added on top,
    # and the pre-RL checkpoint is kept so the split can be shown.
    bc_path = ARTIFACTS / "policy_bc.pt"
    if args.skip_bc and bc_path.exists():
        policy.load_state_dict(torch.load(bc_path, map_location="cpu"))
        print(f"\n  Reusing existing {bc_path} (behaviour cloning skipped)")
    else:
        print("\n  Warm start: behaviour cloning the strongest non-RL scheduler")
        warm_start(
            policy,
            predictor,
            n_slots=1500 if args.quick else args.bc_slots,
            epochs=3 if args.quick else args.bc_epochs,
            seed=args.seed,
        )
        torch.save(policy.state_dict(), bc_path)
        print("  saved artifacts/policy_bc.pt (pre-RL ablation checkpoint)")

    print("\n  Reinforcement learning (PPO)")

    ppo_cfg = PPOConfig(
        rollout_slots=256 if args.quick else args.rollout,
        iterations=4 if args.quick else args.iterations,
        epochs=2 if args.quick else 4,
        minibatch=128 if args.quick else 256,
        episode_slots=250 if args.quick else 500,
        lr=args.lr,
        seed=args.seed,
    )
    trainer = PPOTrainer(policy, predictor, ppo_cfg)

    # Written every iteration so an interrupted run is never wasted.
    def checkpoint(iteration: int, stats: dict) -> None:
        torch.save(policy.state_dict(), ARTIFACTS / "policy_last.pt")
        with open(ARTIFACTS / "ppo_history.json", "w", encoding="utf-8") as fh:
            json.dump(trainer.history, fh, indent=2)

    history = trainer.train(verbose=True, callback=checkpoint)

    with open(ARTIFACTS / "ppo_history.json", "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    # policy.pt is the final iterate, full stop. No checkpoint is selected on
    # the training reward, and none is selected on a handful of extra episodes
    # either: both were tried and both turned out to be noise-mining. Selecting
    # the best 3-iteration smoothed reward shipped a checkpoint worse than the
    # final iterate on held-out seeds, and selecting between the clone and the
    # final iterate on 12 held-out episodes reversed its verdict on the 32-episode
    # benchmark. RL's actual contribution is small enough here that only a paired
    # test over the full benchmark can see it, so that is where it is measured:
    # evaluate.py carries the clone as an ablation arm and reports the comparison.
    torch.save(policy.state_dict(), ARTIFACTS / "policy.pt")
    torch.save(policy.state_dict(), ARTIFACTS / "policy_last.pt")
    print(
        f"  saved artifacts/policy.pt (final PPO iterate)   "
        f"({time.perf_counter() - t0:.0f}s)"
    )
    return policy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="fast smoke run")
    ap.add_argument("--iterations", type=int, default=120, help="PPO iterations")
    ap.add_argument("--rollout", type=int, default=1500,
                    help="slots per PPO rollout; keep it a whole multiple of "
                         "episode length x number of congestion regimes")
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="PPO learning rate; 3e-4 was tuned against a KL anchor "
                         "that never released and overshoots once it does")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bc-slots", type=int, default=12000, help="teacher demonstration slots")
    ap.add_argument("--bc-epochs", type=int, default=30)
    ap.add_argument("--skip-predictor", action="store_true",
                    help="reuse artifacts/predictor.pt instead of retraining it")
    ap.add_argument("--skip-bc", action="store_true",
                    help="reuse artifacts/policy_bc.pt and go straight to PPO")
    args = ap.parse_args()

    ARTIFACTS.mkdir(exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    if args.skip_predictor and (ARTIFACTS / "predictor.pt").exists():
        predictor = GRUPredictor()
        predictor.load_state_dict(torch.load(ARTIFACTS / "predictor.pt"))
        predictor.eval()
        print("Loaded existing artifacts/predictor.pt")
    else:
        predictor = stage1_predictor(args)

    for p in predictor.parameters():
        p.requires_grad_(False)

    stage2_policy(predictor, args)

    print("\nDone. Run `python evaluate.py` to produce the comparison tables.")


if __name__ == "__main__":
    main()

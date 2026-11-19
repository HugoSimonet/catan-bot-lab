"""Evaluate a trained policy against strategy bots.

Every env plays the same number of games, and the lineup rotates one seat
per game, so the learner plays each seat equally often. Seeds are a fixed
function of (seed, env index, game number), so two checkpoints evaluated with
the same settings face exactly the same boards and dice.

    python -m catanbot.evaluate --checkpoint runs/main/checkpoints/latest.pt \\
        --opponents balanced,ows,road_builder --games 100000
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from catanbot.env import CatanVecEnv, policy_code
from catanbot.model import load_policy


def wilson(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class EvalResult:
    opponents: list[str]
    games: int
    wins: int
    draws: int
    win_rate: float
    ci_low: float
    ci_high: float
    avg_vp: float
    avg_turns_in_wins: float
    #: Win rate of each opponent bot (per seat it occupied).
    opponent_win_rate: dict[str, float] = field(default_factory=dict)
    seat_win_rate: list[float] = field(default_factory=list)
    seconds: float = 0.0

    def line(self) -> str:
        return (f"{self.games} games vs {'/'.join(self.opponents)}: win {100 * self.win_rate:.2f}% "
                f"[{100 * self.ci_low:.2f}-{100 * self.ci_high:.2f}], avg VP {self.avg_vp:.2f}")


def evaluate(net: torch.nn.Module, opponents: list[str], games: int = 1000, num_envs: int = 256,
             seed: int = 10_000_000, device: str | torch.device = "cuda",
             deterministic: bool = True) -> EvalResult:
    """Learner (policy 0) vs 3 bots. ``opponents`` is the 3 bot names (repeats allowed)."""
    assert len(opponents) == 3, "need exactly 3 opponents"
    specs = sorted(set(opponents))
    index = {n: i for i, n in enumerate(specs)}
    base = [policy_code(0)] + [index[o] for o in opponents]
    num_envs = max(1, min(num_envs, games))
    per_env = math.ceil(games / num_envs)
    # Start each env at a different rotation so seats are balanced even for small games/env.
    lineups = [base[-(i % 4):] + base[:-(i % 4)] if i % 4 else list(base) for i in range(num_envs)]
    env = CatanVecEnv(specs, lineups, seed=seed)
    was_training = net.training
    net.eval()
    t0 = time.time()

    done_count = np.zeros(num_envs, dtype=np.int64)
    wins = draws = total = 0
    vp_sum = 0
    win_turns = 0
    seat_games = np.zeros(4)
    seat_wins = np.zeros(4)
    opp_games: dict[str, int] = defaultdict(int)
    opp_wins: dict[str, int] = defaultdict(int)

    step = env.reset()
    while (done_count < per_env).any():
        obs = torch.from_numpy(step.obs).to(device)
        mask = torch.from_numpy(step.mask).to(device)
        actions, _, _ = net.act(obs, mask, deterministic=deterministic)
        step = env.step(actions.cpu().numpy())
        for e in np.nonzero(step.done)[0]:
            if done_count[e] >= per_env:
                continue
            done_count[e] += 1
            lineup = step.lineup[e]
            seat = int(np.nonzero(lineup == policy_code(0))[0][0])
            w = int(step.winner[e])
            total += 1
            seat_games[seat] += 1
            vp_sum += int(step.final_vp[e, seat])
            if w == -1:
                draws += 1
            elif w == seat:
                wins += 1
                seat_wins[seat] += 1
                win_turns += int(step.turns[e])
            for s in range(4):
                if s != seat:
                    name = specs[int(lineup[s])]
                    opp_games[name] += 1
                    opp_wins[name] += int(w == s)

    if was_training:
        net.train()
    lo, hi = wilson(wins, total)
    return EvalResult(
        opponents=list(opponents),
        games=total,
        wins=wins,
        draws=draws,
        win_rate=wins / max(total, 1),
        ci_low=lo,
        ci_high=hi,
        avg_vp=vp_sum / max(total, 1),
        avg_turns_in_wins=win_turns / max(wins, 1),
        opponent_win_rate={k: opp_wins[k] / opp_games[k] for k in opp_games},
        seat_win_rate=(seat_wins / np.maximum(seat_games, 1)).tolist(),
        seconds=time.time() - t0,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--opponents", default="balanced,ows,road_builder",
                    help="3 comma-separated bots, or several tables separated by ';'")
    ap.add_argument("--games", type=int, default=10_000)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--seed", type=int, default=10_000_000)
    ap.add_argument("--stochastic", action="store_true", help="sample actions instead of argmax")
    ap.add_argument("--out", help="write results JSON here")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_policy(args.checkpoint, device)
    results = []
    for table in args.opponents.split(";"):
        opps = [o.strip() for o in table.split(",")]
        r = evaluate(net, opps, games=args.games, num_envs=args.envs, seed=args.seed,
                     device=device, deterministic=not args.stochastic)
        print(r.line(), f"({r.seconds:.0f}s)")
        print("   opponent win rates:", {k: round(v, 4) for k, v in r.opponent_win_rate.items()})
        print("   learner win rate by seat:", [round(x, 4) for x in r.seat_win_rate])
        results.append(asdict(r))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

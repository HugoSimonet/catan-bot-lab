"""Parameter sweeps for the strategy bots.

"How much does a slight change to a strategy change its win rate?"

    # One parameter, several values, against a field of other strategies:
    python -m catanbot.sweep one --base ows --param priority_city --values 1.0,1.3,1.6,1.9,2.2

    # Tune every preset (coordinate ascent), keeping each strategy's identity:
    python -m catanbot.sweep tune --rounds 2 --seeds 4000 --out results/tuning.json

Parameters are addressed by name; list entries use a dot: ``resource_weights.4``
is the ore weight.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import polars as pl

from catanbot.tournament import STRATEGIES, field_win_rate, preset_spec, wilson

#: Parameters each preset may tune, with (min, max) bounds. Everything not
#: listed - notably the resource weights - is the strategy's identity and stays fixed.
COMMON = {
    "diversity_weight": (0.0, 3.0),
    "port_generic_weight": (0.0, 3.0),
    "port_specific_weight": (0.0, 1.0),
    "priority_city": (0.3, 3.0),
    "priority_settlement": (0.3, 3.0),
    "priority_road": (0.1, 2.0),
    "priority_dev": (0.1, 3.0),
    "longest_road_weight": (0.0, 2.0),
    "largest_army_weight": (0.0, 2.5),
    "knight_aggression": (0.0, 1.5),
    "robber_leader_weight": (0.0, 3.0),
    "trade_aggression": (0.0, 1.0),
    "discard_avoidance": (0.0, 1.0),
}
TUNABLE = {
    "balanced": COMMON,
    "ows": {**COMMON, "priority_city": (1.4, 3.0), "longest_road_weight": (0.0, 0.3)},
    "road_builder": {**COMMON, "longest_road_weight": (0.9, 2.0), "priority_settlement": (1.2, 3.0),
                     "priority_road": (0.8, 2.0)},
    "dev_rush": {**COMMON, "priority_dev": (1.5, 3.0), "largest_army_weight": (1.0, 2.5)},
    "port_master": {k: v for k, v in COMMON.items()
                    if k not in ("port_generic_weight", "port_specific_weight", "trade_aggression")},
}
MULTIPLIERS = [0.6, 0.8, 1.25, 1.5]


def get_param(spec: dict, name: str) -> float:
    if "." in name:
        key, i = name.split(".")
        return spec[key][int(i)]
    return spec[name]


def with_param(spec: dict, name: str, value: float, rename: str | None = None) -> dict:
    s = copy.deepcopy(spec)
    if "." in name:
        key, i = name.split(".")
        s[key][int(i)] = value
    else:
        s[name] = value
    if rename:
        s["name"] = rename
    return s


def evaluate_spec(spec: dict, field: list, seeds: int, base_seed: int) -> dict:
    wins, games = field_win_rate(spec, field, seeds, base_seed)
    lo, hi = wilson(wins, games)
    return {"wins": wins, "games": games, "win_rate": wins / games, "ci_low": lo, "ci_high": hi}


def sweep_one(base: str | dict, param: str, values: list[float], field: list, seeds: int = 4000,
              base_seed: int = 1) -> pl.DataFrame:
    spec = preset_spec(base) if isinstance(base, str) else base
    rows = []
    for v in values:
        variant = with_param(spec, param, v, rename=f"{spec['name']}[{param}={v:g}]")
        r = evaluate_spec(variant, field, seeds, base_seed)
        rows.append({"param": param, "value": v, **r})
        print(f"  {param}={v:<8g} win {100 * r['win_rate']:.2f}% [{100 * r['ci_low']:.2f}-{100 * r['ci_high']:.2f}]")
    return pl.DataFrame(rows)


def tune(name: str, field: list, seeds: int, rounds: int = 2, min_gain: float = 0.004,
         base_seed: int = 1, log: list | None = None) -> dict:
    """Coordinate ascent on one preset's tunable parameters against ``field``."""
    spec = preset_spec(name)
    best = evaluate_spec(spec, field, seeds, base_seed)
    print(f"[{name}] start: {100 * best['win_rate']:.2f}%")
    start_rate = best["win_rate"]
    for rnd in range(rounds):
        improved = False
        for param, (lo, hi) in TUNABLE[name].items():
            cur = get_param(spec, param)
            candidates = sorted({round(min(hi, max(lo, cur * m if cur else m * 0.5)), 3) for m in MULTIPLIERS} - {cur})
            for v in candidates:
                variant = with_param(spec, param, v)
                r = evaluate_spec(variant, field, seeds, base_seed)
                if log is not None:
                    log.append({"preset": name, "round": rnd, "param": param, "from": cur, "value": v,
                                **r, "baseline": best["win_rate"]})
                if r["win_rate"] > best["win_rate"] + min_gain:
                    print(f"[{name}] {param}: {cur} -> {v}  {100 * best['win_rate']:.2f}% -> {100 * r['win_rate']:.2f}%")
                    spec, best, cur, improved = variant, r, v, True
        if not improved:
            break
    print(f"[{name}] done: {100 * start_rate:.2f}% -> {100 * best['win_rate']:.2f}%")
    return {"name": name, "spec": spec, "start_win_rate": start_rate, **best}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    one = sub.add_parser("one")
    one.add_argument("--base", required=True)
    one.add_argument("--param", required=True)
    one.add_argument("--values", required=True)
    one.add_argument("--field", default=None, help="comma-separated opponents (default: other strategies)")
    one.add_argument("--seeds", type=int, default=4000)
    one.add_argument("--out")
    t = sub.add_parser("tune")
    t.add_argument("--presets", default=",".join(STRATEGIES))
    t.add_argument("--rounds", type=int, default=2)
    t.add_argument("--seeds", type=int, default=4000)
    t.add_argument("--out", default="results/tuning.json")
    args = ap.parse_args()

    if args.cmd == "one":
        field = args.field.split(",") if args.field else [s for s in STRATEGIES if s != args.base]
        df = sweep_one(args.base, args.param, [float(v) for v in args.values.split(",")], field, args.seeds)
        if args.out:
            df.write_csv(args.out)
        return

    t0 = time.time()
    results, log = [], []
    for name in args.presets.split(","):
        field = [s for s in STRATEGIES if s != name]
        results.append(tune(name, field, args.seeds, args.rounds, log=log))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"results": results, "log": log, "seeds": args.seeds, "seconds": time.time() - t0}, f, indent=2)
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

"""Analysis pipeline: produces everything the notebooks and the website show.

    python -m catanbot.pipeline bench
    python -m catanbot.pipeline tournament --seeds 250000
    python -m catanbot.pipeline sweeps
    python -m catanbot.pipeline evaluate --checkpoint runs/main/checkpoints/latest.pt --games 40000
    python -m catanbot.pipeline style --checkpoint runs/main/checkpoints/latest.pt
    python -m catanbot.pipeline replays [--checkpoint ...]
    python -m catanbot.pipeline improvements --before runs/main/checkpoints/latest.pt \\
        --checkpoint runs/phase2/checkpoints/latest.pt --games 40000
    python -m catanbot.pipeline lookahead --checkpoint runs/main/checkpoints/latest.pt --games 8192
    python -m catanbot.pipeline export

Add ``--lookahead`` (optionally with JSON overrides, e.g. ``--lookahead '{"sims": 32}'``)
to evaluate/style/replays/agent to play the agent with decision-time look-ahead.

Outputs go to ``results/`` (small JSON summaries; raw per-game data in
``results/raw/``, gitignored) and to ``web/`` (``web/data/summary.json`` and
``web/public/data/replays/``).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import torch
import yaml

from catanbot import play_replay
from catanbot.env import CatanVecEnv, policy_code
from catanbot.evaluate import evaluate, wilson
from catanbot.lookahead import Lookahead, LookaheadConfig
from catanbot.model import load_policy
from catanbot.sweep import get_param, sweep_one
from catanbot.tournament import (
    STRATEGIES,
    openskill_ratings,
    pool,
    preset_spec,
    resource_mix,
    rotated,
    seat_advantage,
    summarize,
)

RESULTS = Path("results")
RAW = RESULTS / "raw"
WEB_DATA = Path("web/data")
WEB_REPLAYS = Path("web/public/data/replays")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NAMES = {"balanced": "Balanced", "ows": "Ore-Wheat-Sheep", "road_builder": "Road Builder",
         "dev_rush": "Dev-Card Rush", "port_master": "Port Master", "greedy": "Greedy",
         "random": "Random", "learned": "Learned agent"}


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    print(f"wrote {path}")


def _load(path: Path, default=None):
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------------------
def bench() -> dict:
    t = time.time()
    rotated(["random"] * 4, 25_000)
    random_gps = 100_000 / (time.time() - t)
    t = time.time()
    rotated(["balanced", "ows", "road_builder", "dev_rush"], 10_000)
    bot_gps = 40_000 / (time.time() - t)
    out = {"random_games_per_sec": random_gps, "bot_games_per_sec": bot_gps, "threads": os.cpu_count(),
           "games": 140_000}
    print(f"random: {random_gps:,.0f} games/s   strategy bots: {bot_gps:,.0f} games/s")
    _dump(RESULTS / "bench.json", out)
    return out


def tournament(seeds: int, base_seed: int = 2024) -> dict:
    """Random 4-of-5 tables of the strategy bots, every table in all 4 rotations."""
    t = time.time()
    df = pool(STRATEGIES, seeds, base_seed=base_seed, distinct=True)
    secs = time.time() - t
    games = df["game_id"].n_unique()
    print(f"{games:,} games in {secs:.0f}s")
    RAW.mkdir(parents=True, exist_ok=True)
    df.write_parquet(RAW / "tournament.parquet")
    summ = summarize(df)
    print(summ.select(["bot", "win_rate", "ci_low", "ci_high", "avg_vp", "cities", "dev_bought"]))
    ratings = openskill_ratings(df, max_games=40_000)
    out = {
        "games": games,
        "seconds": secs,
        "bots": summ.to_dicts(),
        "seat_win_rate": seat_advantage(df)["win_rate"].to_list(),
        "ratings": ratings.to_dicts(),
        "resource_mix": resource_mix(df).sort("bot").to_dicts(),
    }
    _dump(RESULTS / "tournament.json", out)
    return out


SWEEPS = [
    ("ows", "priority_city", "Ore-Wheat-Sheep: how hard to rush cities"),
    ("ows", "resource_weights.4", "Ore-Wheat-Sheep: how much to value ore"),
    ("road_builder", "longest_road_weight", "Road Builder: chasing Longest Road"),
    ("road_builder", "priority_settlement", "Road Builder: settlement priority"),
    ("dev_rush", "priority_dev", "Dev-Card Rush: how many dev cards to buy"),
    ("dev_rush", "knight_aggression", "Dev-Card Rush: when to play knights"),
    ("port_master", "port_specific_weight", "Port Master: value of 2:1 harbours"),
    ("port_master", "specialization_weight", "Port Master: concentrating on one resource"),
    ("balanced", "diversity_weight", "Balanced: resource diversity bonus"),
    ("balanced", "robber_leader_weight", "Balanced: aiming the robber at the leader"),
    ("balanced", "trade_aggression", "Balanced: bank-trade aggression"),
    ("balanced", "discard_avoidance", "Balanced: avoiding 7-card discards"),
]
UNIT_PARAMS = {"trade_aggression": [0.0, 0.34, 0.67, 1.0],
               "discard_avoidance": [0.0, 0.25, 0.5, 0.75, 1.0]}


def sweeps(seeds: int = 4000) -> list[dict]:
    out = []
    for preset, param, label in SWEEPS:
        spec = preset_spec(preset)
        default = float(get_param(spec, param))
        if param in UNIT_PARAMS:
            values = UNIT_PARAMS[param]
        elif default == 0:
            values = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
        else:
            values = sorted({round(default * m, 3) for m in (0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2)})
        print(f"{label} (default {default})")
        field = [s for s in STRATEGIES if s != preset]
        df = sweep_one(spec, param, values, field, seeds)
        out.append({"preset": preset, "param": param, "label": label, "default": default,
                    "games": int(df["games"].sum()),
                    "points": df.select(["value", "win_rate", "ci_low", "ci_high"]).to_dicts()})
    _dump(RESULTS / "sweeps.json", out)
    return out


def _top3() -> list[str]:
    t = _load(RESULTS / "tournament.json")
    if not t:
        return ["balanced", "ows", "road_builder"]
    return [b["bot"] for b in t["bots"]][:3]


#: Seeds of the evaluation games. The look-ahead settings were chosen on games from
#: EVAL_SEED (`pipeline lookahead`); the improvement results use FRESH_SEED, games
#: those settings were never tuned on.
EVAL_SEED = 10_000_000
FRESH_SEED = 50_000_000


def _baseline(bot: str, table: list[str], games: int, seed: int = EVAL_SEED) -> dict:
    """``bot`` in the agent's seat against ``table``: the same opponents and seed count."""
    df = rotated([bot] + table, games // 4, base_seed=seed)
    # The baseline is lineup position 0. Game k is played in rotation k % 4,
    # which puts lineup position 0 in seat (4 - rotation) % 4. (Matching by
    # name would be ambiguous: the baseline bot can also sit at the table.)
    rot = df["game_id"] % 4
    base_rows = df.filter(df["seat"] == (4 - rot) % 4)
    bw, bn = int(base_rows["won"].sum()), base_rows.height
    lo, hi = wilson(bw, bn)
    return {"bot": bot, "win_rate": bw / bn, "ci_low": lo, "ci_high": hi, "games": bn}


def improvements(before: str, after: str | None, games: int, lookahead: LookaheadConfig,
                 seed: int = FRESH_SEED) -> dict:
    """Each improvement step on the same top-3 table and the same seeds.

    Steps already in results/improvements.json with the same checkpoint, games,
    seed and look-ahead settings are reused, and the phase-2 steps are skipped
    while ``after`` is None, so the phase-1 steps can be computed early.
    """
    top = _top3()
    plan = [("phase1", "First agent (8,000 updates)", before, None),
            ("phase1_lookahead", "First agent + look-ahead", before, lookahead),
            ("phase2", "After phase-2 training", after, None),
            ("phase2_lookahead", "Phase 2 + look-ahead", after, lookahead)]
    prev = {st["key"]: st for st in (_load(RESULTS / "improvements.json") or {}).get("steps", [])}
    steps = []
    for key, label, ckpt, la in plan:
        if ckpt is None:
            continue
        settings = {"checkpoint": ckpt, "requested_games": games, "seed": seed,
                    "config": asdict(la) if la else None}
        old = prev.get(key)
        if old and all(old.get(k) == v for k, v in settings.items()):
            print(f"{label:28s} reused: {100 * old['win_rate']:.2f}%")
            steps.append({**old, "label": label})
            continue
        net = load_policy(ckpt, DEVICE)
        r = evaluate(net, top, games=games, num_envs=512, device=DEVICE, lookahead=la, seed=seed)
        print(f"{label:28s} {r.line()} ({r.seconds:.0f}s)")
        steps.append({"key": key, "label": label, **settings, "games": r.games,
                      "win_rate": r.win_rate, "ci_low": r.ci_low, "ci_high": r.ci_high, "avg_vp": r.avg_vp,
                      "opponent_win_rate": r.opponent_win_rate, "seat_win_rate": r.seat_win_rate,
                      "seconds": r.seconds, "lookahead": r.lookahead})
    out = {"table": top, "steps": steps, "baseline": _baseline(top[0], top, games, seed), "seed": seed}
    _dump(RESULTS / "improvements.json", out)
    return out


#: Look-ahead variants compared by `pipeline lookahead` (label, LookaheadConfig overrides; None = no look-ahead).
LOOKAHEAD_VARIANTS: list[tuple[str, str, dict | None]] = [
    ("none", "Policy alone", None),
    ("decision", "Score at the next decision",
     {"horizon": "decision", "k": 4, "sims": 16, "threshold": 0.9, "min_prior": 0.02}),
    ("decision_48", "Next decision, 48 futures per move",
     {"horizon": "decision", "k": 4, "sims": 48, "threshold": 0.9, "min_prior": 0.02}),
    ("decision_margin", "Next decision, clear gains only",
     {"horizon": "decision", "k": 4, "sims": 16, "threshold": 0.9, "min_prior": 0.02, "margin": 0.02}),
    ("turn", "Score at the next turn (unsure moves only)",
     {"horizon": "turn", "k": 4, "sims": 16, "threshold": 0.9, "min_prior": 0.02}),
    ("turn_all", "Next turn, every plausible decision",
     {"horizon": "turn", "k": 4, "sims": 16, "min_prior": 0.02}),
    ("final", "Next turn, wider shortlist (final)", {}),
]


def lookahead_sweep(checkpoint: str, games: int) -> dict:
    """Compare look-ahead variants for one checkpoint on the top-3 table (same seeds for all)."""
    top = _top3()
    net = load_policy(checkpoint, DEVICE)
    rows = []
    for key, label, overrides in LOOKAHEAD_VARIANTS:
        la = LookaheadConfig(**overrides) if overrides is not None else None
        r = evaluate(net, top, games=games, num_envs=512, device=DEVICE, lookahead=la)
        print(f"{label:40s} {r.line()} ({r.seconds:.0f}s)")
        rows.append({"key": key, "label": label, "games": r.games, "win_rate": r.win_rate, "ci_low": r.ci_low,
                     "ci_high": r.ci_high, "seconds": r.seconds, "lookahead": r.lookahead})
    out = {"checkpoint": checkpoint, "table": top, "variants": rows}
    _dump(RESULTS / "lookahead.json", out)
    return out


def evaluate_checkpoint(checkpoint: str, games: int, lookahead: LookaheadConfig | None = None,
                        seed: int = EVAL_SEED) -> list[dict]:
    net = load_policy(checkpoint, DEVICE)
    top = _top3()
    rows = []
    tables = [top] + [list(c) for c in itertools.combinations(STRATEGIES, 3) if list(c) != top]
    # The improvements step may already have played the top-3 table with these exact settings.
    settings = {"checkpoint": checkpoint, "requested_games": games, "seed": seed,
                "config": asdict(lookahead) if lookahead else None}
    done = next((st for st in (_load(RESULTS / "improvements.json") or {}).get("steps", [])
                 if all(st.get(k) == v for k, v in settings.items()) and "seat_win_rate" in st), None)
    for i, table in enumerate(tables):
        n = games if i == 0 else max(500, games // 10)
        base = _baseline(top[0], table, n, seed)
        if i == 0 and done:
            r = {k: done[k] for k in ("games", "win_rate", "ci_low", "ci_high", "avg_vp", "opponent_win_rate",
                                      "seat_win_rate", "lookahead")}
            print(f"{table}: reused from improvements.json: {100 * r['win_rate']:.2f}%")
        else:
            res = evaluate(net, table, games=n, num_envs=512, device=DEVICE, lookahead=lookahead, seed=seed)
            print(f"{res.line()}   | baseline {base['bot']}: {100 * base['win_rate']:.2f}%")
            r = {k: getattr(res, k) for k in ("games", "win_rate", "ci_low", "ci_high", "avg_vp",
                                                "opponent_win_rate", "seat_win_rate", "lookahead")}
        rows.append({"table": table, **r, "baseline": base, "checkpoint": checkpoint, "seed": seed})
    _dump(RESULTS / "evaluation.json", rows)
    return rows


def _agent_games(checkpoint: str, opponents: list[str], games: int, record: int = 0,
                 seed: int = 20_000_000, lookahead: LookaheadConfig | None = None,
                 ) -> tuple[pl.DataFrame, list[dict]]:
    """Play the agent vs ``opponents``; return per-seat stats and (optionally) replays."""
    net = load_policy(checkpoint, DEVICE)
    la = Lookahead(net, lookahead, DEVICE, seed=seed) if lookahead else None
    specs = sorted(set(opponents))
    idx = {n: i for i, n in enumerate(specs)}
    base = [policy_code(0)] + [idx[o] for o in opponents]
    n_env = min(256, games)
    lineups = [base[-(i % 4):] + base[:-(i % 4)] if i % 4 else list(base) for i in range(n_env)]
    env = CatanVecEnv(specs, lineups, seed=seed, record=record > 0)
    all_envs = np.arange(n_env)
    rows: list[dict] = []
    replays: list[dict] = []
    step = env.reset()
    # A fixed quota of games per env: stopping after the first N games to
    # *finish* would over-represent short games.
    quota = -(-games // n_env)
    per_env = np.zeros(n_env, dtype=np.int64)
    done = 0
    while (per_env < quota).any():
        obs = torch.from_numpy(step.obs).to(DEVICE)
        mask = torch.from_numpy(step.mask).to(DEVICE)
        if la is None:
            a, _, _ = net.act(obs, mask, deterministic=True)
            a = a.cpu().numpy()
        else:
            with torch.no_grad():
                logits, _ = net(obs, mask)
            a = la.choose(env._env, all_envs, logits, step.obs)
        step = env.step(a)
        for e in np.nonzero(step.done)[0]:
            if per_env[e] >= quota:
                continue
            per_env[e] += 1
            res = env.last_result(int(e))
            names = ["learned" if c < 0 else specs[c] for c in res["bot"]]
            for s in range(4):
                rows.append({k: v[s] for k, v in res.items()} | {"bot": names[s], "game_id": done})
            if record and len(replays) < record:
                replays.append({"names": names, "winner": int(step.winner[e]), "turns": int(step.turns[e]),
                                "json": env.last_replay(int(e), names)})
            done += 1
    return pl.DataFrame(rows), replays


STYLE_METRICS = [
    ("win_rate", "Win rate", "pct"),
    ("opening_ore_wheat", "Opening ore+wheat", "pct"),
    ("opening_wood_brick", "Opening wood+brick", "pct"),
    ("opening_port", "Opens on a harbour", "pct"),
    ("settlements", "Settlements", "num"),
    ("cities", "Cities", "num"),
    ("roads", "Roads", "num"),
    ("dev_bought", "Dev cards", "num"),
    ("trades", "Bank trades", "num"),
    ("longest_road", "Holds Longest Road", "pct"),
    ("largest_army", "Holds Largest Army", "pct"),
    ("first_city_turn", "First city (turn)", "num"),
]


def _style_rows(df: pl.DataFrame) -> pl.DataFrame:
    total = sum(pl.col(f"opening_{r}") for r in ["wood", "brick", "sheep", "wheat", "ore"])
    return (
        df.with_columns(
            ((pl.col("opening_ore") + pl.col("opening_wheat")) / total).alias("opening_ore_wheat"),
            ((pl.col("opening_wood") + pl.col("opening_brick")) / total).alias("opening_wood_brick"),
            pl.when(pl.col("first_city_turn") > 0).then(pl.col("first_city_turn")).otherwise(None).alias("fct"),
        )
        .group_by("bot")
        .agg(
            pl.col("won").mean().alias("win_rate"),
            pl.col("opening_ore_wheat").mean(),
            pl.col("opening_wood_brick").mean(),
            pl.col("opening_port").mean(),
            pl.col("settlements").mean(),
            pl.col("cities").mean(),
            pl.col("roads").mean(),
            pl.col("dev_bought").mean(),
            pl.col("trades").mean(),
            pl.col("longest_road").mean(),
            pl.col("largest_army").mean(),
            pl.col("fct").mean().alias("first_city_turn"),
        )
    )


def style(checkpoint: str, games: int, lookahead: LookaheadConfig | None = None) -> dict:
    top = _top3()
    agent, _ = _agent_games(checkpoint, top, games, lookahead=lookahead)
    RAW.mkdir(parents=True, exist_ok=True)
    agent.write_parquet(RAW / "agent_games.parquet")
    return style_summary(agent, checkpoint, games, lookahead)


def style_summary(agent: pl.DataFrame, checkpoint: str, games: int,
                  lookahead: LookaheadConfig | None = None) -> dict:
    """results/style.json from the agent's games (``results/raw/agent_games.parquet``)."""
    agent_style = _style_rows(agent.filter(pl.col("bot") == "learned"))
    bots_df = pl.read_parquet(RAW / "tournament.parquet")
    bot_style = _style_rows(bots_df)
    table = pl.concat([agent_style, bot_style.sort("win_rate", descending=True)], how="vertical_relaxed")
    print(table)
    out = {"metrics": [{"key": k, "label": l, "format": f} for k, l, f in STYLE_METRICS],
           "rows": table.to_dicts(), "games": agent["game_id"].n_unique(), "opponents": _top3(),
           "checkpoint": checkpoint, "lookahead": asdict(lookahead) if lookahead else None,
           "note": "Learned agent measured against the top-3 table; strategies measured in the main tournament."}
    _dump(RESULTS / "style.json", out)
    return out


def replays(checkpoint: str | None, lookahead: LookaheadConfig | None = None) -> list[dict]:
    WEB_REPLAYS.mkdir(parents=True, exist_ok=True)
    for f in WEB_REPLAYS.glob("*.json"):
        f.unlink()
    index: list[dict] = []

    if checkpoint:
        top = _top3()
        _, recs = _agent_games(checkpoint, top, games=96, record=96, seed=30_000_000, lookahead=lookahead)
        wins = [r for r in recs if r["winner"] >= 0 and r["names"][r["winner"]] == "learned"]
        losses = [r for r in recs if r["winner"] >= 0 and r["names"][r["winner"]] != "learned"]
        # Prefer typical games: closest to the median length.
        for group in (wins, losses):
            if group:
                med = float(np.median([r["turns"] for r in group]))
                group.sort(key=lambda r: abs(r["turns"] - med))
        picks = [(r, "win") for r in wins[:3]] + [(r, "loss") for r in losses[:1]]
        for k, (r, kind) in enumerate(picks):
            rid = f"ai-{kind}-{k + 1}"
            (WEB_REPLAYS / f"{rid}.json").write_text(r["json"])
            winner = r["winner"]
            title = ("Learned agent wins" if kind == "win" else
                     f"{NAMES[r['names'][winner]]} beats the learned agent")
            index.append({"id": rid, "title": f"{title} ({r['turns']} turns)",
                          "description": "Reinforcement-learning agent vs the three strongest strategy bots.",
                          "players": r["names"], "winner": winner, "turns": r["turns"],
                          "featured": k == 0 and kind == "win"})

    df = pl.read_parquet(RAW / "tournament.parquet")
    for strat in STRATEGIES:
        wins = df.filter((pl.col("bot") == strat) & (pl.col("won") == 1))
        med = wins["turns"].median()
        g = wins.with_columns((pl.col("turns") - med).abs().alias("d")).sort(["d", "game_id"]).row(0, named=True)
        game = df.filter(pl.col("game_id") == g["game_id"]).sort("seat")
        names = game["bot"].to_list()
        js = play_replay(names, int(g["seed"]))
        rid = f"{strat}-win"
        (WEB_REPLAYS / f"{rid}.json").write_text(js)
        winner = names.index(strat)
        index.append({"id": rid, "title": f"A typical {NAMES[strat]} win",
                      "description": "Strategy bots only.", "players": names, "winner": winner,
                      "turns": int(g["turns"]), "featured": False})
    if not any(r["featured"] for r in index):
        next(r for r in index if r["id"] == f"{STRATEGIES[0]}-win")["featured"] = True
    _dump(WEB_REPLAYS / "index.json", index)
    return index


def _slim_metrics(text: str) -> str:
    """Training log for the repo: last record per update, no per-snapshot beat rates, 5 significant digits."""
    by_update = {}
    for line in text.splitlines():
        if line.strip():
            m = json.loads(line)
            by_update[m["update"]] = m
    out = []
    for k in sorted(by_update):
        rec = by_update[k]
        snaps = [v for key, v in rec.items() if key.startswith("beat/snap_")]
        m = {key: (float(f"{v:.5g}") if isinstance(v, float) else v)
             for key, v in rec.items() if not key.startswith("beat/snap")}
        if snaps:
            m["beat/snapshots_mean"] = float(f"{sum(snaps) / len(snaps):.5g}")
        out.append(json.dumps(m, separators=(",", ":")))
    return "\n".join(out) + "\n"


def export() -> dict:
    bench_d = _load(RESULTS / "bench.json", {"random_games_per_sec": 0, "bot_games_per_sec": 0, "threads": 0, "games": 0})
    tour = _load(RESULTS / "tournament.json")
    if tour is None:
        raise SystemExit("run `pipeline tournament` first")
    tuning = _load(RESULTS / "tuning.json")
    sw = _load(RESULTS / "sweeps.json", [])
    ev = _load(RESULTS / "evaluation.json")
    st = _load(RESULTS / "style.json")

    # Game counts below are all per game (not per seat).
    games = tour["games"] + bench_d.get("games", 0) + sum(s.get("games", 0) for s in sw)
    summary: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine": {k: bench_d[k] for k in ("random_games_per_sec", "bot_games_per_sec", "threads")},
        "tournament": {k: tour[k] for k in ("games", "bots", "seat_win_rate", "ratings", "resource_mix")},
    }

    if tuning:
        games += sum(e["games"] for e in tuning["log"])
        first_from: dict[tuple, float] = {}
        for e in tuning["log"]:
            first_from.setdefault((e["preset"], e["param"]), e["from"])
        rows = []
        for r in tuning["results"]:
            changes = []
            for (preset, param), orig in first_from.items():
                if preset != r["name"]:
                    continue
                final = get_param(r["spec"], param)
                if abs(final - orig) > 1e-9:
                    changes.append({"param": param, "from": orig, "to": final})
            rows.append({"preset": r["name"], "start_win_rate": r["start_win_rate"],
                         "end_win_rate": r["win_rate"], "changes": changes})
        summary["tuning"] = rows
    if sw:
        summary["sweeps"] = [{k: s[k] for k in ("preset", "param", "label", "default", "points")} for s in sw]

    # Training metrics: refresh the committed copies from the run folders when present.
    # Phase 2 continues from phase 1's final weights, so its updates are appended.
    recs = []
    phases = []
    for run in ("main", "phase2"):
        metrics_path = RESULTS / "training" / f"{run}_metrics.jsonl"
        run_metrics = Path(f"runs/{run}/metrics.jsonl")
        if run_metrics.exists():
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(_slim_metrics(run_metrics.read_text()))
        if not metrics_path.exists():
            continue
        lines = [json.loads(x) for x in metrics_path.read_text().splitlines() if x.strip()]
        # Keep the last record per update (a resumed run may repeat updates).
        by_update = {m["update"]: m for m in lines}
        offset = {k: recs[-1][k] if recs else 0 for k in ("update", "decisions", "games", "elapsed_s")}
        for k in sorted(by_update):
            m = dict(by_update[k])
            for key, off in offset.items():
                m[key] = m.get(key, 0) + off
            recs.append(m)
        last = by_update[max(by_update)]
        # Pairwise beat rates are measured on each update's finished games, so a single
        # update is noisy: average the phase's last 100 updates.
        tail = [by_update[k] for k in sorted(by_update)[-100:]]
        beat_keys = sorted({k for m in tail for k in m if k.startswith("beat/")})
        beat = {k.split("/", 1)[1]: sum(m[k] for m in tail if k in m) / sum(1 for m in tail if k in m)
                for k in beat_keys}
        cfg_path = Path("configs") / f"{run}.yaml"
        cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
        final_stage = (cfg.get("stages") or [{}])[-1]
        snap_p, self_p = final_stage.get("snapshot_prob", 0.0), final_stage.get("self_prob", 0.0)
        phases.append({"run": run, "start_update": offset["update"], "updates": max(by_update),
                       "config": {"batch": cfg.get("num_envs", 0) * cfg.get("rollout_steps", 0),
                                  "lr": cfg.get("lr"), "lr_final": cfg.get("lr_final"),
                                  "final_stage": final_stage.get("name"),
                                  "opponent_mix": {"bots": 1 - snap_p - self_p, "snapshots": snap_p,
                                                   "self": self_p}},
                       "decisions": last["decisions"], "games": last["games"],
                       "hours": last.get("elapsed_s", 0) / 3600,
                       "beat": beat})
    if recs:
        tables = sorted({k.split("/")[1] for m in recs for k in m if k.startswith("eval/")})
        curve = []
        for m in recs:
            point = {"update": m["update"], "decisions": m["decisions"], "games": m["games"],
                     "stage": m["stage"], "win_rate": m["win_rate"]}
            evals = {t: {"win_rate": m[f"eval/{t}/win_rate"], "ci_low": m[f"eval/{t}/ci_low"],
                         "ci_high": m[f"eval/{t}/ci_high"]} for t in tables if f"eval/{t}/win_rate" in m}
            if evals:
                point["eval"] = evals
            if evals or m["update"] % 10 == 0:
                curve.append(point)
        last = recs[-1]
        games += last["games"]
        summary["training"] = {"curve": curve, "decisions": last["decisions"], "games": last["games"],
                               "hours": last.get("elapsed_s", 0) / 3600,
                               "eval_tables": [_split_table(t) for t in tables], "phases": phases}
    las = _load(RESULTS / "lookahead.json")
    if las:
        games += sum(v["games"] for v in las["variants"])
        summary["lookahead_variants"] = las
    imp = _load(RESULTS / "improvements.json")
    if imp:
        games += sum(st["games"] for st in imp["steps"]) + imp["baseline"]["games"]
        summary["improvements"] = imp
    if ev:
        # A table reused from improvements.json was only played once.
        played = {(st["checkpoint"], st.get("seed"), st["games"], st["win_rate"]) for st in (imp or {}).get("steps", [])}
        games += sum(e["baseline"]["games"] + (0 if (e.get("checkpoint"), e.get("seed"), e["games"], e["win_rate"])
                                                  in played else e["games"]) for e in ev)
        summary["evaluation"] = []
        for e in ev:
            row = {k: e[k] for k in ("table", "games", "win_rate", "ci_low", "ci_high", "avg_vp", "baseline")}
            row |= {k: e[k] for k in ("checkpoint", "lookahead") if e.get(k)}
            # Each opponent sits at every game, so its win rate has the same sample size.
            row["opponents"] = [{"bot": b, "win_rate": w, "ci_low": wilson(w * e["games"], e["games"])[0],
                                 "ci_high": wilson(w * e["games"], e["games"])[1]}
                                for b, w in e.get("opponent_win_rate", {}).items()]
            summary["evaluation"].append(row)
    if st:
        games += st.get("games", 0)
        summary["style"] = {"metrics": st["metrics"], "rows": st["rows"]}
    summary["totals"] = {"games_simulated": int(games)}
    _dump(WEB_DATA / "summary.json", summary)
    return summary


def _split_table(key: str) -> list[str]:
    """'balanced_ows_road_builder' -> ['balanced', 'ows', 'road_builder']."""
    names = ["road_builder", "dev_rush", "port_master", "balanced", "ows", "greedy", "random"]
    out, rest = [], key
    while rest:
        for n in names:
            if rest.startswith(n):
                out.append(n)
                rest = rest[len(n):].lstrip("_")
                break
        else:
            out.append(rest)
            break
    return out


def main() -> None:
    # Tables printed by polars use box-drawing characters, which the Windows code page
    # can't encode when output is redirected to a file.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["bench", "tournament", "sweeps", "evaluate", "style", "replays", "export",
                                     "agent", "improvements", "lookahead"],
                    help="'agent' = evaluate + style + replays + export for --checkpoint")
    ap.add_argument("--seeds", type=int, default=250_000)
    ap.add_argument("--games", type=int, default=40_000)
    ap.add_argument("--checkpoint")
    ap.add_argument("--before", help="improvements: the checkpoint to compare against")
    ap.add_argument("--seed", type=int, help="evaluation seed (default: 10,000,000; improvements: 50,000,000)")
    ap.add_argument("--lookahead", nargs="?", const="{}", default=None, metavar="JSON",
                    help="play the agent with look-ahead (optional JSON overrides of LookaheadConfig)")
    args = ap.parse_args()
    la = LookaheadConfig(**json.loads(args.lookahead)) if args.lookahead is not None else None
    if args.step == "bench":
        bench()
    elif args.step == "tournament":
        tournament(args.seeds)
    elif args.step == "sweeps":
        sweeps()
    elif args.step == "evaluate":
        evaluate_checkpoint(args.checkpoint, args.games, la, args.seed or EVAL_SEED)
    elif args.step == "style":
        style(args.checkpoint, args.games, la)
    elif args.step == "replays":
        replays(args.checkpoint, la)
    elif args.step == "export":
        export()
    elif args.step == "lookahead":
        lookahead_sweep(args.checkpoint, args.games)
    elif args.step == "improvements":
        if not args.before:
            raise SystemExit("--before is required (and --checkpoint for the phase-2 steps)")
        improvements(args.before, args.checkpoint, args.games, la or LookaheadConfig(), args.seed or FRESH_SEED)
    elif args.step == "agent":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required")
        evaluate_checkpoint(args.checkpoint, args.games, la, args.seed or EVAL_SEED)
        style(args.checkpoint, max(2000, args.games // 5), la)
        replays(args.checkpoint, la)
        export()


if __name__ == "__main__":
    main()

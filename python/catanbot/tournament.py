"""Tournament helpers: run games through the Rust engine and summarise them.

Every function here uses seat rotation (each lineup is played in all 4
rotations on each seed) and common random numbers (the same seeds for every
variant), so comparisons between bots are as fair and low-variance as possible.
"""

from __future__ import annotations

import itertools
import json
import math

import numpy as np
import polars as pl

from catanbot import presets, run_pool, run_rotated, spec_name

RESOURCES = ["wood", "brick", "sheep", "wheat", "ore"]
STRATEGIES = ["balanced", "ows", "road_builder", "dev_rush", "port_master"]
BASELINES = ["random", "greedy"]


def preset_spec(name: str) -> dict:
    """The JSON bot spec of a preset, as a dict (editable)."""
    for n, js in presets():
        if n == name:
            return json.loads(js)
    raise KeyError(name)


def as_spec(bot: str | dict) -> str:
    return bot if isinstance(bot, str) else json.dumps(bot)


def wilson(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def to_frame(d: dict) -> pl.DataFrame:
    """Engine output (numpy columns, one row per game x seat) -> DataFrame with bot names."""
    names = d["bot_names"]
    cols = {k: v for k, v in d.items() if k != "bot_names"}
    df = pl.DataFrame(cols)
    return df.with_columns(
        pl.col("bot").map_elements(lambda i: names[i], return_dtype=pl.String).alias("bot"),
        (pl.col("produced_wood") + pl.col("produced_brick") + pl.col("produced_sheep")
         + pl.col("produced_wheat") + pl.col("produced_ore")).alias("produced"),
    )


def rotated(bots: list[str | dict], seeds: int, base_seed: int = 1) -> pl.DataFrame:
    """Play a fixed 4-bot table in all seat rotations on ``seeds`` seeds."""
    assert len(bots) == 4
    return to_frame(run_rotated([as_spec(b) for b in bots], [0, 1, 2, 3], seeds, base_seed))


def pool(bots: list[str | dict], seeds: int, base_seed: int = 1, distinct: bool = True) -> pl.DataFrame:
    """Random 4-bot tables from a pool, each in all seat rotations."""
    return to_frame(run_pool([as_spec(b) for b in bots], seeds, base_seed, distinct))


def summarize(df: pl.DataFrame) -> pl.DataFrame:
    """Per-bot win rate (with 95% Wilson CI) and play-style statistics."""
    s = (
        df.group_by("bot")
        .agg(
            pl.len().alias("games"),
            pl.col("won").sum().alias("wins"),
            pl.col("vp").mean().alias("avg_vp"),
            pl.col("settlements").mean().alias("settlements"),
            pl.col("cities").mean().alias("cities"),
            pl.col("roads").mean().alias("roads"),
            pl.col("dev_bought").mean().alias("dev_bought"),
            pl.col("knights").mean().alias("knights"),
            pl.col("trades").mean().alias("trades"),
            pl.col("produced").mean().alias("produced"),
            (pl.col("longest_road").filter(pl.col("won") == 1).mean()).alias("lr_in_wins"),
            (pl.col("largest_army").filter(pl.col("won") == 1).mean()).alias("la_in_wins"),
            (pl.col("turns").filter(pl.col("won") == 1).mean()).alias("turns_in_wins"),
        )
        .with_columns((pl.col("wins") / pl.col("games")).alias("win_rate"))
    )
    lo, hi = zip(*[wilson(w, n) for w, n in zip(s["wins"].to_list(), s["games"].to_list())]) if len(s) else ([], [])
    return s.with_columns(pl.Series("ci_low", lo), pl.Series("ci_high", hi)).sort("win_rate", descending=True)


def field_win_rate(bot: str | dict, field: list[str | dict], seeds: int, base_seed: int = 1) -> tuple[int, int]:
    """(wins, games) of ``bot`` against every 3-bot combination of ``field``.

    ``seeds`` is split across the combinations; each seed is played in all 4
    rotations. The same seeds are used for every ``bot``, so two variants
    are compared on identical boards and dice.
    """
    combos = list(itertools.combinations(range(len(field)), 3))
    per = max(1, seeds // len(combos))
    wins = games = 0
    for k, combo in enumerate(combos):
        specs = [as_spec(bot)] + [as_spec(field[i]) for i in combo]
        d = run_rotated(specs, [0, 1, 2, 3], per, base_seed + k * 1_000_000)
        mine = d["bot"] == 0
        wins += int(d["won"][mine].sum())
        games += int(mine.sum())
    return wins, games


def openskill_ratings(df: pl.DataFrame, max_games: int = 50_000) -> pl.DataFrame:
    """OpenSkill (Plackett-Luce) ratings from finished games, ranked by VP (winner first)."""
    from openskill.models import PlackettLuce

    model = PlackettLuce()
    ratings: dict[str, object] = {}
    games = df.filter(pl.col("draw") == 0).sort(["game_id", "seat"])
    ids = games["game_id"].unique(maintain_order=True)[:max_games]
    games = games.filter(pl.col("game_id").is_in(ids.implode()))
    bots = games["bot"].to_list()
    vps = games["vp"].to_list()
    won = games["won"].to_list()
    for g in range(0, len(bots), 4):
        names = bots[g:g + 4]
        if len(set(names)) < 4:
            continue  # PlackettLuce needs distinct players per game
        # Rank: winner first, then by VP.
        scores = [vps[g + i] + (100 if won[g + i] else 0) for i in range(4)]
        teams = [[ratings.setdefault(n, model.rating(name=n))] for n in names]
        new = model.rate(teams, scores=scores)
        for n, t in zip(names, new):
            ratings[n] = t[0]
    rows = [{"bot": n, "mu": r.mu, "sigma": r.sigma, "ordinal": r.ordinal()} for n, r in ratings.items()]
    return pl.DataFrame(rows).sort("ordinal", descending=True)


def seat_advantage(df: pl.DataFrame) -> pl.DataFrame:
    return df.group_by("seat").agg(pl.col("won").mean().alias("win_rate"), pl.len().alias("games")).sort("seat")


def resource_mix(df: pl.DataFrame) -> pl.DataFrame:
    """Average share of each resource in what a bot produced."""
    cols = [f"produced_{r}" for r in RESOURCES]
    out = df.group_by("bot").agg([pl.col(c).sum() for c in cols])
    total = sum(out[c] for c in cols)
    return out.with_columns([(pl.col(c) / total).alias(c.replace("produced_", "")) for c in cols]).select(
        ["bot"] + RESOURCES
    )


def np_summary_dict(df: pl.DataFrame) -> list[dict]:
    """JSON-friendly summary (for the web viewer)."""
    return [{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()}
            for row in summarize(df).to_dicts()]

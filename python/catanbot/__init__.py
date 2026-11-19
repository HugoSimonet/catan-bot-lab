"""catanbot: Settlers of Catan strategy bots, mass simulation and RL.

The game engine is written in Rust (``catanbot._engine``); this package holds
the Python side: the RL environment wrapper, the PPO/league trainer,
tournaments, parameter sweeps and replay export.
"""

from catanbot._engine import (
    NUM_ACTIONS,
    NUM_PLAYERS,
    OBS_SIZE,
    PRESET_NAMES,
    VecEnv,
    layout,
    play_replay,
    presets,
    replay_from_actions,
    run_pool,
    run_rotated,
    spec_name,
    topology_json,
)

__all__ = [
    "NUM_ACTIONS",
    "NUM_PLAYERS",
    "OBS_SIZE",
    "PRESET_NAMES",
    "VecEnv",
    "layout",
    "play_replay",
    "presets",
    "replay_from_actions",
    "run_pool",
    "run_rotated",
    "spec_name",
    "topology_json",
]

"""Thin, typed wrapper around the Rust ``VecEnv``.

Seat codes in a lineup: ``>= 0`` is an index into the bot spec list,
``-1 - k`` is external policy ``k``. Policy 0 is, by convention, the learner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from catanbot import VecEnv


def policy_code(k: int) -> int:
    """Seat code for external policy ``k``."""
    return -1 - k


def code_policy(code: int) -> int | None:
    """Policy id for a seat code, or None if it is a bot."""
    return -1 - code if code < 0 else None


@dataclass
class Step:
    obs: np.ndarray        # [N, OBS_SIZE] float32 - observation of the acting player
    mask: np.ndarray       # [N, NUM_ACTIONS] bool - legal actions
    policy: np.ndarray     # [N] int32 - which external policy must act
    seat: np.ndarray       # [N] int32 - which seat is acting
    reward: np.ndarray     # [N, 4] float32 - reward per seat since the previous step
    done: np.ndarray       # [N] bool - a game finished during this step (env already reset)
    winner: np.ndarray     # [N] int32 - winner seat of the finished game, -1 draw, -2 n/a
    final_vp: np.ndarray   # [N, 4] int32 - final VP of the finished game
    turns: np.ndarray      # [N] int32 - length of the finished game
    lineup: np.ndarray     # [N, 4] int32 - seat codes of the finished game


class CatanVecEnv:
    def __init__(self, bot_specs: list[str], lineups: list[list[int]], seed: int = 0,
                 record: bool = False) -> None:
        self.bot_specs = list(bot_specs)
        self._env = VecEnv(self.bot_specs, [list(l) for l in lineups], seed, record)

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    def set_rewards(self, win: float | None = None, loss: float | None = None,
                    shaping: float | None = None) -> None:
        self._env.set_rewards(win, loss, shaping)

    def set_lineup(self, i: int, lineup: list[int]) -> None:
        """Queue the lineup for env ``i``'s *next* game."""
        self._env.set_lineup(i, list(lineup))

    def lineup(self, i: int) -> list[int]:
        """Lineup of the game currently running in env ``i``."""
        return self._env.lineup(i)

    def reset(self) -> Step:
        return Step(*self._env.reset())

    def step(self, actions: np.ndarray) -> Step:
        return Step(*self._env.step(np.asarray(actions, dtype=np.int64).tolist()))

    def last_replay(self, i: int, players: list[str]) -> str:
        return self._env.last_replay(i, players)

    def last_game(self, i: int) -> tuple[int, list[int]]:
        return self._env.last_game(i)

    def last_result(self, i: int) -> dict[str, list[int]]:
        """Per-seat statistics of env ``i``'s last finished game ({column: [4 values]})."""
        return self._env.last_result(i)

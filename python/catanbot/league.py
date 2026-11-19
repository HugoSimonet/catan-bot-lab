"""Opponent selection: curriculum stages, frozen snapshots and PFSP.

Each game seats the learner (policy 0) against three opponents. An opponent
is one of:

- a Rust strategy bot (``("bot", name)``),
- a frozen past snapshot of the learner, loaded in one of the snapshot slots
  (policies 1..S),
- the current learner itself (self-play; that seat's data is also trained on).

Opponents are drawn with *prioritised fictitious self-play* (PFSP): the more
often an opponent beats the learner, the more often it is picked, so training
time goes where the learner is weakest. "Beats" is measured pairwise from
final victory points, which is much less noisy than 4-player win rate.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

from catanbot.env import code_policy, policy_code

LEARNER = 0


@dataclass
class Stage:
    name: str
    bots: list[str]
    #: Probability that an opponent seat is a frozen snapshot / the learner itself.
    snapshot_prob: float = 0.0
    self_prob: float = 0.0
    #: Advance when the learner's win rate over the last ``window`` games reaches this.
    until_win_rate: float | None = None
    min_updates: int = 0
    max_updates: int | None = None
    #: Prioritise opponents the learner struggles against (else uniform).
    #: Off during the curriculum, so the learner isn't swamped by the strongest bot.
    pfsp: bool = False


@dataclass
class Snapshot:
    id: int
    update: int
    path: str
    state_dict: dict | None = None


@dataclass
class League:
    stages: list[Stage]
    bot_index: dict[str, int]          # bot name -> index in the env's bot spec list
    snapshot_slots: int = 4
    window: int = 500
    pfsp_power: float = 1.5
    seed: int = 0
    stage_idx: int = 0
    stage_start_update: int = 0
    snapshots: list[Snapshot] = field(default_factory=list)
    slots: list[int | None] = field(default_factory=list)   # snapshot id in slot j (policy j+1)
    beat_rate: dict[tuple, float] = field(default_factory=dict)
    recent: deque = field(default_factory=lambda: deque(maxlen=500))

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.recent = deque(maxlen=self.window)
        if not self.slots:
            self.slots = [None] * self.snapshot_slots

    # ------------------------------------------------------------------
    @property
    def stage(self) -> Stage:
        return self.stages[self.stage_idx]

    def _key_of_code(self, code: int) -> tuple:
        pid = code_policy(code)
        if pid is None:
            name = next(n for n, i in self.bot_index.items() if i == code)
            return ("bot", name)
        if pid == LEARNER:
            return ("self",)
        return ("snap", self.slots[pid - 1])

    def _weight(self, key: tuple) -> float:
        b = self.beat_rate.get(key, 0.5)
        return (1.0 - b) ** self.pfsp_power + 0.05

    def _pick(self, keys: list[tuple]) -> tuple:
        if not self.stage.pfsp:
            return self.rng.choice(keys)
        weights = [self._weight(k) for k in keys]
        return self.rng.choices(keys, weights=weights, k=1)[0]

    def sample_lineup(self) -> list[int]:
        """Seat codes for one game: the learner plus 3 opponents, shuffled."""
        st = self.stage
        loaded = [j for j, s in enumerate(self.slots) if s is not None]
        seats = [policy_code(LEARNER)]
        for _ in range(3):
            u = self.rng.random()
            if u < st.snapshot_prob and loaded:
                j = self._pick([("snap", self.slots[j]) for j in loaded])
                slot = self.slots.index(j[1])
                seats.append(policy_code(slot + 1))
            elif u < st.snapshot_prob + st.self_prob:
                seats.append(policy_code(LEARNER))
            else:
                key = self._pick([("bot", b) for b in st.bots])
                seats.append(self.bot_index[key[1]])
        self.rng.shuffle(seats)
        return seats

    # ------------------------------------------------------------------
    def record_game(self, lineup: list[int], final_vp: list[int], winner: int) -> None:
        """Update win statistics from a finished game."""
        learner_seats = [s for s, c in enumerate(lineup) if code_policy(c) == LEARNER]
        if not learner_seats:
            return
        # Learner win rate: did *a* learner seat win? Count once per learner seat
        # so self-play games don't inflate it: use the first learner seat.
        ls = learner_seats[0]
        self.recent.append(1.0 if winner == ls else 0.0)
        for s, code in enumerate(lineup):
            if s in learner_seats:
                continue
            key = self._key_of_code(code)
            if winner == ls:
                beat = 1.0
            elif winner == s:
                beat = 0.0
            else:
                beat = 1.0 if final_vp[ls] > final_vp[s] else 0.5 if final_vp[ls] == final_vp[s] else 0.0
            old = self.beat_rate.get(key, 0.5)
            self.beat_rate[key] = 0.99 * old + 0.01 * beat

    def recent_win_rate(self) -> float:
        return sum(self.recent) / len(self.recent) if self.recent else 0.0

    def maybe_advance(self, update: int) -> bool:
        """Move to the next stage when the current one's goal is met."""
        st = self.stage
        if self.stage_idx + 1 >= len(self.stages):
            return False
        done_updates = update - self.stage_start_update
        if done_updates < st.min_updates:
            return False
        reached = (
            st.until_win_rate is not None
            and len(self.recent) >= self.recent.maxlen
            and self.recent_win_rate() >= st.until_win_rate
        )
        timed_out = st.max_updates is not None and done_updates >= st.max_updates
        if reached or timed_out:
            self.stage_idx += 1
            self.stage_start_update = update
            self.recent.clear()
            return True
        return False

    # ------------------------------------------------------------------
    def next_snapshot_id(self) -> int:
        return self.snapshots[-1].id + 1 if self.snapshots else 0

    def add_snapshot(self, update: int, state_dict: dict, path: str, max_keep: int = 30) -> Snapshot:
        snap = Snapshot(id=self.next_snapshot_id(), update=update, path=path,
                        state_dict={k: v.detach().cpu().clone() for k, v in state_dict.items()})
        self.snapshots.append(snap)
        if len(self.snapshots) > max_keep:
            # Keep the first snapshot (diversity) and the most recent ones.
            self.snapshots = self.snapshots[:1] + self.snapshots[-(max_keep - 1):]
        return snap

    def refill_slots(self) -> dict[int, Snapshot]:
        """Choose which snapshots occupy the slots (PFSP-weighted, newest always in).

        Returns {slot: snapshot} for slots whose content changed.
        """
        if not self.snapshots:
            return {}
        newest = self.snapshots[-1]
        pool = self.snapshots[:-1]
        chosen = [newest]
        while len(chosen) < self.snapshot_slots and pool:
            keys = [("snap", s.id) for s in pool]
            k = self._pick(keys)
            s = next(x for x in pool if x.id == k[1])
            chosen.append(s)
            pool.remove(s)
        changed = {}
        for j in range(self.snapshot_slots):
            sid = chosen[j].id if j < len(chosen) else None
            if self.slots[j] != sid:
                self.slots[j] = sid
                if sid is not None:
                    changed[j] = chosen[j]
        return changed

    def snapshot_by_id(self, sid: int) -> Snapshot | None:
        return next((s for s in self.snapshots if s.id == sid), None)

    # ------------------------------------------------------------------
    def state(self) -> dict:
        return {
            "stage_idx": self.stage_idx,
            "stage_start_update": self.stage_start_update,
            "slots": self.slots,
            "beat_rate": [[list(k), v] for k, v in self.beat_rate.items()],
            "recent": list(self.recent),
            # Weights live in their own files; only metadata goes in the checkpoint.
            "snapshots": [{"id": s.id, "update": s.update, "path": s.path} for s in self.snapshots],
        }

    def load_state(self, d: dict, load_weights) -> None:
        """Restore from ``state()``; ``load_weights(path)`` returns a state dict."""
        self.stage_idx = d["stage_idx"]
        self.stage_start_update = d["stage_start_update"]
        self.slots = list(d["slots"])
        self.beat_rate = {tuple(k): v for k, v in d["beat_rate"]}
        self.recent = deque(d.get("recent", []), maxlen=self.window)
        self.snapshots = [Snapshot(id=s["id"], update=s["update"], path=s["path"],
                                   state_dict=load_weights(s["path"])) for s in d["snapshots"]]

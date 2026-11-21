"""Decision-time look-ahead ("imagined futures") on top of a trained policy.

At every decision with at least two plausible moves (policy probability of at
least ``min_prior``; optionally only when the favourite is below ``threshold``),
the top ``k`` moves are each tried in ``sims`` imagined continuations,
simulated in Rust (``catan_core::search``):

- hidden information is re-dealt from the agent's point of view (opponents'
  cards, the dev deck) and the dice are re-seeded, so nothing is peeked at;
- the agent's move is applied, the agent finishes its turn with its own
  policy, opponents are played by a stand-in bot, and the continuation stops
  at the agent's first decision after theirs (``horizon="turn"``, normally the
  start of its next turn) or at the end of the game;
- the agent's own value network scores where each continuation ended up.

Every candidate sees the same continuations (same re-dealt cards and dice).
The move with the best average score is played, if it beats the policy's
favourite by more than ``margin``; otherwise the policy's favourite is kept.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from catanbot import layout

#: Game phases in the order of the observation's phase one-hot.
PHASES = ("SetupSettlement", "SetupRoad", "PreRoll", "Discard", "MoveRobber", "Steal", "Main",
          "RoadBuilding", "GameOver")
_PHASE_OFFSET = layout()["global"][0] + 6


@dataclass
class LookaheadConfig:
    # Defaults chosen on 8,192 games of the first agent (`pipeline lookahead`).
    k: int = 6                  # candidate moves per decision
    sims: int = 16              # imagined continuations per candidate
    threshold: float = 1.01     # search only when the top move's probability is below this (>1: always)
    min_prior: float = 0.005    # ignore candidates the policy gives less than this
    margin: float = 0.0         # required value gain over the policy's favourite
    opponent: str = "balanced"  # stand-in bot for opponents in imagined futures
    win: float = 1.0            # terminal values on the value head's scale
    loss: float = -1.0 / 3.0
    leaf_batch: int = 8192      # value-network batch size for leaves
    #: Only search decisions in these game phases (None = all), e.g. ("MoveRobber", "Discard").
    phases: tuple[str, ...] | None = None
    #: "turn": play every candidate to the same point (the agent finishes its turn with its
    #: own policy, the opponents play, scored at its next decision after theirs).
    #: "decision": score at the agent's very next decision (cheaper, but a move that keeps the
    #: turn going is then scored earlier in the game than "end turn").
    horizon: str = "turn"
    max_own: int = 40           # turn horizon: cap on the agent's own moves per continuation

    def label(self) -> str:
        return f"k{self.k}_s{self.sims}_t{self.threshold:g}_m{self.margin:g}"


@torch.no_grad()
def _forward(net: torch.nn.Module, obs: np.ndarray, mask: np.ndarray, device, batch: int,
             want: str) -> np.ndarray:
    """Values (``want="value"``) or argmax actions (``want="action"``) in batches."""
    out = np.empty(len(obs), dtype=np.float32 if want == "value" else np.int64)
    for i in range(0, len(obs), batch):
        o = torch.from_numpy(obs[i:i + batch]).to(device, non_blocking=True)
        m = torch.from_numpy(mask[i:i + batch]).to(device, non_blocking=True)
        logits, v = net(o, m)
        out[i:i + batch] = (v.float() if want == "value" else logits.argmax(-1)).cpu().numpy()
    return out


def values_of(net: torch.nn.Module, obs: np.ndarray, mask: np.ndarray, device, batch: int) -> np.ndarray:
    return _forward(net, obs, mask, device, batch, "value")


class Lookahead:
    """Chooses actions for a batch of pending decisions of one policy."""

    def __init__(self, net: torch.nn.Module, cfg: LookaheadConfig, device, seed: int = 0) -> None:
        self.net, self.cfg, self.device = net, cfg, device
        self.calls = 0
        self.seed = seed
        self.searched = 0
        self.changed = 0
        self.decisions = 0

    @torch.no_grad()
    def choose(self, raw_env, env_idx: np.ndarray, logits: torch.Tensor,
               obs: np.ndarray | None = None) -> np.ndarray:
        """Actions for games ``env_idx`` whose (masked) policy logits are ``logits``.

        ``obs`` (the same rows' observations) is needed when ``cfg.phases`` is set.
        """
        c = self.cfg
        probs = torch.softmax(logits.float(), -1)
        greedy = probs.argmax(-1)
        actions = greedy.cpu().numpy().copy()
        self.decisions += len(env_idx)
        top_p, top_a = probs.topk(c.k, dim=-1)
        top_p, top_a = top_p.cpu().numpy(), top_a.cpu().numpy()
        eligible = top_p[:, 0] < c.threshold
        if c.phases is not None:
            phase = obs[:, _PHASE_OFFSET:_PHASE_OFFSET + len(PHASES)].argmax(1)
            eligible &= np.isin(phase, [PHASES.index(x) for x in c.phases])
        sel = np.nonzero(eligible)[0]
        if len(sel) == 0:
            return actions
        cands = np.where(top_p[sel] >= c.min_prior, top_a[sel], -1)
        cands[:, 0] = top_a[sel, 0]
        # Candidates are contiguous from the left (top-k is sorted by probability).
        many = (cands >= 0).sum(1) >= 2
        sel, cands = sel[many], cands[many]
        if len(sel) == 0:
            return actions
        self.calls += 1
        args = (env_idx[sel].tolist(), cands.reshape(-1).tolist(), c.k, c.sims, c.opponent)
        seed = self.seed * 1_000_003 + self.calls
        if c.horizon == "turn":
            run = raw_env.rollouts(*args, seed=seed, win=c.win, loss=c.loss, max_own=c.max_own)
            # The agent plays out its turn in every imagined continuation with its own policy.
            while True:
                ids, p_obs, p_mask = run.pending()
                if len(ids) == 0:
                    break
                acts = _forward(self.net, p_obs, p_mask, self.device, c.leaf_batch, "action")
                run.step(ids.tolist(), acts.tolist())
            obs, mask, slot, term, count = run.leaves()
        elif c.horizon == "decision":
            obs, mask, slot, term, count = raw_env.search(*args, seed=seed, win=c.win, loss=c.loss)
        else:
            raise ValueError(f"unknown horizon {c.horizon!r}")
        total = term.astype(np.float64)
        if len(slot):
            v = values_of(self.net, obs, mask, self.device, c.leaf_batch)
            np.add.at(total, slot, v)
        q = np.where(count > 0, total / np.maximum(count, 1), -np.inf).reshape(len(sel), c.k)
        best = q.argmax(1)
        gain = q[np.arange(len(sel)), best] - q[:, 0]
        switch = (best != 0) & (gain > c.margin)
        chosen = np.where(switch, cands[np.arange(len(sel)), best], cands[:, 0])
        actions[sel] = chosen
        self.searched += len(sel)
        self.changed += int(switch.sum())
        return actions

    def stats(self) -> dict:
        return {"decisions": self.decisions, "searched": self.searched, "changed": self.changed,
                "searched_frac": self.searched / max(self.decisions, 1),
                "changed_frac": self.changed / max(self.decisions, 1)}

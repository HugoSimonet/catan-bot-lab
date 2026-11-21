"""PPO + league self-play training.

    python -m catanbot.train --config configs/main.yaml
    python -m catanbot.train --config configs/main.yaml --resume
    python -m catanbot.train --config configs/phase2.yaml   # continues from main's weights

Data flow per update:
  1. Roll out ``rollout_steps`` decisions in ``num_envs`` parallel games. Each
     pending decision belongs to the learner (policy 0) or a frozen snapshot
     (policies 1..S); Rust bots and forced moves are advanced inside Rust.
  2. Learner transitions are linked per (game, seat) so GAE runs along each
     seat's own trajectory, even though seats take turns.
  3. A few epochs of clipped PPO on the learner's transitions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from catanbot import NUM_ACTIONS, OBS_SIZE, PRESET_NAMES
from catanbot.env import CatanVecEnv
from catanbot.evaluate import evaluate
from catanbot.league import LEARNER, League, Stage
from catanbot.model import make_policy


@dataclass
class Config:
    run_name: str = "main"
    runs_dir: str = "runs"
    seed: int = 0
    device: str = "cuda"
    # Rollouts
    num_envs: int = 256
    rollout_steps: int = 128
    total_updates: int = 3000
    # PPO
    lr: float = 2.5e-4
    lr_final: float = 5e-5
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 4096
    ent_coef: float = 0.01
    ent_coef_final: float = 0.003
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    # Rewards
    loss_reward: float = -1.0 / 3.0
    shaping_start: float = 0.05
    shaping_end_update: int = 400
    # Model: "graph" (entity-based, see model.GraphPolicyNet) or "mlp"
    arch: str = "mlp"
    hidden: int = 512
    blocks: int = 4
    ent: int = 64
    rounds: int = 2
    # League
    stages: list[dict] = field(default_factory=lambda: [{"name": "random", "bots": ["random"]}])
    snapshot_every: int = 50
    snapshot_slots: int = 4
    win_rate_window: int = 500
    # Evaluation / checkpoints
    eval_every: int = 25
    eval_games: int = 512
    eval_opponents: list[list[str]] = field(
        default_factory=lambda: [["balanced", "ows", "road_builder"]])
    checkpoint_every: int = 25
    #: Keep update_XXXXX.pt only every this many updates (0 = keep all); latest.pt is always written.
    checkpoint_keep_every: int = 0
    #: Start from the weights of this checkpoint (a new phase continuing an earlier run).
    #: Only the network is copied: optimizer, schedules and league start fresh.
    init_from: str | None = None


def load_config(path: str | os.PathLike) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    unknown = set(raw) - set(Config.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return Config(**raw)


class Buffer:
    """Learner transitions for one rollout, stored on the training device."""

    def __init__(self, capacity: int, device: torch.device) -> None:
        self.obs = torch.zeros(capacity, OBS_SIZE, device=device)
        self.mask = torch.zeros(capacity, NUM_ACTIONS, dtype=torch.bool, device=device)
        self.action = torch.zeros(capacity, dtype=torch.long, device=device)
        self.logp = torch.zeros(capacity, device=device)
        self.value = torch.zeros(capacity, device=device)
        self.reward = np.zeros(capacity, dtype=np.float32)
        # Index of the same seat's next transition; -1 = game ended; -2 = cut by
        # the end of the rollout (excluded from training, used for bootstrapping).
        self.next = np.full(capacity, -2, dtype=np.int64)
        self.n = 0

    def reset(self) -> None:
        self.reward[: self.n] = 0
        self.next[: self.n] = -2
        self.n = 0


def compute_gae(buf: Buffer, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = buf.n
    values = buf.value[:n].float().cpu().numpy()
    nxt = buf.next[:n]
    rew = buf.reward[:n]
    adv = np.zeros(n, dtype=np.float32)
    for j in range(n - 1, -1, -1):
        k = nxt[j]
        if k == -2:
            continue
        if k == -1:
            delta = rew[j] - values[j]
            adv[j] = delta
        else:
            delta = rew[j] + gamma * values[k] - values[j]
            adv[j] = delta + gamma * lam * adv[k]
    valid = nxt != -2
    returns = adv + values
    return adv, returns, valid


class Trainer:
    def __init__(self, cfg: Config, resume: bool = False) -> None:
        self.cfg = cfg
        self.dir = Path(cfg.runs_dir) / cfg.run_name
        (self.dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.dir / "snapshots").mkdir(parents=True, exist_ok=True)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        self.bot_names = list(PRESET_NAMES)
        self.bot_index = {n: i for i, n in enumerate(self.bot_names)}
        stages = [Stage(**s) for s in cfg.stages]
        for st in stages:
            for b in st.bots:
                if b not in self.bot_index:
                    raise ValueError(f"stage {st.name}: unknown bot {b}")
        self.league = League(stages=stages, bot_index=self.bot_index,
                             snapshot_slots=cfg.snapshot_slots, window=cfg.win_rate_window,
                             seed=cfg.seed)

        self.net = self._make_net().to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)
        self.opponents = {j + 1: self._make_net().to(self.device).eval()
                          for j in range(cfg.snapshot_slots)}
        self.update = 0
        self.decisions = 0
        self.games = 0
        self.elapsed = 0.0
        latest = self.dir / "checkpoints" / "latest.pt"
        if resume and latest.exists():
            self._load(latest)
        elif resume and not cfg.init_from:
            raise FileNotFoundError(f"nothing to resume: {latest} does not exist")
        elif cfg.init_from:
            # A fresh run, or --resume before the first checkpoint: start from init_from.
            init = torch.load(cfg.init_from, map_location=self.device, weights_only=False)
            self.net.load_state_dict(init["model"])
            print(f"initialised from {cfg.init_from} (update {init.get('update', '?')})")

        lineups = [self.league.sample_lineup() for _ in range(cfg.num_envs)]
        self.env = CatanVecEnv(self.bot_names, lineups, seed=cfg.seed * 1_000_003 + self.update * 7919)
        self.env.set_rewards(loss=cfg.loss_reward, shaping=self._shaping())
        for e in range(cfg.num_envs):
            self.env.set_lineup(e, self.league.sample_lineup())
        self.buf = Buffer(cfg.num_envs * cfg.rollout_steps, self.device)
        self.step = self.env.reset()
        self.recent_vp: deque = deque(maxlen=cfg.win_rate_window)
        self.recent_turns: deque = deque(maxlen=cfg.win_rate_window)
        self.metrics_file = open(self.dir / "metrics.jsonl", "a")
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(str(self.dir / "tb"))
        except Exception:  # tensorboard is optional
            self.tb = None
        with open(self.dir / "config.yaml", "w") as f:
            yaml.safe_dump(asdict(cfg), f, sort_keys=False)

    def _make_net(self) -> torch.nn.Module:
        c = self.cfg
        if c.arch == "graph":
            return make_policy("graph", hidden=c.hidden, blocks=c.blocks, ent=c.ent, rounds=c.rounds)
        return make_policy("mlp", hidden=c.hidden, blocks=c.blocks)

    # ------------------------------------------------------------------
    def _frac(self) -> float:
        return min(1.0, self.update / max(1, self.cfg.total_updates))

    def _shaping(self) -> float:
        c = self.cfg
        return c.shaping_start * max(0.0, 1.0 - self.update / max(1, c.shaping_end_update))

    def _schedules(self) -> tuple[float, float]:
        c = self.cfg
        f = self._frac()
        lr = c.lr + (c.lr_final - c.lr) * f
        ent = c.ent_coef + (c.ent_coef_final - c.ent_coef) * f
        return lr, ent

    # ------------------------------------------------------------------
    def rollout(self) -> dict:
        c = self.cfg
        buf = self.buf
        buf.reset()
        n_env = c.num_envs
        pending = np.full((n_env, 4), -1, dtype=np.int64)
        finished = 0
        step = self.step
        self.net.eval()
        for _ in range(c.rollout_steps):
            obs_t = torch.from_numpy(step.obs).to(self.device, non_blocking=True)
            mask_t = torch.from_numpy(step.mask).to(self.device, non_blocking=True)
            actions = np.zeros(n_env, dtype=np.int64)
            for pid in np.unique(step.policy):
                idx = np.nonzero(step.policy == pid)[0]
                idx_t = torch.from_numpy(idx).to(self.device)
                o, m = obs_t[idx_t], mask_t[idx_t]
                if pid == LEARNER:
                    a, logp, v = self.net.act(o, m)
                    k = len(idx)
                    j0 = buf.n
                    js = np.arange(j0, j0 + k)
                    buf.obs[j0:j0 + k] = o
                    buf.mask[j0:j0 + k] = m
                    buf.action[j0:j0 + k] = a
                    buf.logp[j0:j0 + k] = logp
                    buf.value[j0:j0 + k] = v
                    seats = step.seat[idx]
                    prev = pending[idx, seats]
                    has = prev >= 0
                    buf.next[prev[has]] = js[has]
                    pending[idx, seats] = js
                    buf.n += k
                else:
                    a, _, _ = self.opponents[int(pid)].act(o, m)
                actions[idx] = a.cpu().numpy()
            step = self.env.step(actions)
            self.decisions += n_env

            # Credit rewards to each seat's latest transition.
            e_idx, s_idx = np.nonzero((pending >= 0) & (step.reward != 0))
            buf.reward[pending[e_idx, s_idx]] += step.reward[e_idx, s_idx]

            for e in np.nonzero(step.done)[0]:
                for s in range(4):
                    if pending[e, s] >= 0:
                        buf.next[pending[e, s]] = -1
                        pending[e, s] = -1
                lineup = step.lineup[e].tolist()
                self.league.record_game(lineup, step.final_vp[e].tolist(), int(step.winner[e]))
                seat = lineup.index(-1)
                self.recent_vp.append(int(step.final_vp[e, seat]))
                self.recent_turns.append(int(step.turns[e]))
                finished += 1
                # Queue the opponents for the game after the one that just started.
                self.env.set_lineup(int(e), self.league.sample_lineup())
        self.step = step
        self.games += finished
        return {"games_finished": finished}

    def learn(self) -> dict:
        c = self.cfg
        buf = self.buf
        adv, ret, valid = compute_gae(buf, c.gamma, c.gae_lambda)
        idx = np.nonzero(valid)[0]
        if len(idx) < 2:
            return {}
        adv_t = torch.from_numpy(adv).to(self.device)
        ret_t = torch.from_numpy(ret).to(self.device)
        a = adv_t[idx]
        adv_t[idx] = (a - a.mean()) / (a.std() + 1e-8)
        lr, ent_coef = self._schedules()
        for g in self.opt.param_groups:
            g["lr"] = lr

        self.net.train()
        stats = {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}
        batches = 0
        stop = False
        for _ in range(c.epochs):
            perm = torch.from_numpy(np.random.permutation(idx)).to(self.device)
            for start in range(0, len(perm), c.minibatch):
                mb = perm[start:start + c.minibatch]
                logits, v = self.net(buf.obs[mb], buf.mask[mb])
                logp_all = torch.log_softmax(logits, -1)
                logp = logp_all.gather(-1, buf.action[mb][:, None]).squeeze(-1)
                ratio = torch.exp(logp - buf.logp[mb])
                adv_mb = adv_t[mb]
                pg = -torch.min(ratio * adv_mb, ratio.clamp(1 - c.clip, 1 + c.clip) * adv_mb).mean()
                v_loss = 0.5 * (v - ret_t[mb]).pow(2).mean()
                entropy = -(logp_all.exp() * logp_all).sum(-1).mean()
                loss = pg + c.vf_coef * v_loss - ent_coef * entropy
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), c.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    log_ratio = logp - buf.logp[mb]
                    kl = ((ratio - 1) - log_ratio).mean().item()
                    stats["pg_loss"] += pg.item()
                    stats["v_loss"] += v_loss.item()
                    stats["entropy"] += entropy.item()
                    stats["approx_kl"] += kl
                    stats["clip_frac"] += ((ratio - 1).abs() > c.clip).float().mean().item()
                batches += 1
                if kl > 1.5 * c.target_kl:
                    stop = True
                    break
            if stop:
                break
        out = {k: v / max(batches, 1) for k, v in stats.items()}
        values = buf.value[:buf.n].float().cpu().numpy()[idx]
        var = np.var(ret[idx])
        out["explained_var"] = float(1 - np.var(ret[idx] - values) / var) if var > 0 else 0.0
        out["samples"] = int(len(idx))
        out["lr"] = lr
        out["ent_coef"] = ent_coef
        out["early_stop"] = stop
        return out

    # ------------------------------------------------------------------
    def _snapshot(self) -> None:
        sid = self.league.next_snapshot_id()
        path = str(self.dir / "snapshots" / f"snap_{sid:04d}.pt")
        torch.save({"model": self.net.state_dict(), "model_config": self.net.config(),
                    "update": self.update}, path)
        self.league.add_snapshot(self.update, self.net.state_dict(), path)
        self._refill_slots()

    def _refill_slots(self) -> None:
        for slot, snap in self.league.refill_slots().items():
            self.opponents[slot + 1].load_state_dict(snap.state_dict)

    def save(self) -> None:
        ckpt = {
            "model": self.net.state_dict(),
            "model_config": self.net.config(),
            "optimizer": self.opt.state_dict(),
            "update": self.update,
            "decisions": self.decisions,
            "games": self.games,
            "elapsed": self.elapsed,
            "config": asdict(self.cfg),
            "league": self.league.state(),
        }
        keep = self.cfg.checkpoint_keep_every
        if keep <= 0 or self.update % keep == 0 or self.update == self.cfg.total_updates:
            torch.save(ckpt, self.dir / "checkpoints" / f"update_{self.update:05d}.pt")
        # Write then rename, so an interruption mid-save can't leave a broken latest.pt.
        latest = self.dir / "checkpoints" / "latest.pt"
        tmp = latest.with_suffix(".tmp")
        torch.save(ckpt, tmp)
        os.replace(tmp, latest)

    def _load(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["optimizer"])
        self.update = ckpt["update"]
        self.decisions = ckpt["decisions"]
        self.games = ckpt["games"]
        self.elapsed = ckpt.get("elapsed", 0.0)

        def weights(p: str) -> dict:
            return torch.load(p, map_location="cpu", weights_only=False)["model"]

        self.league.load_state(ckpt["league"], weights)
        for j, sid in enumerate(self.league.slots):
            if sid is not None:
                snap = self.league.snapshot_by_id(sid)
                if snap is not None:
                    self.opponents[j + 1].load_state_dict(snap.state_dict)
        print(f"resumed from {path} at update {self.update}")

    def log(self, metrics: dict) -> None:
        metrics = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                   for k, v in metrics.items()}
        self.metrics_file.write(json.dumps(metrics) + "\n")
        self.metrics_file.flush()
        if self.tb is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    self.tb.add_scalar(k, v, self.update)

    # ------------------------------------------------------------------
    def run(self) -> None:
        c = self.cfg
        print(f"device={self.device}  envs={c.num_envs}  rollout={c.rollout_steps}  "
              f"stage={self.league.stage.name}")
        start = time.time() - self.elapsed
        while self.update < c.total_updates:
            t0 = time.time()
            self.env.set_rewards(shaping=self._shaping())
            roll = self.rollout()
            t1 = time.time()
            stats = self.learn()
            t2 = time.time()
            self.update += 1

            metrics = {
                "update": self.update,
                "decisions": self.decisions,
                "games": self.games,
                "elapsed_s": time.time() - start,
                "stage": self.league.stage.name,
                "win_rate": self.league.recent_win_rate(),
                "avg_vp": float(np.mean(self.recent_vp)) if self.recent_vp else 0.0,
                "avg_turns": float(np.mean(self.recent_turns)) if self.recent_turns else 0.0,
                "shaping": self._shaping(),
                "sps": c.num_envs * c.rollout_steps / max(t2 - t0, 1e-9),
                "rollout_s": t1 - t0,
                "learn_s": t2 - t1,
                **roll,
                **stats,
            }
            for k, v in self.league.beat_rate.items():
                metrics[f"beat/{'_'.join(map(str, k))}"] = v

            if self.league.maybe_advance(self.update):
                print(f"== update {self.update}: advancing to stage '{self.league.stage.name}'")
                metrics["stage_advanced"] = self.league.stage.name
            st = self.league.stage
            if st.snapshot_prob > 0 and (not self.league.snapshots or self.update % c.snapshot_every == 0):
                self._snapshot()

            if self.update % c.eval_every == 0:
                for opps in c.eval_opponents:
                    r = evaluate(self.net, opps, games=c.eval_games, num_envs=min(256, c.eval_games),
                                 device=self.device)
                    tag = "_".join(opps)
                    metrics[f"eval/{tag}/win_rate"] = r.win_rate
                    metrics[f"eval/{tag}/ci_low"] = r.ci_low
                    metrics[f"eval/{tag}/ci_high"] = r.ci_high
                    metrics[f"eval/{tag}/avg_vp"] = r.avg_vp
                    print(f"   eval: {r.line()}")
            self.log(metrics)
            print(f"upd {self.update:5d} | {st.name:>10} | win {metrics['win_rate']:.3f} "
                  f"vp {metrics['avg_vp']:.2f} turns {metrics['avg_turns']:.0f} | "
                  f"ent {stats.get('entropy', 0):.3f} kl {stats.get('approx_kl', 0):.4f} "
                  f"ev {stats.get('explained_var', 0):.2f} | {metrics['sps']:.0f} sps "
                  f"({metrics['rollout_s']:.1f}s+{metrics['learn_s']:.1f}s)")
            self.elapsed = metrics["elapsed_s"]
            if self.update % c.checkpoint_every == 0 or self.update == c.total_updates:
                self.save()
        self.metrics_file.close()
        if self.tb is not None:
            self.tb.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--updates", type=int, help="override total_updates")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.updates:
        cfg.total_updates = args.updates
    Trainer(cfg, resume=args.resume).run()


if __name__ == "__main__":
    main()

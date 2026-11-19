"""Tests for the Python side: bindings, environment, model, PPO pieces, league."""

import json

import numpy as np
import pytest
import torch

import catanbot as cb
from catanbot.env import CatanVecEnv, policy_code
from catanbot.league import League, Stage
from catanbot.model import PolicyNet
from catanbot.train import Buffer, compute_gae


def test_constants():
    assert cb.NUM_ACTIONS == 252
    assert cb.OBS_SIZE > 1000
    assert "ows" in cb.PRESET_NAMES


def test_tournament_columns_and_one_winner_per_game():
    d = cb.run_rotated(["ows", "road_builder", "dev_rush", "balanced"], [0, 1, 2, 3], 50)
    n = len(d["game_id"])
    assert n == 50 * 4 * 4  # seeds x rotations x seats
    games = d["game_id"].reshape(-1, 4)
    assert (games == games[:, :1]).all()
    wins = d["won"].reshape(-1, 4).sum(1)
    draws = d["draw"].reshape(-1, 4)[:, 0]
    assert ((wins == 1) | (draws == 1)).all()
    # Every bot plays every seat equally often (seat rotation).
    for b in range(4):
        seats = d["seat"][d["bot"] == b]
        assert np.bincount(seats, minlength=4).tolist() == [50] * 4


def test_same_seed_same_result():
    a = cb.run_rotated(["ows", "balanced", "dev_rush", "port_master"], [0, 1, 2, 3], 20, 99)
    b = cb.run_rotated(["ows", "balanced", "dev_rush", "port_master"], [0, 1, 2, 3], 20, 99)
    for k in ("won", "vp", "turns"):
        assert (a[k] == b[k]).all()


def test_custom_json_spec():
    spec = json.loads(dict(cb.presets())["ows"])
    spec["name"] = "ows_variant"
    spec["priority_city"] = 2.0
    assert cb.spec_name(json.dumps(spec)) == "ows_variant"
    d = cb.run_rotated([json.dumps(spec), "balanced", "dev_rush", "road_builder"], [0, 1, 2, 3], 5)
    assert d["bot_names"][0] == "ows_variant"


def test_play_replay_is_consistent():
    r = json.loads(cb.play_replay(["ows", "balanced", "dev_rush", "road_builder"], 7))
    assert r["frames"], "replay has frames"
    assert len(r["topology"]["node_xy"]) == 54
    last = r["frames"][-1]
    if r["winner"] is not None:
        assert last["vp"][r["winner"]] >= 10


def _random_legal(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.empty(len(mask), dtype=np.int64)
    for i, m in enumerate(mask):
        legal = np.flatnonzero(m)
        assert len(legal) >= 2, "forced moves should be auto-played"
        out[i] = rng.choice(legal)
    return out


def test_vec_env_runs_and_reports_games():
    specs = ["random", "balanced"]
    lineups = [[policy_code(0), 0, 1, 0]] * 8
    env = CatanVecEnv(specs, lineups, seed=3)
    env.set_rewards(shaping=0.01)
    step = env.reset()
    assert step.obs.shape == (8, cb.OBS_SIZE)
    assert step.mask.shape == (8, cb.NUM_ACTIONS)
    assert (step.policy == 0).all()
    rng = np.random.default_rng(0)
    finished = 0
    for _ in range(1500):
        step = env.step(_random_legal(step.mask, rng))
        for e in np.flatnonzero(step.done):
            finished += 1
            assert step.winner[e] >= -1
            assert policy_code(0) in step.lineup[e].tolist()
            res = env.last_result(int(e))
            assert len(res["vp"]) == 4
    assert finished > 0


def test_illegal_action_raises():
    env = CatanVecEnv(["random"], [[policy_code(0), 0, 0, 0]], seed=1)
    step = env.reset()
    illegal = int(np.flatnonzero(~step.mask[0])[0])
    with pytest.raises(ValueError):
        env.step(np.array([illegal]))


def test_lineup_needs_a_policy_seat():
    with pytest.raises(ValueError):
        CatanVecEnv(["random"], [[0, 0, 0, 0]])


def test_policy_never_samples_illegal_actions():
    net = PolicyNet(hidden=64, blocks=1)
    obs = torch.randn(32, cb.OBS_SIZE)
    mask = torch.zeros(32, cb.NUM_ACTIONS, dtype=torch.bool)
    for i in range(32):
        mask[i, torch.randperm(cb.NUM_ACTIONS)[:3]] = True
    for _ in range(20):
        a, logp, v = net.act(obs, mask)
        assert mask[torch.arange(32), a].all()
        assert torch.isfinite(logp).all() and torch.isfinite(v).all()


def test_graph_policy_shapes_and_masking():
    from catanbot.model import GraphPolicyNet, load_policy

    net = GraphPolicyNet(hidden=64, blocks=1, ent=16, rounds=1)
    assert sorted(net.action_order.tolist()) == list(range(cb.NUM_ACTIONS))
    env = CatanVecEnv(["balanced"], [[policy_code(0), 0, 0, 0]] * 6, seed=5)
    step = env.reset()
    obs = torch.from_numpy(step.obs)
    mask = torch.from_numpy(step.mask)
    logits, value = net(obs, mask)
    assert logits.shape == (6, cb.NUM_ACTIONS) and value.shape == (6,)
    for _ in range(10):
        a, logp, _ = net.act(obs, mask)
        assert mask[torch.arange(6), a].all()
    # Settlement logits during setup come from the per-node head: changing one
    # node's features must change that node's logit.
    lay = cb.layout()
    off, n, f = lay["nodes"]
    obs2 = obs.clone()
    obs2[:, off + 7 * f + 14] += 1.0  # "buildable" flag of node 7
    l1, _ = net(obs, torch.ones_like(mask))
    l2, _ = net(obs2, torch.ones_like(mask))
    sb = lay["settlement_base"]
    assert not torch.allclose(l1[:, sb + 7], l2[:, sb + 7])


def test_gae_follows_seat_trajectories():
    buf = Buffer(8, torch.device("cpu"))
    # Two interleaved trajectories: A = 0 -> 2 -> end, B = 1 -> 3 -> (cut)
    buf.n = 4
    buf.value[:4] = torch.tensor([0.5, 0.2, 0.6, 0.3])
    buf.reward[:4] = [0.0, 0.0, 1.0, 0.0]
    buf.next[:4] = [2, 3, -1, -2]
    adv, ret, valid = compute_gae(buf, gamma=1.0, lam=1.0)
    assert valid.tolist() == [True, True, True, False]
    assert adv[2] == pytest.approx(1.0 - 0.6)
    assert adv[0] == pytest.approx((0.6 - 0.5) + adv[2])
    assert adv[1] == pytest.approx(0.3 - 0.2)  # bootstraps from the cut transition's value
    assert ret[0] == pytest.approx(1.0)


def test_league_lineups_and_stats():
    names = list(cb.PRESET_NAMES)
    league = League(stages=[Stage(name="s", bots=["ows", "balanced"])],
                    bot_index={n: i for i, n in enumerate(names)}, seed=1)
    for _ in range(50):
        lineup = league.sample_lineup()
        assert lineup.count(policy_code(0)) == 1
        assert all(c == policy_code(0) or names[c] in ("ows", "balanced") for c in lineup)
    lineup = [policy_code(0), names.index("ows"), names.index("balanced"), names.index("ows")]
    for _ in range(100):
        league.record_game(lineup, [10, 5, 6, 7], winner=0)
    assert league.recent_win_rate() == 1.0
    assert league.beat_rate[("bot", "ows")] > 0.5

"""Policy/value network for the learning agent."""

from __future__ import annotations

import torch
from torch import nn

from catanbot import NUM_ACTIONS, OBS_SIZE

# Large negative (finite) logit for illegal actions: exp() underflows to 0
# without producing NaNs in log-softmax / entropy.
MASK_VALUE = -1e9


class ResBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class PolicyNet(nn.Module):
    """Residual MLP: observation -> masked action logits + state value.

    The value head predicts the expected return of the acting player
    (win = +1, loss = -1/3, plus any shaping), i.e. roughly its win chance.
    """

    def __init__(self, hidden: int = 512, blocks: int = 4, obs_size: int = OBS_SIZE,
                 num_actions: int = NUM_ACTIONS) -> None:
        super().__init__()
        self.hidden = hidden
        self.blocks_n = blocks
        self.embed = nn.Sequential(nn.Linear(obs_size, hidden), nn.ReLU())
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(blocks)])
        self.norm = nn.LayerNorm(hidden)
        self.policy_head = nn.Linear(hidden, num_actions)
        self.value_head = nn.Sequential(nn.Linear(hidden, 256), nn.ReLU(), nn.Linear(256, 1))
        # Small initial policy logits -> near-uniform initial policy.
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.norm(self.blocks(self.embed(obs)))
        logits = self.policy_head(h).float().masked_fill(~mask, MASK_VALUE)
        value = self.value_head(h).squeeze(-1).float()
        return logits, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor, mask: torch.Tensor, deterministic: bool = False):
        """Sample (or argmax) actions. Returns (actions, log_probs, values)."""
        logits, value = self(obs, mask)
        if deterministic:
            actions = logits.argmax(-1)
        else:
            actions = torch.distributions.Categorical(logits=logits).sample()
        logp = torch.log_softmax(logits, -1).gather(-1, actions[:, None]).squeeze(-1)
        return actions, logp, value

    def config(self) -> dict:
        return {"arch": "mlp", "hidden": self.hidden, "blocks": self.blocks_n}


def _mlp(i: int, o: int, hidden: int | None = None) -> nn.Sequential:
    h = hidden or o
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o), nn.LayerNorm(o))


class GraphPolicyNet(nn.Module):
    """Entity-based policy: the board as a graph of tiles, intersections and roads.

    A global residual MLP reads the whole observation (hands, phase, opponents).
    In parallel, every tile, intersection and road slot is embedded by a small
    network *shared* across the board; intersections exchange information
    with their neighbours (message passing), so an embedding can describe
    "a good spot two roads away". Settlement/city/road/robber logits are scored
    per entity by shared heads, so what the network learns about one spot
    transfers to every spot. The remaining actions (roll, trades, cards...)
    come from the global vector.
    """

    def __init__(self, hidden: int = 384, blocks: int = 3, ent: int = 64, rounds: int = 2) -> None:
        super().__init__()
        import json

        from catanbot import layout, topology_json

        self.hidden, self.blocks_n, self.ent, self.rounds = hidden, blocks, ent, rounds
        lay = layout()
        topo = json.loads(topology_json())
        self.tiles_sl = (lay["tiles"][0], lay["tiles"][1], lay["tiles"][2])
        self.nodes_sl = (lay["nodes"][0], lay["nodes"][1], lay["nodes"][2])
        self.edges_sl = (lay["edges"][0], lay["edges"][1], lay["edges"][2])
        nt, nn_, ne = self.tiles_sl[1], self.nodes_sl[1], self.edges_sl[1]

        def padded(rows: list[list[int]], pad: int) -> torch.Tensor:
            return torch.tensor([[pad if x == 255 else x for x in r] for r in rows], dtype=torch.long)

        # Index tensors; padding points at an extra all-zero row.
        self.register_buffer("node_tiles", padded(topo["node_tiles"], nt), persistent=False)
        self.register_buffer("node_nbrs", padded(topo["node_neighbors"], nn_), persistent=False)
        self.register_buffer("edge_nodes", torch.tensor(topo["edge_nodes"], dtype=torch.long), persistent=False)
        self.register_buffer("tile_nodes", torch.tensor(topo["tile_nodes"], dtype=torch.long), persistent=False)
        self.register_buffer("node_tile_cnt", (padded(topo["node_tiles"], nt) < nt).sum(1, keepdim=True).float(), persistent=False)
        self.register_buffer("node_nbr_cnt", (padded(topo["node_neighbors"], nn_) < nn_).sum(1, keepdim=True).float(), persistent=False)

        # Action layout: entity actions, and everything else from the global head.
        sb, cb, rb, ob = lay["settlement_base"], lay["city_base"], lay["road_base"], lay["robber_base"]
        ent_idx = list(range(sb, sb + nn_)) + list(range(cb, cb + nn_)) + list(range(rb, rb + ne)) + list(range(ob, ob + nt))
        other = [i for i in range(NUM_ACTIONS) if i not in set(ent_idx)]
        self.register_buffer("action_order", torch.tensor(other + ent_idx, dtype=torch.long), persistent=False)

        self.embed = nn.Sequential(nn.Linear(OBS_SIZE, hidden), nn.ReLU())
        self.trunk = nn.Sequential(*[ResBlock(hidden) for _ in range(blocks)])
        self.norm = nn.LayerNorm(hidden)
        self.g_to_ent = nn.Linear(hidden, ent)

        self.tile_enc = _mlp(self.tiles_sl[2], ent)
        self.node_enc = _mlp(self.nodes_sl[2] + ent, ent)
        self.mp = nn.ModuleList([_mlp(2 * ent, ent) for _ in range(rounds)])
        self.edge_enc = _mlp(self.edges_sl[2] + ent, ent)
        self.tile_out = _mlp(2 * ent, ent)

        def head() -> nn.Sequential:
            return nn.Sequential(nn.Linear(ent, ent), nn.ReLU(), nn.Linear(ent, 1))

        self.settle_head, self.city_head, self.road_head, self.robber_head = head(), head(), head(), head()
        self.other_head = nn.Linear(hidden, len(other))
        self.value_head = nn.Sequential(nn.Linear(hidden + ent, 256), nn.ReLU(), nn.Linear(256, 1))
        for h in (self.settle_head, self.city_head, self.road_head, self.robber_head):
            nn.init.orthogonal_(h[-1].weight, gain=0.01)
            nn.init.zeros_(h[-1].bias)
        nn.init.orthogonal_(self.other_head.weight, gain=0.01)
        nn.init.zeros_(self.other_head.bias)

    @staticmethod
    def _block(obs: torch.Tensor, sl: tuple[int, int, int]) -> torch.Tensor:
        off, n, f = sl
        return obs[:, off:off + n * f].reshape(obs.shape[0], n, f)

    @staticmethod
    def _pad(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, x.new_zeros(x.shape[0], 1, x.shape[2])], dim=1)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = obs.shape[0]
        g = self.norm(self.trunk(self.embed(obs)))
        gc = self.g_to_ent(g)[:, None, :]

        tiles = self.tile_enc(self._block(obs, self.tiles_sl))                       # [B,19,E]
        adj_tiles = self._pad(tiles)[:, self.node_tiles].sum(2) / self.node_tile_cnt  # [B,54,E]
        h = self.node_enc(torch.cat([self._block(obs, self.nodes_sl), adj_tiles], -1)) + gc
        for layer in self.mp:
            nbr = self._pad(h)[:, self.node_nbrs].sum(2) / self.node_nbr_cnt
            h = h + layer(torch.cat([h, nbr], -1))
        ends = h[:, self.edge_nodes].mean(2)                                           # [B,72,E]
        e = self.edge_enc(torch.cat([self._block(obs, self.edges_sl), ends], -1)) + gc
        corners = h[:, self.tile_nodes].mean(2)                                        # [B,19,E]
        t = self.tile_out(torch.cat([tiles, corners], -1)) + gc

        ent_logits = torch.cat([
            self.settle_head(h).squeeze(-1),
            self.city_head(h).squeeze(-1),
            self.road_head(e).squeeze(-1),
            self.robber_head(t).squeeze(-1),
        ], dim=1)
        # Float32 from here on (the mask value overflows fp16 under autocast).
        flat = torch.cat([self.other_head(g), ent_logits], dim=1).float()
        logits = flat.new_zeros(B, NUM_ACTIONS).index_copy(1, self.action_order, flat)
        logits = logits.masked_fill(~mask, MASK_VALUE)
        value = self.value_head(torch.cat([g, h.mean(1)], -1)).squeeze(-1).float()
        return logits, value

    act = PolicyNet.act

    def config(self) -> dict:
        return {"arch": "graph", "hidden": self.hidden, "blocks": self.blocks_n, "ent": self.ent,
                "rounds": self.rounds}


def make_policy(arch: str = "mlp", **kw) -> nn.Module:
    if arch == "graph":
        return GraphPolicyNet(**kw)
    if arch == "mlp":
        return PolicyNet(**kw)
    raise ValueError(f"unknown architecture {arch!r}")


def load_policy(path: str, device: str | torch.device = "cpu") -> nn.Module:
    """Load a policy from a training checkpoint or exported weights file."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("model_config", {"hidden": 512, "blocks": 4}))
    arch = cfg.pop("arch", "mlp")
    net = make_policy(arch, **cfg).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net

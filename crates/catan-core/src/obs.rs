//! Observation encoding for the learning agent.
//!
//! Everything is from the acting player's perspective: players are listed
//! in relative seat order (0 = me, 1 = next player, ...). Only information a
//! human player would have is included: own hand and dev cards, but just the
//! *counts* of opponents' cards.
//!
//! The board is encoded per tile, per node and per edge in fixed index order,
//! so position i of the vector always means the same place on the board.
//! Per-node resource pips are included explicitly because a flat MLP finds
//! node-tile adjacency hard to learn on its own.

use crate::board::{
    DESERT, NONE, NUM_EDGES, NUM_NODES, NUM_RESOURCES, NUM_TILES, PORT_GENERIC, pips, topology,
};
use crate::state::{GameState, NUM_DEV_TYPES, NUM_PLAYERS, Phase};

pub const TILE_FEATS: usize = 6 + 1 + 1; // resource one-hot (incl. desert), pips, robber
pub const NODE_FEATS: usize = 8 + 6 + 1 + 5; // owner x {settlement, city}, port type, buildable, resource pips
pub const EDGE_FEATS: usize = 4; // road owner (relative)
pub const PLAYER_FEATS: usize = 11 + 6 + 5; // public stats, ports, production
pub const SELF_FEATS: usize = 5 + 5 + 5 + 1 + 5; // hand, dev hand, new dev, total VP, trade rates
pub const GLOBAL_FEATS: usize = 5 + 1 + 9 + 5 + 3; // bank, dev deck, phase, flags, steal targets

/// Start of each block in the observation vector (blocks are contiguous, in this order).
pub const TILES_OFFSET: usize = 0;
pub const NODES_OFFSET: usize = TILES_OFFSET + NUM_TILES * TILE_FEATS;
pub const EDGES_OFFSET: usize = NODES_OFFSET + NUM_NODES * NODE_FEATS;
pub const PLAYERS_OFFSET: usize = EDGES_OFFSET + NUM_EDGES * EDGE_FEATS;
pub const SELF_OFFSET: usize = PLAYERS_OFFSET + NUM_PLAYERS * PLAYER_FEATS;
pub const GLOBAL_OFFSET: usize = SELF_OFFSET + SELF_FEATS;

pub const OBS_SIZE: usize = GLOBAL_OFFSET + GLOBAL_FEATS;

#[inline]
fn rel(owner: u8, me: u8) -> usize {
    ((owner + NUM_PLAYERS as u8 - me) % NUM_PLAYERS as u8) as usize
}

/// Encode `s` from the point of view of `s.acting` into `out` (length `OBS_SIZE`).
pub fn encode(s: &GameState, out: &mut [f32]) {
    assert_eq!(out.len(), OBS_SIZE);
    out.fill(0.0);
    let topo = topology();
    let me = s.acting;
    let mut i = 0;

    // Tiles.
    for t in 0..NUM_TILES {
        let res = s.board.tile_resource[t] as usize;
        out[i + res] = 1.0;
        out[i + 6] = pips(s.board.tile_number[t]) as f32 / 5.0;
        out[i + 7] = (s.robber as usize == t) as u8 as f32;
        i += TILE_FEATS;
    }

    // Nodes.
    for n in 0..NUM_NODES {
        let owner = s.node_owner[n];
        if owner != NONE {
            let k = rel(owner, me);
            out[i + if s.node_city[n] { 4 + k } else { k }] = 1.0;
        }
        let port = s.board.node_port[n];
        if port != NONE {
            out[i + 8 + port as usize] = 1.0;
        }
        out[i + 14] = s.node_buildable(n as u8) as u8 as f32;
        for &t in &topo.node_tiles[n] {
            if t == NONE {
                continue;
            }
            let res = s.board.tile_resource[t as usize];
            if res != DESERT {
                out[i + 15 + res as usize] += pips(s.board.tile_number[t as usize]) as f32 / 5.0;
            }
        }
        i += NODE_FEATS;
    }

    // Edges.
    for e in 0..NUM_EDGES {
        let owner = s.edge_owner[e];
        if owner != NONE {
            out[i + rel(owner, me)] = 1.0;
        }
        i += EDGE_FEATS;
    }

    // Players, in relative order.
    let prod = s.production_pips();
    for k in 0..NUM_PLAYERS {
        let p = ((me as usize + k) % NUM_PLAYERS) as u8;
        let pi = p as usize;
        let f = &mut out[i..i + PLAYER_FEATS];
        f[0] = s.public_vp(p) as f32 / 10.0;
        f[1] = s.hand_size(p) as f32 / 10.0;
        f[2] = s.dev_cards_held(p) as f32 / 5.0;
        f[3] = s.knights_played[pi] as f32 / 5.0;
        f[4] = s.road_length[pi] as f32 / 15.0;
        f[5] = (s.longest_road_owner == p) as u8 as f32;
        f[6] = (s.largest_army_owner == p) as u8 as f32;
        f[7] = s.settlements_left[pi] as f32 / 5.0;
        f[8] = s.cities_left[pi] as f32 / 4.0;
        f[9] = s.roads_left[pi] as f32 / 15.0;
        f[10] = (s.hand_size(p) > 7) as u8 as f32;
        for port in 0..=PORT_GENERIC as usize {
            f[11 + port] = (s.ports[pi] & (1 << port) != 0) as u8 as f32;
        }
        for r in 0..NUM_RESOURCES {
            f[17 + r] = prod[pi][r] as f32 / 20.0;
        }
        i += PLAYER_FEATS;
    }

    // My private information.
    let mi = me as usize;
    for r in 0..NUM_RESOURCES {
        out[i + r] = s.hands[mi][r] as f32 / 10.0;
    }
    for c in 0..NUM_DEV_TYPES {
        out[i + 5 + c] = s.dev_hand[mi][c] as f32 / 3.0;
        out[i + 10 + c] = s.dev_new[mi][c] as f32 / 3.0;
    }
    out[i + 15] = s.total_vp(me) as f32 / 10.0;
    for r in 0..NUM_RESOURCES {
        out[i + 16 + r] = (s.trade_rate(me, r as u8) - 2) as f32 / 2.0;
    }
    i += SELF_FEATS;

    // Global.
    for r in 0..NUM_RESOURCES {
        out[i + r] = s.bank[r] as f32 / 19.0;
    }
    out[i + 5] = s.dev_deck_remaining() as f32 / 25.0;
    out[i + 6 + s.phase.index()] = 1.0;
    let g = i + 6 + Phase::ALL.len();
    out[g] = s.has_rolled as u8 as f32;
    out[g + 1] = s.dev_played as u8 as f32;
    out[g + 2] = s.free_roads as f32 / 2.0;
    out[g + 3] = s.discard_left[mi] as f32 / 10.0;
    out[g + 4] = (s.turn as f32 / 200.0).min(1.0);
    for k in 1..NUM_PLAYERS {
        let p = ((me as usize + k) % NUM_PLAYERS) as u8;
        out[g + 5 + k - 1] = (s.steal_candidates & (1 << p) != 0) as u8 as f32;
    }
    i = g + 8;
    debug_assert_eq!(i, OBS_SIZE);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bots::BotSpec;
    use crate::sim::build_bots;

    #[test]
    fn encoding_fills_exactly_obs_size_and_is_finite() {
        let spec = BotSpec::preset("balanced").unwrap();
        let mut bots = build_bots(&[&spec; 4], 5);
        let mut s = GameState::new(5);
        let mut buf = Vec::new();
        let mut obs = vec![0.0f32; OBS_SIZE];
        while !s.is_over() {
            encode(&s, &mut obs);
            assert!(obs.iter().all(|x| x.is_finite() && *x >= 0.0 && *x <= 4.0));
            s.legal_actions(&mut buf);
            let a = bots[s.acting as usize].act(&s, &buf);
            s.apply(a);
        }
    }

    #[test]
    fn opponents_hidden_cards_do_not_leak() {
        // Two states that differ only in the *composition* of an opponent's
        // hand (same size) must encode identically for player 0.
        let mut a = GameState::new(9);
        a.phase = Phase::Main;
        a.hands[1] = [3, 0, 0, 0, 0];
        a.bank[0] -= 3;
        let mut b = GameState::new(9);
        b.phase = Phase::Main;
        b.hands[1] = [0, 0, 1, 1, 1];
        b.bank[2] -= 1;
        b.bank[3] -= 1;
        b.bank[4] -= 1;
        a.bank = b.bank; // bank differs in reality; neutralise it for this check
        a.dev_hand[1] = [1, 0, 0, 0, 0];
        b.dev_hand[1] = [0, 1, 0, 0, 0];
        let mut oa = vec![0.0; OBS_SIZE];
        let mut ob = vec![0.0; OBS_SIZE];
        encode(&a, &mut oa);
        encode(&b, &mut ob);
        assert_eq!(oa, ob);
    }
}

//! Actions and their fixed integer encoding (used for RL action masks).
//!
//! The steal target is encoded *relative* to the acting player (1 = next seat,
//! 2 = seat after that, 3 = previous seat) so that the network sees the same
//! action meaning regardless of which seat it occupies.

use serde::Serialize;

use crate::board::{NUM_EDGES, NUM_NODES, NUM_RESOURCES, NUM_TILES, RESOURCE_NAMES};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(tag = "type", content = "args")]
pub enum Action {
    Roll,
    EndTurn,
    BuildSettlement(u8),
    BuildCity(u8),
    BuildRoad(u8),
    BuyDev,
    PlayKnight,
    PlayRoadBuilding,
    /// Take two resources from the bank (a <= b).
    PlayYearOfPlenty(u8, u8),
    PlayMonopoly(u8),
    MoveRobber(u8),
    /// Steal from this (absolute) player.
    Steal(u8),
    /// Trade with the bank/harbour: give `rate` of the first resource for one of the second.
    MaritimeTrade(u8, u8),
    Discard(u8),
}

pub const ROLL: usize = 0;
pub const END_TURN: usize = 1;
pub const SETTLEMENT_BASE: usize = 2;
pub const CITY_BASE: usize = SETTLEMENT_BASE + NUM_NODES;
pub const ROAD_BASE: usize = CITY_BASE + NUM_NODES;
pub const BUY_DEV: usize = ROAD_BASE + NUM_EDGES;
pub const PLAY_KNIGHT: usize = BUY_DEV + 1;
pub const PLAY_ROAD_BUILDING: usize = PLAY_KNIGHT + 1;
pub const YOP_BASE: usize = PLAY_ROAD_BUILDING + 1;
pub const MONOPOLY_BASE: usize = YOP_BASE + 15;
pub const ROBBER_BASE: usize = MONOPOLY_BASE + NUM_RESOURCES;
pub const STEAL_BASE: usize = ROBBER_BASE + NUM_TILES;
pub const TRADE_BASE: usize = STEAL_BASE + 3;
pub const DISCARD_BASE: usize = TRADE_BASE + 20;
pub const NUM_ACTIONS: usize = DISCARD_BASE + NUM_RESOURCES;

/// The 15 unordered pairs (a <= b) of resources for Year of Plenty, in index order.
pub const YOP_PAIRS: [(u8, u8); 15] = {
    let mut out = [(0u8, 0u8); 15];
    let mut k = 0;
    let mut a = 0;
    while a < 5 {
        let mut b = a;
        while b < 5 {
            out[k] = (a as u8, b as u8);
            k += 1;
            b += 1;
        }
        a += 1;
    }
    out
};

#[inline]
fn yop_index(a: u8, b: u8) -> usize {
    let (a, b) = if a <= b { (a, b) } else { (b, a) };
    YOP_PAIRS.iter().position(|&p| p == (a, b)).unwrap()
}

impl Action {
    /// Encode to `0..NUM_ACTIONS`. `acting` is needed for the relative steal target.
    pub fn to_index(self, acting: u8) -> usize {
        match self {
            Action::Roll => ROLL,
            Action::EndTurn => END_TURN,
            Action::BuildSettlement(n) => SETTLEMENT_BASE + n as usize,
            Action::BuildCity(n) => CITY_BASE + n as usize,
            Action::BuildRoad(e) => ROAD_BASE + e as usize,
            Action::BuyDev => BUY_DEV,
            Action::PlayKnight => PLAY_KNIGHT,
            Action::PlayRoadBuilding => PLAY_ROAD_BUILDING,
            Action::PlayYearOfPlenty(a, b) => YOP_BASE + yop_index(a, b),
            Action::PlayMonopoly(r) => MONOPOLY_BASE + r as usize,
            Action::MoveRobber(t) => ROBBER_BASE + t as usize,
            Action::Steal(victim) => {
                let rel = (victim + 4 - acting) % 4;
                debug_assert!(rel != 0);
                STEAL_BASE + rel as usize - 1
            }
            Action::MaritimeTrade(give, get) => {
                let g = if get < give { get } else { get - 1 };
                TRADE_BASE + give as usize * 4 + g as usize
            }
            Action::Discard(r) => DISCARD_BASE + r as usize,
        }
    }

    pub fn from_index(idx: usize, acting: u8) -> Option<Action> {
        Some(match idx {
            ROLL => Action::Roll,
            END_TURN => Action::EndTurn,
            i if i < CITY_BASE => Action::BuildSettlement((i - SETTLEMENT_BASE) as u8),
            i if i < ROAD_BASE => Action::BuildCity((i - CITY_BASE) as u8),
            i if i < BUY_DEV => Action::BuildRoad((i - ROAD_BASE) as u8),
            BUY_DEV => Action::BuyDev,
            PLAY_KNIGHT => Action::PlayKnight,
            PLAY_ROAD_BUILDING => Action::PlayRoadBuilding,
            i if i < MONOPOLY_BASE => {
                let (a, b) = YOP_PAIRS[i - YOP_BASE];
                Action::PlayYearOfPlenty(a, b)
            }
            i if i < ROBBER_BASE => Action::PlayMonopoly((i - MONOPOLY_BASE) as u8),
            i if i < STEAL_BASE => Action::MoveRobber((i - ROBBER_BASE) as u8),
            i if i < TRADE_BASE => Action::Steal((acting + (i - STEAL_BASE) as u8 + 1) % 4),
            i if i < DISCARD_BASE => {
                let k = (i - TRADE_BASE) as u8;
                let give = k / 4;
                let g = k % 4;
                let get = if g < give { g } else { g + 1 };
                Action::MaritimeTrade(give, get)
            }
            i if i < NUM_ACTIONS => Action::Discard((i - DISCARD_BASE) as u8),
            _ => return None,
        })
    }

    /// Short human-readable description (used in replays).
    pub fn describe(self) -> String {
        let r = |x: u8| RESOURCE_NAMES[x as usize];
        match self {
            Action::Roll => "rolls the dice".into(),
            Action::EndTurn => "ends the turn".into(),
            Action::BuildSettlement(n) => format!("builds a settlement (node {n})"),
            Action::BuildCity(n) => format!("upgrades to a city (node {n})"),
            Action::BuildRoad(e) => format!("builds a road (edge {e})"),
            Action::BuyDev => "buys a development card".into(),
            Action::PlayKnight => "plays a Knight".into(),
            Action::PlayRoadBuilding => "plays Road Building".into(),
            Action::PlayYearOfPlenty(a, b) => format!("plays Year of Plenty ({} + {})", r(a), r(b)),
            Action::PlayMonopoly(x) => format!("plays Monopoly on {}", r(x)),
            Action::MoveRobber(t) => format!("moves the robber (tile {t})"),
            Action::Steal(p) => format!("steals from player {p}"),
            Action::MaritimeTrade(give, get) => format!("trades {} for {}", r(give), r(get)),
            Action::Discard(x) => format!("discards {}", r(x)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout() {
        assert_eq!(NUM_ACTIONS, 252);
    }

    #[test]
    fn index_roundtrip() {
        for acting in 0..4u8 {
            for i in 0..NUM_ACTIONS {
                let a = Action::from_index(i, acting).unwrap();
                assert_eq!(a.to_index(acting), i, "{a:?}");
                if let Action::Steal(v) = a {
                    assert_ne!(v, acting);
                }
                if let Action::MaritimeTrade(give, get) = a {
                    assert_ne!(give, get);
                }
            }
            assert!(Action::from_index(NUM_ACTIONS, acting).is_none());
        }
    }
}

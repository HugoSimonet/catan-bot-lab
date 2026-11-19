use super::Bot;
use super::heuristic::{node_pips, robber_hurts};
use crate::actions::Action;
use crate::board::{NONE, NUM_NODES, pips, topology};
use crate::rng::Rng;
use crate::state::{GameState, Phase};

/// Simple baseline: settles on the most pips, always builds the most VP it
/// can afford right now (city > settlement > dev card > road), never trades.
pub struct GreedyBot {
    rng: Rng,
}

impl GreedyBot {
    pub fn new(seed: u64) -> Self {
        GreedyBot {
            rng: Rng::new(seed),
        }
    }

    fn random(&mut self, legal: &[Action]) -> Action {
        legal[self.rng.below(legal.len() as u32) as usize]
    }

    fn best_by<F: Fn(&Action) -> Option<f32>>(&mut self, legal: &[Action], f: F) -> Option<Action> {
        let mut best = None;
        let mut best_v = f32::NEG_INFINITY;
        for a in legal {
            if let Some(v) = f(a) {
                let v = v + self.rng.unit() * 1e-3;
                if v > best_v {
                    best_v = v;
                    best = Some(*a);
                }
            }
        }
        best
    }
}

impl Bot for GreedyBot {
    fn act(&mut self, s: &GameState, legal: &[Action]) -> Action {
        let p = s.acting;
        match s.phase {
            Phase::SetupSettlement | Phase::Main => {}
            Phase::PreRoll => {
                if legal.contains(&Action::PlayKnight) && robber_hurts(s, p) {
                    return Action::PlayKnight;
                }
                return Action::Roll;
            }
            Phase::MoveRobber => {
                let topo = topology();
                return self
                    .best_by(legal, |a| match *a {
                        Action::MoveRobber(t) => {
                            let mut v = 0.0;
                            for &n in &topo.tile_nodes[t as usize] {
                                let o = s.node_owner[n as usize];
                                if o == p {
                                    v -= 100.0;
                                } else if o != NONE {
                                    v += pips(s.board.tile_number[t as usize]) as f32;
                                }
                            }
                            Some(v)
                        }
                        _ => None,
                    })
                    .unwrap();
            }
            _ => return self.random(legal),
        }

        if s.phase == Phase::SetupSettlement {
            return self
                .best_by(legal, |a| match *a {
                    Action::BuildSettlement(n) => Some(node_pips(s, n) as f32),
                    _ => None,
                })
                .unwrap();
        }

        // Main phase.
        if let Some(a) = self.best_by(legal, |a| match *a {
            Action::BuildCity(n) => Some(node_pips(s, n) as f32),
            _ => None,
        }) {
            return a;
        }
        if let Some(a) = self.best_by(legal, |a| match *a {
            Action::BuildSettlement(n) => Some(node_pips(s, n) as f32),
            _ => None,
        }) {
            return a;
        }
        if legal.contains(&Action::BuyDev) {
            return Action::BuyDev;
        }
        if legal.contains(&Action::PlayKnight) && robber_hurts(s, p) {
            return Action::PlayKnight;
        }
        let has_spot = (0..NUM_NODES as u8).any(|n| s.can_build_settlement_at(p, n));
        if !has_spot && s.settlements_left[p as usize] > 0 {
            let roads: Vec<Action> = legal
                .iter()
                .copied()
                .filter(|a| matches!(a, Action::BuildRoad(_)))
                .collect();
            if !roads.is_empty() {
                return self.random(&roads);
            }
        }
        Action::EndTurn
    }
}

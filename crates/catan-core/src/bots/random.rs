use super::Bot;
use crate::actions::Action;
use crate::rng::Rng;
use crate::state::GameState;

/// Picks uniformly among legal actions. The weakest possible baseline.
pub struct RandomBot {
    rng: Rng,
}

impl RandomBot {
    pub fn new(seed: u64) -> Self {
        RandomBot {
            rng: Rng::new(seed),
        }
    }
}

impl Bot for RandomBot {
    fn act(&mut self, _state: &GameState, legal: &[Action]) -> Action {
        legal[self.rng.below(legal.len() as u32) as usize]
    }
}

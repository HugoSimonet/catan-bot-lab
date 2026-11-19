//! Bots: a uniform random baseline, a simple greedy baseline, and a
//! parameterised heuristic bot whose presets encode the community strategies.
//!
//! Bots only read public information plus their own hand and dev cards; they
//! never look at opponents' hidden cards, so they play under the same
//! information as the learning agent.

pub mod greedy;
pub mod heuristic;
pub mod params;
pub mod random;

use serde::{Deserialize, Serialize};

pub use params::StrategyParams;

use crate::actions::Action;
use crate::state::GameState;

pub trait Bot: Send {
    /// Choose one of `legal` (never empty) for `state.acting`.
    fn act(&mut self, state: &GameState, legal: &[Action]) -> Action;
}

/// Serializable description of a bot, used by the CLI, tournaments and the
/// Python bindings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BotSpec {
    Random,
    Greedy,
    Heuristic(StrategyParams),
}

pub const PRESET_NAMES: [&str; 7] = [
    "random",
    "greedy",
    "balanced",
    "ows",
    "road_builder",
    "dev_rush",
    "port_master",
];

impl BotSpec {
    pub fn name(&self) -> String {
        match self {
            BotSpec::Random => "random".into(),
            BotSpec::Greedy => "greedy".into(),
            BotSpec::Heuristic(p) => p.name.clone(),
        }
    }

    /// A named preset, or `None` if the name is unknown.
    pub fn preset(name: &str) -> Option<BotSpec> {
        match name {
            "random" => Some(BotSpec::Random),
            "greedy" => Some(BotSpec::Greedy),
            other => StrategyParams::preset(other).map(BotSpec::Heuristic),
        }
    }

    /// Parse a preset name, or a JSON bot spec (`{"kind": "heuristic", ...}`).
    pub fn parse(text: &str) -> Result<BotSpec, String> {
        let t = text.trim();
        if t.starts_with('{') {
            serde_json::from_str(t).map_err(|e| format!("invalid bot spec JSON: {e}"))
        } else {
            BotSpec::preset(t)
                .ok_or_else(|| format!("unknown bot '{t}' (presets: {})", PRESET_NAMES.join(", ")))
        }
    }

    pub fn build(&self, seed: u64) -> Box<dyn Bot> {
        match self {
            BotSpec::Random => Box::new(random::RandomBot::new(seed)),
            BotSpec::Greedy => Box::new(greedy::GreedyBot::new(seed)),
            BotSpec::Heuristic(p) => Box::new(heuristic::HeuristicBot::new(p.clone(), seed)),
        }
    }
}

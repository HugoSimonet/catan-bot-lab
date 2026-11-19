//! Strategy parameters for the heuristic bot, and the named presets.
//!
//! Resource order everywhere: [wood, brick, sheep, wheat, ore].
//!
//! The presets below are the *tuned* versions: each started from a
//! hand-written parameter set (kept in `configs/presets_handwritten.json`)
//! and was improved by coordinate ascent (`python -m catanbot.sweep tune`)
//! against a field of the other strategies. Each strategy's identity
//! parameters - its resource weights and the drives that define it - were held
//! fixed or bounded during tuning (see `TUNABLE` in `python/catanbot/sweep.py`).

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StrategyParams {
    pub name: String,

    // --- Where to settle -------------------------------------------------
    /// Multiplier on the pips of each resource when scoring a spot.
    pub resource_weights: [f32; 5],
    /// Bonus per resource type a spot adds that the player doesn't produce yet.
    pub diversity_weight: f32,
    /// Bonus for concentrating on resources the player already produces.
    pub specialization_weight: f32,
    /// Value of a 3:1 harbour.
    pub port_generic_weight: f32,
    /// Value of a 2:1 harbour, per pip of matching production.
    pub port_specific_weight: f32,

    // --- What to build ---------------------------------------------------
    pub priority_city: f32,
    pub priority_settlement: f32,
    pub priority_road: f32,
    pub priority_dev: f32,
    /// How hard to chase Longest Road.
    pub longest_road_weight: f32,
    /// How hard to chase Largest Army.
    pub largest_army_weight: f32,
    /// Tendency to play knights even when the robber isn't hurting us.
    pub knight_aggression: f32,

    // --- Interaction -----------------------------------------------------
    /// How much the robber targets the leader (vs. just the best tile).
    pub robber_leader_weight: f32,
    /// 0 = never trade with the bank; 1 = trade up to 3 times to finish a build.
    pub trade_aggression: f32,
    /// 0 = happy to hold big hands; 1 = spend/trade to stay at 7 cards or fewer.
    pub discard_avoidance: f32,
}

impl Default for StrategyParams {
    fn default() -> Self {
        StrategyParams::balanced()
    }
}

impl StrategyParams {
    pub fn preset(name: &str) -> Option<StrategyParams> {
        Some(match name {
            "balanced" => Self::balanced(),
            "ows" => Self::ows(),
            "road_builder" => Self::road_builder(),
            "dev_rush" => Self::dev_rush(),
            "port_master" => Self::port_master(),
            _ => return None,
        })
    }

    /// Maximise total production and resource diversity; build whatever is best.
    pub fn balanced() -> Self {
        StrategyParams {
            name: "balanced".into(),
            resource_weights: [1.0, 1.0, 0.9, 1.05, 1.05],
            diversity_weight: 1.875,
            specialization_weight: 0.0,
            port_generic_weight: 1.5,
            port_specific_weight: 0.225,
            priority_city: 0.8,
            priority_settlement: 1.1,
            priority_road: 0.875,
            priority_dev: 0.5,
            longest_road_weight: 0.3,
            largest_army_weight: 0.3,
            knight_aggression: 0.3,
            robber_leader_weight: 1.0,
            trade_aggression: 0.67,
            discard_avoidance: 1.0,
        }
    }

    /// Ore-Wheat-Sheep: settle on ore and wheat, rush cities, then dev cards.
    pub fn ows() -> Self {
        StrategyParams {
            name: "ows".into(),
            resource_weights: [0.55, 0.55, 1.0, 1.45, 1.5],
            diversity_weight: 1.125,
            specialization_weight: 0.0,
            port_generic_weight: 0.8,
            port_specific_weight: 0.225,
            priority_city: 1.4,
            priority_settlement: 1.406,
            priority_road: 0.525,
            priority_dev: 0.88,
            longest_road_weight: 0.0,
            largest_army_weight: 1.5,
            knight_aggression: 0.5,
            robber_leader_weight: 1.0,
            trade_aggression: 0.67,
            discard_avoidance: 1.0,
        }
    }

    /// Wood-Brick expansion: grab spots early, chase Longest Road.
    pub fn road_builder() -> Self {
        StrategyParams {
            name: "road_builder".into(),
            resource_weights: [1.5, 1.5, 0.9, 0.85, 0.5],
            diversity_weight: 1.8,
            specialization_weight: 0.0,
            port_generic_weight: 1.0,
            port_specific_weight: 0.338,
            priority_city: 0.9,
            priority_settlement: 1.5,
            priority_road: 0.8,
            priority_dev: 0.25,
            longest_road_weight: 0.9,
            largest_army_weight: 0.0,
            knight_aggression: 0.2,
            robber_leader_weight: 1.0,
            trade_aggression: 0.67,
            discard_avoidance: 1.0,
        }
    }

    /// Development-card rush: buy lots of dev cards, take Largest Army.
    pub fn dev_rush() -> Self {
        StrategyParams {
            name: "dev_rush".into(),
            resource_weights: [0.6, 0.6, 1.35, 1.3, 1.35],
            diversity_weight: 0.6,
            specialization_weight: 0.0,
            port_generic_weight: 0.8,
            port_specific_weight: 0.225,
            priority_city: 1.5,
            priority_settlement: 1.5,
            priority_road: 0.3,
            priority_dev: 1.5,
            longest_road_weight: 0.0,
            largest_army_weight: 1.0,
            knight_aggression: 1.5,
            robber_leader_weight: 1.2,
            trade_aggression: 0.67,
            discard_avoidance: 1.0,
        }
    }

    /// Harbour strategy: concentrate on one resource and trade it at 2:1.
    pub fn port_master() -> Self {
        StrategyParams {
            name: "port_master".into(),
            resource_weights: [1.0, 1.0, 1.0, 1.0, 1.0],
            diversity_weight: 0.45,
            specialization_weight: 0.8,
            port_generic_weight: 2.0,
            port_specific_weight: 0.7,
            priority_city: 1.0,
            priority_settlement: 1.0,
            priority_road: 0.8,
            priority_dev: 0.5,
            longest_road_weight: 0.2,
            largest_army_weight: 0.2,
            knight_aggression: 0.3,
            robber_leader_weight: 1.0,
            trade_aggression: 1.0,
            discard_avoidance: 1.0,
        }
    }
}

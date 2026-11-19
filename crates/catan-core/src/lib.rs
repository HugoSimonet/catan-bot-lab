//! catan-core: a fast Settlers of Catan engine (base game, 4 players,
//! bank/harbour trading only), strategy bots and a parallel tournament runner.

pub mod actions;
pub mod board;
pub mod bots;
pub mod env;
pub mod obs;
pub mod replay;
pub mod rng;
pub mod sim;
pub mod state;

pub use actions::{Action, NUM_ACTIONS};
pub use board::{Board, topology};
pub use bots::{Bot, BotSpec, StrategyParams};
pub use state::{GameState, NUM_PLAYERS, Phase};

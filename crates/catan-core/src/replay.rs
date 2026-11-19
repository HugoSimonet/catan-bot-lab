//! Replay export for the web viewer.
//!
//! A game is fully determined by its seed and its action list, so a replay is
//! rebuilt by re-applying the actions. The JSON contains the board, the
//! topology needed to draw it, and one frame per action with the public
//! state the viewer shows (hands are included: replays are omniscient).

use serde::Serialize;

use crate::actions::Action;
use crate::board::{Board, NONE, RESOURCE_NAMES, Topology, topology};
use crate::state::{GameState, NUM_DEV_TYPES, NUM_PLAYERS, Phase};

#[derive(Serialize)]
pub struct Replay<'a> {
    pub version: u32,
    pub seed: u64,
    pub players: Vec<String>,
    pub topology: &'a Topology,
    pub board: Board,
    pub robber_start: u8,
    pub winner: Option<u8>,
    pub turns: u16,
    pub frames: Vec<Frame>,
}

#[derive(Serialize)]
pub struct Frame {
    /// Player-turn number.
    pub turn: u16,
    pub actor: u8,
    pub action: Action,
    pub text: String,
    /// Dice result, for Roll actions.
    pub roll: Option<u8>,
    pub robber: u8,
    pub phase: Phase,
    pub hands: [[u8; 5]; NUM_PLAYERS],
    pub dev_cards: [[u8; NUM_DEV_TYPES]; NUM_PLAYERS],
    pub vp: [u8; NUM_PLAYERS],
    pub knights: [u8; NUM_PLAYERS],
    pub road_length: [u8; NUM_PLAYERS],
    pub longest_road: Option<u8>,
    pub largest_army: Option<u8>,
}

fn tile_label(s: &GameState, t: u8) -> String {
    let res = RESOURCE_NAMES[s.board.tile_resource[t as usize] as usize];
    match s.board.tile_number[t as usize] {
        0 => res.to_string(),
        n => format!("{res} {n}"),
    }
}

fn describe(before: &GameState, after: &GameState, action: Action) -> String {
    match action {
        Action::Roll => format!("rolls {}", after.last_roll),
        Action::MoveRobber(t) => format!("moves the robber to {}", tile_label(before, t)),
        Action::Steal(v) => format!("steals a card from P{}", v + 1),
        Action::BuildSettlement(_) if before.is_setup() => "places a settlement".into(),
        Action::BuildRoad(_) if before.is_setup() => "places a road".into(),
        Action::BuildRoad(_) if before.phase == Phase::RoadBuilding => "places a free road".into(),
        Action::BuildSettlement(_) => "builds a settlement".into(),
        Action::BuildCity(_) => "builds a city".into(),
        Action::BuildRoad(_) => "builds a road".into(),
        other => other.describe(),
    }
}

/// Re-play `actions` (absolute actions in order) on `seed` and build the replay.
/// Returns an error if an action is illegal at its point in the game.
pub fn build_replay(
    seed: u64,
    players: Vec<String>,
    actions: &[Action],
) -> Result<Replay<'static>, String> {
    let mut state = GameState::new(seed);
    let robber_start = state.robber;
    let mut frames = Vec::with_capacity(actions.len());
    for (i, &action) in actions.iter().enumerate() {
        if state.is_over() {
            return Err(format!("action {i} given after the game ended"));
        }
        if !state.is_legal(action) {
            return Err(format!("action {i} ({action:?}) is illegal"));
        }
        let before = state;
        let actor = state.acting;
        state.apply(action);
        let opt = |x: u8| if x == NONE { None } else { Some(x) };
        frames.push(Frame {
            turn: before.turn,
            actor,
            action,
            text: describe(&before, &state, action),
            roll: matches!(action, Action::Roll).then_some(state.last_roll),
            robber: state.robber,
            phase: state.phase,
            hands: state.hands,
            dev_cards: std::array::from_fn(|p| {
                std::array::from_fn(|c| state.dev_hand[p][c] + state.dev_new[p][c])
            }),
            vp: std::array::from_fn(|p| state.total_vp(p as u8)),
            knights: state.knights_played,
            road_length: state.road_length,
            longest_road: opt(state.longest_road_owner),
            largest_army: opt(state.largest_army_owner),
        });
    }
    Ok(Replay {
        version: 1,
        seed,
        players,
        topology: topology(),
        board: GameState::new(seed).board,
        robber_start,
        winner: (state.winner != NONE).then_some(state.winner),
        turns: state.turn,
        frames,
    })
}

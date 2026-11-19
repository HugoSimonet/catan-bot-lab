//! Playing games between bots, and parallel tournaments.
//!
//! Seat rotation and common random numbers: a "table" of 4 bots is played on
//! the same seed once per rotation, so every bot plays every seat on exactly
//! the same board and dice sequence. That removes most of the luck from
//! head-to-head comparisons.

use rayon::prelude::*;
use serde::Serialize;

use crate::actions::Action;
use crate::board::NONE;
use crate::bots::{Bot, BotSpec};
use crate::rng::Rng;
use crate::state::{GameState, NUM_PLAYERS, Stats};

/// Seed for the bot in `seat` of game `seed` (independent of the game's streams).
pub fn bot_seed(seed: u64, seat: usize) -> u64 {
    Rng::stream(seed, 100 + seat as u64).next_u64()
}

pub fn build_bots(specs: &[&BotSpec; NUM_PLAYERS], seed: u64) -> Vec<Box<dyn Bot>> {
    (0..NUM_PLAYERS)
        .map(|seat| specs[seat].build(bot_seed(seed, seat)))
        .collect()
}

/// Play one game to the end. If `log` is given, every (actor, action) is appended.
pub fn play_game(
    seed: u64,
    bots: &mut [Box<dyn Bot>],
    mut log: Option<&mut Vec<(u8, Action)>>,
) -> GameState {
    let mut state = GameState::new(seed);
    let mut legal = Vec::with_capacity(128);
    while !state.is_over() {
        state.legal_actions(&mut legal);
        let actor = state.acting;
        let action = if legal.len() == 1 {
            legal[0]
        } else {
            bots[actor as usize].act(&state, &legal)
        };
        debug_assert!(
            legal.contains(&action),
            "bot chose illegal action {action:?}"
        );
        if let Some(l) = log.as_deref_mut() {
            l.push((actor, action));
        }
        state.apply(action);
    }
    state
}

#[derive(Clone, Debug, Serialize)]
pub struct GameResult {
    pub seed: u64,
    /// Winning seat, or `None` for a draw (turn limit).
    pub winner: Option<u8>,
    pub turns: u16,
    pub num_actions: u32,
    pub vp: [u8; NUM_PLAYERS],
    pub settlements: [u8; NUM_PLAYERS],
    pub cities: [u8; NUM_PLAYERS],
    pub roads: [u8; NUM_PLAYERS],
    pub knights: [u8; NUM_PLAYERS],
    pub vp_cards: [u8; NUM_PLAYERS],
    pub longest_road: Option<u8>,
    pub largest_army: Option<u8>,
    pub stats: Stats,
}

impl GameResult {
    pub fn from_state(s: &GameState) -> Self {
        let per =
            |f: &dyn Fn(u8) -> u8| -> [u8; NUM_PLAYERS] { std::array::from_fn(|p| f(p as u8)) };
        let opt = |x: u8| if x == NONE { None } else { Some(x) };
        GameResult {
            seed: s.seed,
            winner: opt(s.winner),
            turns: s.turn,
            num_actions: s.num_actions,
            vp: per(&|p| s.total_vp(p)),
            settlements: per(&|p| s.settlements_built(p)),
            cities: per(&|p| s.cities_built(p)),
            roads: per(&|p| s.roads_built(p)),
            knights: per(&|p| s.knights_played[p as usize]),
            vp_cards: per(&|p| s.vp_cards(p)),
            longest_road: opt(s.longest_road_owner),
            largest_army: opt(s.largest_army_owner),
            stats: s.stats,
        }
    }
}

/// One game of a tournament: which bot (index into the spec list) sat where.
#[derive(Clone, Debug, Serialize)]
pub struct TournamentGame {
    pub game_id: u64,
    pub seats: [u16; NUM_PLAYERS],
    pub result: GameResult,
}

/// A planned game: seed + bot index per seat.
pub type Table = (u64, [u16; NUM_PLAYERS]);

/// Every rotation of `lineup` on each of `num_seeds` seeds (4 games per seed).
pub fn rotated_tables(lineup: [u16; NUM_PLAYERS], num_seeds: u64, base_seed: u64) -> Vec<Table> {
    let mut out = Vec::with_capacity(num_seeds as usize * NUM_PLAYERS);
    for i in 0..num_seeds {
        let seed = base_seed.wrapping_add(i);
        for rot in 0..NUM_PLAYERS {
            let seats = std::array::from_fn(|s| lineup[(s + rot) % NUM_PLAYERS]);
            out.push((seed, seats));
        }
    }
    out
}

/// Random 4-bot tables drawn from a pool of `pool_size` bots, each played in
/// all 4 rotations. `distinct` forbids the same bot twice at one table.
pub fn pool_tables(pool_size: u16, num_seeds: u64, base_seed: u64, distinct: bool) -> Vec<Table> {
    assert!(pool_size > 0);
    assert!(
        !distinct || pool_size as usize >= NUM_PLAYERS,
        "need at least 4 bots for distinct tables"
    );
    let mut rng = Rng::stream(base_seed, 7);
    let mut out = Vec::with_capacity(num_seeds as usize * NUM_PLAYERS);
    for i in 0..num_seeds {
        let mut lineup = [0u16; NUM_PLAYERS];
        let mut k = 0;
        while k < NUM_PLAYERS {
            let b = rng.below(pool_size as u32) as u16;
            if distinct && lineup[..k].contains(&b) {
                continue;
            }
            lineup[k] = b;
            k += 1;
        }
        out.extend(rotated_tables(lineup, 1, base_seed.wrapping_add(i)));
    }
    out
}

/// Play all tables in parallel (on the current rayon pool).
pub fn run_tables(specs: &[BotSpec], tables: &[Table]) -> Vec<TournamentGame> {
    tables
        .par_iter()
        .enumerate()
        .map(|(i, &(seed, seats))| {
            let lineup: [&BotSpec; NUM_PLAYERS] =
                std::array::from_fn(|s| &specs[seats[s] as usize]);
            let mut bots = build_bots(&lineup, seed);
            let state = play_game(seed, &mut bots, None);
            TournamentGame {
                game_id: i as u64,
                seats,
                result: GameResult::from_state(&state),
            }
        })
        .collect()
}

/// Wilson score interval for a binomial proportion (95% by default with z = 1.96).
pub fn wilson_interval(successes: u64, n: u64, z: f64) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let n_f = n as f64;
    let p = successes as f64 / n_f;
    let denom = 1.0 + z * z / n_f;
    let centre = (p + z * z / (2.0 * n_f)) / denom;
    let half = z * ((p * (1.0 - p) / n_f + z * z / (4.0 * n_f * n_f)).sqrt()) / denom;
    ((centre - half).max(0.0), (centre + half).min(1.0))
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct BotSummary {
    pub name: String,
    pub games: u64,
    pub wins: u64,
    pub draws: u64,
    pub win_rate: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub avg_vp: f64,
    pub avg_turns_in_wins: f64,
    /// Share of this bot's wins in which it held Longest Road / Largest Army.
    pub wins_with_longest_road: f64,
    pub wins_with_largest_army: f64,
    pub avg_cities: f64,
    pub avg_settlements: f64,
    pub avg_dev_bought: f64,
    pub avg_trades: f64,
    /// Win rate by seat (0 = first player).
    pub seat_win_rate: [f64; NUM_PLAYERS],
}

/// Aggregate per-bot statistics (each seat appearance counts as one game).
pub fn summarize(specs: &[BotSpec], games: &[TournamentGame]) -> Vec<BotSummary> {
    #[derive(Default, Clone)]
    struct Acc {
        games: u64,
        wins: u64,
        draws: u64,
        vp: u64,
        win_turns: u64,
        win_lr: u64,
        win_la: u64,
        cities: u64,
        settlements: u64,
        dev: u64,
        trades: u64,
        seat_games: [u64; NUM_PLAYERS],
        seat_wins: [u64; NUM_PLAYERS],
    }
    let mut acc = vec![Acc::default(); specs.len()];
    for g in games {
        let r = &g.result;
        for seat in 0..NUM_PLAYERS {
            let a = &mut acc[g.seats[seat] as usize];
            a.games += 1;
            a.seat_games[seat] += 1;
            a.vp += r.vp[seat] as u64;
            a.cities += r.cities[seat] as u64;
            a.settlements += r.settlements[seat] as u64;
            a.dev += r.stats.dev_bought[seat] as u64;
            a.trades += r.stats.maritime_trades[seat] as u64;
            match r.winner {
                None => a.draws += 1,
                Some(w) if w as usize == seat => {
                    a.wins += 1;
                    a.seat_wins[seat] += 1;
                    a.win_turns += r.turns as u64;
                    if r.longest_road == Some(w) {
                        a.win_lr += 1;
                    }
                    if r.largest_army == Some(w) {
                        a.win_la += 1;
                    }
                }
                _ => {}
            }
        }
    }
    acc.iter()
        .zip(specs)
        .map(|(a, spec)| {
            let n = a.games.max(1) as f64;
            let w = a.wins.max(1) as f64;
            let (lo, hi) = wilson_interval(a.wins, a.games, 1.96);
            BotSummary {
                name: spec.name(),
                games: a.games,
                wins: a.wins,
                draws: a.draws,
                win_rate: a.wins as f64 / n,
                ci_low: lo,
                ci_high: hi,
                avg_vp: a.vp as f64 / n,
                avg_turns_in_wins: a.win_turns as f64 / w,
                wins_with_longest_road: a.win_lr as f64 / w,
                wins_with_largest_army: a.win_la as f64 / w,
                avg_cities: a.cities as f64 / n,
                avg_settlements: a.settlements as f64 / n,
                avg_dev_bought: a.dev as f64 / n,
                avg_trades: a.trades as f64 / n,
                seat_win_rate: std::array::from_fn(|s| {
                    a.seat_wins[s] as f64 / a.seat_games[s].max(1) as f64
                }),
            }
        })
        .collect()
}

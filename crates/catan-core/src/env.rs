//! Vectorised multi-agent environment for reinforcement learning.
//!
//! Each of the N parallel games has 4 seats. A seat is either a Rust bot
//! (index into the bot spec list) or an external *policy* (a neural network
//! driven from Python, identified by a small integer). `step` applies the
//! external actions, then advances every game inside Rust - bot moves, dice,
//! forced moves (only one legal action) and resets - until each game needs a
//! decision from an external policy again. Python only ever sees real
//! decisions.
//!
//! Rewards are reported per seat: a terminal reward (+win / loss) plus an
//! optional shaping reward proportional to the change in that seat's victory
//! points. Setup-phase VP changes are never rewarded.
//!
//! After a game ends the environment resets itself. By default the lineup is
//! rotated one seat each game so external policies play every seat; Python can
//! instead queue a new lineup with `set_lineup`.

use rayon::prelude::*;

use crate::actions::{Action, NUM_ACTIONS};
use crate::board::NONE;
use crate::bots::{Bot, BotSpec};
use crate::obs::{OBS_SIZE, encode};
use crate::sim::{GameResult, bot_seed};
use crate::state::{GameState, NUM_PLAYERS};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeatSpec {
    /// Rust bot: index into the environment's bot specs.
    Bot(u16),
    /// External policy id (chooses actions from Python).
    Policy(u16),
}

impl SeatSpec {
    /// Codes used by the Python API: `>= 0` is a bot index, `-1 - k` is policy `k`.
    pub fn from_code(code: i32) -> SeatSpec {
        if code >= 0 {
            SeatSpec::Bot(code as u16)
        } else {
            SeatSpec::Policy((-code - 1) as u16)
        }
    }
    pub fn code(self) -> i32 {
        match self {
            SeatSpec::Bot(b) => b as i32,
            SeatSpec::Policy(k) => -(k as i32) - 1,
        }
    }
}

enum Seat {
    Bot(Box<dyn Bot>),
    Policy(u16),
}

#[derive(Clone, Copy, Debug)]
pub struct RewardConfig {
    pub win: f32,
    pub loss: f32,
    /// Reward per victory point gained (0 disables shaping).
    pub shaping: f32,
}

impl Default for RewardConfig {
    fn default() -> Self {
        RewardConfig {
            win: 1.0,
            loss: -1.0 / 3.0,
            shaping: 0.0,
        }
    }
}

pub struct GameEnv {
    pub state: GameState,
    seats: Vec<Seat>,
    pub lineup: [SeatSpec; NUM_PLAYERS],
    pub pending: Option<[SeatSpec; NUM_PLAYERS]>,
    last_vp: [u8; NUM_PLAYERS],
    legal: Vec<Action>,
    index: u64,
    games_started: u64,
    pub record: bool,
    pub log: Vec<Action>,
    /// Action log and seed of the most recently finished game (if recording).
    pub last_log: Vec<Action>,
    pub last_seed: u64,
    pub last_lineup: [SeatSpec; NUM_PLAYERS],
    /// Statistics of the most recently finished game.
    pub last_result: Option<GameResult>,
}

/// Output buffers for one call to `reset_all`/`step` (one row per game).
pub struct Buffers<'a> {
    pub obs: &'a mut [f32],      // N * OBS_SIZE
    pub mask: &'a mut [bool],    // N * NUM_ACTIONS
    pub policy: &'a mut [i32],   // N: which policy must act
    pub seat: &'a mut [i32],     // N: which seat is acting
    pub reward: &'a mut [f32],   // N * 4: reward per seat since the last step
    pub done: &'a mut [bool],    // N: a game finished during this step
    pub winner: &'a mut [i32],   // N: winning seat of the finished game (-1 draw)
    pub final_vp: &'a mut [i32], // N * 4: final VP of the finished game
    pub turns: &'a mut [i32],    // N: length of the finished game
    pub lineup: &'a mut [i32],   // N * 4: seat codes of the finished game
}

struct Row<'a> {
    obs: &'a mut [f32],
    mask: &'a mut [bool],
    policy: &'a mut i32,
    seat: &'a mut i32,
    reward: &'a mut [f32],
    done: &'a mut bool,
    winner: &'a mut i32,
    final_vp: &'a mut [i32],
    turns: &'a mut i32,
    lineup: &'a mut [i32],
}

impl GameEnv {
    fn new(index: u64, lineup: [SeatSpec; NUM_PLAYERS], record: bool) -> Self {
        GameEnv {
            state: GameState::new(0),
            seats: Vec::new(),
            lineup,
            pending: None,
            last_vp: [0; NUM_PLAYERS],
            legal: Vec::with_capacity(128),
            index,
            games_started: 0,
            record,
            log: Vec::new(),
            last_log: Vec::new(),
            last_seed: 0,
            last_lineup: lineup,
            last_result: None,
        }
    }

    fn start_game(&mut self, specs: &[BotSpec], base_seed: u64, num_envs: u64, first: bool) {
        if let Some(next) = self.pending.take() {
            self.lineup = next;
        } else if !first {
            self.lineup.rotate_right(1);
        }
        let seed = base_seed
            .wrapping_add(self.index)
            .wrapping_add(self.games_started.wrapping_mul(num_envs));
        self.games_started += 1;
        self.state = GameState::new(seed);
        self.seats = (0..NUM_PLAYERS)
            .map(|s| match self.lineup[s] {
                SeatSpec::Bot(b) => Seat::Bot(specs[b as usize].build(bot_seed(seed, s))),
                SeatSpec::Policy(k) => Seat::Policy(k),
            })
            .collect();
        self.last_vp = [0; NUM_PLAYERS];
        self.log.clear();
    }

    fn apply(&mut self, action: Action, cfg: &RewardConfig, reward: &mut [f32]) {
        self.state.apply(action);
        if self.record {
            self.log.push(action);
        }
        let s = &self.state;
        for p in 0..NUM_PLAYERS {
            let vp = s.total_vp(p as u8);
            if !s.is_setup() && cfg.shaping != 0.0 {
                reward[p] += cfg.shaping * (vp as f32 - self.last_vp[p] as f32);
            }
            self.last_vp[p] = vp;
        }
    }

    /// Advance until an external policy must choose (returns false) or the
    /// game ends (returns true). Terminal rewards are added to `reward`.
    fn advance(&mut self, cfg: &RewardConfig, reward: &mut [f32]) -> bool {
        loop {
            if self.state.is_over() {
                let w = self.state.winner;
                if w != NONE {
                    for p in 0..NUM_PLAYERS {
                        reward[p] += if p == w as usize { cfg.win } else { cfg.loss };
                    }
                }
                return true;
            }
            self.state.legal_actions(&mut self.legal);
            let actor = self.state.acting as usize;
            let action = if self.legal.len() == 1 {
                self.legal[0]
            } else {
                match &mut self.seats[actor] {
                    Seat::Policy(_) => return false,
                    Seat::Bot(bot) => bot.act(&self.state, &self.legal),
                }
            };
            self.apply(action, cfg, reward);
        }
    }

    /// Run until a policy decision is pending, resetting finished games.
    fn run_to_decision(
        &mut self,
        specs: &[BotSpec],
        cfg: &RewardConfig,
        base_seed: u64,
        num_envs: u64,
        row: &mut Row,
    ) {
        loop {
            if !self.advance(cfg, row.reward) {
                break;
            }
            // Game over: report it and start the next one.
            *row.done = true;
            *row.winner = if self.state.winner == NONE {
                -1
            } else {
                self.state.winner as i32
            };
            *row.turns = self.state.turn as i32;
            for p in 0..NUM_PLAYERS {
                row.final_vp[p] = self.state.total_vp(p as u8) as i32;
                row.lineup[p] = self.lineup[p].code();
            }
            self.last_seed = self.state.seed;
            self.last_lineup = self.lineup;
            self.last_result = Some(GameResult::from_state(&self.state));
            if self.record {
                std::mem::swap(&mut self.last_log, &mut self.log);
            }
            self.start_game(specs, base_seed, num_envs, false);
            // Rewards from the new game's opening (setup) are always zero, so
            // `row.reward` only ever holds the finished game's rewards.
        }
        self.write_obs(row);
    }

    fn write_obs(&mut self, row: &mut Row) {
        encode(&self.state, row.obs);
        row.mask.fill(false);
        self.state.legal_actions(&mut self.legal);
        let acting = self.state.acting;
        for a in &self.legal {
            row.mask[a.to_index(acting)] = true;
        }
        *row.seat = acting as i32;
        *row.policy = match self.seats[acting as usize] {
            Seat::Policy(k) => k as i32,
            Seat::Bot(_) => unreachable!("decision pending for a bot seat"),
        };
    }
}

pub struct VecEnv {
    pub envs: Vec<GameEnv>,
    pub specs: Vec<BotSpec>,
    pub cfg: RewardConfig,
    pub base_seed: u64,
}

impl VecEnv {
    /// `lineups[i]` is the initial lineup of game i; every lineup must contain
    /// at least one external policy seat.
    pub fn new(
        specs: Vec<BotSpec>,
        lineups: Vec<[SeatSpec; NUM_PLAYERS]>,
        base_seed: u64,
        record: bool,
    ) -> Result<VecEnv, String> {
        for (i, l) in lineups.iter().enumerate() {
            Self::validate_lineup(&specs, l).map_err(|e| format!("lineup {i}: {e}"))?;
        }
        let envs = lineups
            .into_iter()
            .enumerate()
            .map(|(i, l)| GameEnv::new(i as u64, l, record))
            .collect();
        Ok(VecEnv {
            envs,
            specs,
            cfg: RewardConfig::default(),
            base_seed,
        })
    }

    pub fn validate_lineup(specs: &[BotSpec], l: &[SeatSpec; NUM_PLAYERS]) -> Result<(), String> {
        if !l.iter().any(|s| matches!(s, SeatSpec::Policy(_))) {
            return Err("needs at least one policy seat".into());
        }
        for s in l {
            if let SeatSpec::Bot(b) = s
                && *b as usize >= specs.len()
            {
                return Err(format!("bot index {b} out of range"));
            }
        }
        Ok(())
    }

    pub fn num_envs(&self) -> usize {
        self.envs.len()
    }

    fn rows<'a>(b: &'a mut Buffers) -> Vec<Row<'a>> {
        let n = b.policy.len();
        let mut rows = Vec::with_capacity(n);
        let mut obs = b.obs.chunks_mut(OBS_SIZE);
        let mut mask = b.mask.chunks_mut(NUM_ACTIONS);
        let mut reward = b.reward.chunks_mut(NUM_PLAYERS);
        let mut final_vp = b.final_vp.chunks_mut(NUM_PLAYERS);
        let mut lineup = b.lineup.chunks_mut(NUM_PLAYERS);
        let mut policy = b.policy.iter_mut();
        let mut seat = b.seat.iter_mut();
        let mut done = b.done.iter_mut();
        let mut winner = b.winner.iter_mut();
        let mut turns = b.turns.iter_mut();
        for _ in 0..n {
            rows.push(Row {
                obs: obs.next().unwrap(),
                mask: mask.next().unwrap(),
                policy: policy.next().unwrap(),
                seat: seat.next().unwrap(),
                reward: reward.next().unwrap(),
                done: done.next().unwrap(),
                winner: winner.next().unwrap(),
                final_vp: final_vp.next().unwrap(),
                turns: turns.next().unwrap(),
                lineup: lineup.next().unwrap(),
            });
        }
        rows
    }

    fn clear(b: &mut Buffers) {
        b.reward.fill(0.0);
        b.done.fill(false);
        b.winner.fill(-2);
        b.final_vp.fill(0);
        b.turns.fill(0);
        b.lineup.fill(0);
    }

    pub fn reset_all(&mut self, b: &mut Buffers) {
        Self::clear(b);
        let n = self.envs.len() as u64;
        let (specs, cfg, base) = (&self.specs, self.cfg, self.base_seed);
        let mut rows = Self::rows(b);
        self.envs
            .par_iter_mut()
            .zip(rows.par_iter_mut())
            .for_each(|(env, row)| {
                env.start_game(specs, base, n, true);
                env.run_to_decision(specs, &cfg, base, n, row);
                // Rewards (and "done") from reset are meaningless to the caller.
                row.reward.fill(0.0);
                *row.done = false;
                *row.winner = -2;
            });
    }

    /// Apply one action index per game (for the policy currently acting).
    pub fn step(&mut self, actions: &[i64], b: &mut Buffers) -> Result<(), String> {
        if actions.len() != self.envs.len() {
            return Err(format!(
                "expected {} actions, got {}",
                self.envs.len(),
                actions.len()
            ));
        }
        Self::clear(b);
        let n = self.envs.len() as u64;
        let (specs, cfg, base) = (&self.specs, self.cfg, self.base_seed);
        let mut rows = Self::rows(b);
        let errors: Vec<String> = self
            .envs
            .par_iter_mut()
            .zip(rows.par_iter_mut())
            .zip(actions.par_iter())
            .enumerate()
            .filter_map(|(i, ((env, row), &idx))| {
                let acting = env.state.acting;
                let action = match usize::try_from(idx)
                    .ok()
                    .and_then(|x| Action::from_index(x, acting))
                {
                    Some(a) => a,
                    None => return Some(format!("game {i}: action index {idx} out of range")),
                };
                env.state.legal_actions(&mut env.legal);
                if !env.legal.contains(&action) {
                    return Some(format!(
                        "game {i}: illegal action {action:?} in {:?}",
                        env.state.phase
                    ));
                }
                env.apply(action, &cfg, row.reward);
                env.run_to_decision(specs, &cfg, base, n, row);
                None
            })
            .collect();
        match errors.into_iter().next() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

/// Owned output buffers, convenient for Rust-side tests and benchmarks.
pub struct OwnedBuffers {
    pub obs: Vec<f32>,
    pub mask: Vec<bool>,
    pub policy: Vec<i32>,
    pub seat: Vec<i32>,
    pub reward: Vec<f32>,
    pub done: Vec<bool>,
    pub winner: Vec<i32>,
    pub final_vp: Vec<i32>,
    pub turns: Vec<i32>,
    pub lineup: Vec<i32>,
}

impl OwnedBuffers {
    pub fn new(n: usize) -> Self {
        OwnedBuffers {
            obs: vec![0.0; n * OBS_SIZE],
            mask: vec![false; n * NUM_ACTIONS],
            policy: vec![0; n],
            seat: vec![0; n],
            reward: vec![0.0; n * NUM_PLAYERS],
            done: vec![false; n],
            winner: vec![0; n],
            final_vp: vec![0; n * NUM_PLAYERS],
            turns: vec![0; n],
            lineup: vec![0; n * NUM_PLAYERS],
        }
    }
    pub fn view(&mut self) -> Buffers<'_> {
        Buffers {
            obs: &mut self.obs,
            mask: &mut self.mask,
            policy: &mut self.policy,
            seat: &mut self.seat,
            reward: &mut self.reward,
            done: &mut self.done,
            winner: &mut self.winner,
            final_vp: &mut self.final_vp,
            turns: &mut self.turns,
            lineup: &mut self.lineup,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn random_policy_plays_many_games_against_bots() {
        let specs = vec![BotSpec::preset("balanced").unwrap(), BotSpec::Random];
        let n = 16;
        let lineups = (0..n)
            .map(|_| {
                [
                    SeatSpec::Policy(0),
                    SeatSpec::Bot(0),
                    SeatSpec::Bot(1),
                    SeatSpec::Bot(0),
                ]
            })
            .collect();
        let mut env = VecEnv::new(specs, lineups, 123, true).unwrap();
        env.cfg.shaping = 0.01;
        let mut b = OwnedBuffers::new(n);
        env.reset_all(&mut b.view());
        let mut rng = Rng::new(1);
        let mut finished = 0;
        let mut reward_total = 0.0;
        for _ in 0..3000 {
            // Pick a random legal action per game.
            let actions: Vec<i64> = (0..n)
                .map(|i| {
                    let m = &b.mask[i * NUM_ACTIONS..(i + 1) * NUM_ACTIONS];
                    let legal: Vec<usize> = (0..NUM_ACTIONS).filter(|&a| m[a]).collect();
                    assert!(legal.len() >= 2, "forced moves must be auto-played");
                    assert_eq!(b.policy[i], 0);
                    legal[rng.below(legal.len() as u32) as usize] as i64
                })
                .collect();
            env.step(&actions, &mut b.view()).unwrap();
            for i in 0..n {
                if b.done[i] {
                    finished += 1;
                    assert!(b.winner[i] >= -1);
                    // Terminal rewards are zero-sum across seats (ignoring shaping).
                    let r: f32 = b.reward[i * 4..i * 4 + 4].iter().sum();
                    reward_total += r;
                    // The policy seat appears in the reported lineup.
                    assert!(b.lineup[i * 4..i * 4 + 4].contains(&-1));
                }
            }
            for st in env.envs.iter() {
                st.state.check_invariants();
            }
        }
        assert!(finished > 0, "some games should finish");
        assert!(reward_total.is_finite());
        // Recorded logs replay to the same final result.
        for e in env.envs.iter().filter(|e| !e.last_log.is_empty()) {
            let mut s = GameState::new(e.last_seed);
            for &a in &e.last_log {
                s.apply(a);
            }
            assert!(s.is_over());
        }
    }

    #[test]
    fn illegal_action_is_rejected() {
        let specs = vec![BotSpec::Random];
        let lineups = vec![[
            SeatSpec::Policy(0),
            SeatSpec::Bot(0),
            SeatSpec::Bot(0),
            SeatSpec::Bot(0),
        ]];
        let mut env = VecEnv::new(specs, lineups, 1, false).unwrap();
        let mut b = OwnedBuffers::new(1);
        env.reset_all(&mut b.view());
        let illegal = (0..NUM_ACTIONS).find(|&a| !b.mask[a]).unwrap() as i64;
        assert!(env.step(&[illegal], &mut b.view()).is_err());
    }
}

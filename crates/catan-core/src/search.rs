//! Decision-time look-ahead for the learning agent ("imagined futures").
//!
//! For a decision of player `me`, each candidate action is tried in `sims`
//! imagined continuations of the game:
//!
//! 1. **Determinise** the hidden information from `me`'s point of view:
//!    opponents' resource cards are re-dealt from the cards `me` can't see
//!    (keeping each opponent's hand size), opponents' development cards and
//!    the deck order are re-dealt from the cards that haven't been played, and
//!    the dice and steal streams are re-seeded. The search never peeks at the
//!    real hidden cards or the real future dice.
//! 2. Apply the candidate, then let the game run: opponents are played by a
//!    stand-in bot, forced moves are auto-played, until `me` has a real
//!    decision again (a *leaf*) or the game ends.
//! 3. Leaves are returned as observations for the value network to score;
//!    finished games are scored directly (win / loss / draw).
//!
//! Imagined game `j` uses the same seed for every candidate (common random
//! numbers: same hidden cards, same dice), so candidates are compared on the
//! same futures. A candidate whose first continuation used no randomness at
//! all (e.g. building a road and still having moves left) reaches the same
//! leaf in every continuation, so it is simulated once.
//!
//! Two horizons are available:
//!
//! - [`search`]: stop at `me`'s very next decision. Cheap, but a move that keeps
//!   the turn going (a trade) is scored mid-turn while "end turn" is scored
//!   after the opponents have played, so the two scores are taken at different
//!   points of the game.
//! - [`Rollouts`]: a common horizon. `me` finishes the turn with its own policy
//!   (decisions are handed back to Python), the opponents play, and every
//!   candidate is scored at `me`'s first decision after the others have moved
//!   (normally the start of its next turn).

use rayon::prelude::*;

use crate::actions::{Action, NUM_ACTIONS};
use crate::board::NONE;
use crate::bots::{Bot, BotSpec};
use crate::obs::{OBS_SIZE, encode};
use crate::rng::Rng;
use crate::sim::bot_seed;
use crate::state::{DEV_DECK_SIZE, GameState, NUM_DEV_TYPES, NUM_PLAYERS};

/// One decision to search: the state, who decides, and the candidate actions.
pub struct SearchRequest<'a> {
    pub state: &'a GameState,
    pub candidates: Vec<Action>,
}

#[derive(Clone, Copy, Debug)]
pub struct SearchConfig {
    pub sims: usize,
    pub win: f32,
    pub loss: f32,
    pub seed: u64,
    /// Safety cap on actions per continuation (scored as a draw if reached).
    pub max_steps: u32,
}

/// Flat result for `R` requests with up to `K` candidates each; slot = r * K + k.
pub struct SearchBatch {
    /// Leaves for the value network: `M * OBS_SIZE` observations and masks.
    pub obs: Vec<f32>,
    pub mask: Vec<bool>,
    /// Slot of each leaf.
    pub leaf_slot: Vec<i32>,
    /// Sum of terminal values per slot (`R * K`).
    pub terminal_sum: Vec<f32>,
    /// Number of continuations per slot (leaves + finished games); 0 = no candidate.
    pub count: Vec<i32>,
}

enum Outcome {
    Terminal(f32),
    Leaf(Box<GameState>),
}

/// Re-deal everything `me` cannot see. The result is consistent with all
/// public information (hand sizes, dev card counts, bank, played cards).
pub fn determinize(s: &GameState, me: u8, rng: &mut Rng) -> GameState {
    let mut d = *s;
    let mi = me as usize;

    // Resource cards held by opponents: the multiset is public (19 of each,
    // minus the bank, minus my hand); who holds which card is not.
    let mut pool = [0u8; 19 * 5];
    let mut n = 0;
    for o in (0..NUM_PLAYERS).filter(|&o| o != mi) {
        for (r, &c) in s.hands[o].iter().enumerate() {
            for _ in 0..c {
                pool[n] = r as u8;
                n += 1;
            }
        }
    }
    rng.shuffle(&mut pool[..n]);
    let mut k = 0;
    for o in (0..NUM_PLAYERS).filter(|&o| o != mi) {
        let size = s.hand_size(o as u8) as usize;
        d.hands[o] = [0; 5];
        for &r in &pool[k..k + size] {
            d.hands[o][r as usize] += 1;
        }
        k += size;
    }

    // Development cards not visible to me: opponents' cards + the rest of the deck.
    let mut dev = [0u8; DEV_DECK_SIZE];
    let mut m = 0;
    for o in (0..NUM_PLAYERS).filter(|&o| o != mi) {
        for c in 0..NUM_DEV_TYPES {
            for _ in 0..s.dev_hand[o][c] + s.dev_new[o][c] {
                dev[m] = c as u8;
                m += 1;
            }
        }
    }
    let pos = s.dev_deck_pos as usize;
    for &c in &s.dev_deck[pos..] {
        dev[m] = c;
        m += 1;
    }
    rng.shuffle(&mut dev[..m]);
    let mut k = 0;
    for o in (0..NUM_PLAYERS).filter(|&o| o != mi) {
        let (held, new) = (
            s.dev_hand[o].iter().sum::<u8>() as usize,
            s.dev_new[o].iter().sum::<u8>() as usize,
        );
        d.dev_hand[o] = [0; NUM_DEV_TYPES];
        d.dev_new[o] = [0; NUM_DEV_TYPES];
        for &c in &dev[k..k + held] {
            d.dev_hand[o][c as usize] += 1;
        }
        for &c in &dev[k + held..k + held + new] {
            d.dev_new[o][c as usize] += 1;
        }
        k += held + new;
    }
    d.dev_deck[pos..].copy_from_slice(&dev[k..m]);

    d.dice_rng = Rng::new(rng.next_u64());
    d.steal_rng = Rng::new(rng.next_u64());
    d
}

/// Play one imagined continuation. Returns the outcome and whether any
/// randomness or hidden information could have influenced it.
fn continuation(
    root: &GameState,
    me: u8,
    action: Action,
    seed: u64,
    opp: &BotSpec,
    cfg: &SearchConfig,
) -> (Outcome, bool) {
    let mut rng = Rng::new(seed);
    let mut s = determinize(root, me, &mut rng);
    let (dice0, steal0, deck0) = (s.dice_rng, s.steal_rng, s.dev_deck_pos);
    // Monopoly's haul depends on the (re-dealt) hidden hands.
    let mut random = matches!(action, Action::PlayMonopoly(_));
    s.apply(action);
    let mut bots: [Option<Box<dyn Bot>>; NUM_PLAYERS] = Default::default();
    let mut legal = Vec::with_capacity(64);
    let mut steps = 0;
    let outcome = loop {
        if s.is_over() {
            let v = if s.winner == me {
                cfg.win
            } else if s.winner == NONE {
                0.0
            } else {
                cfg.loss
            };
            break Outcome::Terminal(v);
        }
        if steps >= cfg.max_steps {
            break Outcome::Terminal(0.0);
        }
        s.legal_actions(&mut legal);
        let a = if legal.len() == 1 {
            legal[0]
        } else if s.acting == me {
            break Outcome::Leaf(Box::new(s));
        } else {
            random = true;
            let seat = s.acting as usize;
            let bot = bots[seat].get_or_insert_with(|| opp.build(bot_seed(seed, seat)));
            bot.act(&s, &legal)
        };
        s.apply(a);
        steps += 1;
    };
    let random = random || s.dice_rng != dice0 || s.steal_rng != steal0 || s.dev_deck_pos != deck0;
    (outcome, random)
}

/// Seed of imagined game `j` for request `r` (shared by all its candidates).
fn sim_seed(cfg: &SearchConfig, r: usize, j: usize) -> u64 {
    Rng::stream(cfg.seed ^ (r as u64).wrapping_mul(0x9E37_79B9), j as u64).next_u64()
}

/// Run the look-ahead for a batch of decisions in parallel.
pub fn search(
    requests: &[SearchRequest],
    k_max: usize,
    opp: &BotSpec,
    cfg: &SearchConfig,
) -> SearchBatch {
    let slots = requests.len() * k_max;
    let jobs0: Vec<(usize, usize)> = requests
        .iter()
        .enumerate()
        .flat_map(|(r, req)| (0..req.candidates.len().min(k_max)).map(move |k| (r, k)))
        .collect();
    let run = |r: usize, k: usize, j: usize| {
        let req = &requests[r];
        continuation(
            req.state,
            req.state.acting,
            req.candidates[k],
            sim_seed(cfg, r, j),
            opp,
            cfg,
        )
    };

    // First continuation of every candidate; then more only where randomness mattered.
    let first: Vec<(Outcome, bool)> = jobs0.par_iter().map(|&(r, k)| run(r, k, 0)).collect();
    let more: Vec<(usize, usize, usize)> = jobs0
        .iter()
        .zip(&first)
        .filter(|(_, (_, random))| *random)
        .flat_map(|(&(r, k), _)| (1..cfg.sims).map(move |j| (r, k, j)))
        .collect();
    let rest: Vec<Outcome> = more.par_iter().map(|&(r, k, j)| run(r, k, j).0).collect();

    let mut out = SearchBatch {
        obs: Vec::new(),
        mask: Vec::new(),
        leaf_slot: Vec::new(),
        terminal_sum: vec![0.0; slots],
        count: vec![0; slots],
    };
    let mut legal = Vec::with_capacity(64);
    let outcomes = jobs0
        .iter()
        .map(|&(r, k)| (r, k))
        .zip(first.into_iter().map(|(o, _)| o))
        .chain(more.iter().map(|&(r, k, _)| (r, k)).zip(rest));
    for ((r, k), o) in outcomes {
        let slot = r * k_max + k;
        out.count[slot] += 1;
        match o {
            Outcome::Terminal(v) => out.terminal_sum[slot] += v,
            Outcome::Leaf(s) => {
                let start = out.obs.len();
                out.obs.resize(start + OBS_SIZE, 0.0);
                encode(&s, &mut out.obs[start..]);
                let start = out.mask.len();
                out.mask.resize(start + NUM_ACTIONS, false);
                s.action_mask(&mut legal, &mut out.mask[start..]);
                out.leaf_slot.push(slot as i32);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::build_bots;

    /// Play `n` actions of a bot game and return the state.
    fn midgame(seed: u64, n: usize) -> GameState {
        let spec = BotSpec::preset("balanced").unwrap();
        let mut bots = build_bots(&[&spec; 4], seed);
        let mut s = GameState::new(seed);
        let mut buf = Vec::new();
        for _ in 0..n {
            if s.is_over() {
                break;
            }
            s.legal_actions(&mut buf);
            let a = bots[s.acting as usize].act(&s, &buf);
            s.apply(a);
        }
        s
    }

    #[test]
    fn determinize_keeps_public_information() {
        for seed in 0..40 {
            let s = midgame(seed, 300 + seed as usize * 7);
            if s.is_over() {
                continue;
            }
            let me = s.acting;
            let mut rng = Rng::new(seed);
            let d = determinize(&s, me, &mut rng);
            d.check_invariants();
            for p in 0..NUM_PLAYERS as u8 {
                assert_eq!(d.hand_size(p), s.hand_size(p));
                assert_eq!(d.dev_cards_held(p), s.dev_cards_held(p));
                assert_eq!(d.public_vp(p), s.public_vp(p));
            }
            assert_eq!(d.hands[me as usize], s.hands[me as usize]);
            assert_eq!(d.dev_hand[me as usize], s.dev_hand[me as usize]);
            assert_eq!(d.bank, s.bank);
            // Same multiset of unseen dev cards.
            let unseen = |g: &GameState| {
                let mut c = [0u8; NUM_DEV_TYPES];
                for o in (0..NUM_PLAYERS).filter(|&o| o != me as usize) {
                    for t in 0..NUM_DEV_TYPES {
                        c[t] += g.dev_hand[o][t] + g.dev_new[o][t];
                    }
                }
                for &x in &g.dev_deck[g.dev_deck_pos as usize..] {
                    c[x as usize] += 1;
                }
                c
            };
            assert_eq!(unseen(&d), unseen(&s));
            // The observation of `me` is unchanged.
            let (mut a, mut b) = (vec![0.0; OBS_SIZE], vec![0.0; OBS_SIZE]);
            encode(&s, &mut a);
            encode(&d, &mut b);
            assert_eq!(a, b);
        }
    }

    #[test]
    fn search_scores_every_candidate() {
        let opp = BotSpec::preset("balanced").unwrap();
        let cfg = SearchConfig {
            sims: 8,
            win: 1.0,
            loss: -1.0 / 3.0,
            seed: 7,
            max_steps: 5000,
        };
        let states: Vec<GameState> = (0..12)
            .map(|i| midgame(100 + i, 150 + 13 * i as usize))
            .collect();
        let mut buf = Vec::new();
        let reqs: Vec<SearchRequest> = states
            .iter()
            .filter(|s| !s.is_over())
            .map(|s| {
                s.legal_actions(&mut buf);
                SearchRequest {
                    state: s,
                    candidates: buf.iter().copied().take(3).collect(),
                }
            })
            .collect();
        let b = search(&reqs, 3, &opp, &cfg);
        assert_eq!(b.obs.len(), b.leaf_slot.len() * OBS_SIZE);
        assert_eq!(b.mask.len(), b.leaf_slot.len() * NUM_ACTIONS);
        for (r, req) in reqs.iter().enumerate() {
            for k in 0..3 {
                let c = b.count[r * 3 + k];
                if k < req.candidates.len() {
                    assert!(c == 1 || c == cfg.sims as i32, "count {c}");
                } else {
                    assert_eq!(c, 0);
                }
            }
        }
        // Every leaf is a real decision of the searching player.
        for (i, &slot) in b.leaf_slot.iter().enumerate() {
            let m = &b.mask[i * NUM_ACTIONS..(i + 1) * NUM_ACTIONS];
            assert!(m.iter().filter(|&&x| x).count() >= 2, "slot {slot}");
        }
        // Deterministic: same inputs, same outputs.
        let b2 = search(&reqs, 3, &opp, &cfg);
        assert_eq!(b.obs, b2.obs);
        assert_eq!(b.count, b2.count);
    }
}

// ----------------------------------------------------------------------
// Common-horizon rollouts
// ----------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
enum Status {
    /// `me` must decide before the horizon: waiting for the policy.
    Pending,
    /// Reached the horizon: to be scored by the value network.
    Leaf,
    Terminal(f32),
}

struct Job {
    slot: usize,
    me: u8,
    seed: u64,
    state: GameState,
    bots: [Option<Box<dyn Bot>>; NUM_PLAYERS],
    others_moved: bool,
    own_steps: u32,
    steps: u32,
    status: Status,
}

impl Job {
    fn advance(
        &mut self,
        opp: &BotSpec,
        cfg: &SearchConfig,
        max_own: u32,
        legal: &mut Vec<Action>,
    ) {
        loop {
            let s = &mut self.state;
            if s.is_over() {
                let v = if s.winner == self.me {
                    cfg.win
                } else if s.winner == NONE {
                    0.0
                } else {
                    cfg.loss
                };
                self.status = Status::Terminal(v);
                return;
            }
            if self.steps >= cfg.max_steps {
                self.status = Status::Terminal(0.0);
                return;
            }
            s.legal_actions(legal);
            if s.acting == self.me && legal.len() > 1 {
                self.status = if self.others_moved || self.own_steps >= max_own {
                    Status::Leaf
                } else {
                    Status::Pending
                };
                return;
            }
            let a = if legal.len() == 1 {
                legal[0]
            } else {
                let seat = s.acting as usize;
                let seed = self.seed;
                self.bots[seat]
                    .get_or_insert_with(|| opp.build(bot_seed(seed, seat)))
                    .act(s, legal)
            };
            if s.acting != self.me {
                self.others_moved = true;
            }
            s.apply(a);
            self.steps += 1;
        }
    }
}

/// A batch of imagined continuations played to a common horizon. Usage:
/// `new`, then `pending` / `step` until nothing is pending (the policy plays
/// `me`'s remaining moves of the turn), then `leaves`.
pub struct Rollouts {
    jobs: Vec<Job>,
    slots: usize,
    opp: BotSpec,
    cfg: SearchConfig,
    max_own: u32,
}

impl Rollouts {
    /// `max_own` caps `me`'s own decisions per continuation before it is scored anyway.
    pub fn new(
        requests: &[SearchRequest],
        k_max: usize,
        opp: BotSpec,
        cfg: SearchConfig,
        max_own: u32,
    ) -> Rollouts {
        let mut jobs = Vec::new();
        for (r, req) in requests.iter().enumerate() {
            let me = req.state.acting;
            for j in 0..cfg.sims {
                let seed = sim_seed(&cfg, r, j);
                // Same re-dealt hidden cards and dice for every candidate.
                let base = determinize(req.state, me, &mut Rng::new(seed));
                for (k, &a) in req.candidates.iter().take(k_max).enumerate() {
                    let mut state = base;
                    state.apply(a);
                    jobs.push(Job {
                        slot: r * k_max + k,
                        me,
                        seed,
                        state,
                        bots: Default::default(),
                        others_moved: false,
                        own_steps: 0,
                        steps: 1,
                        status: Status::Pending,
                    });
                }
            }
        }
        let mut out = Rollouts {
            jobs,
            slots: requests.len() * k_max,
            opp,
            cfg,
            max_own,
        };
        let (opp, cfg, max_own) = (&out.opp, &out.cfg, out.max_own);
        out.jobs.par_iter_mut().for_each_init(
            || Vec::with_capacity(64),
            |legal, job| job.advance(opp, cfg, max_own, legal),
        );
        out
    }

    pub fn num_jobs(&self) -> usize {
        self.jobs.len()
    }

    /// Jobs waiting for `me`'s policy: (job ids, observations, masks).
    pub fn pending(&self) -> (Vec<i64>, Vec<f32>, Vec<bool>) {
        let ids: Vec<i64> = (0..self.jobs.len())
            .filter(|&i| self.jobs[i].status == Status::Pending)
            .map(|i| i as i64)
            .collect();
        let (obs, mask) = self.encode(&ids);
        (ids, obs, mask)
    }

    fn encode(&self, ids: &[i64]) -> (Vec<f32>, Vec<bool>) {
        // Bots are not Sync, so hand the parallel encoder plain state references.
        let states: Vec<&GameState> = ids.iter().map(|&i| &self.jobs[i as usize].state).collect();
        let mut obs = vec![0.0; ids.len() * OBS_SIZE];
        let mut mask = vec![false; ids.len() * NUM_ACTIONS];
        obs.par_chunks_mut(OBS_SIZE)
            .zip(mask.par_chunks_mut(NUM_ACTIONS))
            .zip(states.par_iter())
            .for_each_init(
                || Vec::with_capacity(64),
                |legal, ((o, m), s)| {
                    encode(s, o);
                    s.action_mask(legal, m);
                },
            );
        (obs, mask)
    }

    /// Apply the policy's action index for each pending job, then advance.
    pub fn step(&mut self, ids: &[i64], actions: &[i64]) -> Result<(), String> {
        if ids.len() != actions.len() {
            return Err(format!("{} ids but {} actions", ids.len(), actions.len()));
        }
        let mut chosen = vec![None; self.jobs.len()];
        for (&i, &a) in ids.iter().zip(actions) {
            let job = self
                .jobs
                .get(i as usize)
                .ok_or_else(|| format!("job {i} out of range"))?;
            if job.status != Status::Pending {
                return Err(format!("job {i} is not waiting for a decision"));
            }
            let action = usize::try_from(a)
                .ok()
                .and_then(|x| Action::from_index(x, job.me))
                .filter(|&x| job.state.is_legal(x))
                .ok_or_else(|| format!("job {i}: illegal action {a}"))?;
            chosen[i as usize] = Some(action);
        }
        let (opp, cfg, max_own) = (&self.opp, &self.cfg, self.max_own);
        self.jobs
            .par_iter_mut()
            .zip(chosen.par_iter())
            .for_each_init(
                || Vec::with_capacity(64),
                |legal, (job, a)| {
                    if let Some(a) = *a {
                        job.state.apply(a);
                        job.steps += 1;
                        job.own_steps += 1;
                        job.advance(opp, cfg, max_own, legal);
                    }
                },
            );
        Ok(())
    }

    /// Final result (call once nothing is pending; pending jobs are scored as leaves).
    pub fn leaves(&self) -> SearchBatch {
        let ids: Vec<i64> = (0..self.jobs.len())
            .filter(|&i| !matches!(self.jobs[i].status, Status::Terminal(_)))
            .map(|i| i as i64)
            .collect();
        let (obs, mask) = self.encode(&ids);
        let mut out = SearchBatch {
            obs,
            mask,
            leaf_slot: ids
                .iter()
                .map(|&i| self.jobs[i as usize].slot as i32)
                .collect(),
            terminal_sum: vec![0.0; self.slots],
            count: vec![0; self.slots],
        };
        for job in &self.jobs {
            out.count[job.slot] += 1;
            if let Status::Terminal(v) = job.status {
                out.terminal_sum[job.slot] += v;
            }
        }
        out
    }
}

#[cfg(test)]
mod rollout_tests {
    use super::*;
    use crate::sim::build_bots;

    #[test]
    fn rollouts_reach_a_common_horizon() {
        let spec = BotSpec::preset("balanced").unwrap();
        let cfg = SearchConfig {
            sims: 4,
            win: 1.0,
            loss: -1.0 / 3.0,
            seed: 3,
            max_steps: 20_000,
        };
        let mut states = Vec::new();
        for seed in 0..10u64 {
            let mut bots = build_bots(&[&spec; 4], seed);
            let mut s = GameState::new(seed);
            let mut buf = Vec::new();
            for _ in 0..(40 + 37 * seed as usize) {
                if s.is_over() {
                    break;
                }
                s.legal_actions(&mut buf);
                if buf.len() > 1 && s.acting == 0 && s.phase == crate::state::Phase::Main {
                    break;
                }
                let a = bots[s.acting as usize].act(&s, &buf);
                s.apply(a);
            }
            if !s.is_over() {
                states.push(s);
            }
        }
        let mut buf = Vec::new();
        let reqs: Vec<SearchRequest> = states
            .iter()
            .map(|s| {
                s.legal_actions(&mut buf);
                SearchRequest {
                    state: s,
                    candidates: buf.iter().copied().take(3).collect(),
                }
            })
            .collect();
        let mut ro = Rollouts::new(&reqs, 3, spec.clone(), cfg, 40);
        // Play `me`'s own moves with a stand-in (first legal action) until the horizon.
        for _ in 0..100 {
            let (ids, _, mask) = ro.pending();
            if ids.is_empty() {
                break;
            }
            let acts: Vec<i64> = mask
                .chunks(NUM_ACTIONS)
                .map(|m| m.iter().position(|&x| x).unwrap() as i64)
                .collect();
            ro.step(&ids, &acts).unwrap();
        }
        assert!(ro.pending().0.is_empty());
        let b = ro.leaves();
        for (r, req) in reqs.iter().enumerate() {
            for k in 0..req.candidates.len() {
                assert_eq!(b.count[r * 3 + k], cfg.sims as i32);
            }
        }
        // Every leaf is a decision of `me` after the others moved.
        for job in ro.jobs.iter().filter(|j| j.status == Status::Leaf) {
            assert_eq!(job.state.acting, job.me);
            assert!(job.others_moved || job.own_steps >= 40);
        }
        // Illegal actions are rejected.
        let mut ro = Rollouts::new(&reqs, 3, spec, cfg, 40);
        let (ids, _, mask) = ro.pending();
        if let Some(&id) = ids.first() {
            let illegal = mask[..NUM_ACTIONS].iter().position(|&x| !x).unwrap() as i64;
            assert!(ro.step(&[id], &[illegal]).is_err());
        }
    }
}

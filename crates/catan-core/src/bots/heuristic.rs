//! A parameterised, goal-driven heuristic bot.
//!
//! Each turn it picks a *goal* (city, settlement, road or dev card) by
//! weighing how valuable each build is against how far it is from affording
//! it, then: builds the goal if it can, trades with the bank/harbours toward
//! it, spends spare cards on other builds, and finally protects its hand from
//! a 7. The `StrategyParams` weights turn this one bot into the different
//! community strategies.

use std::cell::Cell;

use super::{Bot, StrategyParams};
use crate::actions::Action;
use crate::board::{
    DESERT, NONE, NUM_EDGES, NUM_NODES, NUM_RESOURCES, PORT_GENERIC, pips, topology,
};
use crate::rng::Rng;
use crate::state::{
    COST_CITY, COST_DEV, COST_ROAD, COST_SETTLEMENT, Cost, GameState, HAND_LIMIT, KNIGHT, MONOPOLY,
    NUM_PLAYERS, Phase, VP_TO_WIN, YEAR_OF_PLENTY,
};

/// Total pips of the (non-desert) tiles around a node.
pub fn node_pips(s: &GameState, n: u8) -> u8 {
    topology().node_tiles[n as usize]
        .iter()
        .filter(|&&t| t != NONE)
        .map(|&t| pips(s.board.tile_number[t as usize]))
        .sum()
}

/// Whether the robber currently blocks one of `p`'s buildings.
pub fn robber_hurts(s: &GameState, p: u8) -> bool {
    topology().tile_nodes[s.robber as usize]
        .iter()
        .any(|&n| s.node_owner[n as usize] == p)
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Goal {
    City(u8),
    Settlement(u8),
    Road(u8),
    Dev,
    Nothing,
}

impl Goal {
    fn cost(self) -> Cost {
        match self {
            Goal::City(_) => COST_CITY,
            Goal::Settlement(_) => COST_SETTLEMENT,
            Goal::Road(_) => COST_ROAD,
            Goal::Dev => COST_DEV,
            Goal::Nothing => [0; NUM_RESOURCES],
        }
    }

    fn action(self) -> Option<Action> {
        match self {
            Goal::City(n) => Some(Action::BuildCity(n)),
            Goal::Settlement(n) => Some(Action::BuildSettlement(n)),
            Goal::Road(e) => Some(Action::BuildRoad(e)),
            Goal::Dev => Some(Action::BuyDev),
            Goal::Nothing => None,
        }
    }
}

/// Per-decision cache.
struct Ctx<'a> {
    s: &'a GameState,
    p: u8,
    prod: [[u8; NUM_RESOURCES]; NUM_PLAYERS],
}

impl<'a> Ctx<'a> {
    fn new(s: &'a GameState) -> Self {
        Ctx {
            s,
            p: s.acting,
            prod: s.production_pips(),
        }
    }
    fn hand(&self) -> &[u8; NUM_RESOURCES] {
        &self.s.hands[self.p as usize]
    }
}

fn missing(hand: &[u8; NUM_RESOURCES], cost: &Cost) -> [u8; NUM_RESOURCES] {
    let mut m = [0; NUM_RESOURCES];
    for r in 0..NUM_RESOURCES {
        m[r] = cost[r].saturating_sub(hand[r]);
    }
    m
}

pub struct HeuristicBot {
    w: StrategyParams,
    /// In a `Cell` so evaluation helpers can take `&self` (closures passed to
    /// `argmax` borrow `self` immutably).
    rng: Cell<Rng>,
}

impl HeuristicBot {
    pub fn new(params: StrategyParams, seed: u64) -> Self {
        HeuristicBot {
            w: params,
            rng: Cell::new(Rng::new(seed)),
        }
    }

    pub fn params(&self) -> &StrategyParams {
        &self.w
    }

    fn noise(&self) -> f32 {
        let mut r = self.rng.get();
        let x = r.unit();
        self.rng.set(r);
        x * 1e-3
    }

    fn pick(&self, n: usize) -> usize {
        let mut r = self.rng.get();
        let x = r.below(n as u32) as usize;
        self.rng.set(r);
        x
    }

    /// Pick the action maximising `f` (ignoring actions where it returns None).
    fn argmax<F: FnMut(&Action) -> Option<f32>>(
        &self,
        legal: &[Action],
        mut f: F,
    ) -> Option<Action> {
        let mut best = None;
        let mut best_v = f32::NEG_INFINITY;
        for a in legal {
            if let Some(v) = f(a) {
                let v = v + self.noise();
                if v > best_v {
                    best_v = v;
                    best = Some(*a);
                }
            }
        }
        best
    }

    // ------------------------------------------------------------------
    // Evaluation
    // ------------------------------------------------------------------

    /// How much `c.p` would like a settlement at node `n`.
    fn node_value(&self, c: &Ctx, n: u8) -> f32 {
        let s = c.s;
        let w = &self.w;
        let mine = &c.prod[c.p as usize];
        let mut v = 0.0;
        let mut new_types = [false; NUM_RESOURCES];
        let mut node_res_pips = [0u8; NUM_RESOURCES];
        for &t in &topology().node_tiles[n as usize] {
            if t == NONE {
                continue;
            }
            let res = s.board.tile_resource[t as usize];
            if res == DESERT {
                continue;
            }
            let r = res as usize;
            let pip = pips(s.board.tile_number[t as usize]) as f32;
            node_res_pips[r] += pip as u8;
            v += pip * w.resource_weights[r];
            v += w.specialization_weight * pip * (mine[r] as f32 / 10.0);
            if mine[r] == 0 && !new_types[r] {
                new_types[r] = true;
                v += w.diversity_weight;
            }
        }
        let port = s.board.node_port[n as usize];
        if port != NONE && s.ports[c.p as usize] & (1 << port) == 0 {
            if port == PORT_GENERIC {
                v += w.port_generic_weight;
            } else {
                let r = port as usize;
                v += w.port_specific_weight * (mine[r] + node_res_pips[r]) as f32;
            }
        }
        v
    }

    /// Weighted production of an existing settlement (value of upgrading it).
    fn city_value(&self, c: &Ctx, n: u8) -> f32 {
        let s = c.s;
        topology().node_tiles[n as usize]
            .iter()
            .filter(|&&t| t != NONE && s.board.tile_resource[t as usize] != DESERT)
            .map(|&t| {
                pips(s.board.tile_number[t as usize]) as f32
                    * self.w.resource_weights[s.board.tile_resource[t as usize] as usize]
            })
            .sum()
    }

    fn best_settlement_spot(&self, c: &Ctx) -> Option<(u8, f32)> {
        let mut best: Option<(u8, f32)> = None;
        for n in 0..NUM_NODES as u8 {
            if c.s.can_build_settlement_at(c.p, n) {
                let v = self.node_value(c, n);
                if best.is_none_or(|(_, bv)| v > bv) {
                    best = Some((n, v));
                }
            }
        }
        best
    }

    fn best_city_spot(&self, c: &Ctx) -> Option<(u8, f32)> {
        let mut best: Option<(u8, f32)> = None;
        for n in 0..NUM_NODES as u8 {
            if c.s.node_owner[n as usize] == c.p && !c.s.node_city[n as usize] {
                let v = self.city_value(c, n);
                if best.is_none_or(|(_, bv)| v > bv) {
                    best = Some((n, v));
                }
            }
        }
        best
    }

    /// Breadth-first search (up to 3 new roads) for the best open settlement
    /// spot; returns the first road to build toward it and its discounted value.
    fn road_plan(&self, c: &Ctx) -> Option<(u8, f32)> {
        let s = c.s;
        let p = c.p;
        let topo = topology();
        let mut dist = [u8::MAX; NUM_NODES];
        let mut first = [NONE; NUM_NODES];
        let mut queue = [0u8; NUM_NODES];
        let (mut head, mut tail) = (0, 0);
        for n in 0..NUM_NODES as u8 {
            let owner = s.node_owner[n as usize];
            if owner == p || (owner == NONE && s.node_has_road(p, n)) {
                dist[n as usize] = 0;
                queue[tail] = n;
                tail += 1;
            }
        }
        let mut best: Option<(u8, f32)> = None;
        while head < tail {
            let n = queue[head];
            head += 1;
            let d = dist[n as usize];
            if d >= 3 {
                continue;
            }
            for k in 0..3 {
                let e = topo.node_edges[n as usize][k];
                if e == NONE {
                    break;
                }
                if s.edge_owner[e as usize] != NONE {
                    continue;
                }
                let m = topo.node_neighbors[n as usize][k];
                if dist[m as usize] != u8::MAX {
                    continue;
                }
                let mo = s.node_owner[m as usize];
                if mo != NONE && mo != p {
                    continue;
                }
                dist[m as usize] = d + 1;
                first[m as usize] = if d == 0 { e } else { first[n as usize] };
                queue[tail] = m;
                tail += 1;
                if s.node_buildable(m) {
                    let v = self.node_value(c, m) * 0.7f32.powi(d as i32);
                    if best.is_none_or(|(_, bv)| v > bv) {
                        best = Some((first[m as usize], v));
                    }
                }
            }
        }
        best
    }

    /// Best road for Longest Road, valued in "VP-equivalents" (2 = takes it now).
    fn lr_road(&self, c: &Ctx) -> Option<(u8, f32)> {
        if self.w.longest_road_weight <= 0.0 {
            return None;
        }
        let s = c.s;
        let p = c.p;
        let cur = s.road_length[p as usize];
        let others = (0..NUM_PLAYERS as u8)
            .filter(|&o| o != p)
            .map(|o| s.road_length[o as usize])
            .max()
            .unwrap_or(0);
        if s.longest_road_owner == p && cur > others + 1 {
            return None;
        }
        let target = (others + 1).max(5);
        let mut best: Option<(u8, f32)> = None;
        for e in 0..NUM_EDGES as u8 {
            if !s.can_build_road_at(p, e) {
                continue;
            }
            let mut s2 = *s;
            s2.edge_owner[e as usize] = p;
            let len = s2.longest_road(p);
            if len > cur {
                let deficit = target.saturating_sub(len) as f32;
                let v = self.w.longest_road_weight * 2.0 / (1.0 + deficit);
                if best.is_none_or(|(_, bv)| v > bv) {
                    best = Some((e, v));
                }
            }
        }
        best
    }

    fn choose_goal(&self, c: &Ctx) -> Goal {
        let s = c.s;
        let p = c.p;
        let pi = p as usize;
        let w = &self.w;
        let hand = c.hand();
        let mut options: Vec<(Goal, f32)> = Vec::with_capacity(4);

        if s.cities_left[pi] > 0
            && let Some((n, v)) = self.best_city_spot(c)
        {
            options.push((Goal::City(n), w.priority_city * (1.0 + v / 10.0)));
        }
        let spot = if s.settlements_left[pi] > 0 {
            self.best_settlement_spot(c)
        } else {
            None
        };
        if let Some((n, v)) = spot {
            options.push((
                Goal::Settlement(n),
                w.priority_settlement * (1.0 + v / 10.0),
            ));
        }
        if s.roads_left[pi] > 0 {
            let mut road: Option<(u8, f32)> = None;
            if spot.is_none()
                && s.settlements_left[pi] > 0
                && let Some((e, v)) = self.road_plan(c)
            {
                road = Some((e, w.priority_road * (1.0 + v / 10.0)));
            }
            if let Some((e, v)) = self.lr_road(c)
                && road.is_none_or(|(_, rv)| v > rv)
            {
                road = Some((e, v));
            }
            if let Some((e, v)) = road {
                options.push((Goal::Road(e), v));
            }
        }
        if s.dev_deck_remaining() > 0 && w.priority_dev > 0.0 {
            let knights = s.knights_played[pi] + s.dev_hand[pi][KNIGHT] + s.dev_new[pi][KNIGHT];
            let holder = s.largest_army_owner;
            let la_close = if holder == p {
                0.2
            } else {
                let need = if holder == NONE {
                    3
                } else {
                    s.knights_played[holder as usize] + 1
                };
                1.0 / (1.0 + need.saturating_sub(knights) as f32)
            };
            options.push((
                Goal::Dev,
                w.priority_dev * (1.0 + w.largest_army_weight * la_close),
            ));
        }

        let mut best = Goal::Nothing;
        let mut best_v = 0.0;
        for (g, d) in options {
            let miss: u8 = missing(hand, &g.cost()).iter().sum();
            let v = d / (1.0 + 0.35 * miss as f32);
            if v > best_v {
                best_v = v;
                best = g;
            }
        }
        best
    }

    // ------------------------------------------------------------------
    // Decisions
    // ------------------------------------------------------------------

    fn should_play_knight(&self, c: &Ctx) -> bool {
        let s = c.s;
        let p = c.p;
        let k = s.knights_played[p as usize] + 1;
        let holder = s.largest_army_owner;
        let takes_la =
            holder != p && k >= 3 && (holder == NONE || k > s.knights_played[holder as usize]);
        if takes_la && s.total_vp(p) + 2 >= VP_TO_WIN {
            return true;
        }
        let mut score = 0.0;
        if robber_hurts(s, p) {
            score += 1.0;
        }
        score += self.w.largest_army_weight
            * if takes_la {
                1.0
            } else if holder != p {
                0.15 * k as f32
            } else {
                0.0
            };
        score += self.w.knight_aggression * 0.35;
        score >= 0.6
    }

    fn setup_road(&self, c: &Ctx, legal: &[Action]) -> Action {
        let s = c.s;
        let topo = topology();
        let from = s.last_settlement;
        self.argmax(legal, |a| match *a {
            Action::BuildRoad(e) => {
                let m = topo.other_end(e, from);
                let mut best = 0.0f32;
                for &x in &topo.node_neighbors[m as usize] {
                    if x != NONE && x != from && s.node_buildable(x) {
                        best = best.max(self.node_value(c, x));
                    }
                }
                Some(best)
            }
            _ => None,
        })
        .unwrap_or(legal[0])
    }

    fn free_road(&self, c: &Ctx, legal: &[Action]) -> Action {
        let plan = self.lr_road(c).or_else(|| self.road_plan(c));
        if let Some((e, _)) = plan
            && legal.contains(&Action::BuildRoad(e))
        {
            return Action::BuildRoad(e);
        }
        legal[self.pick(legal.len())]
    }

    fn robber(&self, c: &Ctx, legal: &[Action]) -> Action {
        let s = c.s;
        let p = c.p;
        let leader_w = self.w.robber_leader_weight;
        let topo = topology();
        self.argmax(legal, |a| match *a {
            Action::MoveRobber(t) => {
                let pip = pips(s.board.tile_number[t as usize]) as f32;
                let mut v = 0.0;
                for &n in &topo.tile_nodes[t as usize] {
                    let o = s.node_owner[n as usize];
                    if o == p {
                        v -= 100.0;
                    } else if o != NONE {
                        let mult = if s.node_city[n as usize] { 2.0 } else { 1.0 };
                        v += pip * mult * (1.0 + leader_w * s.public_vp(o) as f32 / 5.0);
                        if s.hand_size(o) > 0 {
                            v += 0.5;
                        }
                    }
                }
                Some(v)
            }
            _ => None,
        })
        .unwrap()
    }

    fn steal(&self, c: &Ctx, legal: &[Action]) -> Action {
        let s = c.s;
        self.argmax(legal, |a| match *a {
            Action::Steal(v) => Some(s.public_vp(v) as f32 + 0.3 * s.hand_size(v) as f32),
            _ => None,
        })
        .unwrap()
    }

    fn discard(&self, c: &Ctx, legal: &[Action]) -> Action {
        let cost = self.choose_goal(c).cost();
        let hand = *c.hand();
        let weights = self.w.resource_weights;
        // Discard the card we value least (min of keep-value).
        self.argmax(legal, |a| match *a {
            Action::Discard(r) => {
                let r = r as usize;
                let needed = if hand[r] <= cost[r] { 10.0 } else { 0.0 };
                Some(-(needed + weights[r] - 0.5 * hand[r] as f32))
            }
            _ => None,
        })
        .unwrap()
    }

    fn play_dev_main(&self, c: &Ctx, legal: &[Action], goal: Goal) -> Option<Action> {
        let s = c.s;
        let p = c.p;
        let pi = p as usize;
        if s.dev_played {
            return None;
        }
        if legal.contains(&Action::PlayKnight) && self.should_play_knight(c) {
            return Some(Action::PlayKnight);
        }
        if legal.contains(&Action::PlayRoadBuilding) {
            let wants_roads = matches!(goal, Goal::Road(_))
                || (s.settlements_left[pi] > 0 && self.best_settlement_spot(c).is_none());
            if wants_roads {
                return Some(Action::PlayRoadBuilding);
            }
        }
        let hand = c.hand();
        if s.dev_hand[pi][YEAR_OF_PLENTY] > 0 && goal != Goal::Nothing {
            let miss = missing(hand, &goal.cost());
            let mut wants: Vec<u8> = Vec::new();
            for r in 0..NUM_RESOURCES {
                for _ in 0..miss[r] {
                    wants.push(r as u8);
                }
            }
            if !wants.is_empty() {
                let a = wants[0];
                let b = if wants.len() > 1 {
                    wants[1]
                } else {
                    // Fill with the most valuable resource we hold least of.
                    (0..NUM_RESOURCES as u8)
                        .max_by(|&x, &y| {
                            let vx =
                                self.w.resource_weights[x as usize] - 0.3 * hand[x as usize] as f32;
                            let vy =
                                self.w.resource_weights[y as usize] - 0.3 * hand[y as usize] as f32;
                            vx.partial_cmp(&vy).unwrap()
                        })
                        .unwrap()
                };
                let act = Action::PlayYearOfPlenty(a.min(b), a.max(b));
                if legal.contains(&act) {
                    return Some(act);
                }
            }
        }
        if s.dev_hand[pi][MONOPOLY] > 0 {
            let need = missing(hand, &goal.cost());
            let mut best: Option<(u8, f32)> = None;
            for r in 0..NUM_RESOURCES {
                let mut est = 0.0;
                for o in 0..NUM_PLAYERS {
                    if o == pi {
                        continue;
                    }
                    let total: u32 = c.prod[o].iter().map(|&x| x as u32).sum();
                    let share = if total == 0 {
                        0.2
                    } else {
                        c.prod[o][r] as f32 / total as f32
                    };
                    est += s.hand_size(o as u8) as f32 * share;
                }
                if need[r] > 0 {
                    est *= 1.5;
                }
                if best.is_none_or(|(_, bv)| est > bv) {
                    best = Some((r as u8, est));
                }
            }
            if let Some((r, est)) = best
                && est >= 2.5
            {
                return Some(Action::PlayMonopoly(r));
            }
        }
        None
    }

    fn trade_toward(&self, c: &Ctx, legal: &[Action], goal: Goal) -> Option<Action> {
        if self.w.trade_aggression <= 0.0 || goal == Goal::Nothing {
            return None;
        }
        let s = c.s;
        let hand = c.hand();
        let cost = goal.cost();
        let miss = missing(hand, &cost);
        let total_missing: u8 = miss.iter().sum();
        let max_trades = (self.w.trade_aggression * 3.0).round() as u8;
        if total_missing == 0 || total_missing > max_trades {
            return None;
        }
        let mut conversions = 0u8;
        let mut give_best: Option<(u8, u8, u8)> = None; // (resource, rate, surplus)
        for give in 0..NUM_RESOURCES as u8 {
            if miss[give as usize] > 0 {
                continue;
            }
            let surplus = hand[give as usize].saturating_sub(cost[give as usize]);
            let rate = s.trade_rate(c.p, give);
            conversions += surplus / rate;
            if surplus >= rate {
                let better = match give_best {
                    None => true,
                    Some((_, br, bs)) => rate < br || (rate == br && surplus > bs),
                };
                if better {
                    give_best = Some((give, rate, surplus));
                }
            }
        }
        if conversions < total_missing {
            return None;
        }
        let (give, _, _) = give_best?;
        let get = (0..NUM_RESOURCES as u8)
            .max_by_key(|&r| miss[r as usize])
            .unwrap();
        let a = Action::MaritimeTrade(give, get);
        legal.contains(&a).then_some(a)
    }

    fn opportunistic(&self, c: &Ctx, legal: &[Action], goal: Goal) -> Option<Action> {
        let s = c.s;
        let p = c.p;
        let pi = p as usize;
        let hand = c.hand();
        let cost = goal.cost();
        let mut spare = [0u8; NUM_RESOURCES];
        for r in 0..NUM_RESOURCES {
            spare[r] = hand[r] - hand[r].min(cost[r]);
        }
        let fits = |k: &Cost| (0..NUM_RESOURCES).all(|r| spare[r] >= k[r]);

        if fits(&COST_SETTLEMENT)
            && let Some(a) = self.argmax(legal, |a| match *a {
                Action::BuildSettlement(n) => Some(self.node_value(c, n)),
                _ => None,
            })
        {
            return Some(a);
        }
        if fits(&COST_CITY)
            && let Some(a) = self.argmax(legal, |a| match *a {
                Action::BuildCity(n) => Some(self.city_value(c, n)),
                _ => None,
            })
        {
            return Some(a);
        }
        if fits(&COST_DEV) && self.w.priority_dev >= 0.4 && legal.contains(&Action::BuyDev) {
            return Some(Action::BuyDev);
        }
        if fits(&COST_ROAD) && !matches!(goal, Goal::Road(_)) && s.roads_left[pi] > 0 {
            let plan = self.lr_road(c).or_else(|| {
                if self.w.priority_road >= 0.5 && s.settlements_left[pi] > 0 {
                    self.road_plan(c)
                } else {
                    None
                }
            });
            if let Some((e, _)) = plan {
                let a = Action::BuildRoad(e);
                if legal.contains(&a) {
                    return Some(a);
                }
            }
        }
        None
    }

    fn protect_hand(&self, c: &Ctx, legal: &[Action], goal: Goal) -> Option<Action> {
        let s = c.s;
        let p = c.p;
        let limit = HAND_LIMIT as f32 + ((1.0 - self.w.discard_avoidance) * 4.0).round();
        if (s.hand_size(p) as f32) <= limit {
            return None;
        }
        // Any build at all.
        if let Some(a) = self.argmax(legal, |a| match *a {
            Action::BuildSettlement(n) => Some(100.0 + self.node_value(c, n)),
            Action::BuildCity(n) => Some(90.0 + self.city_value(c, n)),
            Action::BuyDev => Some(50.0),
            _ => None,
        }) {
            return Some(a);
        }
        // Otherwise trade our most plentiful card for what the goal needs
        // (or our scarcest resource).
        let hand = c.hand();
        let miss = missing(hand, &goal.cost());
        let get = if miss.iter().any(|&m| m > 0) {
            (0..NUM_RESOURCES as u8)
                .max_by_key(|&r| miss[r as usize])
                .unwrap()
        } else {
            (0..NUM_RESOURCES as u8)
                .min_by_key(|&r| hand[r as usize])
                .unwrap()
        };
        let give = (0..NUM_RESOURCES as u8)
            .filter(|&r| r != get && hand[r as usize] >= s.trade_rate(p, r))
            .max_by_key(|&r| hand[r as usize])?;
        let a = Action::MaritimeTrade(give, get);
        legal.contains(&a).then_some(a)
    }

    fn main_phase(&self, c: &Ctx, legal: &[Action]) -> Action {
        let goal = self.choose_goal(c);
        if let Some(a) = self.play_dev_main(c, legal, goal) {
            return a;
        }
        if let Some(a) = goal.action()
            && legal.contains(&a)
        {
            return a;
        }
        if let Some(a) = self.trade_toward(c, legal, goal) {
            return a;
        }
        if let Some(a) = self.opportunistic(c, legal, goal) {
            return a;
        }
        if let Some(a) = self.protect_hand(c, legal, goal) {
            return a;
        }
        Action::EndTurn
    }
}

impl Bot for HeuristicBot {
    fn act(&mut self, s: &GameState, legal: &[Action]) -> Action {
        if legal.len() == 1 {
            return legal[0];
        }
        let c = Ctx::new(s);
        match s.phase {
            Phase::SetupSettlement => self
                .argmax(legal, |a| match *a {
                    Action::BuildSettlement(n) => Some(self.node_value(&c, n)),
                    _ => None,
                })
                .unwrap(),
            Phase::SetupRoad => self.setup_road(&c, legal),
            Phase::PreRoll => {
                if legal.contains(&Action::PlayKnight) && self.should_play_knight(&c) {
                    Action::PlayKnight
                } else {
                    Action::Roll
                }
            }
            Phase::Discard => self.discard(&c, legal),
            Phase::MoveRobber => self.robber(&c, legal),
            Phase::Steal => self.steal(&c, legal),
            Phase::RoadBuilding => self.free_road(&c, legal),
            Phase::Main => self.main_phase(&c, legal),
            Phase::GameOver => legal[0],
        }
    }
}

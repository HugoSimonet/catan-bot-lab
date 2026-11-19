//! Game state and the full base-game rules (bank/harbour trading only).
//!
//! `GameState` is `Copy` and heap-free, so cloning it for search or for
//! "what if" evaluation by bots is cheap.
//!
//! Every random event draws from a stream seeded by the game seed:
//! stream 0 = board, 1 = dice, 2 = dev deck, 3 = steals. A game is therefore
//! fully determined by its seed plus the list of actions taken, and two
//! matchups played on the same seed see the same board and the same dice.

use serde::Serialize;

use crate::actions::{Action, NUM_ACTIONS, YOP_PAIRS};
use crate::board::{
    Board, DESERT, NONE, NUM_EDGES, NUM_NODES, NUM_RESOURCES, NUM_TILES, PORT_GENERIC, topology,
};
use crate::rng::Rng;

pub const NUM_PLAYERS: usize = 4;
pub const VP_TO_WIN: u8 = 10;
/// Player-turns after which the game is recorded as a draw.
pub const MAX_TURNS: u16 = 1000;
/// Hard cap on the number of actions (safety net against degenerate policies).
pub const MAX_ACTIONS: u32 = 30_000;
pub const HAND_LIMIT: u8 = 7;

pub const KNIGHT: usize = 0;
pub const VICTORY_POINT: usize = 1;
pub const ROAD_BUILDING: usize = 2;
pub const YEAR_OF_PLENTY: usize = 3;
pub const MONOPOLY: usize = 4;
pub const NUM_DEV_TYPES: usize = 5;
pub const DEV_NAMES: [&str; NUM_DEV_TYPES] = [
    "knight",
    "victory_point",
    "road_building",
    "year_of_plenty",
    "monopoly",
];
const DEV_DECK: [(usize, u8); NUM_DEV_TYPES] = [
    (KNIGHT, 14),
    (VICTORY_POINT, 5),
    (ROAD_BUILDING, 2),
    (YEAR_OF_PLENTY, 2),
    (MONOPOLY, 2),
];
pub const DEV_DECK_SIZE: usize = 25;

pub type Cost = [u8; NUM_RESOURCES];
pub const COST_ROAD: Cost = [1, 1, 0, 0, 0];
pub const COST_SETTLEMENT: Cost = [1, 1, 1, 1, 0];
pub const COST_CITY: Cost = [0, 0, 0, 2, 3];
pub const COST_DEV: Cost = [0, 0, 1, 1, 1];

/// The order players place their initial settlements (snake draft).
pub const SETUP_ORDER: [u8; 8] = [0, 1, 2, 3, 3, 2, 1, 0];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Phase {
    SetupSettlement,
    SetupRoad,
    /// Start of a turn: roll, or play a Knight first.
    PreRoll,
    /// After a 7: `acting` must discard one card at a time.
    Discard,
    MoveRobber,
    /// Choose which adjacent opponent to steal from.
    Steal,
    Main,
    /// Placing the free roads from a Road Building card.
    RoadBuilding,
    GameOver,
}

impl Phase {
    pub const ALL: [Phase; 9] = [
        Phase::SetupSettlement,
        Phase::SetupRoad,
        Phase::PreRoll,
        Phase::Discard,
        Phase::MoveRobber,
        Phase::Steal,
        Phase::Main,
        Phase::RoadBuilding,
        Phase::GameOver,
    ];
    pub fn index(self) -> usize {
        self as usize
    }
}

/// Per-player counters kept for analysis (not used by the rules).
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Stats {
    pub produced: [[u16; NUM_RESOURCES]; NUM_PLAYERS],
    /// Production lost because the robber sat on the tile.
    pub robbed_production: [u16; NUM_PLAYERS],
    pub dev_bought: [u8; NUM_PLAYERS],
    pub maritime_trades: [u16; NUM_PLAYERS],
    pub cards_discarded: [u16; NUM_PLAYERS],
    pub cards_stolen_from: [u16; NUM_PLAYERS],
    /// Turn number of each player's first city (0 = never).
    pub first_city_turn: [u16; NUM_PLAYERS],
    /// Production pips per resource right after the setup phase (opening choice).
    pub opening_pips: [[u8; NUM_RESOURCES]; NUM_PLAYERS],
    /// Whether each player's opening settlements touch a harbour.
    pub opening_port: [bool; NUM_PLAYERS],
}

#[derive(Clone, Copy, Debug)]
pub struct GameState {
    pub seed: u64,
    pub board: Board,
    pub node_owner: [u8; NUM_NODES],
    pub node_city: [bool; NUM_NODES],
    pub edge_owner: [u8; NUM_EDGES],
    pub robber: u8,
    pub hands: [[u8; NUM_RESOURCES]; NUM_PLAYERS],
    pub bank: [u8; NUM_RESOURCES],
    /// Playable development cards (bought before this turn).
    pub dev_hand: [[u8; NUM_DEV_TYPES]; NUM_PLAYERS],
    /// Development cards bought this turn (not yet playable).
    pub dev_new: [[u8; NUM_DEV_TYPES]; NUM_PLAYERS],
    pub dev_deck: [u8; DEV_DECK_SIZE],
    pub dev_deck_pos: u8,
    pub knights_played: [u8; NUM_PLAYERS],
    pub settlements_left: [u8; NUM_PLAYERS],
    pub cities_left: [u8; NUM_PLAYERS],
    pub roads_left: [u8; NUM_PLAYERS],
    /// Bitmask of harbours each player owns: bit r = 2:1 of resource r, bit 5 = 3:1.
    pub ports: [u8; NUM_PLAYERS],
    pub road_length: [u8; NUM_PLAYERS],
    pub longest_road_owner: u8,
    pub largest_army_owner: u8,
    /// Whose turn it is.
    pub current: u8,
    /// Who must make the next decision (differs from `current` while discarding).
    pub acting: u8,
    pub phase: Phase,
    pub has_rolled: bool,
    pub dev_played: bool,
    pub free_roads: u8,
    pub discard_left: [u8; NUM_PLAYERS],
    /// Bitmask of players that can be stolen from (Steal phase).
    pub steal_candidates: u8,
    /// Number of completed (settlement, road) placements in the setup phase.
    pub setup_step: u8,
    pub last_settlement: u8,
    pub last_roll: u8,
    /// Player-turns completed (0 during setup).
    pub turn: u16,
    pub num_actions: u32,
    pub winner: u8,
    pub dice_rng: Rng,
    pub steal_rng: Rng,
    pub stats: Stats,
}

impl GameState {
    pub fn new(seed: u64) -> Self {
        let board = Board::random(&mut Rng::stream(seed, 0));
        Self::with_board(board, seed)
    }

    pub fn with_board(board: Board, seed: u64) -> Self {
        let mut dev_deck = [0u8; DEV_DECK_SIZE];
        let mut k = 0;
        for (card, count) in DEV_DECK {
            for _ in 0..count {
                dev_deck[k] = card as u8;
                k += 1;
            }
        }
        Rng::stream(seed, 2).shuffle(&mut dev_deck);

        GameState {
            seed,
            board,
            node_owner: [NONE; NUM_NODES],
            node_city: [false; NUM_NODES],
            edge_owner: [NONE; NUM_EDGES],
            robber: board.desert(),
            hands: [[0; NUM_RESOURCES]; NUM_PLAYERS],
            bank: [19; NUM_RESOURCES],
            dev_hand: [[0; NUM_DEV_TYPES]; NUM_PLAYERS],
            dev_new: [[0; NUM_DEV_TYPES]; NUM_PLAYERS],
            dev_deck,
            dev_deck_pos: 0,
            knights_played: [0; NUM_PLAYERS],
            settlements_left: [5; NUM_PLAYERS],
            cities_left: [4; NUM_PLAYERS],
            roads_left: [15; NUM_PLAYERS],
            ports: [0; NUM_PLAYERS],
            road_length: [0; NUM_PLAYERS],
            longest_road_owner: NONE,
            largest_army_owner: NONE,
            current: SETUP_ORDER[0],
            acting: SETUP_ORDER[0],
            phase: Phase::SetupSettlement,
            has_rolled: false,
            dev_played: false,
            free_roads: 0,
            discard_left: [0; NUM_PLAYERS],
            steal_candidates: 0,
            setup_step: 0,
            last_settlement: NONE,
            last_roll: 0,
            turn: 0,
            num_actions: 0,
            winner: NONE,
            dice_rng: Rng::stream(seed, 1),
            steal_rng: Rng::stream(seed, 3),
            stats: Stats::default(),
        }
    }

    // ------------------------------------------------------------------
    // Queries
    // ------------------------------------------------------------------

    #[inline]
    pub fn is_over(&self) -> bool {
        self.phase == Phase::GameOver
    }

    #[inline]
    pub fn is_setup(&self) -> bool {
        matches!(self.phase, Phase::SetupSettlement | Phase::SetupRoad)
    }

    #[inline]
    pub fn hand_size(&self, p: u8) -> u8 {
        self.hands[p as usize].iter().sum()
    }

    /// Total development cards held (public information: the count, not the types).
    #[inline]
    pub fn dev_cards_held(&self, p: u8) -> u8 {
        let p = p as usize;
        self.dev_hand[p].iter().sum::<u8>() + self.dev_new[p].iter().sum::<u8>()
    }

    #[inline]
    pub fn settlements_built(&self, p: u8) -> u8 {
        5 - self.settlements_left[p as usize]
    }

    #[inline]
    pub fn cities_built(&self, p: u8) -> u8 {
        4 - self.cities_left[p as usize]
    }

    #[inline]
    pub fn roads_built(&self, p: u8) -> u8 {
        15 - self.roads_left[p as usize]
    }

    /// Victory points visible to everyone (excludes hidden VP cards).
    pub fn public_vp(&self, p: u8) -> u8 {
        self.settlements_built(p)
            + 2 * self.cities_built(p)
            + if self.longest_road_owner == p { 2 } else { 0 }
            + if self.largest_army_owner == p { 2 } else { 0 }
    }

    #[inline]
    pub fn vp_cards(&self, p: u8) -> u8 {
        self.dev_hand[p as usize][VICTORY_POINT] + self.dev_new[p as usize][VICTORY_POINT]
    }

    #[inline]
    pub fn total_vp(&self, p: u8) -> u8 {
        self.public_vp(p) + self.vp_cards(p)
    }

    #[inline]
    pub fn can_afford(&self, p: u8, cost: &Cost) -> bool {
        let h = &self.hands[p as usize];
        (0..NUM_RESOURCES).all(|r| h[r] >= cost[r])
    }

    /// Best bank/harbour rate for giving resource `give`.
    #[inline]
    pub fn trade_rate(&self, p: u8, give: u8) -> u8 {
        let ports = self.ports[p as usize];
        if ports & (1 << give) != 0 {
            2
        } else if ports & (1 << PORT_GENERIC) != 0 {
            3
        } else {
            4
        }
    }

    #[inline]
    pub fn dev_deck_remaining(&self) -> u8 {
        DEV_DECK_SIZE as u8 - self.dev_deck_pos
    }

    #[inline]
    pub fn can_play_dev(&self, p: u8, card: usize) -> bool {
        !self.dev_played && self.dev_hand[p as usize][card] > 0
    }

    /// Node is empty and no neighbouring node is occupied (distance rule).
    pub fn node_buildable(&self, n: u8) -> bool {
        if self.node_owner[n as usize] != NONE {
            return false;
        }
        topology().node_neighbors[n as usize]
            .iter()
            .all(|&m| m == NONE || self.node_owner[m as usize] == NONE)
    }

    /// Player has a road touching node `n`.
    pub fn node_has_road(&self, p: u8, n: u8) -> bool {
        topology().node_edges[n as usize]
            .iter()
            .any(|&e| e != NONE && self.edge_owner[e as usize] == p)
    }

    pub fn can_build_settlement_at(&self, p: u8, n: u8) -> bool {
        self.node_buildable(n) && self.node_has_road(p, n)
    }

    /// Edge is empty and connects to the player's network (not through an
    /// opponent's building).
    pub fn can_build_road_at(&self, p: u8, e: u8) -> bool {
        if self.edge_owner[e as usize] != NONE {
            return false;
        }
        let topo = topology();
        for &n in &topo.edge_nodes[e as usize] {
            let owner = self.node_owner[n as usize];
            if owner == p {
                return true;
            }
            if owner == NONE
                && topo.node_edges[n as usize]
                    .iter()
                    .any(|&e2| e2 != NONE && e2 != e && self.edge_owner[e2 as usize] == p)
            {
                return true;
            }
        }
        false
    }

    pub fn any_road_buildable(&self, p: u8) -> bool {
        (0..NUM_EDGES as u8).any(|e| self.can_build_road_at(p, e))
    }

    /// Length of the player's longest continuous road (a trail may end at,
    /// but not pass through, an opponent's building).
    pub fn longest_road(&self, p: u8) -> u8 {
        let topo = topology();
        let mut best = 0;
        for n in 0..NUM_NODES as u8 {
            let mine = topo.node_edges[n as usize]
                .iter()
                .filter(|&&e| e != NONE && self.edge_owner[e as usize] == p)
                .count();
            // Trails worth starting from: dead ends, junctions, or blocked nodes.
            // (Starting from every node with a road is also correct, just slower.)
            if mine == 0 {
                continue;
            }
            best = best.max(self.road_dfs(p, n, 0, true));
        }
        best
    }

    fn road_dfs(&self, p: u8, node: u8, used: u128, is_start: bool) -> u8 {
        if !is_start {
            let owner = self.node_owner[node as usize];
            if owner != NONE && owner != p {
                return 0;
            }
        }
        let topo = topology();
        let mut best = 0;
        for k in 0..3 {
            let e = topo.node_edges[node as usize][k];
            if e == NONE {
                break;
            }
            if self.edge_owner[e as usize] == p && used & (1u128 << e) == 0 {
                let next = topo.node_neighbors[node as usize][k];
                best = best.max(1 + self.road_dfs(p, next, used | (1u128 << e), false));
            }
        }
        best
    }

    /// Resource production in pips (dice ways out of 36) for each player and
    /// resource, ignoring the robber. Cities count double.
    pub fn production_pips(&self) -> [[u8; NUM_RESOURCES]; NUM_PLAYERS] {
        let topo = topology();
        let mut out = [[0u8; NUM_RESOURCES]; NUM_PLAYERS];
        for n in 0..NUM_NODES {
            let owner = self.node_owner[n];
            if owner == NONE {
                continue;
            }
            let mult = if self.node_city[n] { 2 } else { 1 };
            for &t in &topo.node_tiles[n] {
                if t == NONE {
                    continue;
                }
                let res = self.board.tile_resource[t as usize];
                if res != DESERT {
                    out[owner as usize][res as usize] +=
                        mult * crate::board::pips(self.board.tile_number[t as usize]);
                }
            }
        }
        out
    }

    // ------------------------------------------------------------------
    // Legal actions
    // ------------------------------------------------------------------

    /// Fill `out` with every legal action for `self.acting`.
    pub fn legal_actions(&self, out: &mut Vec<Action>) {
        out.clear();
        let p = self.acting;
        let pi = p as usize;
        let topo = topology();
        match self.phase {
            Phase::SetupSettlement => {
                for n in 0..NUM_NODES as u8 {
                    if self.node_buildable(n) {
                        out.push(Action::BuildSettlement(n));
                    }
                }
            }
            Phase::SetupRoad => {
                for &e in &topo.node_edges[self.last_settlement as usize] {
                    if e != NONE && self.edge_owner[e as usize] == NONE {
                        out.push(Action::BuildRoad(e));
                    }
                }
            }
            Phase::PreRoll => {
                out.push(Action::Roll);
                if self.can_play_dev(p, KNIGHT) {
                    out.push(Action::PlayKnight);
                }
            }
            Phase::Discard => {
                for r in 0..NUM_RESOURCES as u8 {
                    if self.hands[pi][r as usize] > 0 {
                        out.push(Action::Discard(r));
                    }
                }
            }
            Phase::MoveRobber => {
                for t in 0..NUM_TILES as u8 {
                    if t != self.robber {
                        out.push(Action::MoveRobber(t));
                    }
                }
            }
            Phase::Steal => {
                for v in 0..NUM_PLAYERS as u8 {
                    if self.steal_candidates & (1 << v) != 0 {
                        out.push(Action::Steal(v));
                    }
                }
            }
            Phase::RoadBuilding => {
                for e in 0..NUM_EDGES as u8 {
                    if self.can_build_road_at(p, e) {
                        out.push(Action::BuildRoad(e));
                    }
                }
            }
            Phase::Main => {
                out.push(Action::EndTurn);
                if self.roads_left[pi] > 0 && self.can_afford(p, &COST_ROAD) {
                    for e in 0..NUM_EDGES as u8 {
                        if self.can_build_road_at(p, e) {
                            out.push(Action::BuildRoad(e));
                        }
                    }
                }
                if self.settlements_left[pi] > 0 && self.can_afford(p, &COST_SETTLEMENT) {
                    for n in 0..NUM_NODES as u8 {
                        if self.can_build_settlement_at(p, n) {
                            out.push(Action::BuildSettlement(n));
                        }
                    }
                }
                if self.cities_left[pi] > 0 && self.can_afford(p, &COST_CITY) {
                    for n in 0..NUM_NODES as u8 {
                        if self.node_owner[n as usize] == p && !self.node_city[n as usize] {
                            out.push(Action::BuildCity(n));
                        }
                    }
                }
                if self.dev_deck_remaining() > 0 && self.can_afford(p, &COST_DEV) {
                    out.push(Action::BuyDev);
                }
                if !self.dev_played {
                    let dh = &self.dev_hand[pi];
                    if dh[KNIGHT] > 0 {
                        out.push(Action::PlayKnight);
                    }
                    if dh[ROAD_BUILDING] > 0
                        && self.roads_left[pi] > 0
                        && self.any_road_buildable(p)
                    {
                        out.push(Action::PlayRoadBuilding);
                    }
                    if dh[YEAR_OF_PLENTY] > 0 {
                        for &(a, b) in &YOP_PAIRS {
                            let ok = if a == b {
                                self.bank[a as usize] >= 2
                            } else {
                                self.bank[a as usize] >= 1 && self.bank[b as usize] >= 1
                            };
                            if ok {
                                out.push(Action::PlayYearOfPlenty(a, b));
                            }
                        }
                    }
                    if dh[MONOPOLY] > 0 {
                        for r in 0..NUM_RESOURCES as u8 {
                            out.push(Action::PlayMonopoly(r));
                        }
                    }
                }
                for give in 0..NUM_RESOURCES as u8 {
                    if self.hands[pi][give as usize] >= self.trade_rate(p, give) {
                        for get in 0..NUM_RESOURCES as u8 {
                            if get != give && self.bank[get as usize] > 0 {
                                out.push(Action::MaritimeTrade(give, get));
                            }
                        }
                    }
                }
            }
            Phase::GameOver => {}
        }
    }

    /// Write a boolean mask of length `NUM_ACTIONS` (true = legal).
    pub fn action_mask(&self, buf: &mut Vec<Action>, mask: &mut [bool]) {
        debug_assert_eq!(mask.len(), NUM_ACTIONS);
        mask.fill(false);
        self.legal_actions(buf);
        for a in buf.iter() {
            mask[a.to_index(self.acting)] = true;
        }
    }

    pub fn is_legal(&self, action: Action) -> bool {
        let mut buf = Vec::with_capacity(64);
        self.legal_actions(&mut buf);
        buf.contains(&action)
    }

    // ------------------------------------------------------------------
    // Transitions
    // ------------------------------------------------------------------

    /// Apply a legal action for `self.acting`. Behaviour is unspecified for
    /// illegal actions (use `is_legal` first when the action is untrusted).
    pub fn apply(&mut self, action: Action) {
        debug_assert!(!self.is_over(), "game is over");
        let p = self.acting;
        let pi = p as usize;
        self.num_actions += 1;
        match action {
            Action::Roll => self.roll(),
            Action::EndTurn => self.end_turn(),
            Action::BuildSettlement(n) => {
                if self.phase == Phase::SetupSettlement {
                    self.place_settlement(p, n);
                    if self.setup_step >= 4 {
                        // Second settlement: collect one of each adjacent resource.
                        for &t in &topology().node_tiles[n as usize] {
                            if t == NONE {
                                continue;
                            }
                            let res = self.board.tile_resource[t as usize];
                            if res != DESERT {
                                self.gain_from_bank(p, res, 1);
                            }
                        }
                    }
                    self.last_settlement = n;
                    self.phase = Phase::SetupRoad;
                } else {
                    self.pay(p, &COST_SETTLEMENT);
                    self.place_settlement(p, n);
                }
            }
            Action::BuildCity(n) => {
                self.pay(p, &COST_CITY);
                self.node_city[n as usize] = true;
                self.cities_left[pi] -= 1;
                self.settlements_left[pi] += 1;
                if self.stats.first_city_turn[pi] == 0 {
                    self.stats.first_city_turn[pi] = self.turn.max(1);
                }
            }
            Action::BuildRoad(e) => match self.phase {
                Phase::SetupRoad => {
                    self.place_road(p, e);
                    self.advance_setup();
                }
                Phase::RoadBuilding => {
                    self.place_road(p, e);
                    self.free_roads -= 1;
                    if self.free_roads == 0
                        || self.roads_left[pi] == 0
                        || !self.any_road_buildable(p)
                    {
                        self.free_roads = 0;
                        self.phase = Phase::Main;
                    }
                }
                _ => {
                    self.pay(p, &COST_ROAD);
                    self.place_road(p, e);
                }
            },
            Action::BuyDev => {
                self.pay(p, &COST_DEV);
                let card = self.dev_deck[self.dev_deck_pos as usize] as usize;
                self.dev_deck_pos += 1;
                self.dev_new[pi][card] += 1;
                self.stats.dev_bought[pi] += 1;
            }
            Action::PlayKnight => {
                self.use_dev(p, KNIGHT);
                self.knights_played[pi] += 1;
                self.update_largest_army(p);
                self.phase = Phase::MoveRobber;
            }
            Action::PlayRoadBuilding => {
                self.use_dev(p, ROAD_BUILDING);
                self.free_roads = self.roads_left[pi].min(2);
                self.phase = Phase::RoadBuilding;
            }
            Action::PlayYearOfPlenty(a, b) => {
                self.use_dev(p, YEAR_OF_PLENTY);
                self.gain_from_bank(p, a, 1);
                self.gain_from_bank(p, b, 1);
            }
            Action::PlayMonopoly(r) => {
                self.use_dev(p, MONOPOLY);
                let r = r as usize;
                for o in 0..NUM_PLAYERS {
                    if o != pi {
                        let amount = self.hands[o][r];
                        self.hands[o][r] = 0;
                        self.hands[pi][r] += amount;
                    }
                }
            }
            Action::MoveRobber(t) => self.move_robber(p, t),
            Action::Steal(v) => {
                self.steal(p, v);
                self.finish_robber();
            }
            Action::MaritimeTrade(give, get) => {
                let rate = self.trade_rate(p, give);
                self.hands[pi][give as usize] -= rate;
                self.bank[give as usize] += rate;
                self.gain_from_bank(p, get, 1);
                self.stats.maritime_trades[pi] += 1;
            }
            Action::Discard(r) => {
                self.hands[pi][r as usize] -= 1;
                self.bank[r as usize] += 1;
                self.discard_left[pi] -= 1;
                self.stats.cards_discarded[pi] += 1;
                if self.discard_left[pi] == 0 {
                    self.next_discarder_or_robber();
                }
            }
        }
        self.check_game_end();
    }

    fn check_game_end(&mut self) {
        if self.is_setup() || self.phase == Phase::GameOver {
            return;
        }
        if self.total_vp(self.current) >= VP_TO_WIN {
            self.winner = self.current;
            self.phase = Phase::GameOver;
        } else if self.num_actions >= MAX_ACTIONS {
            self.phase = Phase::GameOver;
        }
    }

    #[inline]
    fn pay(&mut self, p: u8, cost: &Cost) {
        let pi = p as usize;
        for r in 0..NUM_RESOURCES {
            self.hands[pi][r] -= cost[r];
            self.bank[r] += cost[r];
        }
    }

    #[inline]
    fn gain_from_bank(&mut self, p: u8, res: u8, amount: u8) {
        self.bank[res as usize] -= amount;
        self.hands[p as usize][res as usize] += amount;
    }

    fn use_dev(&mut self, p: u8, card: usize) {
        self.dev_hand[p as usize][card] -= 1;
        self.dev_played = true;
    }

    fn place_settlement(&mut self, p: u8, n: u8) {
        let topo = topology();
        self.node_owner[n as usize] = p;
        self.settlements_left[p as usize] -= 1;
        let port = self.board.node_port[n as usize];
        if port != NONE {
            self.ports[p as usize] |= 1 << port;
        }
        // A new settlement can cut an opponent's road.
        let mut affected = 0u8;
        for &e in &topo.node_edges[n as usize] {
            if e != NONE {
                let owner = self.edge_owner[e as usize];
                if owner != NONE && owner != p {
                    affected |= 1 << owner;
                }
            }
        }
        if affected != 0 {
            for o in 0..NUM_PLAYERS as u8 {
                if affected & (1 << o) != 0 {
                    self.road_length[o as usize] = self.longest_road(o);
                }
            }
            self.update_longest_road_owner();
        }
    }

    fn place_road(&mut self, p: u8, e: u8) {
        self.edge_owner[e as usize] = p;
        self.roads_left[p as usize] -= 1;
        self.road_length[p as usize] = self.longest_road(p);
        self.update_longest_road_owner();
    }

    /// Longest Road: needs at least 5; the holder keeps it on ties; if the
    /// holder is overtaken by a break and several players tie, nobody holds it.
    fn update_longest_road_owner(&mut self) {
        let lens = self.road_length;
        let max = *lens.iter().max().unwrap();
        let holder = self.longest_road_owner;
        if holder != NONE && lens[holder as usize] == max && max >= 5 {
            return;
        }
        if max < 5 {
            self.longest_road_owner = NONE;
            return;
        }
        let mut leader = NONE;
        let mut count = 0;
        for (p, &l) in lens.iter().enumerate() {
            if l == max {
                leader = p as u8;
                count += 1;
            }
        }
        self.longest_road_owner = if count == 1 { leader } else { NONE };
    }

    fn update_largest_army(&mut self, p: u8) {
        let k = self.knights_played[p as usize];
        let holder = self.largest_army_owner;
        if k >= 3 && holder != p && (holder == NONE || k > self.knights_played[holder as usize]) {
            self.largest_army_owner = p;
        }
    }

    fn advance_setup(&mut self) {
        self.setup_step += 1;
        if self.setup_step as usize == SETUP_ORDER.len() {
            self.stats.opening_pips = self.production_pips();
            for p in 0..NUM_PLAYERS {
                self.stats.opening_port[p] = self.ports[p] != 0;
            }
            self.current = 0;
            self.acting = 0;
            self.phase = Phase::PreRoll;
        } else {
            let next = SETUP_ORDER[self.setup_step as usize];
            self.current = next;
            self.acting = next;
            self.phase = Phase::SetupSettlement;
        }
    }

    fn roll(&mut self) {
        let roll = self.dice_rng.roll_2d6();
        self.resolve_roll(roll);
    }

    pub(crate) fn resolve_roll(&mut self, roll: u8) {
        self.last_roll = roll;
        self.has_rolled = true;
        if roll == 7 {
            for o in 0..NUM_PLAYERS as u8 {
                let h = self.hand_size(o);
                if h > HAND_LIMIT {
                    self.discard_left[o as usize] = h / 2;
                }
            }
            self.next_discarder_or_robber();
        } else {
            self.produce(roll);
            self.phase = Phase::Main;
        }
    }

    fn produce(&mut self, roll: u8) {
        let topo = topology();
        let mut demand = [[0u8; NUM_RESOURCES]; NUM_PLAYERS];
        for t in 0..NUM_TILES {
            if self.board.tile_number[t] != roll {
                continue;
            }
            let res = self.board.tile_resource[t] as usize;
            for &n in &topo.tile_nodes[t] {
                let owner = self.node_owner[n as usize];
                if owner == NONE {
                    continue;
                }
                let amount = if self.node_city[n as usize] { 2 } else { 1 };
                if t as u8 == self.robber {
                    self.stats.robbed_production[owner as usize] += amount as u16;
                } else {
                    demand[owner as usize][res] += amount;
                }
            }
        }
        for r in 0..NUM_RESOURCES {
            let total: u8 = (0..NUM_PLAYERS).map(|p| demand[p][r]).sum();
            if total == 0 {
                continue;
            }
            let claimants = (0..NUM_PLAYERS).filter(|&p| demand[p][r] > 0).count();
            for p in 0..NUM_PLAYERS {
                let want = demand[p][r];
                if want == 0 {
                    continue;
                }
                // Bank shortage: if the bank can't pay everyone, nobody gets that
                // resource, unless only one player is owed it (they get the rest).
                let give = if total <= self.bank[r] {
                    want
                } else if claimants == 1 {
                    self.bank[r]
                } else {
                    0
                };
                self.bank[r] -= give;
                self.hands[p][r] += give;
                self.stats.produced[p][r] += give as u16;
            }
        }
    }

    fn next_discarder_or_robber(&mut self) {
        for k in 0..NUM_PLAYERS as u8 {
            let o = (self.current + k) % NUM_PLAYERS as u8;
            if self.discard_left[o as usize] > 0 {
                self.acting = o;
                self.phase = Phase::Discard;
                return;
            }
        }
        self.acting = self.current;
        self.phase = Phase::MoveRobber;
    }

    fn move_robber(&mut self, p: u8, t: u8) {
        self.robber = t;
        let mut candidates = 0u8;
        for &n in &topology().tile_nodes[t as usize] {
            let owner = self.node_owner[n as usize];
            if owner != NONE && owner != p && self.hand_size(owner) > 0 {
                candidates |= 1 << owner;
            }
        }
        match candidates.count_ones() {
            0 => self.finish_robber(),
            1 => {
                self.steal(p, candidates.trailing_zeros() as u8);
                self.finish_robber();
            }
            _ => {
                self.steal_candidates = candidates;
                self.phase = Phase::Steal;
            }
        }
    }

    fn steal(&mut self, thief: u8, victim: u8) {
        let total = self.hand_size(victim);
        if total == 0 {
            return;
        }
        let mut k = self.steal_rng.below(total as u32) as u8;
        let vi = victim as usize;
        for r in 0..NUM_RESOURCES {
            let have = self.hands[vi][r];
            if k < have {
                self.hands[vi][r] -= 1;
                self.hands[thief as usize][r] += 1;
                self.stats.cards_stolen_from[vi] += 1;
                return;
            }
            k -= have;
        }
    }

    fn finish_robber(&mut self) {
        self.steal_candidates = 0;
        self.acting = self.current;
        self.phase = if self.has_rolled {
            Phase::Main
        } else {
            Phase::PreRoll
        };
    }

    fn end_turn(&mut self) {
        let pi = self.current as usize;
        for c in 0..NUM_DEV_TYPES {
            self.dev_hand[pi][c] += self.dev_new[pi][c];
        }
        self.dev_new[pi] = [0; NUM_DEV_TYPES];
        self.dev_played = false;
        self.has_rolled = false;
        self.turn += 1;
        if self.turn >= MAX_TURNS {
            self.phase = Phase::GameOver;
            return;
        }
        self.current = (self.current + 1) % NUM_PLAYERS as u8;
        self.acting = self.current;
        self.phase = Phase::PreRoll;
    }

    /// Panics with a description if any internal invariant is violated.
    /// Used by tests and debug tooling.
    pub fn check_invariants(&self) {
        let topo = topology();
        for r in 0..NUM_RESOURCES {
            let total: u32 = self.bank[r] as u32
                + (0..NUM_PLAYERS)
                    .map(|p| self.hands[p][r] as u32)
                    .sum::<u32>();
            assert_eq!(total, 19, "resource {r} not conserved");
        }
        let held: u32 = (0..NUM_PLAYERS as u8)
            .map(|p| self.dev_cards_held(p) as u32)
            .sum();
        let knights: u32 = self.knights_played.iter().map(|&k| k as u32).sum();
        assert!(
            held + knights <= self.dev_deck_pos as u32,
            "more dev cards than drawn"
        );
        for p in 0..NUM_PLAYERS as u8 {
            let pi = p as usize;
            let settlements = (0..NUM_NODES)
                .filter(|&n| self.node_owner[n] == p && !self.node_city[n])
                .count() as u8;
            let cities = (0..NUM_NODES)
                .filter(|&n| self.node_owner[n] == p && self.node_city[n])
                .count() as u8;
            let roads = self.edge_owner.iter().filter(|&&o| o == p).count() as u8;
            assert_eq!(
                settlements + self.settlements_left[pi],
                5,
                "settlement pieces of P{p}"
            );
            assert_eq!(cities + self.cities_left[pi], 4, "city pieces of P{p}");
            assert_eq!(roads + self.roads_left[pi], 15, "road pieces of P{p}");
            assert_eq!(
                self.road_length[pi],
                self.longest_road(p),
                "stale road length of P{p}"
            );
        }
        for n in 0..NUM_NODES {
            if self.node_owner[n] != NONE {
                for &m in &topo.node_neighbors[n] {
                    assert!(
                        m == NONE || self.node_owner[m as usize] == NONE,
                        "distance rule broken at node {n}"
                    );
                }
            } else {
                assert!(!self.node_city[n]);
            }
        }
        // Every road touches its owner's network.
        for e in 0..NUM_EDGES as u8 {
            let p = self.edge_owner[e as usize];
            if p == NONE {
                continue;
            }
            let connected = topo.edge_nodes[e as usize].iter().any(|&n| {
                self.node_owner[n as usize] == p
                    || topo.node_edges[n as usize]
                        .iter()
                        .any(|&e2| e2 != NONE && e2 != e && self.edge_owner[e2 as usize] == p)
            });
            assert!(connected, "road {e} of P{p} is disconnected");
        }
        let lr = self.longest_road_owner;
        if lr != NONE {
            let max = *self.road_length.iter().max().unwrap();
            assert!(self.road_length[lr as usize] >= 5 && self.road_length[lr as usize] == max);
        }
        let la = self.largest_army_owner;
        if la != NONE {
            let max = *self.knights_played.iter().max().unwrap();
            assert!(
                self.knights_played[la as usize] >= 3 && self.knights_played[la as usize] == max
            );
        }
        if self.winner != NONE {
            assert!(self.total_vp(self.winner) >= VP_TO_WIN);
            assert!(self.is_over());
        }
        if !self.is_over() {
            let mut buf = Vec::new();
            self.legal_actions(&mut buf);
            assert!(!buf.is_empty(), "no legal actions in {:?}", self.phase);
        }
    }
}

#[cfg(test)]
mod tests;

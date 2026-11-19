use super::*;
use crate::actions::Action;
use crate::board::{NUM_TILES, topology};
use crate::bots::BotSpec;
use crate::sim::{build_bots, play_game};

fn edge_between(a: u8, b: u8) -> u8 {
    let t = topology();
    let k = t.node_neighbors[a as usize]
        .iter()
        .position(|&x| x == b)
        .expect("not adjacent");
    t.node_edges[a as usize][k]
}

/// A fresh state that skips setup: Main phase, player 0 to act.
fn main_phase_state(seed: u64) -> GameState {
    let mut s = GameState::new(seed);
    s.phase = Phase::Main;
    s.has_rolled = true;
    s.setup_step = 8;
    s
}

fn give(s: &mut GameState, p: u8, cards: [u8; 5]) {
    for r in 0..5 {
        s.bank[r] -= cards[r];
        s.hands[p as usize][r] += cards[r];
    }
}

fn legal(s: &GameState) -> Vec<Action> {
    let mut v = Vec::new();
    s.legal_actions(&mut v);
    v
}

#[test]
fn setup_follows_snake_order_and_pays_second_settlement() {
    let mut s = GameState::new(3);
    let mut order = Vec::new();
    while s.is_setup() {
        let acts = legal(&s);
        if s.phase == Phase::SetupSettlement {
            order.push(s.acting);
            // Distance rule: every offered node has empty neighbours.
            for a in &acts {
                if let Action::BuildSettlement(n) = *a {
                    assert!(s.node_buildable(n));
                }
            }
        }
        let before = s.hand_size(s.acting);
        let step = s.setup_step;
        let a = acts[0];
        let actor = s.acting;
        s.apply(a);
        if let Action::BuildSettlement(n) = a {
            let gained = s.hand_size(actor) - before;
            let producing = topology().node_tiles[n as usize]
                .iter()
                .filter(|&&t| t != NONE && s.board.tile_resource[t as usize] != DESERT)
                .count() as u8;
            assert_eq!(gained, if step >= 4 { producing } else { 0 });
        }
        s.check_invariants();
    }
    assert_eq!(order, SETUP_ORDER.to_vec());
    assert_eq!(s.phase, Phase::PreRoll);
    assert_eq!(s.current, 0);
}

#[test]
fn longest_road_counts_paths_and_cycles() {
    let mut s = main_phase_state(1);
    let ring = topology().tile_nodes[9]; // the centre hex
    for i in 0..5 {
        let e = edge_between(ring[i], ring[i + 1]);
        s.edge_owner[e as usize] = 0;
    }
    assert_eq!(s.longest_road(0), 5);
    let e = edge_between(ring[5], ring[0]);
    s.edge_owner[e as usize] = 0;
    assert_eq!(s.longest_road(0), 6, "a closed loop counts every edge once");
}

#[test]
fn opponent_settlement_breaks_longest_road() {
    let mut s = main_phase_state(1);
    let ring = topology().tile_nodes[9];
    // Player 0 builds 5 roads in a path around the centre hex, starting from a settlement.
    s.node_owner[ring[0] as usize] = 0;
    s.settlements_left[0] -= 1;
    for i in 0..5 {
        let e = edge_between(ring[i], ring[i + 1]);
        s.place_road(0, e);
    }
    assert_eq!(s.road_length[0], 5);
    assert_eq!(s.longest_road_owner, 0);
    // Player 1 settles in the middle of the path (node ring[3] is two edges
    // from player 0's settlement, so the distance rule allows it).
    s.place_settlement(1, ring[3]);
    assert_eq!(s.road_length[0], 3);
    assert_eq!(s.longest_road_owner, NONE);
}

#[test]
fn longest_road_holder_keeps_it_on_a_tie() {
    let mut s = main_phase_state(1);
    s.road_length = [5, 5, 0, 0];
    s.longest_road_owner = 0;
    s.update_longest_road_owner();
    assert_eq!(s.longest_road_owner, 0);
    s.road_length = [5, 6, 0, 0];
    s.update_longest_road_owner();
    assert_eq!(s.longest_road_owner, 1);
}

#[test]
fn largest_army_needs_three_and_must_be_exceeded() {
    let mut s = main_phase_state(1);
    s.knights_played = [2, 0, 0, 0];
    s.update_largest_army(0);
    assert_eq!(s.largest_army_owner, NONE);
    s.knights_played = [3, 0, 0, 0];
    s.update_largest_army(0);
    assert_eq!(s.largest_army_owner, 0);
    s.knights_played = [3, 3, 0, 0];
    s.update_largest_army(1);
    assert_eq!(s.largest_army_owner, 0, "a tie does not take Largest Army");
    s.knights_played = [3, 4, 0, 0];
    s.update_largest_army(1);
    assert_eq!(s.largest_army_owner, 1);
}

#[test]
fn seven_makes_big_hands_discard_half_then_robber_moves() {
    let mut s = main_phase_state(5);
    s.phase = Phase::PreRoll;
    s.has_rolled = false;
    give(&mut s, 1, [3, 3, 3, 0, 0]); // 9 cards -> discard 4
    give(&mut s, 2, [2, 2, 2, 1, 0]); // 7 cards -> safe
    give(&mut s, 0, [4, 4, 0, 0, 0]); // 8 cards -> discard 4
    s.resolve_roll(7);
    // The roller discards first, then the others in seat order.
    assert_eq!(s.phase, Phase::Discard);
    assert_eq!(s.acting, 0);
    assert_eq!(s.discard_left, [4, 4, 0, 0]);
    while s.phase == Phase::Discard {
        let a = legal(&s)[0];
        s.apply(a);
    }
    assert_eq!(s.hand_size(0), 4);
    assert_eq!(s.hand_size(1), 5);
    assert_eq!(s.hand_size(2), 7);
    assert_eq!(s.phase, Phase::MoveRobber);
    assert_eq!(s.acting, 0);
    assert!(
        !legal(&s).contains(&Action::MoveRobber(s.robber)),
        "robber must move"
    );
    s.check_invariants();
}

#[test]
fn robber_steals_from_single_adjacent_opponent() {
    let mut s = main_phase_state(5);
    s.phase = Phase::MoveRobber;
    let t = (0..NUM_TILES as u8).find(|&t| t != s.robber).unwrap();
    let n = topology().tile_nodes[t as usize][0];
    s.node_owner[n as usize] = 2;
    s.settlements_left[2] -= 1;
    give(&mut s, 2, [0, 0, 0, 0, 3]);
    s.apply(Action::MoveRobber(t));
    assert_eq!(s.hands[0][4], 1);
    assert_eq!(s.hands[2][4], 2);
    assert_eq!(s.phase, Phase::Main);
}

#[test]
fn dev_card_cannot_be_played_the_turn_it_is_bought() {
    let mut s = main_phase_state(9);
    s.dev_deck[0] = KNIGHT as u8;
    give(&mut s, 0, COST_DEV);
    s.apply(Action::BuyDev);
    assert_eq!(s.dev_new[0][KNIGHT], 1);
    assert!(!legal(&s).contains(&Action::PlayKnight));
    s.apply(Action::EndTurn);
    // Advance to player 0's next turn.
    for _ in 0..3 {
        s.apply(Action::Roll);
        while s.phase != Phase::Main {
            let a = legal(&s)[0];
            s.apply(a);
        }
        s.apply(Action::EndTurn);
    }
    assert_eq!(s.current, 0);
    assert!(
        legal(&s).contains(&Action::PlayKnight),
        "knight playable before rolling"
    );
}

#[test]
fn only_one_dev_card_per_turn() {
    let mut s = main_phase_state(9);
    s.dev_hand[0][MONOPOLY] = 1;
    s.dev_hand[0][KNIGHT] = 1;
    s.apply(Action::PlayMonopoly(0));
    assert!(!legal(&s).contains(&Action::PlayKnight));
}

#[test]
fn harbours_set_trade_rates() {
    let mut s = main_phase_state(11);
    assert_eq!(s.trade_rate(0, WOOD_R), 4);
    let topo = topology();
    let generic_slot = s
        .board
        .port_type
        .iter()
        .position(|&t| t == PORT_GENERIC)
        .unwrap();
    let n = topo.edge_nodes[topo.port_edges[generic_slot] as usize][0];
    s.place_settlement(0, n);
    assert_eq!(s.trade_rate(0, WOOD_R), 3);
    let ore_slot = s.board.port_type.iter().position(|&t| t == 4).unwrap();
    let m = topo.edge_nodes[topo.port_edges[ore_slot] as usize][0];
    s.place_settlement(0, m);
    assert_eq!(s.trade_rate(0, 4), 2);
    assert_eq!(s.trade_rate(0, WOOD_R), 3);

    give(&mut s, 0, [0, 0, 0, 0, 2]);
    assert!(legal(&s).contains(&Action::MaritimeTrade(4, 0)));
    s.apply(Action::MaritimeTrade(4, 0));
    assert_eq!(s.hands[0], [1, 0, 0, 0, 0]);
    s.check_invariants();
}
const WOOD_R: u8 = 0;

#[test]
fn bank_shortage_rule() {
    let mut s = main_phase_state(13);
    // Give tile 0 a unique number so only it produces.
    s.board.tile_number = [0; NUM_TILES];
    s.board.tile_resource[0] = 4; // ore
    s.board.tile_number[0] = 5;
    s.robber = 18;
    let corners = topology().tile_nodes[0];
    s.node_owner[corners[0] as usize] = 1;
    s.node_owner[corners[2] as usize] = 2;
    s.node_city[corners[2] as usize] = true;

    // Enough in the bank: everyone is paid (city = 2).
    s.resolve_roll(5);
    assert_eq!((s.hands[1][4], s.hands[2][4]), (1, 2));

    // Bank has 2 ore but 3 are owed to two players: nobody gets ore.
    s.hands[1][4] = 0;
    s.hands[2][4] = 0;
    s.bank[4] = 2;
    s.hands[3][4] = 17;
    s.resolve_roll(5);
    assert_eq!((s.hands[1][4], s.hands[2][4]), (0, 0));

    // Only one player owed: they get whatever is left.
    s.node_owner[corners[0] as usize] = NONE;
    s.resolve_roll(5);
    assert_eq!(s.hands[2][4], 2);
}

#[test]
fn reaching_ten_points_on_your_turn_wins() {
    let mut s = main_phase_state(17);
    s.dev_hand[0][VICTORY_POINT] = 5;
    let nodes = [0u8, 2, 4, 6];
    for &n in &nodes {
        s.node_owner[n as usize] = 0;
        s.settlements_left[0] -= 1;
    }
    assert_eq!(s.total_vp(0), 9);
    // Upgrade one settlement to a city: 10 VP.
    give(&mut s, 0, COST_CITY);
    s.apply(Action::BuildCity(0));
    assert!(s.is_over());
    assert_eq!(s.winner, 0);
}

#[test]
fn random_games_preserve_invariants() {
    let spec = BotSpec::Random;
    for seed in 0..300 {
        let lineup = [&spec; 4];
        let mut bots = build_bots(&lineup, seed);
        let mut s = GameState::new(seed);
        let mut buf = Vec::new();
        let mut mask = vec![false; NUM_ACTIONS];
        while !s.is_over() {
            s.action_mask(&mut buf, &mut mask);
            for a in &buf {
                let idx = a.to_index(s.acting);
                assert_eq!(Action::from_index(idx, s.acting), Some(*a));
            }
            let a = bots[s.acting as usize].act(&s, &buf);
            s.apply(a);
            s.check_invariants();
        }
    }
}

#[test]
fn strategy_bot_games_preserve_invariants_and_finish() {
    let names = [
        "balanced",
        "ows",
        "road_builder",
        "dev_rush",
        "port_master",
        "greedy",
    ];
    for seed in 0..120u64 {
        let specs: Vec<BotSpec> = (0..4)
            .map(|i| BotSpec::preset(names[(seed as usize + i) % names.len()]).unwrap())
            .collect();
        let lineup = [&specs[0], &specs[1], &specs[2], &specs[3]];
        let mut bots = build_bots(&lineup, seed);
        let mut s = GameState::new(seed);
        let mut buf = Vec::new();
        while !s.is_over() {
            s.legal_actions(&mut buf);
            let a = bots[s.acting as usize].act(&s, &buf);
            assert!(buf.contains(&a), "bot chose illegal {a:?}");
            s.apply(a);
            s.check_invariants();
        }
        assert_ne!(
            s.winner, NONE,
            "seed {seed}: strategy bots should not hit the turn limit"
        );
    }
}

#[test]
fn games_are_reproducible_from_seed_and_actions() {
    let spec = BotSpec::preset("balanced").unwrap();
    let lineup = [&spec; 4];
    let mut bots = build_bots(&lineup, 42);
    let mut log = Vec::new();
    let a = play_game(42, &mut bots, Some(&mut log));
    let mut b = GameState::new(42);
    for &(actor, action) in &log {
        assert_eq!(b.acting, actor);
        b.apply(action);
    }
    assert_eq!(a.winner, b.winner);
    assert_eq!(a.hands, b.hands);
    assert_eq!(a.edge_owner, b.edge_owner);
    assert_eq!(a.turn, b.turn);
}

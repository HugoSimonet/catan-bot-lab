//! Python module `catanbot._engine`.
//!
//! Exposes the vectorised RL environment, the tournament runner (returning
//! columnar numpy data) and replay export. Long-running calls release the GIL.

use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use catan_core::actions::{Action, NUM_ACTIONS};
use catan_core::bots::{BotSpec, PRESET_NAMES};
use catan_core::env::{OwnedBuffers, SeatSpec, VecEnv};
use catan_core::obs::OBS_SIZE;
use catan_core::replay::build_replay;
use catan_core::sim::{self, GameResult, Table, TournamentGame};
use catan_core::state::{GameState, NUM_PLAYERS};

fn err(e: impl ToString) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn parse_specs(specs: &[String]) -> PyResult<Vec<BotSpec>> {
    specs
        .iter()
        .map(|s| BotSpec::parse(s).map_err(err))
        .collect()
}

fn parse_lineup(codes: &[i32]) -> PyResult<[SeatSpec; NUM_PLAYERS]> {
    if codes.len() != NUM_PLAYERS {
        return Err(err(format!(
            "a lineup needs {NUM_PLAYERS} seats, got {}",
            codes.len()
        )));
    }
    Ok(std::array::from_fn(|i| SeatSpec::from_code(codes[i])))
}

/// Names and JSON specs of the built-in presets.
#[pyfunction]
fn presets() -> Vec<(String, String)> {
    PRESET_NAMES
        .iter()
        .map(|n| {
            (
                n.to_string(),
                serde_json::to_string(&BotSpec::preset(n).unwrap()).unwrap(),
            )
        })
        .collect()
}

/// Layout of the observation vector and of the action indices, for models
/// that treat tiles, nodes and edges as entities.
#[pyfunction]
fn layout<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    use catan_core::actions as a;
    use catan_core::obs as o;
    let d = PyDict::new(py);
    d.set_item("tiles", (o::TILES_OFFSET, catan_core::board::NUM_TILES, o::TILE_FEATS))?;
    d.set_item("nodes", (o::NODES_OFFSET, catan_core::board::NUM_NODES, o::NODE_FEATS))?;
    d.set_item("edges", (o::EDGES_OFFSET, catan_core::board::NUM_EDGES, o::EDGE_FEATS))?;
    d.set_item("players", (o::PLAYERS_OFFSET, NUM_PLAYERS, o::PLAYER_FEATS))?;
    d.set_item("self", (o::SELF_OFFSET, 1, o::SELF_FEATS))?;
    d.set_item("global", (o::GLOBAL_OFFSET, 1, o::GLOBAL_FEATS))?;
    d.set_item("settlement_base", a::SETTLEMENT_BASE)?;
    d.set_item("city_base", a::CITY_BASE)?;
    d.set_item("road_base", a::ROAD_BASE)?;
    d.set_item("robber_base", a::ROBBER_BASE)?;
    Ok(d)
}

/// The fixed board topology as JSON (tile/node/edge adjacency and coordinates).
#[pyfunction]
fn topology_json() -> String {
    serde_json::to_string(catan_core::board::topology()).unwrap()
}

/// Canonical name of a bot spec (preset name or JSON).
#[pyfunction]
fn spec_name(spec: &str) -> PyResult<String> {
    Ok(BotSpec::parse(spec).map_err(err)?.name())
}

/// Per-seat statistics columns shared by tournaments and environment games.
const COLUMNS: [&str; 33] = [
    "game_id",
    "seed",
    "seat",
    "bot",
    "won",
    "draw",
    "vp",
    "settlements",
    "cities",
    "roads",
    "knights",
    "vp_cards",
    "longest_road",
    "largest_army",
    "dev_bought",
    "trades",
    "produced_wood",
    "produced_brick",
    "produced_sheep",
    "produced_wheat",
    "produced_ore",
    "discarded",
    "stolen_from",
    "robbed_production",
    "first_city_turn",
    "turns",
    "opening_wood",
    "opening_brick",
    "opening_sheep",
    "opening_wheat",
    "opening_ore",
    "opening_port",
    "num_actions",
];

fn seat_row(game_id: i64, bot: i64, seat: usize, r: &GameResult) -> [i64; 33] {
    let st = &r.stats;
    let s8 = seat as u8;
    let o = &st.opening_pips[seat];
    [
        game_id,
        r.seed as i64,
        seat as i64,
        bot,
        (r.winner == Some(s8)) as i64,
        r.winner.is_none() as i64,
        r.vp[seat] as i64,
        r.settlements[seat] as i64,
        r.cities[seat] as i64,
        r.roads[seat] as i64,
        r.knights[seat] as i64,
        r.vp_cards[seat] as i64,
        (r.longest_road == Some(s8)) as i64,
        (r.largest_army == Some(s8)) as i64,
        st.dev_bought[seat] as i64,
        st.maritime_trades[seat] as i64,
        st.produced[seat][0] as i64,
        st.produced[seat][1] as i64,
        st.produced[seat][2] as i64,
        st.produced[seat][3] as i64,
        st.produced[seat][4] as i64,
        st.cards_discarded[seat] as i64,
        st.cards_stolen_from[seat] as i64,
        st.robbed_production[seat] as i64,
        st.first_city_turn[seat] as i64,
        r.turns as i64,
        o[0] as i64,
        o[1] as i64,
        o[2] as i64,
        o[3] as i64,
        o[4] as i64,
        st.opening_port[seat] as i64,
        r.num_actions as i64,
    ]
}

fn games_to_dict<'py>(
    py: Python<'py>,
    games: &[TournamentGame],
    names: &[String],
) -> PyResult<Bound<'py, PyDict>> {
    let rows = games.len() * NUM_PLAYERS;
    let mut cols: Vec<Vec<i64>> = (0..COLUMNS.len())
        .map(|_| Vec::with_capacity(rows))
        .collect();
    for g in games {
        for seat in 0..NUM_PLAYERS {
            let row = seat_row(g.game_id as i64, g.seats[seat] as i64, seat, &g.result);
            for (col, v) in cols.iter_mut().zip(row) {
                col.push(v);
            }
        }
    }
    let d = PyDict::new(py);
    for (name, values) in COLUMNS.iter().zip(cols) {
        d.set_item(*name, values.into_pyarray(py))?;
    }
    d.set_item("bot_names", names.to_vec())?;
    Ok(d)
}

fn run(py: Python<'_>, specs: Vec<String>, tables: Vec<Table>) -> PyResult<Bound<'_, PyDict>> {
    let parsed = parse_specs(&specs)?;
    let names: Vec<String> = parsed.iter().map(|s| s.name()).collect();
    let games = py.detach(|| sim::run_tables(&parsed, &tables));
    games_to_dict(py, &games, &names)
}

/// Play `lineup` (4 indices into `specs`) in every seat rotation on
/// `num_seeds` seeds. Returns one row per (game, seat) as numpy columns.
#[pyfunction]
#[pyo3(signature = (specs, lineup, num_seeds, base_seed = 1))]
fn run_rotated(
    py: Python<'_>,
    specs: Vec<String>,
    lineup: [u16; 4],
    num_seeds: u64,
    base_seed: u64,
) -> PyResult<Bound<'_, PyDict>> {
    if lineup.iter().any(|&b| b as usize >= specs.len()) {
        return Err(err("lineup index out of range"));
    }
    run(py, specs, sim::rotated_tables(lineup, num_seeds, base_seed))
}

/// Random 4-bot tables from a pool of specs, each in every seat rotation.
#[pyfunction]
#[pyo3(signature = (specs, num_seeds, base_seed = 1, distinct = true))]
fn run_pool(
    py: Python<'_>,
    specs: Vec<String>,
    num_seeds: u64,
    base_seed: u64,
    distinct: bool,
) -> PyResult<Bound<'_, PyDict>> {
    if distinct && specs.len() < NUM_PLAYERS {
        return Err(err("need at least 4 specs for distinct tables"));
    }
    let n = specs.len() as u16;
    run(
        py,
        specs,
        sim::pool_tables(n, num_seeds, base_seed, distinct),
    )
}

/// Build a replay JSON from a seed and a list of action *indices*.
#[pyfunction]
fn replay_from_actions(seed: u64, players: Vec<String>, actions: Vec<i64>) -> PyResult<String> {
    let mut state = GameState::new(seed);
    let mut decoded = Vec::with_capacity(actions.len());
    for (i, &idx) in actions.iter().enumerate() {
        let a = usize::try_from(idx)
            .ok()
            .and_then(|x| Action::from_index(x, state.acting))
            .ok_or_else(|| err(format!("bad action index {idx} at position {i}")))?;
        if !state.is_legal(a) {
            return Err(err(format!("illegal action {a:?} at position {i}")));
        }
        state.apply(a);
        decoded.push(a);
    }
    let replay = build_replay(seed, players, &decoded).map_err(err)?;
    Ok(serde_json::to_string(&replay).unwrap())
}

/// Play one game between 4 bot specs and return its replay JSON.
#[pyfunction]
fn play_replay(specs: Vec<String>, seed: u64) -> PyResult<String> {
    let parsed = parse_specs(&specs)?;
    if parsed.len() != NUM_PLAYERS {
        return Err(err("need exactly 4 specs"));
    }
    let lineup: [&BotSpec; 4] = std::array::from_fn(|i| &parsed[i]);
    let mut bots = sim::build_bots(&lineup, seed);
    let mut log = Vec::new();
    sim::play_game(seed, &mut bots, Some(&mut log));
    let actions: Vec<Action> = log.into_iter().map(|(_, a)| a).collect();
    let names = parsed.iter().map(|s| s.name()).collect();
    let replay = build_replay(seed, names, &actions).map_err(err)?;
    Ok(serde_json::to_string(&replay).unwrap())
}

/// Vectorised environment (see `catan_core::env`).
///
/// Seat codes in lineups: `>= 0` is an index into `bot_specs`, `-1 - k` is
/// external policy `k`.
#[pyclass(name = "VecEnv", unsendable)]
struct PyVecEnv {
    inner: VecEnv,
}

type StepTuple<'py> = (
    Bound<'py, PyAny>, // obs [N, OBS]
    Bound<'py, PyAny>, // mask [N, A]
    Bound<'py, PyAny>, // policy [N]
    Bound<'py, PyAny>, // seat [N]
    Bound<'py, PyAny>, // reward [N, 4]
    Bound<'py, PyAny>, // done [N]
    Bound<'py, PyAny>, // winner [N]
    Bound<'py, PyAny>, // final_vp [N, 4]
    Bound<'py, PyAny>, // turns [N]
    Bound<'py, PyAny>, // lineup [N, 4]
);

fn to_numpy<'py>(py: Python<'py>, b: OwnedBuffers, n: usize) -> PyResult<StepTuple<'py>> {
    let p = NUM_PLAYERS;
    Ok((
        PyArray1::from_vec(py, b.obs)
            .reshape([n, OBS_SIZE])?
            .into_any(),
        PyArray1::from_vec(py, b.mask)
            .reshape([n, NUM_ACTIONS])?
            .into_any(),
        PyArray1::from_vec(py, b.policy).into_any(),
        PyArray1::from_vec(py, b.seat).into_any(),
        PyArray1::from_vec(py, b.reward).reshape([n, p])?.into_any(),
        PyArray1::from_vec(py, b.done).into_any(),
        PyArray1::from_vec(py, b.winner).into_any(),
        PyArray1::from_vec(py, b.final_vp)
            .reshape([n, p])?
            .into_any(),
        PyArray1::from_vec(py, b.turns).into_any(),
        PyArray1::from_vec(py, b.lineup).reshape([n, p])?.into_any(),
    ))
}

#[pymethods]
impl PyVecEnv {
    #[new]
    #[pyo3(signature = (bot_specs, lineups, seed = 0, record = false))]
    fn new(
        bot_specs: Vec<String>,
        lineups: Vec<Vec<i32>>,
        seed: u64,
        record: bool,
    ) -> PyResult<Self> {
        let specs = parse_specs(&bot_specs)?;
        let lineups = lineups
            .iter()
            .map(|l| parse_lineup(l))
            .collect::<PyResult<Vec<_>>>()?;
        let inner = VecEnv::new(specs, lineups, seed, record).map_err(err)?;
        Ok(PyVecEnv { inner })
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.inner.num_envs()
    }

    /// Reward settings: `win`, `loss` (terminal) and `shaping` (per VP gained).
    #[pyo3(signature = (win = None, loss = None, shaping = None))]
    fn set_rewards(&mut self, win: Option<f32>, loss: Option<f32>, shaping: Option<f32>) {
        if let Some(w) = win {
            self.inner.cfg.win = w;
        }
        if let Some(l) = loss {
            self.inner.cfg.loss = l;
        }
        if let Some(s) = shaping {
            self.inner.cfg.shaping = s;
        }
    }

    /// Queue the lineup for game `i`'s next game.
    fn set_lineup(&mut self, i: usize, lineup: Vec<i32>) -> PyResult<()> {
        let l = parse_lineup(&lineup)?;
        VecEnv::validate_lineup(&self.inner.specs, &l).map_err(err)?;
        let env = self
            .inner
            .envs
            .get_mut(i)
            .ok_or_else(|| err("game index out of range"))?;
        env.pending = Some(l);
        Ok(())
    }

    /// Current lineup (seat codes) of game `i`.
    fn lineup(&self, i: usize) -> PyResult<Vec<i32>> {
        let env = self
            .inner
            .envs
            .get(i)
            .ok_or_else(|| err("game index out of range"))?;
        Ok(env.lineup.iter().map(|s| s.code()).collect())
    }

    fn reset<'py>(&mut self, py: Python<'py>) -> PyResult<StepTuple<'py>> {
        let n = self.inner.num_envs();
        let mut b = OwnedBuffers::new(n);
        let inner = &mut self.inner;
        py.detach(|| inner.reset_all(&mut b.view()));
        to_numpy(py, b, n)
    }

    /// Apply one action index per game; returns the next decisions.
    fn step<'py>(&mut self, py: Python<'py>, actions: Vec<i64>) -> PyResult<StepTuple<'py>> {
        let n = self.inner.num_envs();
        let mut b = OwnedBuffers::new(n);
        let inner = &mut self.inner;
        py.detach(|| inner.step(&actions, &mut b.view()))
            .map_err(err)?;
        to_numpy(py, b, n)
    }

    /// Replay JSON of the last finished game in env `i` (needs `record=True`).
    fn last_replay(&self, i: usize, players: Vec<String>) -> PyResult<String> {
        let env = self
            .inner
            .envs
            .get(i)
            .ok_or_else(|| err("game index out of range"))?;
        if env.last_log.is_empty() {
            return Err(err(
                "no finished game recorded (create the env with record=True)",
            ));
        }
        let replay = build_replay(env.last_seed, players, &env.last_log).map_err(err)?;
        Ok(serde_json::to_string(&replay).unwrap())
    }

    /// Per-seat statistics of the last finished game in env `i`, as
    /// {column: [4 values]} (the `bot` column holds the seat codes).
    fn last_result<'py>(&self, py: Python<'py>, i: usize) -> PyResult<Bound<'py, PyDict>> {
        let env = self
            .inner
            .envs
            .get(i)
            .ok_or_else(|| err("game index out of range"))?;
        let r = env
            .last_result
            .as_ref()
            .ok_or_else(|| err("no finished game yet"))?;
        let rows: Vec<[i64; 33]> = (0..NUM_PLAYERS)
            .map(|s| seat_row(0, env.last_lineup[s].code() as i64, s, r))
            .collect();
        let d = PyDict::new(py);
        for (c, name) in COLUMNS.iter().enumerate() {
            d.set_item(*name, rows.iter().map(|row| row[c]).collect::<Vec<i64>>())?;
        }
        Ok(d)
    }

    /// (seed, seat codes) of the last finished game in env `i`.
    fn last_game(&self, i: usize) -> PyResult<(u64, Vec<i32>)> {
        let env = self
            .inner
            .envs
            .get(i)
            .ok_or_else(|| err("game index out of range"))?;
        Ok((
            env.last_seed,
            env.last_lineup.iter().map(|s| s.code()).collect(),
        ))
    }
}

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("NUM_ACTIONS", NUM_ACTIONS)?;
    m.add("OBS_SIZE", OBS_SIZE)?;
    m.add("NUM_PLAYERS", NUM_PLAYERS)?;
    m.add("PRESET_NAMES", PRESET_NAMES.to_vec())?;
    m.add_class::<PyVecEnv>()?;
    m.add_function(wrap_pyfunction!(presets, m)?)?;
    m.add_function(wrap_pyfunction!(spec_name, m)?)?;
    m.add_function(wrap_pyfunction!(layout, m)?)?;
    m.add_function(wrap_pyfunction!(topology_json, m)?)?;
    m.add_function(wrap_pyfunction!(run_rotated, m)?)?;
    m.add_function(wrap_pyfunction!(run_pool, m)?)?;
    m.add_function(wrap_pyfunction!(replay_from_actions, m)?)?;
    m.add_function(wrap_pyfunction!(play_replay, m)?)?;
    Ok(())
}

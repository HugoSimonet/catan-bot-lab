//! `catan-sim`: benchmarks, tournaments and replay export.
//!
//! Examples:
//!   catan-sim bench --games 20000
//!   catan-sim tournament --bots ows,road_builder,dev_rush,balanced --seeds 25000 --out results/raw/t.csv
//!   catan-sim pool --pool random,greedy,balanced,ows,road_builder,dev_rush,port_master --seeds 50000
//!   catan-sim replay --bots ows,road_builder,dev_rush,balanced --seed 7 --out web/public/data/replays/r7.json
//!   catan-sim presets

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, Subcommand};

use catan_core::bots::{BotSpec, PRESET_NAMES};
use catan_core::replay::build_replay;
use catan_core::sim::{self, BotSummary, Table, TournamentGame};

#[derive(Parser)]
#[command(name = "catan-sim", version, about = "Settlers of Catan bot simulator")]
struct Cli {
    /// Worker threads (default: all logical cores).
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Measure games per second.
    Bench {
        #[arg(long, default_value = "random,random,random,random")]
        bots: String,
        #[arg(long, default_value_t = 20_000)]
        games: u64,
    },
    /// Play a fixed 4-bot lineup in every seat rotation on N seeds (4 games per seed).
    Tournament {
        /// Four comma-separated bots: preset names or paths to JSON bot specs.
        #[arg(long)]
        bots: String,
        #[arg(long, default_value_t = 10_000)]
        seeds: u64,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Per-seat CSV of every game.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Summary JSON.
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Random tables drawn from a pool of bots, each in every seat rotation.
    Pool {
        #[arg(long)]
        pool: String,
        #[arg(long, default_value_t = 10_000)]
        seeds: u64,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Never seat the same bot twice at one table.
        #[arg(long)]
        distinct: bool,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Play one game and write a replay JSON for the web viewer.
    Replay {
        #[arg(long)]
        bots: String,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long)]
        out: PathBuf,
    },
    /// Print the strategy presets as JSON.
    Presets,
}

fn parse_bots(list: &str) -> Result<Vec<BotSpec>, String> {
    list.split(',')
        .map(|item| {
            let item = item.trim();
            if item.ends_with(".json") {
                let text = std::fs::read_to_string(item).map_err(|e| format!("{item}: {e}"))?;
                BotSpec::parse(&text)
            } else {
                BotSpec::parse(item)
            }
        })
        .collect()
}

fn four(specs: &[BotSpec]) -> Result<[u16; 4], String> {
    if specs.len() != 4 {
        return Err(format!("expected 4 bots, got {}", specs.len()));
    }
    Ok([0, 1, 2, 3])
}

fn print_summary(summaries: &[BotSummary], games: usize, secs: f64) {
    println!(
        "\n{games} games in {secs:.1}s ({:.0} games/s)\n",
        games as f64 / secs.max(1e-9)
    );
    println!(
        "{:<16} {:>9} {:>8} {:>17} {:>7} {:>7} {:>6} {:>6} {:>6} {:>6}",
        "bot", "games", "win%", "95% CI", "avgVP", "draws", "LR%", "LA%", "dev", "trades"
    );
    let mut rows: Vec<&BotSummary> = summaries.iter().filter(|s| s.games > 0).collect();
    rows.sort_by(|a, b| b.win_rate.partial_cmp(&a.win_rate).unwrap());
    for s in rows {
        println!(
            "{:<16} {:>9} {:>7.2}% {:>7.2}%-{:>6.2}% {:>7.2} {:>7} {:>5.0}% {:>5.0}% {:>6.2} {:>6.1}",
            s.name,
            s.games,
            100.0 * s.win_rate,
            100.0 * s.ci_low,
            100.0 * s.ci_high,
            s.avg_vp,
            s.draws,
            100.0 * s.wins_with_longest_road,
            100.0 * s.wins_with_largest_army,
            s.avg_dev_bought,
            s.avg_trades,
        );
    }
}

fn write_csv(path: &PathBuf, specs: &[BotSpec], games: &[TournamentGame]) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(
        w,
        "game_id,seed,seat,bot,won,draw,vp,settlements,cities,roads,knights,vp_cards,\
         longest_road,largest_army,dev_bought,trades,produced,discarded,stolen_from,robbed_production,\
         first_city_turn,turns"
    )?;
    let names: Vec<String> = specs.iter().map(|s| s.name()).collect();
    for g in games {
        let r = &g.result;
        for seat in 0..4 {
            let st = &r.stats;
            let produced: u32 = st.produced[seat].iter().map(|&x| x as u32).sum();
            writeln!(
                w,
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                g.game_id,
                r.seed,
                seat,
                names[g.seats[seat] as usize],
                (r.winner == Some(seat as u8)) as u8,
                r.winner.is_none() as u8,
                r.vp[seat],
                r.settlements[seat],
                r.cities[seat],
                r.roads[seat],
                r.knights[seat],
                r.vp_cards[seat],
                (r.longest_road == Some(seat as u8)) as u8,
                (r.largest_army == Some(seat as u8)) as u8,
                st.dev_bought[seat],
                st.maritime_trades[seat],
                produced,
                st.cards_discarded[seat],
                st.cards_stolen_from[seat],
                st.robbed_production[seat],
                st.first_city_turn[seat],
                r.turns,
            )?;
        }
    }
    w.flush()
}

fn run(
    specs: &[BotSpec],
    tables: &[Table],
    out: Option<PathBuf>,
    json: Option<PathBuf>,
) -> Result<(), String> {
    let start = Instant::now();
    let games = sim::run_tables(specs, tables);
    let secs = start.elapsed().as_secs_f64();
    let summaries = sim::summarize(specs, &games);
    print_summary(&summaries, games.len(), secs);
    if let Some(path) = out {
        write_csv(&path, specs, &games).map_err(|e| e.to_string())?;
        println!("\nwrote {}", path.display());
    }
    if let Some(path) = json {
        let text = serde_json::to_string_pretty(&summaries).unwrap();
        std::fs::write(&path, text).map_err(|e| e.to_string())?;
        println!("wrote {}", path.display());
    }
    Ok(())
}

fn main() -> Result<(), String> {
    let cli = Cli::parse();
    if let Some(n) = cli.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .map_err(|e| e.to_string())?;
    }
    match cli.command {
        Command::Bench { bots, games } => {
            let specs = parse_bots(&bots)?;
            let lineup = four(&specs)?;
            let tables = sim::rotated_tables(lineup, games.div_ceil(4), 1);
            let start = Instant::now();
            let results = sim::run_tables(&specs, &tables);
            let secs = start.elapsed().as_secs_f64();
            let actions: u64 = results.iter().map(|g| g.result.num_actions as u64).sum();
            let turns: u64 = results.iter().map(|g| g.result.turns as u64).sum();
            let draws = results.iter().filter(|g| g.result.winner.is_none()).count();
            let threads = rayon::current_num_threads();
            println!(
                "{} games on {threads} threads in {secs:.2}s: {:.0} games/s ({:.0}/s per thread), \
                 {:.2}M actions/s, avg {:.0} turns, {:.0} actions/game, {draws} draws",
                results.len(),
                results.len() as f64 / secs,
                results.len() as f64 / secs / threads as f64,
                actions as f64 / secs / 1e6,
                turns as f64 / results.len() as f64,
                actions as f64 / results.len() as f64,
            );
            Ok(())
        }
        Command::Tournament {
            bots,
            seeds,
            seed,
            out,
            json,
        } => {
            let specs = parse_bots(&bots)?;
            let lineup = four(&specs)?;
            let tables = sim::rotated_tables(lineup, seeds, seed);
            run(&specs, &tables, out, json)
        }
        Command::Pool {
            pool,
            seeds,
            seed,
            distinct,
            out,
            json,
        } => {
            let specs = parse_bots(&pool)?;
            let tables = sim::pool_tables(specs.len() as u16, seeds, seed, distinct);
            run(&specs, &tables, out, json)
        }
        Command::Replay { bots, seed, out } => {
            let specs = parse_bots(&bots)?;
            four(&specs)?;
            let lineup: [&BotSpec; 4] = std::array::from_fn(|i| &specs[i]);
            let mut b = sim::build_bots(&lineup, seed);
            let mut log = Vec::new();
            sim::play_game(seed, &mut b, Some(&mut log));
            let actions: Vec<_> = log.iter().map(|&(_, a)| a).collect();
            let names = specs.iter().map(|s| s.name()).collect();
            let replay = build_replay(seed, names, &actions)?;
            if let Some(dir) = out.parent() {
                std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
            }
            std::fs::write(&out, serde_json::to_string(&replay).unwrap())
                .map_err(|e| e.to_string())?;
            println!(
                "wrote {} ({} frames, winner {:?}, {} turns)",
                out.display(),
                replay.frames.len(),
                replay.winner,
                replay.turns
            );
            Ok(())
        }
        Command::Presets => {
            let presets: Vec<BotSpec> = PRESET_NAMES
                .iter()
                .map(|n| BotSpec::preset(n).unwrap())
                .collect();
            println!("{}", serde_json::to_string_pretty(&presets).unwrap());
            Ok(())
        }
    }
}

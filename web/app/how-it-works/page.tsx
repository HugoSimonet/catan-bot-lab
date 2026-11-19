import type { Metadata } from "next";
import Link from "next/link";

import { SUMMARY } from "@/lib/data";
import { compact } from "@/lib/theme";

export const metadata: Metadata = {
  title: "How it works",
  description: "The Rust game engine, the strategy bots, the simulation methodology and the reinforcement-learning setup.",
};

export default function HowItWorks() {
  const s = SUMMARY;
  return (
    <article className="prose-body mx-auto max-w-3xl">
      <h1 className="font-display text-4xl font-semibold">How it works</h1>
      <p className="text-lg text-muted">
        Three pieces: a fast and carefully tested game engine, a set of strategy bots, and a learning agent that
        trains against them. Source layout:{" "}
        <code>crates/catan-core</code> (Rust engine), <code>python/catanbot</code> (training and analysis),{" "}
        <code>web/</code> (this site).
      </p>

      <h2>The engine</h2>
      <p>
        The rules engine is written in Rust. A game state is a fixed-size, heap-free struct (about 800 bytes),
        so copying it to look ahead is almost free. It implements the full base game for four players:
        snake-draft setup, production with the bank-shortage rule, the robber with discards and steals, all five
        development cards with their timing rules (not on the turn bought, one per turn, knights before rolling),
        Longest Road with proper road-breaking, Largest Army, and 4:1, 3:1 and 2:1 trading with the bank and
        harbours. The only simplification is that players do not trade with each other.
      </p>
      <ul>
        <li>
          <strong>Speed:</strong> about {compact(s.engine.random_games_per_sec)} random games and{" "}
          {compact(s.engine.bot_games_per_sec)} strategy-bot games per second on a 6-core desktop CPU (
          {s.engine.threads} threads).
        </li>
        <li>
          <strong>Reproducibility:</strong> boards, dice, the dev-card deck and steals each draw from their own
          seeded random stream. A game is fully determined by its seed and its move list, which is how the replays
          on this site work.
        </li>
        <li>
          <strong>Correctness:</strong> unit tests cover the tricky rules, and hundreds of full games re-check
          invariants after every single move: resources are conserved, piece limits hold, Longest Road is never
          stale, and so on.
        </li>
      </ul>

      <h2>The strategy bots</h2>
      <p>
        One goal-driven heuristic bot, configured by about 15 parameters. Each turn it picks a goal (city,
        settlement, road or development card) by weighing how much it wants each one against how far it is from
        affording it. It then builds the goal if it can, trades with the bank toward it, spends spare cards on
        other builds, and keeps its hand small enough to survive a 7. The presets turn that one bot into the
        community strategies:
      </p>
      <ul>
        <li><strong>Ore-Wheat-Sheep</strong>: settle on ore and wheat, rush cities, then buy development cards.</li>
        <li><strong>Road Builder</strong>: wood and brick, expand early, chase Longest Road.</li>
        <li><strong>Dev-Card Rush</strong>: buy development cards heavily and take Largest Army.</li>
        <li><strong>Port Master</strong>: concentrate on one resource and trade it through its 2:1 harbour.</li>
        <li><strong>Balanced</strong>: maximise total production and diversity, and build whatever is best.</li>
      </ul>
      <p>
        Bots only see what a human would: their own hand, plus the number of cards and development cards each
        opponent holds, but never what those cards are.
      </p>

      <h2>Measuring strategies fairly</h2>
      <p>
        Catan is very luck-heavy, so two methods keep comparisons honest. <strong>Seat rotation</strong>: every
        lineup is played four times on each seed, once with each bot in each seat, because going first is a real
        advantage. <strong>Common random numbers</strong>: every variant being compared plays the same seeds, so
        they get identical boards and dice rolls, and differences come from decisions rather than luck. Win rates
        are reported with 95% Wilson confidence intervals.
      </p>
      <p>
        Parameter sweeps vary one setting of a strategy while holding the rest fixed, then measure its win rate
        against the other strategies. Coordinate ascent over those settings produces a tuned version of each
        strategy, with its resource preferences (its identity) held fixed. The learning agent is trained and
        evaluated against these tuned versions.
      </p>

      <h2>The learning agent</h2>
      <h3>What it sees</h3>
      <p>
        A vector of 1,652 numbers from its own point of view: every tile (resource, number, robber), every
        intersection (buildings, harbour, the resource pips around it), every road, each player&apos;s public state
        listed relative to its own seat, its own hand and development cards, the bank, and the game phase.
        Opponents&apos; hidden cards are not included.
      </p>
      <h3>What it can do</h3>
      <p>
        A fixed menu of 252 actions (each settlement spot, city, road, trade, card, robber tile, and so on). At
        every decision the engine provides a <em>mask</em> of the legal ones, and illegal actions get zero
        probability. Moves with only one legal option are played automatically, so the network only sees real
        decisions.
      </p>
      <h3>The network: the board as a graph</h3>
      <p>
        A residual MLP reads the whole observation: hands, phase, opponents. Alongside it, every tile,
        intersection and road slot is embedded by a small network <em>shared across the board</em>, and
        intersections pass messages to their neighbours twice, so an intersection&apos;s embedding can describe
        &quot;a good spot two roads away&quot;. Settlement, city, road and robber choices are scored per entity by shared
        heads, so what the network learns about one spot transfers to every other spot. A plain MLP was my first
        attempt. On identical settings the graph version reached 6.2 average VP after 150 updates, against 4.8 for
        the MLP, and won three times as often.
      </p>
      <h3>How it learns</h3>
      <p>
        Proximal Policy Optimization (PPO). The reward is +1 for a win and −1/3 for a loss, plus a small bonus per
        victory point early in training that is gradually removed. Because the four seats take turns, the trainer
        links each seat&apos;s own decisions into a trajectory and computes advantages along it.
      </p>
      <h3>Who it plays</h3>
      <p>
        A curriculum: first random bots, then simple bots, then the tuned strategies, and finally a{" "}
        <strong>league</strong>. In the league each opponent seat is a strategy bot, a frozen copy of the agent
        from earlier in training, or the agent itself. Opponents are drawn with prioritised fictitious self-play:
        the more often an opponent outscores the agent, the more often it is picked. The agent never sees a
        hand-written strategy rule; whatever style it ends up with is its own.
      </p>
      <h3>Evaluation</h3>
      <p>
        The trained agent plays tens of thousands of games against fixed tables of the strongest strategies, with
        seats rotated and argmax (no sampling) actions. Every checkpoint is evaluated on the same seeds, so
        checkpoints are compared on identical boards and dice. See the <Link href="/results/">results</Link>.
      </p>

      <h2>Limitations</h2>
      <ul>
        <li>No player-to-player trading, which is a large part of human Catan.</li>
        <li>The strategy bots are my implementations of community strategies, tuned in simulation. Strong human players would play each strategy better.</li>
        <li>Four-player Catan is not zero-sum between two sides: an agent can win more often just by exploiting weaker opponents at the table, so results are always reported against specific lineups.</li>
      </ul>
    </article>
  );
}

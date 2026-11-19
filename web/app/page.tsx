import Link from "next/link";

import ReplayPlayer from "@/components/ReplayPlayer";
import { WinRateBars } from "@/components/charts";
import { FEATURED, SUMMARY, strategyRows } from "@/lib/data";
import { botColor, botLabel, compact, fmtInt, pct } from "@/lib/theme";

export default function Home() {
  const s = SUMMARY;
  const strategies = strategyRows();
  const best = strategies[0];
  const headline = s.evaluation?.[0];

  return (
    <div className="flex flex-col gap-16">
      <section className="grid items-center gap-10 lg:grid-cols-[1.05fr_1fr]">
        <div>
          <p className="text-sm font-semibold uppercase tracking-[0.18em] text-accent">
            Settlers of Catan × reinforcement learning
          </p>
          <h1 className="mt-3 font-display text-4xl font-semibold leading-[1.1] sm:text-5xl">
            Can a self-taught AI beat the best Catan strategies?
          </h1>
          <p className="mt-5 max-w-xl text-lg leading-relaxed text-muted">
            I turned the community&apos;s best-known Catan strategies into bots, simulated{" "}
            <strong className="text-fg">{compact(s.totals.games_simulated)} games</strong> to measure them, tuned
            them with parameter sweeps, and then trained a neural network that was never shown a strategy. It was
            rewarded for victory points early on, and after that only for winning.
          </p>
          <div className="mt-7 flex flex-wrap gap-3">
            <Link href="/results/" className="rounded-lg bg-accent px-4 py-2.5 font-semibold text-white hover:opacity-90">
              See the results
            </Link>
            <Link href="/replays/" className="rounded-lg border border-line px-4 py-2.5 font-semibold hover:bg-black/5 dark:hover:bg-white/10">
              Watch replays
            </Link>
          </div>
          <dl className="mt-9 grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Stat value={compact(s.totals.games_simulated)} label="games simulated" />
            <Stat value={compact(s.engine.bot_games_per_sec)} label="bot games / sec" />
            <Stat value={String(strategies.length)} label="strategy bots" />
            {headline ? (
              <Stat value={pct(headline.win_rate)} label="AI win rate vs. top 3" accent />
            ) : (
              <Stat value={best ? pct(best.win_rate) : "–"} label={`best strategy (${best ? botLabel(best.bot) : ""})`} />
            )}
          </dl>
        </div>
        {FEATURED && (
          <div className="card overflow-hidden p-2">
            <ReplayPlayer id={FEATURED.id} autoplay compact />
            <div className="flex items-center justify-between px-2 pb-1 pt-2 text-xs text-muted">
              <span>{FEATURED.title}</span>
              <Link href={`/replays/${FEATURED.id}/`} className="link">open full replay →</Link>
            </div>
          </div>
        )}
      </section>

      <section className="grid gap-5 md:grid-cols-3">
        <Step n={1} title="Strategies as bots">
          Ore-Wheat-Sheep, Road Builder, Dev-Card Rush, Port Master and a Balanced player, each a parameterised
          heuristic bot playing under the same hidden-information rules a human does.
        </Step>
        <Step n={2} title="Millions of games">
          A Rust engine plays the full base game at {compact(s.engine.bot_games_per_sec)} bot games per second. Every
          matchup rotates seats and reuses the same boards and dice, so comparisons are fair and tight.
        </Step>
        <Step n={3} title="An AI that finds its own way">
          A PPO agent with action masking trains through a curriculum and then a self-play league that keeps
          picking the opponents it struggles against most.
        </Step>
      </section>

      {headline?.opponents && headline.opponents.length > 0 && (
        <section>
          <div className="mb-4 flex items-end justify-between gap-4">
            <div>
              <h2 className="font-display text-2xl font-semibold">The AI against the three strongest strategies</h2>
              <p className="text-sm text-muted">
                {fmtInt(headline.games)} games, all seats rotated. {standing(headline.win_rate, headline.opponents)}
              </p>
            </div>
            <Link href="/results/" className="link shrink-0 whitespace-nowrap text-sm">full evaluation →</Link>
          </div>
          <div className="card p-4 sm:p-6">
            <WinRateBars
              reference={0.25}
              rows={[
                { bot: "learned", win_rate: headline.win_rate, ci_low: headline.ci_low, ci_high: headline.ci_high },
                ...headline.opponents,
              ]
                .sort((a, b) => b.win_rate - a.win_rate)
                .map((r) => ({
                  label: botLabel(r.bot),
                  value: r.win_rate,
                  lo: r.ci_low,
                  hi: r.ci_high,
                  color: botColor(r.bot),
                }))}
            />
          </div>
        </section>
      )}

      <section>
        <div className="mb-4 flex items-end justify-between gap-4">
          <div>
            <h2 className="font-display text-2xl font-semibold">Strategy win rates</h2>
            <p className="text-sm text-muted">
              {compact(s.tournament.games)} four-player games between the strategy bots. 25% is break-even.
            </p>
          </div>
          <Link href="/results/" className="link shrink-0 whitespace-nowrap text-sm">all results →</Link>
        </div>
        <div className="card p-4 sm:p-6">
          <WinRateBars
            reference={0.25}
            rows={strategies.map((b) => ({
              label: botLabel(b.bot),
              value: b.win_rate,
              lo: b.ci_low,
              hi: b.ci_high,
              color: botColor(b.bot),
            }))}
          />
        </div>
      </section>
    </div>
  );
}

/** "It finishes level with X, ahead of Y and behind Z." Gaps under one point count as level. */
function standing(agent: number, opponents: { bot: string; win_rate: number }[]): string {
  const names = (xs: { bot: string }[]) => {
    const n = xs.map((x) => botLabel(x.bot));
    return n.length > 1 ? `${n.slice(0, -1).join(", ")} and ${n[n.length - 1]}` : n[0];
  };
  const level = opponents.filter((o) => Math.abs(o.win_rate - agent) <= 0.01);
  const ahead = opponents.filter((o) => o.win_rate < agent - 0.01);
  const behind = opponents.filter((o) => o.win_rate > agent + 0.01);
  const parts = [
    level.length ? `level with ${names(level)}` : "",
    ahead.length ? `ahead of ${names(ahead)}` : "",
    behind.length ? `behind ${names(behind)}` : "",
  ].filter(Boolean);
  if (!parts.length) return "";
  const joined = parts.length > 1 ? `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}` : parts[0];
  return `The AI finishes ${joined}.`;
}

function Stat({ value, label, accent }: { value: string; label: string; accent?: boolean }) {
  return (
    <div>
      <dt className="sr-only">{label}</dt>
      <dd className={`font-display text-3xl font-semibold tabular-nums ${accent ? "text-accent" : ""}`}>{value}</dd>
      <dd className="mt-0.5 text-xs text-muted">{label}</dd>
    </div>
  );
}

function Step({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <div className="card p-5">
      <div className="flex h-8 w-8 items-center justify-center rounded-full bg-accent/15 font-display font-semibold text-accent">{n}</div>
      <h3 className="mt-3 font-semibold">{title}</h3>
      <p className="mt-1.5 text-sm leading-relaxed text-muted">{children}</p>
    </div>
  );
}

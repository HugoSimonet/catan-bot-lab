import type { Metadata } from "next";
import Link from "next/link";

import { Legend, LineChart, WinRateBars } from "@/components/charts";
import { SUMMARY } from "@/lib/data";
import { botColor, botLabel, compact, fmtInt, pct } from "@/lib/theme";
import type { ImprovementStep, TrainingPhase } from "@/lib/types";

export const metadata: Metadata = {
  title: "Making it stronger",
  description:
    "What happened after the first training run: a diagnosis, a second training phase against the strongest strategies, and look-ahead before uncertain moves.",
};

const STEP_COLORS: Record<string, string> = {
  phase1: "#e8a3ad",
  phase1_lookahead: "#dd7684",
  phase2: "#d1495b",
  phase2_lookahead: "#9e2235",
};

/** Look-ahead settings used when the data doesn't say (matches LookaheadConfig defaults). */
const DEFAULT_LA = { k: 4, sims: 8, threshold: 1.01, min_prior: 0.02 };

const prob = (x: number) => pct(x, x < 0.01 ? 1 : 0);

const pts = (d: number) => `${d >= 0 ? "+" : "−"}${Math.abs(100 * d).toFixed(1)} pts`;

export default function Improving() {
  const s = SUMMARY;
  const imp = s.improvements;
  const phases = s.training?.phases ?? [];
  const p1 = phases.find((p) => p.run === "main");
  const p2 = phases.find((p) => p.run === "phase2");
  const step = (key: string) => imp?.steps.find((x) => x.key === key);
  const first = step("phase1");
  const final = imp?.steps[imp.steps.length - 1];
  const base = imp?.baseline;
  const lav = s.lookahead_variants;
  const la = { ...DEFAULT_LA, ...(final?.lookahead ?? lav?.variants[lav.variants.length - 1]?.lookahead ?? {}) };
  const variant = (key: string) => lav?.variants.find((v) => v.key === key);

  return (
    <article className="flex flex-col gap-14">
      <header className="max-w-3xl">
        <p className="text-sm font-semibold uppercase tracking-[0.18em] text-accent">After the first training run</p>
        <h1 className="mt-2 font-display text-4xl font-semibold">Making the AI stronger</h1>
        <p className="mt-4 text-lg leading-relaxed text-muted">
          The first run ended close but short.
          {first && base && (
            <>
              {" "}Against the three strongest strategies the agent won <strong className="text-fg">{pct(first.win_rate)}</strong>{" "}
              of its games, while the tuned {botLabel(base.bot)} bot, put in the same seat, won{" "}
              <strong className="text-fg">{pct(base.win_rate)}</strong>.
            </>
          )}{" "}
          Two changes followed: a second training phase aimed at the opponents it was losing to, and a look-ahead
          that imagines the next turn before an uncertain move.
          {final && first && final !== first && (
            <>
              {" "}Together they take it to <strong className="text-fg">{pct(final.win_rate)}</strong> ({pts(final.win_rate - first.win_rate)}).
            </>
          )}
        </p>
      </header>

      {imp && (
        <section>
          <SectionTitle
            title="Step by step"
            sub={`Every step plays the same ${fmtInt(imp.steps[0].games)} games (same boards and dice) against ${imp.table
              .map(botLabel)
              .join(", ")}, all seats rotated.`}
          />
          <div className="card p-4 sm:p-6">
            <WinRateBars
              reference={0.25}
              wideLabels
              rows={[
                ...imp.steps.map((st) => ({
                  label: st.label,
                  value: st.win_rate,
                  lo: st.ci_low,
                  hi: st.ci_high,
                  color: STEP_COLORS[st.key] ?? botColor("learned"),
                })),
                ...(base
                  ? [{
                      label: `${botLabel(base.bot)} in the same seat`,
                      value: base.win_rate,
                      lo: base.ci_low,
                      hi: base.ci_high,
                      color: botColor(base.bot),
                    }]
                  : []),
              ]}
            />
          </div>
          {final && base && <Verdict final={final} base={base} />}
        </section>
      )}

      {p1 && (
        <section className="grid gap-8 lg:grid-cols-[1fr_1.1fr] lg:items-start">
          <div className="prose-body">
            <SectionTitle title="1 · Diagnosis: who was it practising against?" />
            <p>
              In the first run&apos;s final stage, a self-play league, each opponent seat was a strategy bot only half
              of the time. The other half went to frozen copies of the agent and to the agent itself.
            </p>
            <p>
              The training log keeps a pairwise score for every opponent: how often the agent finished ahead of it.
              Against its own past copies the agent was already ahead about half the time or more. Against the
              opponents it actually had to beat, it was behind: it finished ahead of Balanced in{" "}
              {pct(p1.beat.bot_balanced ?? 0, 0)} of games and of Road Builder in {pct(p1.beat.bot_road_builder ?? 0, 0)}.
              Much of its practice went to opponents it already handled.
            </p>
          </div>
          <div className="flex flex-col gap-4">
            <div className="card p-4 sm:p-5">
              <h3 className="font-semibold">Pairwise score at the end of the first run</h3>
              <p className="mt-1 text-sm text-muted">
                Share of games the agent finished ahead of each opponent (win, or more points), over its last 100 updates.
              </p>
              <div className="mt-4">
                <WinRateBars
                  reference={0.5}
                  referenceLabel="even (50%)"
                  max={1}
                  rows={beatRows(p1)}
                />
              </div>
            </div>
            <MixCard phases={phases} />
          </div>
        </section>
      )}

      {p2 && p1 && (
        <section>
          <SectionTitle
            title="2 · A second training phase where it loses"
            sub={`${fmtInt(p2.updates)} more PPO updates starting from the first run's final weights: ${compact(p2.decisions)} decisions over ${compact(p2.games)} games (${p2.hours.toFixed(1)} h)`}
          />
          <div className="grid gap-4 lg:grid-cols-[1fr_1.4fr]">
            <div className="card overflow-x-auto p-2 sm:p-4">
              <table className="data">
                <thead>
                  <tr>
                    <th />
                    <th>First run (league)</th>
                    <th>Phase 2</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Opponent seats that are strategy bots</td>
                    <td>{pct(p1.config.opponent_mix.bots, 0)}</td>
                    <td className="font-semibold">{pct(p2.config.opponent_mix.bots, 0)}</td>
                  </tr>
                  <tr>
                    <td>Past copies / itself</td>
                    <td>{pct(p1.config.opponent_mix.snapshots, 0)} / {pct(p1.config.opponent_mix.self, 0)}</td>
                    <td>{pct(p2.config.opponent_mix.snapshots, 0)} / {pct(p2.config.opponent_mix.self, 0)}</td>
                  </tr>
                  <tr>
                    <td>Decisions per update</td>
                    <td>{fmtInt(p1.config.batch)}</td>
                    <td>{fmtInt(p2.config.batch)}</td>
                  </tr>
                  <tr>
                    <td>Learning rate</td>
                    <td>{p1.config.lr} → {p1.config.lr_final}</td>
                    <td>{p2.config.lr} → {p2.config.lr_final}</td>
                  </tr>
                </tbody>
              </table>
              <p className="px-2 pb-1 pt-3 text-xs text-muted">
                Strategy bots are still picked by prioritised fictitious self-play, so the ones it beats least come up most.
              </p>
            </div>
            <div className="card p-4 sm:p-6">
              <Phase2Curve phase1={p1} phase2={p2} />
            </div>
          </div>
          <p className="mt-4 max-w-3xl text-muted">
            Over the last 100 updates of phase 2 the agent finished ahead of {botLabel("balanced")} in{" "}
            {pct(p2.beat.bot_balanced ?? 0, 0)} of its training games ({pct(p1.beat.bot_balanced ?? 0, 0)} at the end of
            the first run) and ahead of {botLabel("road_builder")} in {pct(p2.beat.bot_road_builder ?? 0, 0)} (
            {pct(p1.beat.bot_road_builder ?? 0, 0)}).
            {step("phase1") && step("phase2") && (
              <>
                {" "}On the same {fmtInt(step("phase2")!.games)} evaluation games, the network on its own goes from{" "}
                {pct(step("phase1")!.win_rate)} to <strong className="text-fg">{pct(step("phase2")!.win_rate)}</strong>{" "}
                ({pts(step("phase2")!.win_rate - step("phase1")!.win_rate)}).
              </>
            )}
          </p>
        </section>
      )}

      <section>
        <SectionTitle
          title="3 · Look-ahead: imagining the next turn"
          sub="The network is trained to choose instantly. At decision time there's room to think first."
        />
        <LookaheadDiagram k={la.k} sims={la.sims} minPrior={la.min_prior} />
        <div className="prose-body mt-6 max-w-3xl">
          <p>
            Most of the time the agent is sure of its move: its policy puts nearly all the probability on one option.
            {la.threshold > 1
              ? ` At every decision where it sees two or more plausible moves (each at least ${prob(la.min_prior)} likely),`
              : ` When its favourite move is below ${pct(la.threshold, 0)},`}{" "}
            the look-ahead shortlists up to {la.k} of them and plays each one forward in {la.sims} imagined futures.
            Each future re-deals what the agent can&apos;t see (opponents&apos; cards, the development deck, the
            dice), so it never peeks, and every candidate gets the <em>same</em> {la.sims} futures, so they are
            compared on equal luck. The agent finishes its turn with its own policy, stand-in opponents play their
            turns, and the agent&apos;s own value network scores the position at its first decision after theirs,
            normally the start of its next turn. The candidate with the best average score is played.
          </p>
          {first && step("phase1_lookahead") && step("phase2") && step("phase2_lookahead") && (
            <p>
              The two improvements stack. On the same evaluation games, look-ahead adds{" "}
              {pts(step("phase1_lookahead")!.win_rate - first.win_rate)} to the first agent and{" "}
              {pts(step("phase2_lookahead")!.win_rate - step("phase2")!.win_rate)} to the phase-2 network, overriding
              its policy on {pct(step("phase2_lookahead")!.lookahead.changed_frac ?? 0, 0)} of its decisions.
            </p>
          )}
        </div>
        {lav && lav.variants.length > 1 && (
          <div className="mt-6 grid gap-4 lg:grid-cols-[1.2fr_1fr] lg:items-start">
            <div className="card overflow-x-auto p-2 sm:p-4">
              <table className="data">
                <thead>
                  <tr>
                    <th>Variant</th>
                    <th>Win rate</th>
                    <th>95% CI</th>
                    <th>vs policy alone</th>
                    <th>Moves changed</th>
                  </tr>
                </thead>
                <tbody>
                  {lav.variants.map((v) => (
                    <tr key={v.key}>
                      <td>{v.label}</td>
                      <td className="font-semibold">{pct(v.win_rate, 1)}</td>
                      <td>{pct(v.ci_low, 1)} – {pct(v.ci_high, 1)}</td>
                      <td>{v.key === lav.variants[0].key ? "–" : pts(v.win_rate - lav.variants[0].win_rate)}</td>
                      <td>{v.lookahead.changed_frac != null ? pct(v.lookahead.changed_frac, 1) : "–"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="px-2 pb-1 pt-3 text-xs text-muted">
                First-run agent, {fmtInt(lav.variants[0].games)} games per variant on the same seeds, against{" "}
                {lav.table.map(botLabel).join(", ")}. &quot;Moves changed&quot;: share of all its decisions where the
                look-ahead overrode the policy. The settings were chosen on these games; every other result on this page
                uses different games.
              </p>
            </div>
            <div className="prose-body">
              <h3>The first version didn&apos;t work</h3>
              <p>
                The first version scored each candidate at the agent&apos;s very next decision. That compares
                positions taken at different moments: a bank trade keeps the turn going and was scored immediately,
                while &quot;end turn&quot; was scored only after three opponents had rolled and built. It overrode the
                policy on about half of the moves it examined and gained almost nothing; more imagined futures
                didn&apos;t help.
              </p>
              <p>
                The fix was to play every candidate to the same point, the agent&apos;s next turn, on the same
                imagined futures
                {variant("turn") && variant("none") ? ` (${pts(variant("turn")!.win_rate - variant("none")!.win_rate)})` : ""}.
                Only overriding the policy when the gain is clear rescued part of the first version, whose
                comparisons were noisy, but adds nothing once every candidate faces the same futures. The final
                version instead looks at more decisions with fewer futures each.
              </p>
            </div>
          </div>
        )}
      </section>

      <section className="prose-body max-w-3xl">
        <h2 className="font-display text-2xl font-semibold">What the look-ahead is allowed to know</h2>
        <ul>
          <li>
            <strong>No peeking.</strong> Opponents&apos; hidden cards are re-dealt from the cards the agent can&apos;t
            see, keeping each opponent&apos;s hand size; the development deck is reshuffled; the dice are re-seeded.
            The real future is never used.
          </li>
          <li>
            <strong>Its own judgement.</strong> The agent&apos;s moves inside the imagined futures come from its own
            policy, and the score comes from its own value network, both learned in training.
          </li>
          <li>
            <strong>A stand-in for opponents.</strong> In the imagined futures, every opponent is played by the
            Balanced bot. The real opponents can be different strategies: the look-ahead doesn&apos;t know who it is
            playing.
          </li>
          <li>
            <strong>A cost.</strong> Games take longer, because hundreds of imagined turns are simulated in Rust and
            scored on the GPU for each uncertain move.
            {first && step("phase1_lookahead") && (
              <>
                {" "}With 512 games in parallel, an evaluation game takes about{" "}
                {Math.round((1000 * step("phase1_lookahead")!.seconds) / step("phase1_lookahead")!.games)} ms with
                look-ahead instead of {((1000 * first.seconds) / first.games).toFixed(1)} ms: still a fraction of a
                second per game.
              </>
            )}
          </li>
        </ul>
        <p>
          The full results are on the <Link href="/results/">results page</Link>, and the code is in{" "}
          <code>crates/catan-core/src/search.rs</code> and <code>python/catanbot/lookahead.py</code>.
        </p>
      </section>
    </article>
  );
}

function SectionTitle({ title, sub }: { title: string; sub?: string }) {
  return (
    <div className="mb-4">
      <h2 className="font-display text-2xl font-semibold">{title}</h2>
      {sub && <p className="mt-1 text-sm text-muted">{sub}</p>}
    </div>
  );
}

function Verdict({ final, base }: { final: ImprovementStep; base: { bot: string; win_rate: number; ci_low: number; ci_high: number } }) {
  const above = final.ci_low > base.ci_high;
  const below = final.ci_high < base.ci_low;
  const parity = final.ci_low > 0.25;
  let text: string;
  if (above) {
    text = `The improved agent now wins more often than ${botLabel(base.bot)} does from the same seat, and the confidence intervals don't overlap.`;
  } else if (below) {
    text = `The improved agent is still behind ${botLabel(base.bot)} playing from the same seat.`;
  } else {
    text = `The improved agent is now level with ${botLabel(base.bot)} playing from the same seat: the confidence intervals overlap.`;
  }
  return (
    <p className="mt-4 max-w-3xl text-muted">
      {text}{" "}
      {parity ? "It is clearly above the 25% break-even of a four-player table." : "It is not yet clearly above the 25% break-even of a four-player table."}
    </p>
  );
}

function beatRows(p: TrainingPhase) {
  const order: [string, string, string][] = [
    ["snapshots_mean", "Its past copies (avg)", "#e8a3ad"],
    ["bot_dev_rush", botLabel("dev_rush"), botColor("dev_rush")],
    ["bot_port_master", botLabel("port_master"), botColor("port_master")],
    ["bot_ows", botLabel("ows"), botColor("ows")],
    ["bot_balanced", botLabel("balanced"), botColor("balanced")],
    ["bot_road_builder", botLabel("road_builder"), botColor("road_builder")],
  ];
  return order
    .filter(([k]) => p.beat[k] != null)
    .map(([k, label, color]) => ({ label, value: p.beat[k], color }));
}

function MixCard({ phases }: { phases: TrainingPhase[] }) {
  const parts: [keyof TrainingPhase["config"]["opponent_mix"], string, string][] = [
    ["bots", "Strategy bots", "#2a6f97"],
    ["snapshots", "Past copies", "#e8a3ad"],
    ["self", "Itself", "#d1495b"],
  ];
  const name = (run: string) => (run === "main" ? "First run (league stage)" : "Phase 2");
  return (
    <div className="card p-4 sm:p-5">
      <h3 className="font-semibold">Who sat at its table</h3>
      <p className="mt-1 text-sm text-muted">Share of opponent seats by type.</p>
      <div className="mt-4 flex flex-col gap-3">
        {phases.map((ph) => (
          <div key={ph.run} className="flex flex-col gap-1 text-sm sm:flex-row sm:items-center sm:gap-3">
            <span className="w-48 shrink-0 font-medium">{name(ph.run)}</span>
            <div className="flex h-6 flex-1 overflow-hidden rounded">
              {parts.map(([k, label, color]) =>
                ph.config.opponent_mix[k] > 0 ? (
                  <div key={k} className="flex items-center justify-center text-[11px] font-semibold text-white"
                    style={{ width: `${100 * ph.config.opponent_mix[k]}%`, background: color }}
                    title={`${label}: ${pct(ph.config.opponent_mix[k], 0)}`}>
                    {pct(ph.config.opponent_mix[k], 0)}
                  </div>
                ) : null,
              )}
            </div>
          </div>
        ))}
        <div className="flex flex-wrap gap-3 text-xs text-muted">
          {parts.map(([k, label, color]) => (
            <span key={k} className="inline-flex items-center gap-1">
              <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: color }} /> {label}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

function Phase2Curve({ phase1, phase2 }: { phase1: TrainingPhase; phase2: TrainingPhase }) {
  const s = SUMMARY.training!;
  const key = s.eval_tables[0]?.join("_");
  if (!key) return null;
  // The last part of the first run for context, then phase 2.
  const from = phase2.start_update - Math.min(2000, phase1.updates);
  const pts = s.curve.filter((c) => c.update >= from && c.eval?.[key]);
  const split = (a: number, b: number) =>
    pts.filter((c) => c.update >= a && c.update <= b).map((c) => ({
      x: c.update,
      y: c.eval![key].win_rate,
      lo: c.eval![key].ci_low,
      hi: c.eval![key].ci_high,
    }));
  const series = [
    { name: "First run (512-game evaluations)", color: "#e8a3ad", points: split(from, phase2.start_update) },
    { name: "Phase 2 (2,048-game evaluations)", color: "#d1495b", points: split(phase2.start_update, Infinity) },
  ];
  return (
    <>
      <h3 className="font-semibold">Evaluation win rate during training</h3>
      <p className="mt-1 text-sm text-muted">vs {s.eval_tables[0].map(botLabel).join(", ")}; fixed evaluation seeds.</p>
      <div className="mt-3">
        <LineChart
          series={series}
          xLabel="training update"
          yLabel="win rate"
          xFormat={(v) => compact(v)}
          reference={{ y: 0.25, label: "break-even (25%)" }}
          height={260}
        />
      </div>
      <Legend items={series.map((x) => ({ name: x.name, color: x.color }))} />
    </>
  );
}

function LookaheadDiagram({ k, sims, minPrior }: { k: number; sims: number; minPrior: number }) {
  const steps: { title: string; body: string }[] = [
    { title: "Shortlist", body: `Up to ${k} moves the policy finds plausible (at least ${prob(minPrior)} likely).` },
    { title: "Re-deal the unknowns", body: `${sims} imagined futures: opponents' cards, the dev deck and the dice re-dealt. The same ${sims} for every move.` },
    { title: "Play to the next turn", body: "The agent finishes its turn with its own policy; stand-in opponents play theirs." },
    { title: "Score and choose", body: "The agent's value network rates where each future ends up. The best average wins." },
  ];
  return (
    <ol className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      {steps.map((st, i) => (
        <li key={st.title} className="card relative p-4">
          <div className="flex h-7 w-7 items-center justify-center rounded-full bg-accent/15 font-display text-sm font-semibold text-accent">
            {i + 1}
          </div>
          <h3 className="mt-2 font-semibold">{st.title}</h3>
          <p className="mt-1 text-sm leading-relaxed text-muted">{st.body}</p>
          {i < steps.length - 1 && (
            <span aria-hidden className="absolute -right-2.5 top-1/2 hidden -translate-y-1/2 text-lg text-muted lg:block">→</span>
          )}
        </li>
      ))}
    </ol>
  );
}

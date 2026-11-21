import type { Metadata } from "next";

import { Legend, LineChart, WinRateBars } from "@/components/charts";
import { SUMMARY } from "@/lib/data";
import { RESOURCE_FILL, RESOURCE_LABELS, botColor, botLabel, compact, fmtInt, pct } from "@/lib/theme";

export const metadata: Metadata = {
  title: "Results",
  description: "Win rates of Catan strategy bots, parameter sweeps, and the reinforcement-learning agent's training and evaluation.",
};

const PARAM_LABELS: Record<string, string> = {
  diversity_weight: "Diversity bonus",
  specialization_weight: "Specialisation bonus",
  port_generic_weight: "3:1 harbour value",
  port_specific_weight: "2:1 harbour value",
  priority_city: "City priority",
  priority_settlement: "Settlement priority",
  priority_road: "Road priority",
  priority_dev: "Dev-card priority",
  longest_road_weight: "Longest Road drive",
  largest_army_weight: "Largest Army drive",
  knight_aggression: "Knight aggression",
  robber_leader_weight: "Robber targets leader",
  trade_aggression: "Bank-trade aggression",
  discard_avoidance: "Discard avoidance",
  "resource_weights.0": "Wood weight",
  "resource_weights.1": "Brick weight",
  "resource_weights.2": "Sheep weight",
  "resource_weights.3": "Wheat weight",
  "resource_weights.4": "Ore weight",
};
const paramLabel = (p: string) => PARAM_LABELS[p] ?? p;

export default function Results() {
  const s = SUMMARY;
  const bots = s.tournament.bots;
  return (
    <div className="flex flex-col gap-14">
      <header>
        <h1 className="font-display text-4xl font-semibold">Results</h1>
        <p className="mt-3 max-w-3xl text-muted">
          Every number on this page comes from simulated four-player games with seat rotation (each lineup is played
          from all four seats) and common random numbers (every variant sees the same boards and dice). Error bars
          are 95% Wilson confidence intervals. In a four-player game, 25% is break-even.
        </p>
      </header>

      <section>
        <SectionTitle title="1 · Which strategy wins most?" sub={`${compact(s.tournament.games)} games, random 4-bot tables from a pool of ${bots.length} bots`} />
        <div className="card p-4 sm:p-6">
          <WinRateBars
            reference={0.25}
            rows={bots.map((b) => ({ label: botLabel(b.bot), value: b.win_rate, lo: b.ci_low, hi: b.ci_high, color: botColor(b.bot) }))}
          />
        </div>
        <div className="card mt-4 overflow-x-auto p-2 sm:p-4">
          <table className="data">
            <thead>
              <tr>
                <th>Bot</th>
                <th>Win rate</th>
                <th>Avg VP</th>
                <th>Settlements</th>
                <th>Cities</th>
                <th>Roads</th>
                <th>Dev cards</th>
                <th>Bank trades</th>
                <th>Wins with Longest Road</th>
                <th>Wins with Largest Army</th>
                <th>Turns to win</th>
              </tr>
            </thead>
            <tbody>
              {bots.map((b) => (
                <tr key={b.bot}>
                  <td className="whitespace-nowrap font-semibold">
                    <span className="mr-2 inline-block h-2.5 w-2.5 rounded-full" style={{ background: botColor(b.bot) }} />
                    {botLabel(b.bot)}
                  </td>
                  <td>{pct(b.win_rate, 2)}</td>
                  <td>{b.avg_vp.toFixed(2)}</td>
                  <td>{b.settlements?.toFixed(2)}</td>
                  <td>{b.cities?.toFixed(2)}</td>
                  <td>{b.roads?.toFixed(1)}</td>
                  <td>{b.dev_bought?.toFixed(1)}</td>
                  <td>{b.trades?.toFixed(1)}</td>
                  <td>{b.lr_in_wins != null ? pct(b.lr_in_wins, 0) : "–"}</td>
                  <td>{b.la_in_wins != null ? pct(b.la_in_wins, 0) : "–"}</td>
                  <td>{b.turns_in_wins != null ? b.turns_in_wins.toFixed(0) : "–"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-4 grid gap-4 md:grid-cols-2">
          <div className="card p-4 sm:p-5">
            <h3 className="font-semibold">Going first matters</h3>
            <p className="mt-1 text-sm text-muted">Win rate by seat across all games (seat 1 places and rolls first).</p>
            <div className="mt-3">
              <WinRateBars
                reference={0.25}
                referenceLabel="25%"
                rows={s.tournament.seat_win_rate.map((w, i) => ({ label: `Seat ${i + 1}`, value: w, color: "#7d8597" }))}
              />
            </div>
          </div>
          {s.tournament.resource_mix && (
            <div className="card p-4 sm:p-5">
              <h3 className="font-semibold">What each strategy produces</h3>
              <p className="mt-1 text-sm text-muted">Share of each resource in everything a bot collected from dice rolls.</p>
              <div className="mt-4 flex flex-col gap-2.5">
                {s.tournament.resource_mix.map((r) => {
                  const shares = [r.wood, r.brick, r.sheep, r.wheat, r.ore];
                  return (
                    <div key={r.bot} className="flex items-center gap-3 text-sm">
                      <span className="w-32 shrink-0 font-medium">{botLabel(r.bot)}</span>
                      <div className="flex h-5 flex-1 overflow-hidden rounded">
                        {shares.map((v, i) => (
                          <div key={i} style={{ width: `${100 * v}%`, background: RESOURCE_FILL[i] }}
                            title={`${RESOURCE_LABELS[i]} ${pct(v)}`} />
                        ))}
                      </div>
                    </div>
                  );
                })}
                <div className="mt-1 flex flex-wrap gap-3 text-xs text-muted">
                  {RESOURCE_LABELS.slice(0, 5).map((l, i) => (
                    <span key={l} className="inline-flex items-center gap-1">
                      <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: RESOURCE_FILL[i] }} /> {l}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </section>

      {s.tuning && s.tuning.length > 0 && (
        <section>
          <SectionTitle title="2 · Small changes, big swings" sub="Coordinate-ascent tuning of each strategy, keeping its identity (resource preferences) fixed" />
          <div className="card overflow-x-auto p-2 sm:p-4">
            <table className="data">
              <thead>
                <tr>
                  <th>Strategy</th>
                  <th>Hand-written</th>
                  <th>Tuned</th>
                  <th>Changes that helped</th>
                </tr>
              </thead>
              <tbody>
                {s.tuning.map((t) => (
                  <tr key={t.preset}>
                    <td className="font-semibold">{botLabel(t.preset)}</td>
                    <td>{pct(t.start_win_rate, 1)}</td>
                    <td className="font-semibold">
                      {pct(t.end_win_rate, 1)}{" "}
                      <span className="text-xs text-accent">
                        ({t.end_win_rate >= t.start_win_rate ? "+" : ""}
                        {(100 * (t.end_win_rate - t.start_win_rate)).toFixed(1)} pts)
                      </span>
                    </td>
                    <td className="text-xs text-muted">
                      {t.changes.length
                        ? t.changes.map((c) => `${paramLabel(c.param)} ${fmt(c.from)} → ${fmt(c.to)}`).join(" · ")
                        : "already locally optimal"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-sm text-muted">
            Win rates are measured against a field of the other four hand-written strategies. The tuned versions are
            what the learning agent trains and is evaluated against.
          </p>
        </section>
      )}

      {s.sweeps && s.sweeps.length > 0 && (
        <section>
          <SectionTitle title="3 · Parameter sweeps" sub="Win rate of one tuned strategy as one parameter varies; everything else fixed" />
          <div className="grid gap-4 md:grid-cols-2">
            {s.sweeps.map((sw) => (
              <div key={`${sw.preset}-${sw.param}`} className="card p-4">
                <h3 className="font-semibold">{sw.label}</h3>
                <p className="text-xs text-muted">
                  {botLabel(sw.preset)} · {paramLabel(sw.param)} (tuned value {fmt(sw.default)})
                </p>
                <LineChart
                  width={400}
                  height={250}
                  markers
                  xLabel={paramLabel(sw.param)}
                  xFormat={(v) => fmt(v)}
                  yFormat={(v) => pct(v, 0)}
                  reference={{ y: 0.25, label: "25%" }}
                  series={[{
                    name: botLabel(sw.preset),
                    color: botColor(sw.preset),
                    points: sw.points.map((p) => ({ x: p.value, y: p.win_rate, lo: p.ci_low, hi: p.ci_high })),
                  }]}
                />
              </div>
            ))}
          </div>
        </section>
      )}

      {s.training && (
        <section>
          <SectionTitle
            title="4 · Learning to win"
            sub={`${compact(s.training.decisions)} decisions over ${compact(s.training.games)} training games (${s.training.hours.toFixed(1)} h on one GTX 1660 Ti)`}
          />
          <div className="card p-4 sm:p-6">
            {(() => {
              const tables = s.training.eval_tables;
              const colors = ["#d1495b", "#2a6f97", "#3c8d5a"];
              const series = tables.map((t, i) => {
                const key = t.join("_");
                return {
                  name: `vs ${t.map(botLabel).join(", ")}`,
                  color: colors[i % colors.length],
                  points: s.training!.curve
                    .filter((c) => c.eval?.[key])
                    .map((c) => ({ x: c.decisions, y: c.eval![key].win_rate, lo: c.eval![key].ci_low, hi: c.eval![key].ci_high })),
                };
              });
              return (
                <>
                  <LineChart
                    series={series}
                    xLabel="decisions trained on"
                    yLabel="evaluation win rate"
                    xFormat={(v) => compact(v)}
                    reference={{ y: 0.25, label: "break-even (25%)" }}
                  />
                  <Legend items={series.map((x) => ({ name: x.name, color: x.color }))} />
                  <StageStrip curve={s.training.curve} />
                </>
              );
            })()}
          </div>
        </section>
      )}

      {s.evaluation && s.evaluation.length > 0 && (
        <section>
          <SectionTitle
            title="5 · Final evaluation"
            sub={`${s.evaluation[0].lookahead ? "The final agent (both training phases, playing with look-ahead)" : "The trained agent"} in one seat, three strategy bots in the others, all seats rotated`}
          />
          <div className="card p-4 sm:p-6">
            <WinRateBars
              reference={0.25}
              rows={s.evaluation.flatMap((e) => [
                {
                  label: `AI vs ${e.table.map(shortLabel).join("/")}`,
                  value: e.win_rate,
                  lo: e.ci_low,
                  hi: e.ci_high,
                  color: botColor("learned"),
                },
                ...(e.baseline
                  ? [{
                      label: `${botLabel(e.baseline.bot)} in same seat`,
                      value: e.baseline.win_rate,
                      lo: e.baseline.ci_low,
                      hi: e.baseline.ci_high,
                      color: botColor(e.baseline.bot),
                    }]
                  : []),
              ])}
            />
          </div>
          <div className="card mt-4 overflow-x-auto p-2 sm:p-4">
            <table className="data">
              <thead>
                <tr>
                  <th>Opponents</th>
                  <th>Games</th>
                  <th>AI win rate</th>
                  <th>95% CI</th>
                  <th>AI avg VP</th>
                </tr>
              </thead>
              <tbody>
                {s.evaluation.map((e) => (
                  <tr key={e.table.join()}>
                    <td>{e.table.map(botLabel).join(", ")}</td>
                    <td>{fmtInt(e.games)}</td>
                    <td className="font-semibold">{pct(e.win_rate, 2)}</td>
                    <td>{pct(e.ci_low, 2)} – {pct(e.ci_high, 2)}</td>
                    <td>{e.avg_vp.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {s.style && (
        <section>
          <SectionTitle title="6 · What did the AI learn?" sub="How the agent plays compared with the strategies it was trained against" />
          <div className="card overflow-x-auto p-2 sm:p-4">
            <table className="data">
              <thead>
                <tr>
                  <th>Player</th>
                  {s.style.metrics.map((m) => (
                    <th key={m.key}>{m.label}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {s.style.rows.map((r) => (
                  <tr key={r.bot} className={r.bot === "learned" ? "font-semibold" : ""}>
                    <td>
                      <span className="mr-2 inline-block h-2.5 w-2.5 rounded-full" style={{ background: botColor(String(r.bot)) }} />
                      {botLabel(String(r.bot))}
                    </td>
                    {s.style!.metrics.map((m) => {
                      const v = r[m.key];
                      return <td key={m.key}>{typeof v === "number" ? (m.format === "pct" ? pct(v, 0) : v.toFixed(2)) : v}</td>;
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <p className="text-xs text-muted">Data generated {new Date(s.generated_at).toUTCString()}.</p>
    </div>
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

function StageStrip({ curve }: { curve: { decisions: number; stage: string }[] }) {
  if (!curve.length) return null;
  const total = curve[curve.length - 1].decisions || 1;
  const spans: { stage: string; start: number; end: number }[] = [];
  for (const c of curve) {
    const last = spans[spans.length - 1];
    if (last && last.stage === c.stage) last.end = c.decisions;
    else spans.push({ stage: c.stage, start: last ? last.end : 0, end: c.decisions });
  }
  const colors: Record<string, string> = {
    warmup: "#b8bcc4",
    ladder: "#7d8590",
    strategies: "#2a6f97",
    league: "#d1495b",
    focus: "#9e2235",
  };
  const label = (stage: string) => (stage === "focus" ? "phase 2: focus on the strongest bots" : `${stage} curriculum stage`);
  return (
    <div className="mt-4">
      <div className="flex h-3 overflow-hidden rounded">
        {spans.map((sp) => (
          <div key={sp.stage + sp.start} style={{ width: `${(100 * (sp.end - sp.start)) / total}%`, background: colors[sp.stage] ?? "#999" }}
            title={sp.stage} />
        ))}
      </div>
      <div className="mt-1.5 flex flex-wrap gap-3 text-xs text-muted">
        {spans.map((sp) => (
          <span key={sp.stage + sp.start} className="inline-flex items-center gap-1">
            <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: colors[sp.stage] ?? "#999" }} />
            {label(sp.stage)}
          </span>
        ))}
      </div>
    </div>
  );
}

function fmt(v: number): string {
  return Number.isInteger(v) ? String(v) : v.toFixed(Math.abs(v) < 1 ? 2 : 2).replace(/0+$/, "").replace(/\.$/, "");
}

function shortLabel(id: string): string {
  const m: Record<string, string> = {
    balanced: "BAL",
    ows: "OWS",
    road_builder: "ROAD",
    dev_rush: "DEV",
    port_master: "PORT",
    greedy: "GRD",
    random: "RND",
  };
  return m[id] ?? id;
}

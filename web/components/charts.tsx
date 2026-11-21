// Dependency-free SVG charts. Pure components (no hooks), so they render at
// build time in server components and drop into any React app unchanged.

import { pct } from "@/lib/theme";

export interface BarRow {
  label: string;
  value: number;
  lo?: number;
  hi?: number;
  color: string;
  note?: string;
}

/**
 * Horizontal bars with 95% CI whiskers and an optional reference line.
 * HTML/CSS rather than SVG so the text stays readable at phone widths.
 */
export function WinRateBars({
  rows,
  reference,
  referenceLabel = "parity (25%)",
  max,
  wideLabels = false,
}: {
  rows: BarRow[];
  reference?: number;
  referenceLabel?: string;
  max?: number;
  /** Wider label column (for long labels) on screens wider than a phone. */
  wideLabels?: boolean;
}) {
  // Literal class names, so Tailwind generates both.
  const cols = wideLabels ? "sm:grid-cols-[15rem_minmax(0,1fr)_5.5rem]" : "sm:grid-cols-[10rem_minmax(0,1fr)_5.5rem]";
  const hi = max ?? Math.min(1, Math.max(...rows.map((r) => r.hi ?? r.value), reference ?? 0) * 1.12);
  const pos = (v: number) => `${(100 * Math.max(0, v)) / hi}%`;
  return (
    <div role="list" aria-label="Win rates" className="flex flex-col gap-2.5">
      {rows.map((r) => (
        <div key={r.label} role="listitem"
          className={`grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1 ${cols}`}>
          <div className="truncate text-sm font-semibold sm:text-right">{r.label}</div>
          <div className="relative order-3 col-span-2 h-6 sm:order-none sm:col-span-1"
            title={`${r.label}: ${pct(r.value, 2)}${r.lo !== undefined ? ` (95% CI ${pct(r.lo, 2)}–${pct(r.hi!, 2)})` : ""}`}>
            <div className="absolute inset-y-0 left-0 rounded" style={{ width: pos(r.value), background: r.color, opacity: 0.9 }} />
            {r.lo !== undefined && r.hi !== undefined && (
              <div className="absolute top-1/2 h-2.5 -translate-y-1/2 border-x-2 border-fg/80"
                style={{ left: pos(r.lo), width: `calc(${pos(r.hi)} - ${pos(r.lo)})` }}>
                <div className="absolute inset-x-0 top-1/2 h-0.5 -translate-y-1/2 bg-fg/80" />
              </div>
            )}
            {reference !== undefined && (
              <div className="absolute -inset-y-1 w-px bg-fg/50" style={{ left: pos(reference) }} />
            )}
          </div>
          <div className="text-right text-sm tabular-nums sm:text-left">
            {pct(r.value, 1)}
            {r.note ? <span className="ml-1 text-xs text-muted">{r.note}</span> : null}
          </div>
        </div>
      ))}
      {reference !== undefined && (
        <div className={`grid grid-cols-1 sm:gap-x-3 ${cols}`}>
          <div className="hidden sm:block" />
          <div className="relative h-4 text-[11px] text-muted">
            <span className="absolute -translate-x-1/2 whitespace-nowrap" style={{ left: pos(reference) }}>
              ↑ {referenceLabel}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

export interface Series {
  name: string;
  color: string;
  points: { x: number; y: number; lo?: number; hi?: number }[];
  dashed?: boolean;
}

/** Multi-series line chart with optional confidence bands. */
export function LineChart({
  series,
  xLabel,
  yLabel,
  yFormat = (v) => pct(v, 0),
  xFormat = (v) => String(v),
  reference,
  height = 300,
  width = 720,
  yDomain,
  markers = false,
}: {
  series: Series[];
  xLabel?: string;
  yLabel?: string;
  yFormat?: (v: number) => string;
  xFormat?: (v: number) => string;
  reference?: { y: number; label: string };
  height?: number;
  /** Width of the drawing coordinate system: use roughly the rendered width so text stays legible. */
  width?: number;
  yDomain?: [number, number];
  markers?: boolean;
}) {
  const W = width;
  const H = height;
  const m = { l: 56, r: 16, t: 14, b: 44 };
  const all = series.flatMap((s) => s.points);
  if (!all.length) return null;
  const xs = all.map((p) => p.x);
  const ys = all.flatMap((p) => [p.y, p.lo ?? p.y, p.hi ?? p.y]).concat(reference ? [reference.y] : []);
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs);
  let [y0, y1] = yDomain ?? [Math.min(...ys), Math.max(...ys)];
  if (!yDomain) {
    const pad = (y1 - y0) * 0.1 || 0.05;
    y0 = Math.max(0, y0 - pad);
    y1 = y1 + pad;
  }
  const sx = (v: number) => m.l + ((v - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
  const sy = (v: number) => H - m.b - ((v - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);
  const yt = niceTicks(y0, y1, 5);
  const xt = niceTicks(x0, x1, 6);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img" aria-label={yLabel ?? "chart"}>
      {yt.map((t) => (
        <g key={`y${t}`}>
          <line x1={m.l} x2={W - m.r} y1={sy(t)} y2={sy(t)} className="stroke-line" />
          <text x={m.l - 8} y={sy(t) + 4} textAnchor="end" fontSize={11} className="fill-muted">{yFormat(t)}</text>
        </g>
      ))}
      {xt.map((t) => (
        <text key={`x${t}`} x={sx(t)} y={H - m.b + 18} textAnchor="middle" fontSize={11} className="fill-muted">
          {xFormat(t)}
        </text>
      ))}
      {xLabel && <text x={(W + m.l) / 2} y={H - 6} textAnchor="middle" fontSize={12} className="fill-muted">{xLabel}</text>}
      {yLabel && (
        <text transform={`translate(14 ${(H - m.b + m.t) / 2}) rotate(-90)`} textAnchor="middle" fontSize={12} className="fill-muted">
          {yLabel}
        </text>
      )}
      {reference && (
        <g>
          <line x1={m.l} x2={W - m.r} y1={sy(reference.y)} y2={sy(reference.y)} stroke="currentColor" strokeOpacity={0.5} strokeDasharray="6 4" />
          <text x={W - m.r - 4} y={sy(reference.y) - 6} textAnchor="end" fontSize={11} className="fill-muted">{reference.label}</text>
        </g>
      )}
      {series.map((s) => {
        const band = s.points.filter((p) => p.lo !== undefined && p.hi !== undefined);
        const path = s.points.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(" ");
        const area =
          band.length > 1
            ? `M${band.map((p) => `${sx(p.x).toFixed(1)},${sy(p.hi!).toFixed(1)}`).join(" L")} L${[...band]
                .reverse()
                .map((p) => `${sx(p.x).toFixed(1)},${sy(p.lo!).toFixed(1)}`)
                .join(" L")} Z`
            : null;
        return (
          <g key={s.name}>
            {area && <path d={area} fill={s.color} opacity={0.15} />}
            <path d={path} fill="none" stroke={s.color} strokeWidth={2.2} strokeDasharray={s.dashed ? "6 4" : undefined} />
            {markers &&
              s.points.map((p, i) => (
                <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={3.5} fill={s.color}>
                  <title>{`${s.name}: ${xFormat(p.x)} → ${yFormat(p.y)}`}</title>
                </circle>
              ))}
          </g>
        );
      })}
    </svg>
  );
}

export function Legend({ items }: { items: { name: string; color: string; dashed?: boolean }[] }) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
      {items.map((i) => (
        <span key={i.name} className="inline-flex items-center gap-1.5">
          <svg width="18" height="8" aria-hidden>
            <line x1="0" x2="18" y1="4" y2="4" stroke={i.color} strokeWidth="3" strokeDasharray={i.dashed ? "4 3" : undefined} />
          </svg>
          {i.name}
        </span>
      ))}
    </div>
  );
}

function niceTicks(a: number, b: number, n: number): number[] {
  const span = b - a || 1;
  const raw = span / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((s) => s * mag).find((s) => span / s <= n) ?? mag * 10;
  const out: number[] = [];
  for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(Number(v.toPrecision(12)));
  return out;
}

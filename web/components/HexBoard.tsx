import type { Board, Topology } from "@/lib/types";
import { NONE, type Pieces } from "@/lib/replay";
import { PLAYER_COLORS, PLAYER_STROKE, RESOURCE_FILL, RESOURCE_LABELS } from "@/lib/theme";

const PIPS: Record<number, number> = { 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1 };

interface Props {
  topology: Topology;
  board: Board;
  pieces: Pieces;
  robber: number;
  highlight?: { node?: number; edge?: number; tile?: number; player?: number };
  className?: string;
}

const SETTLEMENT: [number, number][] = [
  [-1, 0.8], [-1, -0.25], [0, -1.05], [1, -0.25], [1, 0.8],
];
const CITY: [number, number][] = [
  [-1.35, 0.9], [-1.35, -0.2], [-0.35, -0.2], [-0.35, -0.75], [0.35, -1.35], [1.05, -0.75], [1.05, 0.9],
];

function shape(points: [number, number][], cx: number, cy: number, s: number): string {
  return points.map(([x, y]) => `${(cx + x * s).toFixed(3)},${(cy + y * s).toFixed(3)}`).join(" ");
}

/** Static SVG rendering of a Catan-style board with pieces. Pure: safe in server or client trees. */
export default function HexBoard({ topology, board, pieces, robber, highlight, className }: Props) {
  const t = topology;
  const R = 5.35;
  const sea = Array.from({ length: 6 }, (_, i) => {
    const a = (Math.PI / 3) * i;
    return `${(R * Math.cos(a)).toFixed(3)},${(R * Math.sin(a)).toFixed(3)}`;
  }).join(" ");

  return (
    <svg
      viewBox="-5.6 -5.0 11.2 10.0"
      className={className}
      role="img"
      aria-label="Game board"
    >
      <defs>
        <radialGradient id="sea" cx="50%" cy="45%" r="60%">
          <stop offset="0%" stopColor="#3b82b8" />
          <stop offset="100%" stopColor="#1f5f8f" />
        </radialGradient>
        <filter id="soft" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="0.04" stdDeviation="0.04" floodOpacity="0.35" />
        </filter>
      </defs>
      <polygon points={sea} fill="url(#sea)" stroke="#16486d" strokeWidth={0.08} strokeLinejoin="round" />

      {/* Harbours */}
      {t.port_edges.map((e, slot) => {
        const [a, b] = t.edge_nodes[e];
        const [ax, ay] = t.node_xy[a];
        const [bx, by] = t.node_xy[b];
        const mx = (ax + bx) / 2;
        const my = (ay + by) / 2;
        const len = Math.hypot(mx, my);
        const px = mx + (mx / len) * 0.62;
        const py = my + (my / len) * 0.62;
        const type = board.port_type[slot];
        const generic = type === 5;
        return (
          <g key={`port-${slot}`}>
            <line x1={ax} y1={ay} x2={px} y2={py} stroke="#c9b48a" strokeWidth={0.07} />
            <line x1={bx} y1={by} x2={px} y2={py} stroke="#c9b48a" strokeWidth={0.07} />
            <circle cx={px} cy={py} r={0.28} fill={generic ? "#f6efdf" : RESOURCE_FILL[type]}
              stroke="#5b4a2e" strokeWidth={0.035} />
            <text x={px} y={py + 0.065} textAnchor="middle" fontSize={0.18} fontWeight={700}
              fill={generic ? "#3b3122" : "#fff"} style={{ fontFamily: "var(--font-mono)" }}>
              {generic ? "3:1" : "2:1"}
            </text>
            <title>{generic ? "3:1 harbour (any resource)" : `2:1 ${RESOURCE_LABELS[type]} harbour`}</title>
          </g>
        );
      })}

      {/* Tiles */}
      {t.tile_nodes.map((nodes, tile) => {
        const pts = nodes.map((n) => t.node_xy[n].map((v) => v.toFixed(3)).join(",")).join(" ");
        const res = board.tile_resource[tile];
        return (
          <polygon key={`tile-${tile}`} points={pts} fill={RESOURCE_FILL[res]} stroke="#efe3c4"
            strokeWidth={0.07} strokeLinejoin="round">
            <title>
              {RESOURCE_LABELS[res]}
              {board.tile_number[tile] ? ` ${board.tile_number[tile]}` : ""}
            </title>
          </polygon>
        );
      })}

      {/* Number tokens and robber */}
      {t.tile_xy.map(([x, y], tile) => {
        const n = board.tile_number[tile];
        const red = n === 6 || n === 8;
        const robbed = tile === robber;
        const pips = PIPS[n] ?? 0;
        return (
          <g key={`tok-${tile}`} opacity={robbed && n ? 0.55 : 1}>
            {n > 0 && (
              <>
                <circle cx={x} cy={y} r={0.36} fill="#f6efdf" stroke="#8a7650" strokeWidth={0.03} filter="url(#soft)" />
                <text x={x} y={y + 0.08} textAnchor="middle" fontSize={0.3} fontWeight={800}
                  fill={red ? "#b3261e" : "#2b2418"}>
                  {n}
                </text>
                {Array.from({ length: pips }, (_, i) => (
                  <circle key={i} cx={x + (i - (pips - 1) / 2) * 0.075} cy={y + 0.2} r={0.025}
                    fill={red ? "#b3261e" : "#2b2418"} />
                ))}
              </>
            )}
          </g>
        );
      })}
      {(() => {
        const [x, y] = t.tile_xy[robber];
        const hasToken = board.tile_number[robber] > 0;
        const rx = hasToken ? x - 0.52 : x;
        const ry = hasToken ? y + 0.05 : y;
        return (
          <g filter="url(#soft)">
            <title>Robber</title>
            <path
              d={`M ${rx - 0.16} ${ry + 0.3} Q ${rx - 0.18} ${ry - 0.02} ${rx - 0.07} ${ry - 0.1}
                  A 0.11 0.11 0 1 1 ${rx + 0.07} ${ry - 0.1} Q ${rx + 0.18} ${ry - 0.02} ${rx + 0.16} ${ry + 0.3} Z`}
              fill="#2a2a30" stroke="#0e0e12" strokeWidth={0.03}
            />
          </g>
        );
      })()}

      {/* Roads */}
      {t.edge_nodes.map(([a, b], e) => {
        const owner = pieces.edgeOwner[e];
        if (owner === NONE) return null;
        const [ax, ay] = t.node_xy[a];
        const [bx, by] = t.node_xy[b];
        const x1 = ax + (bx - ax) * 0.17;
        const y1 = ay + (by - ay) * 0.17;
        const x2 = ax + (bx - ax) * 0.83;
        const y2 = ay + (by - ay) * 0.83;
        const hl = highlight?.edge === e;
        return (
          <g key={`road-${e}`}>
            {hl && <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#fff" strokeWidth={0.34} strokeLinecap="round" className="pulse" />}
            <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={PLAYER_STROKE[owner]} strokeWidth={0.21} strokeLinecap="round" />
            <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={PLAYER_COLORS[owner]} strokeWidth={0.13} strokeLinecap="round" />
          </g>
        );
      })}

      {/* Settlements and cities */}
      {t.node_xy.map(([x, y], n) => {
        const owner = pieces.nodeOwner[n];
        if (owner === NONE) return null;
        const city = pieces.nodeCity[n] === 1;
        const hl = highlight?.node === n;
        return (
          <g key={`node-${n}`}>
            {hl && <circle cx={x} cy={y} r={0.36} fill="none" stroke="#fff" strokeWidth={0.07} className="pulse" />}
            <polygon
              points={shape(city ? CITY : SETTLEMENT, x, y, city ? 0.2 : 0.17)}
              fill={PLAYER_COLORS[owner]}
              stroke={PLAYER_STROKE[owner]}
              strokeWidth={0.045}
              strokeLinejoin="round"
              filter="url(#soft)"
            />
          </g>
        );
      })}

      {highlight?.tile !== undefined && (() => {
        const pts = t.tile_nodes[highlight.tile!]
          .map((n) => t.node_xy[n].map((v) => v.toFixed(3)).join(","))
          .join(" ");
        return <polygon points={pts} fill="none" stroke="#fff" strokeWidth={0.09} className="pulse" />;
      })()}
    </svg>
  );
}

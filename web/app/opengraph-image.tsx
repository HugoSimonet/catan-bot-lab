import { ImageResponse } from "next/og";

import { SUMMARY } from "@/lib/data";
import { compact } from "@/lib/theme";

// Rendered once at build time (required with `output: "export"`).
export const dynamic = "force-static";

export const alt = "Catan Bot Lab: strategy bots vs. a self-taught AI";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

const HEX = "0,-1 0.866,-0.5 0.866,0.5 0,1 -0.866,0.5 -0.866,-0.5";
const TILES: [number, number, string][] = [
  [0, 0, "#e3b448"], [1.732, 0, "#2f6b3a"], [-1.732, 0, "#7d8597"],
  [0.866, -1.5, "#b5532f"], [-0.866, -1.5, "#93c062"], [0.866, 1.5, "#93c062"], [-0.866, 1.5, "#2f6b3a"],
];

export default function OpenGraphImage() {
  const ev = SUMMARY.evaluation?.[0];
  const sub = ev
    ? `Learned agent wins ${(100 * ev.win_rate).toFixed(1)}% against the top three strategies (25% = break-even)`
    : `${compact(SUMMARY.totals.games_simulated)} simulated games · 5 community strategies · PPO self-play`;
  return new ImageResponse(
    (
      <div style={{ width: "100%", height: "100%", display: "flex", background: "#0f1722", color: "#ece7dc", padding: 72, alignItems: "center" }}>
        <div style={{ display: "flex", flexDirection: "column", flex: 1, paddingRight: 40 }}>
          <div style={{ fontSize: 26, letterSpacing: 6, color: "#e0714f", textTransform: "uppercase" }}>Catan Bot Lab</div>
          <div style={{ fontSize: 64, fontWeight: 700, lineHeight: 1.1, marginTop: 20 }}>
            Can a self-taught AI beat the best Catan strategies?
          </div>
          <div style={{ fontSize: 28, color: "#9aa3ae", marginTop: 28 }}>{sub}</div>
        </div>
        <svg width="360" height="360" viewBox="-3 -3 6 6">
          {TILES.map(([x, y, c], i) => (
            <polygon key={i} points={HEX} fill={c} stroke="#efe3c4" strokeWidth="0.08" transform={`translate(${x} ${y})`} />
          ))}
        </svg>
      </div>
    ),
    { ...size },
  );
}

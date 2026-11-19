import type { Metadata } from "next";
import Link from "next/link";

import { REPLAYS } from "@/lib/data";
import { PLAYER_COLORS, PLAYER_STROKE, botLabel } from "@/lib/theme";

export const metadata: Metadata = {
  title: "Replays",
  description: "Watch simulated Catan games between strategy bots and the reinforcement-learning agent, move by move.",
};

export default function Replays() {
  return (
    <div>
      <h1 className="font-display text-4xl font-semibold">Replays</h1>
      <p className="mt-3 max-w-2xl text-muted">
        Every game is fully determined by its seed and the list of moves, so any simulated game can be replayed
        exactly. These were picked to show each strategy at its best, and the learning agent in action.
      </p>
      <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {REPLAYS.map((r) => (
          <Link key={r.id} href={`/replays/${r.id}/`} className="card group flex flex-col p-5 transition hover:-translate-y-0.5 hover:shadow-md">
            <h2 className="font-semibold group-hover:text-accent">{r.title}</h2>
            {r.description && <p className="mt-1 text-sm text-muted">{r.description}</p>}
            <ul className="mt-4 flex flex-col gap-1.5 text-sm">
              {r.players.map((p, i) => (
                <li key={i} className="flex items-center gap-2">
                  <span className="dot" style={{ background: PLAYER_COLORS[i], borderColor: PLAYER_STROKE[i] }} />
                  <span className={r.winner === i ? "font-semibold" : ""}>{botLabel(p)}</span>
                  {r.winner === i && <span className="text-xs text-accent">winner</span>}
                </li>
              ))}
            </ul>
            <div className="mt-auto pt-4 text-xs text-muted">{r.turns} turns</div>
          </Link>
        ))}
      </div>
    </div>
  );
}

// Build-time data. Both files are produced by `python -m catanbot.export_web`.
import summaryJson from "@/data/summary.json";
import replayIndexJson from "@/public/data/replays/index.json";

import type { ReplayMeta, Summary } from "./types";

export const SUMMARY = summaryJson as unknown as Summary;
export const REPLAYS = replayIndexJson as unknown as ReplayMeta[];

export const FEATURED: ReplayMeta | undefined = REPLAYS.find((r) => r.featured) ?? REPLAYS[0];

/** Best (highest win-rate) strategy bots in the main tournament, excluding baselines. */
export function strategyRows() {
  return SUMMARY.tournament.bots.filter((b) => b.bot !== "random" && b.bot !== "greedy");
}

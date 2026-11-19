// Board and player palette (original colours; no official artwork is used).

export const RESOURCE_NAMES = ["wood", "brick", "sheep", "wheat", "ore", "desert"] as const;
export const RESOURCE_LABELS = ["Wood", "Brick", "Sheep", "Wheat", "Ore", "Desert"];

export const RESOURCE_FILL = [
  "#2f6b3a", // wood: forest
  "#b5532f", // brick: clay
  "#93c062", // sheep: pasture
  "#e3b448", // wheat: field
  "#7d8597", // ore: mountains
  "#d8c79f", // desert
];

export const PLAYER_COLORS = ["#d1495b", "#2f6fdf", "#f08c2e", "#f2efe6"];
export const PLAYER_STROKE = ["#5a1420", "#10285a", "#6b3505", "#3b3a36"];
export const PLAYER_NAMES = ["Red", "Blue", "Orange", "White"];

export const DEV_LABELS = ["Knight", "Victory Point", "Road Building", "Year of Plenty", "Monopoly"];

/** Human label for a bot / strategy id. */
export function botLabel(id: string): string {
  const known: Record<string, string> = {
    random: "Random",
    greedy: "Greedy",
    balanced: "Balanced",
    ows: "Ore-Wheat-Sheep",
    road_builder: "Road Builder",
    dev_rush: "Dev-Card Rush",
    port_master: "Port Master",
    learned: "Learned (RL)",
    ai: "Learned (RL)",
  };
  return known[id] ?? id;
}

/** Stable accent colour per strategy, for charts. */
export function botColor(id: string): string {
  const map: Record<string, string> = {
    balanced: "#2a6f97",
    ows: "#7d5ba6",
    road_builder: "#b5532f",
    dev_rush: "#3c8d5a",
    port_master: "#c9962b",
    greedy: "#8a8f98",
    random: "#b8bcc4",
    learned: "#d1495b",
    ai: "#d1495b",
  };
  return map[id] ?? "#5f6b7a";
}

export const pct = (x: number, digits = 1) => `${(100 * x).toFixed(digits)}%`;
export const fmtInt = (x: number) => Math.round(x).toLocaleString("en-US");

export function compact(x: number): string {
  if (x >= 1e9) return `${(x / 1e9).toFixed(1)}B`;
  if (x >= 1e6) return `${(x / 1e6).toFixed(x >= 1e7 ? 0 : 1)}M`;
  if (x >= 1e3) return `${(x / 1e3).toFixed(x >= 1e4 ? 0 : 1)}k`;
  return String(Math.round(x));
}

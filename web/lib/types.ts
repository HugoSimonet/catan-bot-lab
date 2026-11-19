// Shapes of the JSON produced by the Rust engine / Python exporter.

export type XY = [number, number];

export interface Topology {
  tile_coords: XY[];
  tile_xy: XY[];
  node_xy: XY[];
  tile_nodes: number[][];
  tile_neighbors: number[][];
  node_tiles: number[][];
  node_neighbors: number[][];
  node_edges: number[][];
  edge_nodes: [number, number][];
  port_edges: number[];
}

export interface Board {
  tile_resource: number[]; // 0..4 resources, 5 desert
  tile_number: number[]; // 0 for desert
  port_type: number[]; // 0..4 = 2:1 of resource, 5 = 3:1
  node_port: number[]; // 255 = none
}

// serde: {"type": "Roll"} | {"type": "BuildRoad", "args": 12} | {"type": "MaritimeTrade", "args": [4, 0]}
export type Action =
  | { type: "Roll" | "EndTurn" | "BuyDev" | "PlayKnight" | "PlayRoadBuilding" }
  | { type: "BuildSettlement" | "BuildCity" | "BuildRoad" | "PlayMonopoly" | "MoveRobber" | "Steal" | "Discard"; args: number }
  | { type: "PlayYearOfPlenty" | "MaritimeTrade"; args: [number, number] };

export type Phase =
  | "SetupSettlement"
  | "SetupRoad"
  | "PreRoll"
  | "Discard"
  | "MoveRobber"
  | "Steal"
  | "Main"
  | "RoadBuilding"
  | "GameOver";

export interface Frame {
  turn: number;
  actor: number;
  action: Action;
  text: string;
  roll: number | null;
  robber: number;
  phase: Phase;
  hands: number[][];
  dev_cards: number[][];
  vp: number[];
  knights: number[];
  road_length: number[];
  longest_road: number | null;
  largest_army: number | null;
}

export interface Replay {
  version: number;
  seed: number;
  players: string[];
  topology: Topology;
  board: Board;
  robber_start: number;
  winner: number | null;
  turns: number;
  frames: Frame[];
}

export interface ReplayMeta {
  id: string;
  title: string;
  description?: string;
  players: string[];
  winner: number | null;
  turns: number;
  featured?: boolean;
}

export interface BotRow {
  bot: string;
  games: number;
  wins: number;
  win_rate: number;
  ci_low: number;
  ci_high: number;
  avg_vp: number;
  settlements?: number;
  cities?: number;
  roads?: number;
  dev_bought?: number;
  knights?: number;
  trades?: number;
  produced?: number;
  lr_in_wins?: number | null;
  la_in_wins?: number | null;
  turns_in_wins?: number | null;
}

export interface SweepPoint {
  value: number;
  win_rate: number;
  ci_low: number;
  ci_high: number;
}

export interface Sweep {
  preset: string;
  param: string;
  label: string;
  default: number;
  points: SweepPoint[];
}

export interface TuningResult {
  preset: string;
  start_win_rate: number;
  end_win_rate: number;
  changes: { param: string; from: number; to: number }[];
}

export interface CurvePoint {
  update: number;
  decisions: number;
  games: number;
  stage: string;
  win_rate: number; // training win rate (rolling)
  eval?: Record<string, { win_rate: number; ci_low: number; ci_high: number }>;
}

export interface EvalRow {
  table: string[];
  games: number;
  win_rate: number;
  ci_low: number;
  ci_high: number;
  avg_vp: number;
  /** Win rate of the best strategy bot when it takes the learner's seat at the same table. */
  baseline?: { bot: string; win_rate: number; ci_low: number; ci_high: number };
  /** Win rates of the three strategy bots in the same games. */
  opponents?: { bot: string; win_rate: number; ci_low: number; ci_high: number }[];
}

export interface StyleRow {
  bot: string;
  [metric: string]: string | number;
}

export interface Summary {
  generated_at: string;
  engine: { random_games_per_sec: number; bot_games_per_sec: number; threads: number };
  totals: { games_simulated: number };
  tournament: {
    games: number;
    bots: BotRow[];
    seat_win_rate: number[];
    ratings?: { bot: string; mu: number; sigma: number; ordinal: number }[];
    resource_mix?: { bot: string; wood: number; brick: number; sheep: number; wheat: number; ore: number }[];
  };
  tuning?: TuningResult[];
  sweeps?: Sweep[];
  training?: { curve: CurvePoint[]; decisions: number; games: number; hours: number; eval_tables: string[][] };
  evaluation?: EvalRow[];
  style?: { metrics: { key: string; label: string; format?: "pct" | "num" }[]; rows: StyleRow[] };
}

import type { Action, Replay } from "./types";

export const NONE = 255;

/** Board pieces after a given number of frames. */
export interface Pieces {
  nodeOwner: Uint8Array; // 255 = empty
  nodeCity: Uint8Array; // 0/1
  edgeOwner: Uint8Array; // 255 = empty
}

/**
 * Rebuild the pieces on the board for every point of the replay.
 * `snapshots[k]` is the board after the first `k` frames (k = 0: empty board).
 */
export function buildSnapshots(replay: Replay): Pieces[] {
  const nodes = replay.topology.node_xy.length;
  const edges = replay.topology.edge_nodes.length;
  let cur: Pieces = {
    nodeOwner: new Uint8Array(nodes).fill(NONE),
    nodeCity: new Uint8Array(nodes),
    edgeOwner: new Uint8Array(edges).fill(NONE),
  };
  const out: Pieces[] = [cur];
  for (const f of replay.frames) {
    const a = f.action;
    if (a.type === "BuildSettlement" || a.type === "BuildCity" || a.type === "BuildRoad") {
      cur = {
        nodeOwner: cur.nodeOwner.slice(),
        nodeCity: cur.nodeCity.slice(),
        edgeOwner: cur.edgeOwner.slice(),
      };
      if (a.type === "BuildSettlement") cur.nodeOwner[a.args] = f.actor;
      if (a.type === "BuildCity") cur.nodeCity[a.args] = 1;
      if (a.type === "BuildRoad") cur.edgeOwner[a.args] = f.actor;
    }
    out.push(cur);
  }
  return out;
}

/** Index of the first frame of each player-turn (for "jump to turn"). */
export function turnStarts(replay: Replay): number[] {
  const starts: number[] = [];
  let last = -1;
  replay.frames.forEach((f, i) => {
    if (f.turn !== last) {
      starts.push(i);
      last = f.turn;
    }
  });
  return starts;
}

/** What to highlight on the board for an action. */
export function actionTarget(a: Action): { node?: number; edge?: number; tile?: number } {
  switch (a.type) {
    case "BuildSettlement":
    case "BuildCity":
      return { node: a.args };
    case "BuildRoad":
      return { edge: a.args };
    case "MoveRobber":
      return { tile: a.args };
    default:
      return {};
  }
}

export function isMilestone(a: Action): boolean {
  return a.type === "BuildCity" || a.type === "BuildSettlement" || a.type === "PlayKnight";
}

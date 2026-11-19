//! Board topology (fixed for the standard 19-hex board) and per-game layouts.
//!
//! Indices:
//! - tiles 0..19, row-major from the top row (3-4-5-4-3)
//! - nodes 0..54 (intersections), sorted top-to-bottom, left-to-right
//! - edges 0..72 (paths between nodes), sorted by their node pair
//! - port slots 0..9, clockwise around the coast
//!
//! Coordinates use pointy-top hexes with y pointing down (screen space), so
//! the same numbers can be used directly by the SVG replay viewer.

use std::sync::OnceLock;

use serde::Serialize;

use crate::rng::Rng;

pub const NUM_TILES: usize = 19;
pub const NUM_NODES: usize = 54;
pub const NUM_EDGES: usize = 72;
pub const NUM_PORTS: usize = 9;
pub const NUM_RESOURCES: usize = 5;
/// Sentinel for "no tile / node / edge / player".
pub const NONE: u8 = u8::MAX;

pub const WOOD: u8 = 0;
pub const BRICK: u8 = 1;
pub const SHEEP: u8 = 2;
pub const WHEAT: u8 = 3;
pub const ORE: u8 = 4;
pub const DESERT: u8 = 5;
/// Port type for a 3:1 harbour. Port types 0..5 are 2:1 harbours of that resource.
pub const PORT_GENERIC: u8 = 5;

pub const RESOURCE_NAMES: [&str; 6] = ["wood", "brick", "sheep", "wheat", "ore", "desert"];

/// Number of dice combinations (out of 36) that roll `number`.
#[inline]
pub const fn pips(number: u8) -> u8 {
    match number {
        2 | 12 => 1,
        3 | 11 => 2,
        4 | 10 => 3,
        5 | 9 => 4,
        6 | 8 => 5,
        _ => 0,
    }
}

#[derive(Debug, Serialize)]
pub struct Topology {
    /// Axial (q, r) coordinate of each tile.
    pub tile_coords: [(i8, i8); NUM_TILES],
    pub tile_xy: [(f32, f32); NUM_TILES],
    #[serde(with = "serde_arrays")]
    pub node_xy: [(f32, f32); NUM_NODES],
    /// The six corner nodes of each tile, clockwise from the upper-right corner.
    pub tile_nodes: [[u8; 6]; NUM_TILES],
    /// Neighbouring tiles (NONE-padded).
    pub tile_neighbors: [[u8; 6]; NUM_TILES],
    /// Tiles touching each node (NONE-padded, 1..=3 entries).
    #[serde(with = "serde_arrays")]
    pub node_tiles: [[u8; 3]; NUM_NODES],
    /// Nodes one edge away (NONE-padded, 2..=3 entries).
    #[serde(with = "serde_arrays")]
    pub node_neighbors: [[u8; 3]; NUM_NODES],
    /// Edges touching each node, aligned with `node_neighbors`.
    #[serde(with = "serde_arrays")]
    pub node_edges: [[u8; 3]; NUM_NODES],
    #[serde(with = "serde_arrays")]
    pub edge_nodes: [[u8; 2]; NUM_EDGES],
    /// Coastal edge hosting each port slot.
    pub port_edges: [u8; NUM_PORTS],
}

static TOPOLOGY: OnceLock<Topology> = OnceLock::new();

/// The shared board topology (built once on first use).
#[inline]
pub fn topology() -> &'static Topology {
    TOPOLOGY.get_or_init(Topology::build)
}

impl Topology {
    fn build() -> Topology {
        let sqrt3 = 3f64.sqrt();

        let mut coords: Vec<(i8, i8)> = Vec::with_capacity(NUM_TILES);
        for r in -2i8..=2 {
            for q in -2i8..=2 {
                if (q + r).abs() <= 2 {
                    coords.push((q, r));
                }
            }
        }
        assert_eq!(coords.len(), NUM_TILES);
        let centers: Vec<(f64, f64)> = coords
            .iter()
            .map(|&(q, r)| (sqrt3 * (q as f64 + r as f64 / 2.0), 1.5 * r as f64))
            .collect();

        // Corner i of a pointy-top hex sits at angle 60*i - 30 degrees
        // (i = 0 is the upper-right corner, then clockwise in screen space).
        let key = |x: f64, y: f64| ((x * 1000.0).round() as i64, (y * 1000.0).round() as i64);
        let mut corner_keys = [[(0i64, 0i64); 6]; NUM_TILES];
        let mut node_keys: Vec<(i64, i64)> = Vec::new();
        for (t, &(cx, cy)) in centers.iter().enumerate() {
            for (i, slot) in corner_keys[t].iter_mut().enumerate() {
                let ang = (60.0 * i as f64 - 30.0).to_radians();
                let k = key(cx + ang.cos(), cy + ang.sin());
                *slot = k;
                if !node_keys.contains(&k) {
                    node_keys.push(k);
                }
            }
        }
        node_keys.sort_by_key(|&(x, y)| (y, x));
        assert_eq!(node_keys.len(), NUM_NODES);
        let node_of = |k: (i64, i64)| node_keys.iter().position(|&n| n == k).unwrap() as u8;

        let mut tile_nodes = [[0u8; 6]; NUM_TILES];
        for t in 0..NUM_TILES {
            for i in 0..6 {
                tile_nodes[t][i] = node_of(corner_keys[t][i]);
            }
        }

        let mut edges: Vec<(u8, u8)> = Vec::new();
        for nodes in &tile_nodes {
            for i in 0..6 {
                let (a, b) = (nodes[i], nodes[(i + 1) % 6]);
                let e = (a.min(b), a.max(b));
                if !edges.contains(&e) {
                    edges.push(e);
                }
            }
        }
        edges.sort();
        assert_eq!(edges.len(), NUM_EDGES);

        let mut edge_nodes = [[0u8; 2]; NUM_EDGES];
        let mut node_neighbors = [[NONE; 3]; NUM_NODES];
        let mut node_edges = [[NONE; 3]; NUM_NODES];
        for (e, &(a, b)) in edges.iter().enumerate() {
            edge_nodes[e] = [a, b];
            for (n, other) in [(a, b), (b, a)] {
                let slot = node_neighbors[n as usize]
                    .iter()
                    .position(|&x| x == NONE)
                    .unwrap();
                node_neighbors[n as usize][slot] = other;
                node_edges[n as usize][slot] = e as u8;
            }
        }

        let mut node_tiles = [[NONE; 3]; NUM_NODES];
        for (t, nodes) in tile_nodes.iter().enumerate() {
            for &n in nodes {
                let slot = node_tiles[n as usize]
                    .iter()
                    .position(|&x| x == NONE)
                    .unwrap();
                node_tiles[n as usize][slot] = t as u8;
            }
        }

        let mut tile_neighbors = [[NONE; 6]; NUM_TILES];
        let dirs = [(1i8, 0i8), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)];
        for (t, &(q, r)) in coords.iter().enumerate() {
            let mut k = 0;
            for (dq, dr) in dirs {
                if let Some(o) = coords.iter().position(|&c| c == (q + dq, r + dr)) {
                    tile_neighbors[t][k] = o as u8;
                    k += 1;
                }
            }
        }

        let mut node_xy = [(0f32, 0f32); NUM_NODES];
        for (n, &(x, y)) in node_keys.iter().enumerate() {
            node_xy[n] = (x as f32 / 1000.0, y as f32 / 1000.0);
        }
        let mut tile_xy = [(0f32, 0f32); NUM_TILES];
        for (t, &(x, y)) in centers.iter().enumerate() {
            tile_xy[t] = (x as f32, y as f32);
        }

        // Coastal edges belong to exactly one tile. Order them clockwise by the
        // angle of their midpoint, then place 9 harbours with gaps of 2,2,3
        // coastal edges (the standard frame pattern; no two harbours share a node).
        let mut coastal: Vec<(f64, u8)> = Vec::new();
        for (e, &[a, b]) in edge_nodes.iter().enumerate() {
            let tiles_on_edge = tile_nodes
                .iter()
                .filter(|nodes| nodes.contains(&a) && nodes.contains(&b))
                .count();
            if tiles_on_edge == 1 {
                let (ax, ay) = node_xy[a as usize];
                let (bx, by) = node_xy[b as usize];
                let ang = ((ay + by) as f64 / 2.0).atan2((ax + bx) as f64 / 2.0);
                coastal.push((ang, e as u8));
            }
        }
        assert_eq!(coastal.len(), 30);
        coastal.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
        let slots = [0usize, 3, 6, 10, 13, 16, 20, 23, 26];
        let mut port_edges = [0u8; NUM_PORTS];
        for (i, &s) in slots.iter().enumerate() {
            port_edges[i] = coastal[s].1;
        }

        let mut tile_coords = [(0i8, 0i8); NUM_TILES];
        tile_coords.copy_from_slice(&coords);

        Topology {
            tile_coords,
            tile_xy,
            node_xy,
            tile_nodes,
            tile_neighbors,
            node_tiles,
            node_neighbors,
            node_edges,
            edge_nodes,
            port_edges,
        }
    }

    /// The node at the other end of `edge` from `node`.
    #[inline]
    pub fn other_end(&self, edge: u8, node: u8) -> u8 {
        let [a, b] = self.edge_nodes[edge as usize];
        if a == node { b } else { a }
    }
}

/// A per-game board layout: which resource and number each tile has, and
/// which harbour each port slot has.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Board {
    /// Resource of each tile (`DESERT` for the desert).
    #[serde(with = "serde_arrays")]
    pub tile_resource: [u8; NUM_TILES],
    /// Number token of each tile (0 for the desert).
    #[serde(with = "serde_arrays")]
    pub tile_number: [u8; NUM_TILES],
    /// Harbour type per port slot: 0..5 = 2:1 of that resource, `PORT_GENERIC` = 3:1.
    pub port_type: [u8; NUM_PORTS],
    /// Harbour type at each node, or `NONE`.
    #[serde(with = "serde_arrays")]
    pub node_port: [u8; NUM_NODES],
}

/// serde only derives arrays up to 32 elements; serialize larger ones as slices.
pub(crate) mod serde_arrays {
    use serde::Serializer;
    pub fn serialize<S: Serializer, T: serde::Serialize, const N: usize>(
        arr: &[T; N],
        s: S,
    ) -> Result<S::Ok, S::Error> {
        s.collect_seq(arr.iter())
    }
}

const TILE_DECK: [u8; NUM_TILES] = [
    WOOD, WOOD, WOOD, WOOD, BRICK, BRICK, BRICK, SHEEP, SHEEP, SHEEP, SHEEP, WHEAT, WHEAT, WHEAT,
    WHEAT, ORE, ORE, ORE, DESERT,
];
const NUMBER_TOKENS: [u8; 18] = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12];
const PORT_DECK: [u8; NUM_PORTS] = [
    WOOD,
    BRICK,
    SHEEP,
    WHEAT,
    ORE,
    PORT_GENERIC,
    PORT_GENERIC,
    PORT_GENERIC,
    PORT_GENERIC,
];

impl Board {
    /// A random layout: shuffled tiles, shuffled number tokens with no two
    /// red numbers (6/8) on adjacent tiles, and shuffled harbours.
    pub fn random(rng: &mut Rng) -> Board {
        let topo = topology();
        let mut tile_resource = TILE_DECK;
        rng.shuffle(&mut tile_resource);

        let mut numbers = NUMBER_TOKENS;
        let mut tile_number = [0u8; NUM_TILES];
        loop {
            rng.shuffle(&mut numbers);
            let mut k = 0;
            for t in 0..NUM_TILES {
                if tile_resource[t] == DESERT {
                    tile_number[t] = 0;
                } else {
                    tile_number[t] = numbers[k];
                    k += 1;
                }
            }
            let red = |n: u8| n == 6 || n == 8;
            let ok = (0..NUM_TILES).all(|t| {
                !red(tile_number[t])
                    || topo.tile_neighbors[t]
                        .iter()
                        .all(|&o| o == NONE || !red(tile_number[o as usize]))
            });
            if ok {
                break;
            }
        }

        let mut port_type = PORT_DECK;
        rng.shuffle(&mut port_type);
        let mut node_port = [NONE; NUM_NODES];
        for (slot, &edge) in topo.port_edges.iter().enumerate() {
            for &n in &topo.edge_nodes[edge as usize] {
                node_port[n as usize] = port_type[slot];
            }
        }

        Board {
            tile_resource,
            tile_number,
            port_type,
            node_port,
        }
    }

    pub fn desert(&self) -> u8 {
        self.tile_resource
            .iter()
            .position(|&r| r == DESERT)
            .unwrap() as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_counts() {
        let t = topology();
        // Every tile has 6 distinct corners.
        for nodes in &t.tile_nodes {
            let mut v = nodes.to_vec();
            v.sort();
            v.dedup();
            assert_eq!(v.len(), 6);
        }
        // Node degree is 2 or 3; edges are consistent with neighbours.
        let mut deg2 = 0;
        for n in 0..NUM_NODES {
            let deg = t.node_neighbors[n].iter().filter(|&&x| x != NONE).count();
            assert!(deg == 2 || deg == 3);
            if deg == 2 {
                deg2 += 1;
            }
            for k in 0..deg {
                let e = t.node_edges[n][k];
                assert_eq!(t.other_end(e, n as u8), t.node_neighbors[n][k]);
            }
            let tiles = t.node_tiles[n].iter().filter(|&&x| x != NONE).count();
            assert!((1..=3).contains(&tiles));
        }
        // 54 nodes, 72 edges: sum of degrees = 2 * edges.
        let deg_sum: usize = (0..NUM_NODES)
            .map(|n| t.node_neighbors[n].iter().filter(|&&x| x != NONE).count())
            .sum();
        assert_eq!(deg_sum, 2 * NUM_EDGES);
        assert_eq!(deg2, 18);
        // Rows of 3-4-5-4-3 tiles.
        let rows: Vec<usize> = (-2..=2)
            .map(|r| t.tile_coords.iter().filter(|c| c.1 == r).count())
            .collect();
        assert_eq!(rows, vec![3, 4, 5, 4, 3]);
        // Harbours never share a node.
        let mut port_nodes: Vec<u8> = t
            .port_edges
            .iter()
            .flat_map(|&e| t.edge_nodes[e as usize])
            .collect();
        port_nodes.sort();
        port_nodes.dedup();
        assert_eq!(port_nodes.len(), 18);
    }

    #[test]
    fn random_board_is_valid() {
        let topo = topology();
        for seed in 0..500 {
            let b = Board::random(&mut Rng::new(seed));
            let mut counts = [0; 6];
            for &r in &b.tile_resource {
                counts[r as usize] += 1;
            }
            assert_eq!(counts, [4, 3, 4, 4, 3, 1]);
            let d = b.desert() as usize;
            assert_eq!(b.tile_number[d], 0);
            for t in 0..NUM_TILES {
                let n = b.tile_number[t];
                if n == 6 || n == 8 {
                    for &o in &topo.tile_neighbors[t] {
                        if o != NONE {
                            let m = b.tile_number[o as usize];
                            assert!(m != 6 && m != 8);
                        }
                    }
                }
            }
            assert_eq!(b.node_port.iter().filter(|&&p| p != NONE).count(), 18);
        }
    }
}

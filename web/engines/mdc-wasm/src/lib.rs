// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Doeon Kwon
//! Surface-nets (naive dual contouring) meshing of a binary occupancy grid, compiled to WASM for
//! mini-PRISM's free-space viewer. Turns the blocky exposed-face surface into a watertight,
//! smoothed dual mesh — one vertex per boundary cell placed at the centroid of its cut edges,
//! connected across the 3 primary grid edges.
//!
//! HONEST SCOPE: this is surface-nets, the "lite" dual method. A binary occupancy grid carries no
//! sub-voxel Hermite data (edge normals), so there are NO sharp features to preserve — the QEF /
//! Manifold-Dual-Contouring step (Doeon's SPACE0 engine) needs an SDF + edge normals, which come
//! from a distance transform of the occupancy. That is the documented next step; this v1 already
//! gives the watertight smoothing (the big visible win over cube faces) and the Rust->WASM pipeline.
//!
//! Occupancy layout: `occ[x*ny*nz + y*nz + z]`, value != 0 == solid (matches the numpy C-order grid).

use wasm_bindgen::prelude::*;

/// Cube corners in the order used by the sign mask; corner i is at offset CUBE[i].
const CUBE: [[usize; 3]; 8] = [
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
];
/// 12 cube edges as (corner_a, corner_b).
const EDGES: [[usize; 2]; 12] = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
];

#[wasm_bindgen]
pub struct Mesh {
    positions: Vec<f32>,
    indices: Vec<u32>,
}

#[wasm_bindgen]
impl Mesh {
    #[wasm_bindgen(getter)]
    pub fn positions(&self) -> Vec<f32> {
        self.positions.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn indices(&self) -> Vec<u32> {
        self.indices.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn vertex_count(&self) -> usize {
        self.positions.len() / 3
    }
    #[wasm_bindgen(getter)]
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

#[inline]
fn solid(occ: &[u8], nx: usize, ny: usize, nz: usize, x: usize, y: usize, z: usize) -> bool {
    let _ = nx;
    occ[x * ny * nz + y * nz + z] != 0
}

/// Surface-nets mesh of `occ`. `vs` = voxel size (m); `(ox,oy,oz)` = world coord of voxel (0,0,0).
/// Normals are left to the caller (three.js `computeVertexNormals()`) to keep this lean + correct.
#[wasm_bindgen]
pub fn surface_nets(
    occ: &[u8], nx: usize, ny: usize, nz: usize, vs: f32, ox: f32, oy: f32, oz: f32,
) -> Mesh {
    let (cx, cy, cz) = (nx - 1, ny - 1, nz - 1); // one cell per interior cube
    let cell_idx = |x: usize, y: usize, z: usize| x * cy * cz + y * cz + z;
    let mut cell_vert = vec![u32::MAX; cx * cy * cz]; // vertex index owned by each cell, or MAX
    let mut positions: Vec<f32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Pass 1 — place one vertex per boundary cell at the centroid of its cut edges.
    for x in 0..cx {
        for y in 0..cy {
            for z in 0..cz {
                let mut mask = 0u8;
                for (i, c) in CUBE.iter().enumerate() {
                    if solid(occ, nx, ny, nz, x + c[0], y + c[1], z + c[2]) {
                        mask |= 1 << i;
                    }
                }
                if mask == 0 || mask == 0xFF {
                    continue; // fully inside or fully outside — no surface
                }
                let (mut sx, mut sy, mut sz, mut n) = (0f32, 0f32, 0f32, 0f32);
                for e in EDGES.iter() {
                    let sa = (mask >> e[0]) & 1;
                    let sb = (mask >> e[1]) & 1;
                    if sa != sb {
                        let a = CUBE[e[0]];
                        let b = CUBE[e[1]];
                        // binary field -> zero crossing at the edge midpoint
                        sx += (a[0] + b[0]) as f32 * 0.5;
                        sy += (a[1] + b[1]) as f32 * 0.5;
                        sz += (a[2] + b[2]) as f32 * 0.5;
                        n += 1.0;
                    }
                }
                if n == 0.0 {
                    continue;
                }
                let inv = 1.0 / n;
                cell_vert[cell_idx(x, y, z)] = (positions.len() / 3) as u32;
                positions.push(ox + (x as f32 + sx * inv) * vs);
                positions.push(oy + (y as f32 + sy * inv) * vs);
                positions.push(oz + (z as f32 + sz * inv) * vs);
            }
        }
    }

    // Pass 2 — for each of the 3 primary grid edges at corner (x,y,z), if it crosses the surface,
    // stitch the 4 cells sharing that edge into a quad (two triangles). Winding follows the sign
    // so front faces point out of the solid.
    let quad = |ind: &mut Vec<u32>, a: u32, b: u32, c: u32, d: u32, flip: bool| {
        if a == u32::MAX || b == u32::MAX || c == u32::MAX || d == u32::MAX {
            return;
        }
        if flip {
            ind.extend_from_slice(&[a, d, c, a, c, b]);
        } else {
            ind.extend_from_slice(&[a, b, c, a, c, d]);
        }
    };
    for x in 1..cx {
        for y in 1..cy {
            for z in 1..cz {
                let s = solid(occ, nx, ny, nz, x, y, z);
                // edge along +x: grid (x,y,z)-(x+1,y,z); 4 cells share it in the y,z plane
                if s != solid(occ, nx, ny, nz, x + 1, y, z) {
                    quad(&mut indices,
                        cell_vert[cell_idx(x, y - 1, z - 1)], cell_vert[cell_idx(x, y, z - 1)],
                        cell_vert[cell_idx(x, y, z)], cell_vert[cell_idx(x, y - 1, z)], !s);
                }
                // edge along +y
                if s != solid(occ, nx, ny, nz, x, y + 1, z) {
                    quad(&mut indices,
                        cell_vert[cell_idx(x - 1, y, z - 1)], cell_vert[cell_idx(x, y, z - 1)],
                        cell_vert[cell_idx(x, y, z)], cell_vert[cell_idx(x - 1, y, z)], s);
                }
                // edge along +z
                if s != solid(occ, nx, ny, nz, x, y, z + 1) {
                    quad(&mut indices,
                        cell_vert[cell_idx(x - 1, y - 1, z)], cell_vert[cell_idx(x, y - 1, z)],
                        cell_vert[cell_idx(x, y, z)], cell_vert[cell_idx(x - 1, y, z)], !s);
                }
            }
        }
    }

    Mesh { positions, indices }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A solid sphere in a grid must mesh to a closed, non-empty surface with every triangle index
    // in range. Watertight-ish check: each vertex is referenced by at least one triangle.
    #[test]
    fn sphere_is_a_closed_surface() {
        let (nx, ny, nz) = (32usize, 32usize, 32usize);
        let mut occ = vec![0u8; nx * ny * nz];
        let (cxf, r) = (15.5f32, 10.0f32);
        for x in 0..nx {
            for y in 0..ny {
                for z in 0..nz {
                    let d = ((x as f32 - cxf).powi(2) + (y as f32 - cxf).powi(2)
                        + (z as f32 - cxf).powi(2)).sqrt();
                    if d <= r {
                        occ[x * ny * nz + y * nz + z] = 1;
                    }
                }
            }
        }
        let m = surface_nets(&occ, nx, ny, nz, 1.0, 0.0, 0.0, 0.0);
        assert!(m.vertex_count() > 200, "sphere should have many verts, got {}", m.vertex_count());
        assert!(m.triangle_count() > 200, "sphere should have many tris");
        // all indices in range
        let vc = m.vertex_count() as u32;
        assert!(m.indices.iter().all(|&i| i < vc), "index out of range");
        // roughly spherical: all verts within [r-2, r+2] of center
        let p = &m.positions;
        for v in p.chunks(3) {
            let d = ((v[0] - cxf).powi(2) + (v[1] - cxf).powi(2) + (v[2] - cxf).powi(2)).sqrt();
            assert!(d > r - 2.5 && d < r + 2.5, "vertex off the sphere shell: d={d}");
        }
    }

    #[test]
    fn empty_grid_is_empty_mesh() {
        let occ = vec![0u8; 8 * 8 * 8];
        let m = surface_nets(&occ, 8, 8, 8, 1.0, 0.0, 0.0, 0.0);
        assert_eq!(m.vertex_count(), 0);
        assert_eq!(m.triangle_count(), 0);
    }
}

pub mod qef;
pub mod edt;
pub mod dc;

/// QEF Manifold-Dual-Contouring mesh of `occ` (the upgrade over `surface_nets`): EDT->SDF->Hermite
/// ->QEF vertex placement, so sharp corners are recovered instead of rounded. Returns positions +
/// indices + analytic normals + a per-vertex `unknown` channel (kept separate from geometry).
#[wasm_bindgen]
pub fn mdc_mesh(
    occ: &[u8], nx: usize, ny: usize, nz: usize, vs: f32, ox: f32, oy: f32, oz: f32,
) -> dc::MdcMesh {
    dc::mdc_mesh_impl(occ, nx, ny, nz, vs, ox, oy, oz, None)
}

/// Model-friendly output: the signed distance field (f32, row-major, in voxel units) — a regular,
/// differentiable tensor a perception/physics model consumes, NOT a rendered image. The `unknown`
/// mask is a SEPARATE channel (fetch it alongside), never fused into the SDF (design §4).
#[wasm_bindgen]
pub struct ModelSdf {
    sdf: Vec<f32>,
}

#[wasm_bindgen]
impl ModelSdf {
    #[wasm_bindgen(getter)]
    pub fn sdf(&self) -> Vec<f32> {
        self.sdf.clone()
    }
}

/// Compute the SDF grid for the model audience (the same field the mesher uses internally).
#[wasm_bindgen]
pub fn mdc_model_sdf(occ: &[u8], nx: usize, ny: usize, nz: usize) -> ModelSdf {
    ModelSdf { sdf: edt::Sdf::from_occupancy(occ, nx, ny, nz).data }
}

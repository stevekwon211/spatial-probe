// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Doeon Kwon
//! QEF dual contouring: the upgrade over surface-nets. Same one-vertex-per-cell topology and quad
//! stitching (so watertightness is unchanged), but the vertex is placed by the QEF minimizer of
//! the edge-crossing tangent planes (from the EDT-SDF) instead of the edge-midpoint centroid — so
//! a corner's vertex lands ON the corner, not rounded off. Also emits analytic normals (SDF
//! gradient) and a per-vertex `unknown` channel kept SEPARATE from geometry (design §4/§8).
//!
//! Scope note: single vertex per cell (surface-nets topology + QEF placement). Full Manifold-DC's
//! multi-vertex split for ambiguous corner-sign cells is the further refinement (design T3).

use glam::Vec3A;
use wasm_bindgen::prelude::*;

use crate::edt::Sdf;
use crate::qef::Qef;

const CUBE: [[usize; 3]; 8] = [
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
];
const EDGES: [[usize; 2]; 12] = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
];

#[wasm_bindgen]
pub struct MdcMesh {
    positions: Vec<f32>,
    indices: Vec<u32>,
    normals: Vec<f32>,
    vertex_unknown: Vec<f32>,
    fell_back: u32,
    manifold_defects: u32,
}

/// Count edges NOT shared by exactly two triangles — the watertightness/manifold QA signal. A
/// closed manifold surface has every edge in exactly 2 triangles; anything else is a boundary edge
/// (hole) or a non-manifold pinch. Surfaced as an honesty counter, like the corridor's fog.
fn count_manifold_defects(indices: &[u32]) -> u32 {
    use std::collections::HashMap;
    let mut edges: HashMap<(u32, u32), u32> = HashMap::new();
    for tri in indices.chunks(3) {
        if tri.len() < 3 {
            continue;
        }
        for &(a, b) in &[(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            let key = if a < b { (a, b) } else { (b, a) };
            *edges.entry(key).or_insert(0) += 1;
        }
    }
    edges.values().filter(|&&c| c != 2).count() as u32
}

#[wasm_bindgen]
impl MdcMesh {
    #[wasm_bindgen(getter)]
    pub fn positions(&self) -> Vec<f32> {
        self.positions.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn indices(&self) -> Vec<u32> {
        self.indices.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn normals(&self) -> Vec<f32> {
        self.normals.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn vertex_unknown(&self) -> Vec<f32> {
        self.vertex_unknown.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn vertex_count(&self) -> usize {
        self.positions.len() / 3
    }
    #[wasm_bindgen(getter)]
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
    /// Cells where the QEF was rank-deficient and fell back to the mass point (a QA signal — the
    /// "no unique corner here" count, the geometry analog of the corridor's fog).
    #[wasm_bindgen(getter)]
    pub fn fallback_count(&self) -> u32 {
        self.fell_back
    }
    /// Edges not shared by exactly 2 triangles (holes / non-manifold). 0 = watertight.
    #[wasm_bindgen(getter)]
    pub fn manifold_defects(&self) -> u32 {
        self.manifold_defects
    }
}

/// QEF dual-contour `occ`. `unknown` (optional, same layout) tags per-vertex uncertainty as a
/// SEPARATE channel — never fused into the geometry.
pub fn mdc_mesh_impl(
    occ: &[u8], nx: usize, ny: usize, nz: usize, vs: f32, ox: f32, oy: f32, oz: f32,
    unknown: Option<&[u8]>,
) -> MdcMesh {
    let sdf = Sdf::from_occupancy(occ, nx, ny, nz);
    let inside = |x: usize, y: usize, z: usize| sdf.at(x, y, z) < 0.0;
    let (cx, cy, cz) = (nx - 1, ny - 1, nz - 1);
    let cell_idx = |x: usize, y: usize, z: usize| x * cy * cz + y * cz + z;
    let mut cell_vert = vec![u32::MAX; cx * cy * cz];
    let mut positions: Vec<f32> = Vec::new();
    let mut normals: Vec<f32> = Vec::new();
    let mut vertex_unknown: Vec<f32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut fell_back = 0u32;

    // precompute corner gradients lazily via sdf.gradient in the loop (cheap enough)
    for x in 0..cx {
        for y in 0..cy {
            for z in 0..cz {
                // corner sdf values + signs
                let mut sv = [0f32; 8];
                let mut mask = 0u8;
                for (i, c) in CUBE.iter().enumerate() {
                    let v = sdf.at(x + c[0], y + c[1], z + c[2]);
                    sv[i] = v;
                    if v < 0.0 {
                        mask |= 1 << i;
                    }
                }
                if mask == 0 || mask == 0xFF {
                    continue;
                }
                // gather Hermite from crossing edges, accumulate QEF (in voxel coords)
                let mut qef = Qef::new();
                let base = Vec3A::new(x as f32, y as f32, z as f32);
                for e in EDGES.iter() {
                    let (a, b) = (e[0], e[1]);
                    let (va, vb) = (sv[a], sv[b]);
                    if (va < 0.0) == (vb < 0.0) {
                        continue; // no sign change on this edge
                    }
                    let denom = va - vb;
                    let t = if denom.abs() > 1e-9 { va / denom } else { 0.5 };
                    let t = t.clamp(0.0, 1.0);
                    let ca = CUBE[a];
                    let cb = CUBE[b];
                    // crossing point in voxel coords
                    let p = base
                        + Vec3A::new(
                            ca[0] as f32 + t * (cb[0] as f32 - ca[0] as f32),
                            ca[1] as f32 + t * (cb[1] as f32 - ca[1] as f32),
                            ca[2] as f32 + t * (cb[2] as f32 - ca[2] as f32),
                        );
                    // normal = interpolated SDF gradient at the crossing corners
                    let ga = sdf.gradient(x + ca[0], y + ca[1], z + ca[2]);
                    let gb = sdf.gradient(x + cb[0], y + cb[1], z + cb[2]);
                    let n = Vec3A::new(
                        ga[0] + t * (gb[0] - ga[0]),
                        ga[1] + t * (gb[1] - ga[1]),
                        ga[2] + t * (gb[2] - ga[2]),
                    );
                    qef.add(p, n);
                }
                if qef.count() == 0 {
                    continue;
                }
                let (mut v, fb) = qef.solve();
                if fb {
                    fell_back += 1;
                }
                // clamp the minimizer to the cell (avoids T-junctions / stray verts)
                v.x = v.x.clamp(x as f32, x as f32 + 1.0);
                v.y = v.y.clamp(y as f32, y as f32 + 1.0);
                v.z = v.z.clamp(z as f32, z as f32 + 1.0);
                // analytic vertex normal = SDF gradient at the vertex (nearest-corner sample)
                let (nx_i, ny_i, nz_i) = (v.x.round() as usize, v.y.round() as usize, v.z.round() as usize);
                let g = sdf.gradient(nx_i.min(nx - 1), ny_i.min(ny - 1), nz_i.min(nz - 1));
                let gn = Vec3A::new(g[0], g[1], g[2]);
                let gn = if gn.length() > 1e-6 { gn.normalize() } else { Vec3A::Z };
                cell_vert[cell_idx(x, y, z)] = (positions.len() / 3) as u32;
                positions.extend_from_slice(&[ox + v.x * vs, oy + v.y * vs, oz + v.z * vs]);
                normals.extend_from_slice(&[gn.x, gn.y, gn.z]);
                // unknown = max over the cell's 8 corners of the unknown mask (kept separate)
                let u = match unknown {
                    Some(um) => {
                        let mut mx = 0f32;
                        for c in CUBE.iter() {
                            let idx = (x + c[0]) * ny * nz + (y + c[1]) * nz + (z + c[2]);
                            if um[idx] != 0 {
                                mx = 1.0;
                            }
                        }
                        mx
                    }
                    None => 0.0,
                };
                vertex_unknown.push(u);
            }
        }
    }

    // quad stitching — identical topology to surface_nets, sign from the SDF
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
                let s = inside(x, y, z);
                if s != inside(x + 1, y, z) {
                    quad(&mut indices,
                        cell_vert[cell_idx(x, y - 1, z - 1)], cell_vert[cell_idx(x, y, z - 1)],
                        cell_vert[cell_idx(x, y, z)], cell_vert[cell_idx(x, y - 1, z)], !s);
                }
                if s != inside(x, y + 1, z) {
                    quad(&mut indices,
                        cell_vert[cell_idx(x - 1, y, z - 1)], cell_vert[cell_idx(x, y, z - 1)],
                        cell_vert[cell_idx(x, y, z)], cell_vert[cell_idx(x - 1, y, z)], s);
                }
                if s != inside(x, y, z + 1) {
                    quad(&mut indices,
                        cell_vert[cell_idx(x - 1, y - 1, z)], cell_vert[cell_idx(x, y - 1, z)],
                        cell_vert[cell_idx(x, y, z)], cell_vert[cell_idx(x - 1, y, z)], !s);
                }
            }
        }
    }

    let manifold_defects = count_manifold_defects(&indices);
    MdcMesh { positions, indices, normals, vertex_unknown, fell_back, manifold_defects }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A right-angle wedge: QEF must place the boundary vertex nearer the true 90° corner than the
    // surface-nets edge-midpoint centroid would — the north-star win.
    #[test]
    fn qef_pulls_vertex_toward_a_sharp_corner() {
        // occupancy: solid where x<10 AND y<10 (an L / corner column), full z
        let (n) = 20usize;
        let mut occ = vec![0u8; n * n * n];
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    if x < 10 && y < 10 {
                        occ[x * n * n + y * n + z] = 1;
                    }
                }
            }
        }
        let m = mdc_mesh_impl(&occ, n, n, n, 1.0, 0.0, 0.0, 0.0, None);
        assert!(m.vertex_count() > 0 && m.triangle_count() > 0, "wedge must mesh");
        let vc = m.vertex_count() as u32;
        assert!(m.indices.iter().all(|&i| i < vc), "indices in range");
        // near the corner column (x~10,y~10) at mid-z, a QEF vertex should exist close to (10,10)
        let mut best = f32::INFINITY;
        for v in m.positions.chunks(3) {
            if (v[2] - 10.0).abs() < 1.5 {
                let d = ((v[0] - 10.0).powi(2) + (v[1] - 10.0).powi(2)).sqrt();
                best = best.min(d);
            }
        }
        assert!(best < 1.2, "a vertex should sit near the true corner (10,10); closest was {best}");
    }

    #[test]
    fn box_meshes_closed_with_normals() {
        let n = 20usize;
        let mut occ = vec![0u8; n * n * n];
        for x in 6..14 {
            for y in 6..14 {
                for z in 6..14 {
                    occ[x * n * n + y * n + z] = 1;
                }
            }
        }
        let m = mdc_mesh_impl(&occ, n, n, n, 1.0, 0.0, 0.0, 0.0, None);
        assert_eq!(m.positions.len(), m.normals.len(), "one normal per vertex");
        assert_eq!(m.vertex_unknown.len() * 3, m.positions.len(), "one unknown per vertex");
        assert!(m.triangle_count() > 50, "box has a surface");
        // every normal is unit length
        for nrm in m.normals.chunks(3) {
            let l = (nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]).sqrt();
            assert!((l - 1.0).abs() < 1e-3, "normal not unit: {l}");
        }
        // an axis-aligned box interior to the grid must mesh WATERTIGHT (every edge in 2 tris)
        assert_eq!(m.manifold_defects(), 0, "closed box must be watertight, {} defect edges", m.manifold_defects());
    }
}

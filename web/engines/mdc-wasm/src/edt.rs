// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Doeon Kwon
//! Occupancy -> signed distance field. A binary occupancy grid has no sub-voxel Hermite data, so
//! to run QEF dual contouring we synthesize an approximate SDF: the signed Euclidean distance to
//! the occupied/free boundary, from which we read edge crossings (sign change) and per-edge normals
//! (SDF gradient). HONEST: this is an EDT-derived approximation — it recovers corners at grid
//! resolution (the field dips near a 90° edge, so QEF pulls the vertex to the corner instead of the
//! centroid), NOT true sub-voxel CAD sharpness (design §8).
//!
//! Uses the Felzenszwalb-Huttenlocher separable 1D transform (exact squared EDT, O(n)) — standard,
//! not homegrown; the same primitive scipy's distance_transform_edt uses.

const INF: f32 = 1e20;

/// Exact 1D squared-distance transform of a sampled function `f` (Felzenszwalb-Huttenlocher).
fn edt_1d(f: &[f32], out: &mut [f32]) {
    let n = f.len();
    if n == 0 {
        return;
    }
    let mut v = vec![0usize; n];
    let mut z = vec![0f32; n + 1];
    let mut k = 0usize;
    v[0] = 0;
    z[0] = -INF;
    z[1] = INF;
    for q in 1..n {
        loop {
            let s = ((f[q] + (q * q) as f32) - (f[v[k]] + (v[k] * v[k]) as f32))
                / (2.0 * q as f32 - 2.0 * v[k] as f32);
            if s <= z[k] {
                if k == 0 {
                    v[0] = q;
                    z[0] = -INF;
                    z[1] = INF;
                    break;
                }
                k -= 1;
            } else {
                k += 1;
                v[k] = q;
                z[k] = s;
                z[k + 1] = INF;
                break;
            }
        }
    }
    let mut k = 0usize;
    for q in 0..n {
        while z[k + 1] < q as f32 {
            k += 1;
        }
        let dq = q as f32 - v[k] as f32;
        out[q] = dq * dq + f[v[k]];
    }
}

/// Squared EDT of a 3D boolean feature grid (`feature[i]` true = distance-0 seed), separable.
fn edt_3d_sq(feature: &dyn Fn(usize) -> bool, nx: usize, ny: usize, nz: usize) -> Vec<f32> {
    let idx = |x: usize, y: usize, z: usize| x * ny * nz + y * nz + z;
    let mut g = vec![0f32; nx * ny * nz];
    for i in 0..nx * ny * nz {
        g[i] = if feature(i) { 0.0 } else { INF };
    }
    // along z
    let (mut col, mut res) = (vec![0f32; nz], vec![0f32; nz]);
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                col[z] = g[idx(x, y, z)];
            }
            edt_1d(&col, &mut res);
            for z in 0..nz {
                g[idx(x, y, z)] = res[z];
            }
        }
    }
    // along y
    let (mut col, mut res) = (vec![0f32; ny], vec![0f32; ny]);
    for x in 0..nx {
        for z in 0..nz {
            for y in 0..ny {
                col[y] = g[idx(x, y, z)];
            }
            edt_1d(&col, &mut res);
            for y in 0..ny {
                g[idx(x, y, z)] = res[y];
            }
        }
    }
    // along x
    let (mut col, mut res) = (vec![0f32; nx], vec![0f32; nx]);
    for y in 0..ny {
        for z in 0..nz {
            for x in 0..nx {
                col[x] = g[idx(x, y, z)];
            }
            edt_1d(&col, &mut res);
            for x in 0..nx {
                g[idx(x, y, z)] = res[x];
            }
        }
    }
    g
}

/// Signed distance field (in voxel units) of `occ` (nonzero = solid). Negative inside the solid,
/// positive outside; the zero level is the occupied/free boundary.
pub struct Sdf {
    pub data: Vec<f32>,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
}

impl Sdf {
    pub fn from_occupancy(occ: &[u8], nx: usize, ny: usize, nz: usize) -> Sdf {
        // distance to nearest solid (valid outside) and to nearest free (valid inside)
        let out_sq = edt_3d_sq(&|i| occ[i] != 0, nx, ny, nz); // seeds = solid
        let in_sq = edt_3d_sq(&|i| occ[i] == 0, nx, ny, nz); // seeds = free
        let mut data = vec![0f32; nx * ny * nz];
        for i in 0..data.len() {
            data[i] = if occ[i] != 0 {
                -in_sq[i].max(0.0).sqrt() // inside -> negative distance to boundary
            } else {
                out_sq[i].max(0.0).sqrt() // outside -> positive
            };
        }
        Sdf { data, nx, ny, nz }
    }

    #[inline]
    pub fn at(&self, x: usize, y: usize, z: usize) -> f32 {
        self.data[x * self.ny * self.nz + y * self.nz + z]
    }

    /// Central-difference gradient at a grid corner (clamped at borders). Points from solid (−)
    /// toward free (+), i.e. the outward surface normal direction.
    pub fn gradient(&self, x: usize, y: usize, z: usize) -> [f32; 3] {
        let cl = |v: usize, n: usize| (v.min(n - 1)).max(0);
        let gx = self.at(cl(x + 1, self.nx), y, z) - self.at(x.saturating_sub(1), y, z);
        let gy = self.at(x, cl(y + 1, self.ny), z) - self.at(x, y.saturating_sub(1), z);
        let gz = self.at(x, y, cl(z + 1, self.nz)) - self.at(x, y, z.saturating_sub(1));
        [gx * 0.5, gy * 0.5, gz * 0.5]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_sdf_matches_analytic_within_a_voxel() {
        let (n, cxf, r) = (40usize, 19.5f32, 12.0f32);
        let mut occ = vec![0u8; n * n * n];
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    let d = ((x as f32 - cxf).powi(2) + (y as f32 - cxf).powi(2)
                        + (z as f32 - cxf).powi(2)).sqrt();
                    if d <= r {
                        occ[x * n * n + y * n + z] = 1;
                    }
                }
            }
        }
        let sdf = Sdf::from_occupancy(&occ, n, n, n);
        // sample many points; signed EDT should track |p-c|-r within ~1 voxel
        let mut maxerr = 0f32;
        for x in 2..n - 2 {
            for y in 2..n - 2 {
                for z in 2..n - 2 {
                    let analytic = ((x as f32 - cxf).powi(2) + (y as f32 - cxf).powi(2)
                        + (z as f32 - cxf).powi(2)).sqrt() - r;
                    let e = (sdf.at(x, y, z) - analytic).abs();
                    if analytic.abs() < 6.0 {
                        maxerr = maxerr.max(e);
                    }
                }
            }
        }
        assert!(maxerr < 1.3, "signed EDT should match analytic sphere within ~1 voxel, max err {maxerr}");
    }

    #[test]
    fn sign_is_negative_inside_positive_outside() {
        let n = 20usize;
        let mut occ = vec![0u8; n * n * n];
        for x in 6..14 {
            for y in 6..14 {
                for z in 6..14 {
                    occ[x * n * n + y * n + z] = 1;
                }
            }
        }
        let sdf = Sdf::from_occupancy(&occ, n, n, n);
        assert!(sdf.at(10, 10, 10) < 0.0, "center of block is inside (negative)");
        assert!(sdf.at(2, 2, 2) > 0.0, "corner of grid is outside (positive)");
        // gradient near the +x face points along +x (outward)
        let g = sdf.gradient(14, 10, 10);
        assert!(g[0] > 0.0 && g[0].abs() > g[1].abs() && g[0].abs() > g[2].abs(),
            "outward gradient on +x face, got {g:?}");
    }
}

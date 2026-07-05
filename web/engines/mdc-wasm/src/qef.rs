// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Doeon Kwon
//! Quadratic Error Function solver for dual contouring — the piece that makes QEF-DC beat
//! surface-nets: instead of averaging edge midpoints (a centroid that drifts on flat regions and
//! rounds off corners), it finds the point minimizing sum of squared distances to the tangent
//! PLANES at the edge crossings, so a corner's vertex is pulled to the corner.
//!
//! Recipe follows the shipped zero-volume engine (facet-1/§6 of the design):
//!   - Schaefer-Warren mass-point form: solve A^T A y = A^T (b - A·mp), x = y + mp. Biasing to the
//!     mass point regularizes the null space so a planar patch collapses to its centroid instead
//!     of shooting off — the single most important stability choice.
//!   - Symmetric 3x3 eigen-decomposition via cyclic Jacobi (4 sweeps), pseudo-inverse with a
//!     relative singular-value cutoff PINV_TOL. No atan2 (deterministic Givens).
//!   - The result is clamped to the cell by the caller.

use glam::Vec3A;

const SVD_SWEEPS: usize = 4;
const PINV_TOL: f32 = 0.1; // singular values below PINV_TOL * s_max are treated as zero

/// Accumulates plane constraints (point p on a surface with unit normal n => n·x = n·p) and solves
/// for the least-squares minimizer, biased to the mass point of the added positions.
#[derive(Clone, Default)]
pub struct Qef {
    // A^T A (symmetric 3x3, upper triangle) and A^T b, plus the running mass point.
    ata: [f32; 6], // 00,01,02,11,12,22
    atb: Vec3A,
    mass: Vec3A,
    n: u32,
}

impl Qef {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one Hermite sample: a surface point `p` with unit normal `n`.
    pub fn add(&mut self, p: Vec3A, n: Vec3A) {
        let len = n.length();
        if !len.is_finite() || len < 1e-8 {
            return; // a zero/NaN normal carries no plane constraint
        }
        let nn = n / len;
        let d = nn.dot(p);
        self.ata[0] += nn.x * nn.x;
        self.ata[1] += nn.x * nn.y;
        self.ata[2] += nn.x * nn.z;
        self.ata[3] += nn.y * nn.y;
        self.ata[4] += nn.y * nn.z;
        self.ata[5] += nn.z * nn.z;
        self.atb += nn * d;
        self.mass += p;
        self.n += 1;
    }

    pub fn count(&self) -> u32 {
        self.n
    }

    /// Returns (minimizer, fell_back_to_mass_point). `fallback` = the null space was rank-deficient
    /// (a planar patch) so the solver returned the mass point — the honest "no unique corner" case.
    pub fn solve(&self) -> (Vec3A, bool) {
        if self.n == 0 {
            return (Vec3A::ZERO, true);
        }
        let mp = self.mass / self.n as f32;
        // b' = A^T b - (A^T A) mp   (solve for the offset from the mass point)
        let atamp = self.mul_ata(mp);
        let bp = self.atb - atamp;
        let (evals, evecs) = jacobi_symmetric(&self.ata);
        // pseudo-inverse: y = sum_i (u_i·bp / lambda_i) u_i, dropping near-zero eigenvalues
        let lmax = evals.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
        let cutoff = PINV_TOL * lmax;
        let mut y = Vec3A::ZERO;
        let mut dropped = 0;
        for i in 0..3 {
            let l = evals[i];
            if l.abs() <= cutoff || l.abs() < 1e-12 {
                dropped += 1;
                continue;
            }
            let u = evecs[i];
            y += u * (u.dot(bp) / l);
        }
        let x = y + mp;
        let fallback = dropped >= 2 || !x.is_finite();
        if fallback {
            (mp, true)
        } else {
            (x, false)
        }
    }

    /// Mass point of the added samples (the caller's fallback + snap anchor).
    pub fn mass_point(&self) -> Vec3A {
        if self.n == 0 {
            Vec3A::ZERO
        } else {
            self.mass / self.n as f32
        }
    }

    fn mul_ata(&self, v: Vec3A) -> Vec3A {
        Vec3A::new(
            self.ata[0] * v.x + self.ata[1] * v.y + self.ata[2] * v.z,
            self.ata[1] * v.x + self.ata[3] * v.y + self.ata[4] * v.z,
            self.ata[2] * v.x + self.ata[4] * v.y + self.ata[5] * v.z,
        )
    }
}

/// Cyclic Jacobi eigen-decomposition of a symmetric 3x3 (upper triangle [00,01,02,11,12,22]).
/// Returns (eigenvalues, eigenvectors). Deterministic (fixed sweeps, no atan2).
fn jacobi_symmetric(sym: &[f32; 6]) -> ([f32; 3], [Vec3A; 3]) {
    let mut a = [
        [sym[0], sym[1], sym[2]],
        [sym[1], sym[3], sym[4]],
        [sym[2], sym[4], sym[5]],
    ];
    let mut v = [
        [1.0f32, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    for _ in 0..SVD_SWEEPS {
        for (p, q) in [(0usize, 1usize), (0, 2), (1, 2)] {
            let apq = a[p][q];
            if apq.abs() < 1e-12 {
                continue;
            }
            // Jacobi rotation angle via the stable half-angle form (no atan2).
            let tau = (a[q][q] - a[p][p]) / (2.0 * apq);
            let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = t * c;
            // apply rotation to A (both sides) and accumulate into V
            for k in 0..3 {
                let akp = a[k][p];
                let akq = a[k][q];
                a[k][p] = c * akp - s * akq;
                a[k][q] = s * akp + c * akq;
            }
            for k in 0..3 {
                let apk = a[p][k];
                let aqk = a[q][k];
                a[p][k] = c * apk - s * aqk;
                a[q][k] = s * apk + c * aqk;
            }
            for k in 0..3 {
                let vkp = v[k][p];
                let vkq = v[k][q];
                v[k][p] = c * vkp - s * vkq;
                v[k][q] = s * vkp + c * vkq;
            }
        }
    }
    let evals = [a[0][0], a[1][1], a[2][2]];
    let evecs = [
        Vec3A::new(v[0][0], v[1][0], v[2][0]),
        Vec3A::new(v[0][1], v[1][1], v[2][1]),
        Vec3A::new(v[0][2], v[1][2], v[2][2]),
    ];
    (evals, evecs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: Vec3A, b: Vec3A, eps: f32) -> bool {
        (a - b).length() < eps
    }

    #[test]
    fn corner_of_three_orthogonal_planes_is_the_corner() {
        // three axis planes through (2,3,4) -> unique minimizer at exactly (2,3,4)
        let mut q = Qef::new();
        q.add(Vec3A::new(2.0, 3.0, 4.0), Vec3A::X);
        q.add(Vec3A::new(2.0, 3.0, 4.0), Vec3A::Y);
        q.add(Vec3A::new(2.0, 3.0, 4.0), Vec3A::Z);
        let (x, fb) = q.solve();
        assert!(!fb, "3 orthogonal planes are well-determined");
        assert!(approx(x, Vec3A::new(2.0, 3.0, 4.0), 1e-3), "got {x:?}");
    }

    #[test]
    fn planar_patch_falls_back_to_mass_point() {
        // all samples on the same plane z=1 with normal +z -> null space rank 2 -> mass point
        let mut q = Qef::new();
        for (px, py) in [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (2.0, 2.0)] {
            q.add(Vec3A::new(px, py, 1.0), Vec3A::Z);
        }
        let (x, fb) = q.solve();
        assert!(fb, "a planar patch has no unique corner -> fallback");
        // mass point is the centroid (1,1,1); z must sit on the plane
        assert!((x.z - 1.0).abs() < 1e-3, "solution must lie on the plane, got {x:?}");
        assert!(approx(x, Vec3A::new(1.0, 1.0, 1.0), 1e-3), "got {x:?}");
    }

    #[test]
    fn edge_of_two_planes_lies_on_the_edge() {
        // two planes (x through 1, y through 2) share an edge; minimizer is on that edge (x=1,y=2)
        let mut q = Qef::new();
        q.add(Vec3A::new(1.0, 0.0, 0.0), Vec3A::X);
        q.add(Vec3A::new(0.0, 2.0, 5.0), Vec3A::Y);
        let (x, _fb) = q.solve();
        assert!((x.x - 1.0).abs() < 1e-2 && (x.y - 2.0).abs() < 1e-2, "on the shared edge, got {x:?}");
    }

    #[test]
    fn empty_is_fallback_zero() {
        let (x, fb) = Qef::new().solve();
        assert!(fb && x == Vec3A::ZERO);
    }
}

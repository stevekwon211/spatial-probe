# Path C — 3D Gaussian Splatting (the photoreal track)

Separate from the deterministic **DC-mesh** free-space pipeline (paths A/B, which give a *legible,
QA-able* surface + honest uncertainty). gsplat is the *photoreal* reconstruction: per-scene GPU
optimization producing Gaussians (not a mesh), rendered as novel views. Use it when the question is
"what did the street actually look like", not "is this cell confidently free".

## Why it's a distinct track (not a toggle on the mesher)

| | DC-mesh (A/B) | gsplat (C) |
|---|---|---|
| output | triangle surface + per-vertex camera color | 3D Gaussians (position, cov, SH color, opacity) |
| compute | client WASM, ~160 ms, deterministic | per-scene **GPU training**, minutes–hours |
| honesty | uncertainty kept as a separate channel (fog corridor) | photoreal but *opaque* — no free/fog/blocked label |
| dynamics | aggregated occupancy (static) | needs box-decomposition or it **ghosts** movers |

The mesh is for *review*; the splat is for *look*. Both read the same nuScenes scene; neither replaces
the other.

## Data we have (verified, CPU, runs now)

`gsplat_prep.py --scene scene-0061` produced:
- **234 posed images** = 39 keyframes × 6 surround cameras, each with cam→world (global frame) from
  `extrinsic ∘ ego_pose`, intrinsics (fl 1266, 1600×900) → `transforms.json` (nerfstudio/instant-ngp).
- **16 419 LiDAR/occupancy init points** (occupied voxel centers → global) → `points3d.ply`.

Ego motion over the ~20 s drive turns the 6 forward-facing cameras into real multi-view coverage —
enough to train, not enough to be dense everywhere (forward bias, thin side coverage).

## Training (the GPU step — kickable like BEVFusion was, not run here)

```
pip install gsplat                       # Nerfstudio's CUDA splatting lib
ns-install-cli / or a minimal gsplat train loop
# dataset = transforms.json (nerfstudio parser) + points3d.ply init
# 30k iters, densify+prune, render held-out keyframe -> PSNR
```
Free-GPU host: Kaggle P100/T4 (same path as `p2_fusion/kaggle`). Expect a static-street v1.

## Honest scope + kill criteria (pre-registered)

- **v1 = static only.** Moving cars/peds ghost (they violate the static-scene assumption). This is the
  *known* failure, reported not hidden — a held-out render will show ghost trails on movers.
- **Falsifiable target:** static-region PSNR should beat a nearest-training-view baseline by a real
  margin on a held-out keyframe. If it doesn't clear the baseline, the sparse forward-view coverage
  was insufficient — a negative worth reporting, not a footnote.
- **v2 (not v1) = dynamics.** StreetGaussians / OmniRe decompose the scene with the 3D boxes
  (`annotations.json` has them): static Gaussians + per-object rigid Gaussians moving on box tracks.
  That removes the ghosting. Multi-day track.
- Coverage caveat is **logged, not silent**: side/rear geometry is thin; do not claim full-scene
  photoreal from forward-biased cameras.

## Where it fits the PRISM role

Product-engineer angle: the mesh view is the *reviewer's* tool (fast, deterministic, labels
uncertainty); the splat is the *stakeholder's* tool (photoreal recall of a scene). A real spatial-data
product surfaces both from the same asset — this scaffold is the seam between them.

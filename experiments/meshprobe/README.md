# mesh probe — Occ3D occupancy → solid terrain mesh + free-space overlay

Tests Doeon's hypothesis: voxel+meshing a scene into a SOLID surface makes free-space / collision
problems pop for a human/agent reviewer, vs a sparse point cloud. VERDICT: confirmed, with a
honesty caveat baked in.

- `build_mesh.py` — loads one Occ3D-nuScenes scene, exposed-face voxel mesh (numpy, 0-dep — honest
  for voxels, no marching-cubes/DC interpolation inventing surface), + the ego corridor + nearest-
  obstacle clearance + the single-sweep confirmed points (the uncertainty overlay). Writes mesh.json.
- `index.html` — Three.js viewer: dense mesh (height-colored) + single-sweep points (cyan) + ego +
  corridor + clearance marker. Toggle mesh off → only the sparse single-sweep points remain =
  the scene's 88.4%-unknown occlusion, made visual. The mesh is legible but is aggregated-GT
  inference; a reviewer must see how little one sweep confirms.

Run:
    ../../.venv/bin/python build_mesh.py
    python3 serve.py   # http://127.0.0.1:5322  (forces JS mime for ES modules)

Upgrade path: exposed-face → Manifold Dual Contouring (SPACE0 zero-mesh: QEF vertex placement,
Uribe-Lobello manifold decomposition, sharp-feature preservation) for smooth curbs/walls. ~600-800
lines Python port; algorithm is language-agnostic (corner-signs → 12-edge crossings → QEF → quads).
Vendored three.js + mesh.json are gitignored (reproducible).

# M2 — Occ3D-nuScenes adapter contract

The data-independent core (predicates, retrieval, metrics, query DSL) is complete and tested. M2
wires it to real scenes by implementing `probe.adapters.occ3d.load_scene` so that
`experiments/occquery_v0/run.py` can swap `synthetic.SCENES` for adapter-loaded scenes **with no
other change**. This file is the setup + contract; it deliberately does NOT download anything (the
data is gated behind a free nuScenes research account + terms acceptance).

## Datasets required

- **nuScenes** (`v1.0-mini` to prototype, `v1.0-trainval` for real numbers) — ego pose + annotations.
- **Occ3D-nuScenes** voxel labels + visibility masks — the occupancy substrate.

Both need a free nuScenes account and acceptance of the nuScenes terms. Nothing is committed
(`data/` is gitignored).

## Expected directory layout (project-local, gitignored) — PROVISIONAL

```
data/
  nuscenes/
    v1.0-mini/            # or v1.0-trainval
    samples/  sweeps/  maps/
  occ3d-nuscenes/
    gts/<scene_name>/<sample_token>/labels.npz   # semantics + mask_camera + mask_lidar
    annotations.json                              # split + token index (per the Occ3D release)
```

Confirm the exact layout against the official Occ3D-nuScenes release before implementing.

## Versions / integrity

- Pin the nuScenes version and the Occ3D release commit in `data/VERSIONS.txt`.
- Record an sha256 of each `labels.npz` you depend on in `data/checksums.txt`, so a
  silently-changed label file is caught (a known reproducibility footgun).

## Occupancy mapping (the contract)

Per Occ3D frame, map each voxel's semantic label + visibility mask to the probe encoding
(`probe.raycast` constants):

| Occ3D voxel | probe encoding |
|---|---|
| not observed (visibility mask = 0) | `UNKNOWN` (-1) |
| free / empty label | `FREE` (0) |
| any non-free, non-ground semantic class | `OCCUPIED` (1) |
| ground classes (drivable surface, etc.) | left as occupancy but excluded via `OccupancyGrid.ground_height` |

Grid spec (PROVISIONAL): 0.4 m voxels, 200 x 200 x 16, range [-40, 40] x [-40, 40] x [-1.0, 5.4] m,
ego-centric. The `UNKNOWN` mapping is what makes the 3-policy unknown-sensitivity report (PLAN §4)
real on actual sensor coverage rather than synthetic.

## Output schema (must match the synthetic generator)

`load_scene(scene_token, data_root) -> probe.scene.Scene`, one `Frame` per keyframe sample:

- `Frame.grid`  — `OccupancyGrid` (occupancy mapped as above; voxel_size / origin / ground_height
  from the grid spec).
- `Frame.ego`   — `EgoPose` (position + yaw from the nuScenes ego pose; speed from consecutive
  sample timestamps; width/length/height from the ego vehicle record).
- `Frame.objects` — `tuple[TrackedBox]` from `sample_annotation` (center, size, yaw, category label,
  velocity).
- `Frame.time`  — sample timestamp in seconds.

## First real-data smoke command

```sh
pip install -e ".[data]"          # adds nuscenes-devkit
python - <<'PY'
from pathlib import Path
from probe.adapters.occ3d import load_scene
scene = load_scene("<a v1.0-mini scene token>", Path("data"))
print(len(scene), "frames")  # then: occupied/free/unknown voxel counts + a 2D slice sanity check
PY
```

Acceptance (PLAN M2): load 1 scene, sanity-check occupied / free / unknown voxel counts and a quick
2D slice render; then run `experiments/occquery_v0/run.py` with `SCENES` sourced from the adapter and
GT from hand-labeling instead of construction.

## Not yet verified (resolve at implementation time)

- The exact Occ3D grid spec (voxel size, dims, range, free-label id) and the on-disk layout above.
- **Substrate vs anchor mismatch:** the occupancy substrate here is Occ3D-**nuScenes**, but the H2
  retrieval anchor (RefAV HOTA-Temporal) is **Argoverse 2**. Decide whether to (a) build the
  denotation arm (H1/H3) on Occ3D-nuScenes and the leaderboard arm (H2) on Argoverse 2 separately, or
  (b) source occupancy for Argoverse 2 scenes. This is the single biggest open question for
  occquery's external evaluation — see `docs/benchmark-anchors.md`.

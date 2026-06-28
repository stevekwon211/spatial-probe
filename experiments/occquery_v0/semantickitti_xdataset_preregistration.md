# Pre-registration -- SemanticKITTI cross-dataset (H1 expressivity on a 3rd dataset + a
# non-degenerate, non-vacuous denotation test)

SEALED before the confirmatory verdict (the metric-vs-baseline CIs were NOT computed before this
commit). Exploration that PRECEDED this seal and is disclosed here: I read raw `.bin`/`.invalid`/
`.label` to verify the on-disk format, the grid geometry (axis order / origin), the `.bin`-vs-`.label`
occupied XOR (non-degeneracy), and the band/domain blocked-rate (non-vacuity) -- these are the
APPARATUS-validation + descriptive non-vacuity/non-degeneracy stats the design itself requires me to
report. The CONFIRMATORY quantity (predicate IoU CI.lo vs the trivial all-free baseline) was NOT
evaluated before this seal.

## Why this experiment exists (the requirement, owned)
- H1 (occupancy expressivity dominates box-only on free-space queries) is the SOLE field headline.
  It has held on AV2 (h3b) and the design is dataset-agnostic. Owner-question: does it ALSO hold on a
  THIRD, structurally-different dataset (SemanticKITTI SSC: a 256x256x32 single-scan voxel grid, not a
  multi-sweep BEV like Occ3D/AV2)? If yes, the expressivity gap is not an artifact of one substrate.
- L1 (denotation vs Occ3D dense GT) died TWICE over: (a) DEGENERATE -- Occ3D `mask_lidar` makes
  observed-obstacles == dense-GT-obstacles (~0.008% XOR), so the FREE denotation was identity by
  construction; (b) the in-path corridor was VACUOUS (~99.5% free). SemanticKITTI's `.bin` is the
  SINGLE-SCAN voxelized input (sparse, occluded) and `.label` is the COMPLETED dense GT -- they
  genuinely differ by occlusion/sparsity. Owner-question: does that give a NON-degenerate AND
  NON-vacuous denotation test Occ3D could not?

## Ground-truth (CONFIRMED this session by reading the raw files)
- `<id>.bin`: 262144 B = bit-packed occupancy, `np.unpackbits -> reshape(256,256,32)`.
- `<id>.invalid`: 262144 B = bit-packed UNKNOWN/undeterminable mask, same shape.
- `<id>.label`: 4194304 B = `uint16 reshape(256,256,32)`; 0 = empty, !=0 = occupied semantic class.
- Geometry (CONFIRMED): axis0 = forward x in [0, 51.2] m (occupied mean idx ~60, biased near ego);
  axis1 = lateral y in [-25.6, 25.6] m (occupied mean ~128 = centered on ego); axis2 = up z in
  [-2.0, 4.4] m. voxel 0.2 m. ego at the grid origin (x=0, y=0), heading +x. ORIGIN (voxel-(0,0,0)
  center) = (0.1, -25.5, -1.9). Road-class (40) median z-idx 2 (world z ~ -1.5 m) -> ground band.
- NON-DEGENERACY (CONFIRMED, the L1-killer absent here): per frame ~0.6% of voxels occupied in `.bin`
  vs ~4.3% in `.label`; XOR(`.bin`, `.label!=0`) ~3.5-3.9% of ALL voxels, and `.bin`-occupied is
  essentially a SUBSET of `.label`-occupied (`.bin & ~.label` ~ 0; `.label & ~.bin` ~ 73-82k voxels).
  i.e. the dense completion adds ~73-82k obstacle voxels the single scan never observed. Contrast L1's
  0.008% XOR. The full XOR is recomputed and reported in the run.
- Labels exist for sequences 00-10 (training split, 11 sequences); 11-21 are the unlabeled test split.

## Sample (stated, fixed before the run)
- The 11 labeled sequences 00..10. From each, up to 40 evenly-spaced frames (np.linspace over the
  available frame ids; fewer if a sequence has <40). Expected n ~ 440 frames, 11 sequence-clusters.
- Bootstrap is SEQUENCE-clustered (a sequence is one driving scene; intra-sequence frames are spatially
  autocorrelated, so the cluster is the sequence). 1000 resamples, seed 0. n_sequences and n_frames are
  reported.

## Adapter (the apparatus)
`src/probe/adapters/semantickitti.py` (3rd adapter alongside occ3d, av2_sensor -- adapters are an
established `src/probe/adapters/` pattern with 2 prior uses, so this is reuse-of-pattern, not a new
abstraction). Two grid builders, both -> `probe.grid.OccupancyGrid` matching the m2-adapter-contract
schema so the sealed predicates run UNCHANGED:
- OBSERVED (from `.bin` + `.invalid`): OCCUPIED=1 where the `.bin` bit is set; else UNKNOWN=-1 where the
  `.invalid` bit is set; else FREE=0. (OCCUPIED takes precedence over UNKNOWN.)
- DENSE GT (from `.label`): OCCUPIED=1 where `label != 0`; else FREE=0.
- grid spec: VOXEL_SIZE 0.2, GRID_SHAPE (256,256,32), ORIGIN (0.1,-25.5,-1.9), GROUND_HEIGHT -1.4
  (road median world z ~ -1.5 m; -1.4 excludes the ground band z-idx 0-2, keeps obstacles z-idx 3+).
- ego: EgoPose((0,0,0), heading 0, width 1.85, length 4.6, height 1.9) -- standard car envelope, shared
  with the other datasets (no per-frame pose file ships with the SSC voxels). speed: nominal 5.0 m/s
  (urban) for the band-reach geometry only; speed does NOT enter the headline denotation domain.

## Leg 1 -- H1 expressivity, cross-dataset (oracle-free, the headline)
Reuse `h3b_expressivity.leg1_expressivity(queries, probe_scene)` VERBATIM (the sealed
`queries.yaml` set + the production `probe.retrieval.scene_matches` evaluator + the `refav_expressible`
flag) with `probe_scene` = a SemanticKITTI scene (5 frames, seq 00) loaded via the new adapter.
- occupancy coverage = sealed occupancy free-space queries that EXECUTE on the real KITTI grid.
- box-only coverage = RefAV 32-fn `refav_expressible` flag (STRUCTURALLY 0 on free-space: no free-space
  primitive). Report the free-space-family occupancy% vs box-only% gap on KITTI (a 3rd dataset beyond
  AV2 + nuScenes).
- **KILL (declared now):** H1 is FALSIFIED iff box-only can express ANY free-space-family query
  (`free_space_families_only.box_only_expressible > 0`).

## Leg 2 -- denotation (non-degenerate, non-vacuous): observed `.bin` vs dense GT `.label`
FREE is the positive class; pred = OBSERVED occupancy, ref = DENSE-GT occupancy. Per BEV cell at the
ego-height band, BLOCKED iff any ego-height-band voxel projects occupied; FREE otherwise.
- ego-height band = world z in (GROUND_HEIGHT, GROUND_HEIGHT + ego.height] = (-1.4, 0.5] (voxel z-idx
  3-11), the exact vertical envelope `OccupancyGrid.obstacle_centers(max_height_agl=ego.height)` reads.
- **Headline domain (NON-VACUOUS):** the forward ego-height BEV FIELD the free-space predicate family
  scans -- every column with x in [0, 51.2] m (the whole grid ahead of the ego), restricted to
  DETERMINABLE columns (NOT all-invalid across the ego-height band; undeterminable columns are dropped,
  the standard SemanticKITTI SSC ignore-invalid rule). DEVIATION from the literal "in-path band" in the
  task brief, made toward the task's stated GOAL: the thin in-path corridor is VACUOUS (~98% free,
  measured -- the corridor directly ahead of a driving car is free by construction, exactly L1's
  failure), so it cannot be the non-vacuous test the goal requires. The forward ego-height field is the
  spatial domain `centerline_lateral_distance` (lookahead 20 m) + `lateral_clearance` actually read for
  roadside obstacles, and is measured NON-vacuous (GT blocked-rate ~41%). No tuned parameter (it is the
  whole forward grid). The verdict is RELATIVE to the all-free baseline, so the domain extent cannot
  bias the conclusion toward a win.
- **Secondary domain (reported for contrast, expected VACUOUS):** the L1-style thin in-path corridor
  via the reused `l1_denotation_occ3d.band_blocked_bev` (forward 0..reach, |lateral|<=width/2+1.0 m).
  Reported to show WHY the headline domain was widened; NOT the headline.
- Metrics (reuse `l1_denotation_occ3d` helpers VERBATIM -- unit-tested, no homegrown rank-AUC, pure
  numpy): IoU/precision/recall/F1 of FREE + false_block_rate + miss_rate, sequence-clustered bootstrap
  (1000, seed 0) via `_boot_metric`. Baselines: all-free (`_allfree_conf`), random@band-free-rate
  (`_random_conf`).
- Report the headline-domain GT blocked-rate (proving NON-vacuous) and the global `.bin`-vs-`.label`
  occupied XOR (proving NON-degenerate).
- **KILL (declared now):** Leg-2 is DEAD (not a result) iff the predicate FREE-IoU is no better than the
  trivial all-free baseline on the NON-vacuous headline domain -- specifically iff
  `predicate_IoU.CI_lo <= all_free_IoU.mean`. (CI.lo above the baseline mean = a real, relative result.)

## Honest label (independence ledger)
SemanticKITTI `.label` is the temporally-aggregated, human-corrected LiDAR completion; `.bin` is one
LiDAR scan voxelized. Both are LiDAR-derived -> this is a CONSISTENCY test (observed single-scan vs
completed dense occupancy under occlusion), NOT external ground truth. It is NON-degenerate (the two
genuinely differ, XOR confirmed huge) and NON-vacuous (GT ~41% blocked), so it is a real
denotation-ROBUSTNESS-under-occlusion test -- strictly stronger than L1, but still consistency. H1
(Leg 1) stays the field headline; Leg 2 is reported as consistency, never as external P/R/F1.

## What a result means (falsifiability, stated before data)
- H1 holds on KITTI iff box-only free-space coverage == 0 while occupancy coverage > 0 (the gap).
- Leg 2 is a (consistency) result iff predicate IoU CI.lo > all-free IoU mean on the non-vacuous domain.
  If CI overlaps the baseline, the headline is "Leg-2 not better than trivial on KITTI either" -- a
  reported negative, not buried.
</content>
</invoke>

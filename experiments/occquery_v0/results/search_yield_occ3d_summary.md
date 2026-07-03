# Search yield + honesty tags — Occ3D-nuScenes corpus

- Pre-reg: `search_yield_occ3d_preregistration.md (SEALED, commit 6f17a1c)`
- Commit: `5c42e111e1a0bdfca83930ddcc844f7d78cf9142`  seed 0
- Result class: corpus DESCRIPTION / consistency (dense arm = Occ3D auto-label; observed arm = single-sweep visibility) — NOT external world-truth. Tag validity vs dense GT NOT checkable here (Occ3D obstacle-visibility degeneracy); mechanism validity = sealed synthetic test 5844786.
- Split: headline 680 scenes / dev 170

## Verdicts (per the sealed criteria): C1 HOLDS / C2 KILLED (exoneration impossible)

## C1 — scene yield (headline split; long-tail band = (0, 20%])
10/20 long-tail, 1/20 zero, 3/20 saturated (>50%).

| query | backend | scene yield | scenes | frame hit rate | band |
|---|---|---|---|---|---|
| grazing_side_clearance | occupancy | 0.2250 CI[0.1941, 0.2559] | 153 | 0.0325 | mid |
| tight_clearance_at_speed | occupancy | 0.0426 CI[0.0294, 0.0588] | 29 | 0.0019 | long_tail |
| moderate_side_clearance_at_speed | occupancy | 0.1059 CI[0.0853, 0.1294] | 72 | 0.0085 | long_tail |
| tight_side_clearance | occupancy | 0.2250 CI[0.1926, 0.2559] | 153 | 0.0325 | mid |
| moderate_side_clearance | occupancy | 0.4956 CI[0.4588, 0.5353] | 337 | 0.1256 | mid |
| centerline_obstacle_grazing | occupancy | 0.6603 CI[0.6279, 0.6971] | 449 | 0.1994 | saturated |
| centerline_obstacle_tight | occupancy | 0.6603 CI[0.6235, 0.6956] | 449 | 0.1994 | saturated |
| centerline_obstacle_moderate_at_urban_speed | occupancy | 0.3529 CI[0.3176, 0.3868] | 240 | 0.0614 | mid |
| centerline_obstacle_moderate | occupancy | 0.7221 CI[0.6882, 0.7544] | 491 | 0.2438 | saturated |
| centerline_obstacle_tight_at_speed | occupancy | 0.0765 CI[0.0574, 0.0971] | 52 | 0.0182 | long_tail |
| free_path_blocked_within_body_length | occupancy | 0.0735 CI[0.0544, 0.0926] | 50 | 0.0059 | long_tail |
| free_path_is_blocked | occupancy | 0.1029 CI[0.0809, 0.1280] | 70 | 0.0147 | long_tail |
| free_path_blocked_at_two_second_horizon | occupancy | 0.2897 CI[0.2574, 0.3235] | 197 | 0.0581 | mid |
| free_path_blocked_at_long_range_high_speed | occupancy | 0.1735 CI[0.1441, 0.2015] | 118 | 0.0458 | long_tail |
| free_path_blocked_or_side_clearance_tight | occupancy | 0.2706 CI[0.2368, 0.3059] | 184 | 0.0442 | mid |
| corridor_pinches_fully_shut | occupancy | 0.0000 CI[0.0000, 0.0000] | 0 | 0.0000 | zero |
| near_corridor_below_vehicle_width | occupancy | 0.0235 CI[0.0132, 0.0368] | 16 | 0.0009 | long_tail |
| corridor_narrows_below_vehicle_width | occupancy | 0.0485 CI[0.0338, 0.0662] | 33 | 0.0025 | long_tail |
| far_corridor_below_vehicle_width | occupancy | 0.1529 CI[0.1279, 0.1794] | 104 | 0.0080 | long_tail |
| corridor_below_half_meter | occupancy | 0.0162 CI[0.0074, 0.0265] | 11 | 0.0005 | long_tail |
| near_a_tracked_vehicle | tracking | 0.0000 CI[0.0000, 0.0000] | 0 | 0.0000 | baseline |
| within_one_carlength_of_a_tracked_vehicle | tracking | 0.7515 CI[0.7191, 0.7853] | 511 | 0.2329 | baseline |
| near_a_tracked_pedestrian | tracking | 0.0059 CI[0.0015, 0.0132] | 4 | 0.0009 | baseline |
| near_a_tracked_bicycle | tracking | 0.0000 CI[0.0000, 0.0000] | 0 | 0.0000 | baseline |

## C2 — honesty tags (observed arm, 10 free_path/corridor queries)
At sealed eps=0.05: UNRESOLVED 1.0000 of non-hits, EXONERATED 0.0000 of non-hits.

| eps | CONFIRMED_HIT | EXONERATED | UNRESOLVED | unresolved rate | exonerated rate |
|---|---|---|---|---|---|
| 0.01 | 4949 | 0 | 269111 | 1.0000 CI[1.0000, 1.0000] | 0.0000 CI[0.0000, 0.0000] |
| 0.05 | 4949 | 0 | 269111 | 1.0000 CI[1.0000, 1.0000] | 0.0000 CI[0.0000, 0.0000] |
| 0.1 | 4949 | 0 | 269111 | 1.0000 CI[1.0000, 1.0000] | 0.0000 CI[0.0000, 0.0000] |
| 0.2 | 4949 | 10 | 269101 | 1.0000 CI[0.9999, 1.0000] | 0.0000 CI[0.0000, 0.0001] |

## Physical-quantity distributions (headline, finite values)
| measurement | n | P5 | P50 | P95 |
|---|---|---|---|---|
| centerline_lateral_distance | 27406 | 0.200 | 2.200 | 5.800 |
| distance_to_nearest_object[object_class=bicycle] | 5838 | 6.648 | 21.107 | 49.437 |
| distance_to_nearest_object[object_class=pedestrian] | 22106 | 5.283 | 15.347 | 47.217 |
| distance_to_nearest_object[object_class=vehicle] | 27235 | 3.551 | 8.335 | 27.076 |
| ego_speed | 27406 | 0.000 | 5.248 | 10.473 |
| lateral_clearance | 27400 | 0.675 | 1.875 | 8.675 |
| min_free_width_along_path[horizon=1.0] | 25684 | 4.400 | 11.200 | 22.800 |
| min_free_width_along_path[horizon=2.0] | 26365 | 4.400 | 10.400 | 21.200 |
| min_free_width_along_path[horizon=4.0] | 26759 | 3.600 | 9.600 | 19.600 |

- 20/20 occupancy queries refav_expressible=false (sealed queries.yaml, verified vs RefAV's 32-function set) — restated, not recomputed.

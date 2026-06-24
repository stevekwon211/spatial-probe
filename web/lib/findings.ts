// The evidence ledger: every claim, its verdict, and how it was graded -- negatives included.
// Sourced VERBATIM from experiments/*/results/summary.md (no re-inflation). The `gradedBy` field has
// no "validated" member by design: no externally-validated F1 exists, so overclaiming is unrepresentable.

export type Verdict = "HOLDS" | "INCONCLUSIVE" | "RETRACTED" | "NOT-STARTED";
export type GradedBy = "oracle-free-witness" | "consistency-only" | "graded-surrogate" | "untested";

export interface Finding {
  id: string;
  claim: string;
  axis: "Search" | "Dynamics";
  verdict: Verdict;
  detail: string; // the one-line gloss of how it was graded
  gradedBy: GradedBy;
  dossier: string[]; // paragraphs, honest, from the summaries
}

export const FINDINGS: Finding[] = [
  {
    id: "h1",
    claim: "H1 — occupancy predicates express box-blind free-space",
    axis: "Search",
    verdict: "HOLDS",
    detail: "oracle-free witness",
    gradedBy: "oracle-free-witness",
    dossier: [
      "Two scenes identical in the tracked-box + map + velocity channel, differing only by an unboxed occupancy obstacle. Any function of the box+map channel must return the same answer; the occupancy predicate separates them. Non-identifiability by construction — no oracle needed.",
      "On the pre-registered 24-query set, occupancy expresses 24/24; RefAV's released 32-function set expresses 4/24 (the box-baseline controls). The 20 occupancy-native queries are inexpressible in the box-only language.",
      "This is the program's sole headline result, and it is oracle-free — it does not depend on any independent measurement oracle.",
    ],
  },
  {
    id: "h3",
    claim: "H3 — the free-space measurement matches independent truth, better than box-only",
    axis: "Search",
    verdict: "INCONCLUSIVE",
    detail: "future-reveal oracle; twice-audited, retracted twice",
    gradedBy: "consistency-only",
    dossier: [
      "H3 asks whether the occupancy predicate's free/occupied verdict matches an INDEPENDENT measurement. The only oracle buildable on this hardware shares the predicate's data source (same Occ3D / LiDAR), so it is a consistency check, not external truth — H3 was demoted on that basis.",
      "A temporal independent oracle WAS built (future-reveal: grade a frame-t claim against a later raw LiDAR sweep). The result was retracted twice before settling: shipped FALSIFIED (cbe404b) — wrong, it omitted a pre-registered control; then thin-HOLDS / INDETERMINATE (ecd0f05) — also wrong, the re-entry control over-dropped PARKED cars (89-95% of the dropped box true-positives were parked, validly gradeable), which was the sole maker of the apparent gap.",
      "Corrected with a motion-faithful control: gap +0.005 / -0.017 / -0.037, CI includes 0 at all k -> NULL. The decisive finding: the revealed-occupied 'truth' is 0.2% real structure (97.5% is ground / free single LiDAR returns), so the 10-scene substrate has no power to test H3. Verdict: INCONCLUSIVE.",
      "An adversarial audit of our own output caught both wrong calls and a false mechanism claim. The honest answer is inconclusive — neither confirmed nor falsified — on a degenerate substrate. A clean independent test needs more structure-bearing scenes or a cross-modal grader.",
    ],
  },
  {
    id: "h2",
    claim: "H2 — competitive on a public retrieval leaderboard (HOTA-Temporal)",
    axis: "Search",
    verdict: "NOT-STARTED",
    detail: "substrate mismatch (RefAV is Argoverse 2, not Occ3D-nuScenes)",
    gradedBy: "untested",
    dossier: [
      "occquery's denotation arm is on Occ3D-nuScenes; the RefAV HOTA-Temporal anchor is on Argoverse 2 — a different dataset. Sourcing occupancy for Argoverse 2 would mean generating a non-released oracle, which undermines the released-oracle property.",
      "H2 is a credibility leg, not required for the H1 headline. Not started.",
    ],
  },
  {
    id: "dyn-sh1",
    claim: "DynField SH1 — a stored motion field is sometimes necessary (necessity-witness)",
    axis: "Dynamics",
    verdict: "HOLDS",
    detail: "oracle-free construction",
    gradedBy: "oracle-free-witness",
    dossier: [
      "Two frames identical in static occupancy but differing in stored motion force a static-only surrogate to act identically while a motion-aware one differs. By construction, no oracle. Scope bound: non-identifiability under the surrogate's static-occupancy observable set, not a claim about every planner.",
    ],
  },
  {
    id: "dyn-sh4",
    claim: "DynField SH4 — the static baseline is not motion-contaminated (leakage gate)",
    axis: "Dynamics",
    verdict: "HOLDS",
    detail: "held-out val split, distance↔velocity corr 0.184 < 0.2",
    gradedBy: "graded-surrogate",
    dossier: [
      "On the held-out official nuScenes val split (150 scenes), the static field (lead distance) does not encode the ablated velocity — correlation 0.184, below the pre-registered 0.2. The static-only baseline is not motion-contaminated.",
    ],
  },
  {
    id: "dyn-redundant",
    claim: "DynField — velocity is action-redundant in safe vehicle-following",
    axis: "Dynamics",
    verdict: "HOLDS",
    detail: "graded IDM on val (n=443); redundant",
    gradedBy: "graded-surrogate",
    dossier: [
      "A graded IDM surrogate (the closing-gap term carries the ablated velocity) shows velocity is action-EQUIVALENT in safe vehicle-following: true decel-delta 0.10, CI [0.06, 0.14], entirely below the shuffled-velocity band [0.22, 0.56], n=443. The 'redundant when calmly following' half of the by-regime thesis, measured cleanly on real data.",
      "Framing is action-sensitivity, not necessity — necessity needs a closed-loop quality oracle (GPU Tier-2).",
    ],
  },
  {
    id: "dyn-danger",
    claim: "DynField — velocity is necessary when dangerous",
    axis: "Dynamics",
    verdict: "NOT-STARTED",
    detail: "untestable on nuScenes (no danger): TTC<2s in ~22/2114 lead-frames",
    gradedBy: "untested",
    dossier: [
      "The complementary 'necessary when dangerous' half is untestable on nuScenes because the dataset is benign — only ~22/2114 lead-frames at TTC<2s, and ~0 genuine fast-closing near-misses (lead closing-speed median +0.07 m/s).",
      "Real AV danger is geometric (fast ego, moderate gap), not dynamic (a lead rushing in); velocity-necessity lives in the fast-closing tail, which is empirically absent. Resolving it needs a danger-bearing substrate (simulation / safety-critical scenarios), where 'necessary' is finally licensed.",
    ],
  },
];

export const VERDICT_COUNTS = FINDINGS.reduce<Record<Verdict, number>>(
  (acc, f) => ({ ...acc, [f.verdict]: (acc[f.verdict] ?? 0) + 1 }),
  { HOLDS: 0, INCONCLUSIVE: 0, RETRACTED: 0, "NOT-STARTED": 0 },
);

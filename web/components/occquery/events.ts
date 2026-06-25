// Pure functions over per-frame FrameMeta data.
// No React, no side-effects.

export type FrameMeta = {
  t: number;
  speed: number;
  n_obstacles_band: number;
  n_obstacles_total: number;
  predicates: {
    ego_width: number;
    min_free_width: number | null;
    lateral_clearance: number | null;
    free_path_blocked: boolean;
    box_distance: number | null;
  };
};

export type DeltaPoint = {
  frameIdx: number;
  t: number;
  // box_distance - lateral_clearance (both on the lateral axis). Null when either is absent.
  delta: number | null;
  lateral_clearance: number | null;
  box_distance: number | null;
};

export type EventKind = "NEAR_MISS" | "VERDICT_FLIP";

export type FrameEvent = {
  kind: EventKind;
  frameIdx: number;
  t: number;
  // NEAR_MISS: lateral_clearance at the local minimum. VERDICT_FLIP: undefined.
  value?: number;
};

// Threshold below which a lateral_clearance local minimum counts as a near-miss.
const NEAR_MISS_THRESHOLD = 2.5; // metres

/**
 * Returns one DeltaPoint per frame.
 * delta = box_distance - lateral_clearance (same lateral axis; NOT min_free_width which is a width).
 * Null when either measurement is absent for that frame.
 */
export function deltaSeries(frames: FrameMeta[]): DeltaPoint[] {
  return frames.map((f, i) => {
    const lc = f.predicates.lateral_clearance;
    const bd = f.predicates.box_distance;
    const delta = lc !== null && bd !== null ? bd - lc : null;
    return { frameIdx: i, t: f.t, delta, lateral_clearance: lc, box_distance: bd };
  });
}

/**
 * Returns the union of two event types:
 *   NEAR_MISS  — a local minimum of lateral_clearance below NEAR_MISS_THRESHOLD.
 *   VERDICT_FLIP — free_path_blocked flips between consecutive frames.
 * Sorted by frameIdx ascending.
 */
export function deriveEvents(frames: FrameMeta[]): FrameEvent[] {
  const events: FrameEvent[] = [];

  // NEAR_MISS: strict local minimum of lateral_clearance below threshold.
  for (let i = 0; i < frames.length; i++) {
    const lc = frames[i].predicates.lateral_clearance;
    if (lc === null || lc >= NEAR_MISS_THRESHOLD) continue;

    const prevLc = i > 0 ? frames[i - 1].predicates.lateral_clearance : null;
    const nextLc = i < frames.length - 1 ? frames[i + 1].predicates.lateral_clearance : null;

    const lowerThanPrev = prevLc === null || lc <= prevLc;
    const lowerThanNext = nextLc === null || lc <= nextLc;

    if (lowerThanPrev && lowerThanNext) {
      events.push({ kind: "NEAR_MISS", frameIdx: i, t: frames[i].t, value: lc });
    }
  }

  // VERDICT_FLIP: free_path_blocked changes from frame i-1 to frame i.
  for (let i = 1; i < frames.length; i++) {
    const prev = frames[i - 1].predicates.free_path_blocked;
    const curr = frames[i].predicates.free_path_blocked;
    if (prev !== curr) {
      events.push({ kind: "VERDICT_FLIP", frameIdx: i, t: frames[i].t });
    }
  }

  events.sort((a, b) => a.frameIdx - b.frameIdx);
  return events;
}

/** Returns the frameIdx of the maximum absolute delta, or 0 when series is empty or all null. */
export function maxAbsDeltaIdx(series: DeltaPoint[]): number {
  let bestIdx = 0;
  let bestAbs = -Infinity;
  for (const pt of series) {
    if (pt.delta === null) continue;
    const abs = Math.abs(pt.delta);
    if (abs > bestAbs) { bestAbs = abs; bestIdx = pt.frameIdx; }
  }
  return bestIdx;
}

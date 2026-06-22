// Bilingual strings for the H3 labeling view. English is CANONICAL (it matches the sealed
// queries.yaml the predicate runs on); Korean is a faithful, meaning-preserving translation for the
// labeler's comfort -- same thresholds, same occupancy/box terms (kept in English inside Korean), so
// switching language never changes what is being judged. queries.yaml is NOT touched (it stays sealed).

export type Lang = "en" | "ko";

export const UI = {
  en: {
    title: "H3 labeling",
    progress: (cur: number, total: number, voted: number) => `task ${cur}/${total} · ${voted} voted`,
    banner:
      "Judge the geometry, not danger. Human-vs-code consistency on the same Occ3D data — not an external oracle. You do not see the predicate's answer until you vote.",
    scope: (s?: string) => `query · scope ${(s ?? "").toUpperCase()}`,
    frame: (cur: number, total: number | string) => `frame ${cur}/${total}`,
    frameStat: (speed: number, obs: number) => `${speed.toFixed(1)} m/s · ${obs} obs`,
    frameHint: (scene: string) => `${scene} · ← → to step frames`,
    prompt: "Does any frame of this scene match the query above?",
    yes: "Yes",
    no: "No",
    skip: "Skip",
    hidden: "predicate answer hidden until you vote",
    youVoted: "You voted",
    predicate: (retrieved: boolean, n: number) =>
      `predicate: ${retrieved ? "retrieved" : "not retrieved"}${n > 0 ? ` · matched ${n} frame(s)` : ""}`,
    agrees: "agrees with your vote (consistency)",
    disagrees: "disagrees — a real signal, kept as-is (no re-vote pressure)",
    skipped: "skipped",
    overlayNote: "reachable free-space overlay now shown (QA only)",
    undo: "Undo",
    next: "Next →",
    save: (n: number) => `Save ${n} verdicts`,
    loading: "loading sealed pool…",
  },
  ko: {
    title: "H3 라벨링",
    progress: (cur: number, total: number, voted: number) => `task ${cur}/${total} · ${voted}개 판정`,
    banner:
      "위험이 아니라 geometry를 판정. 같은 Occ3D 데이터에 대한 human-vs-code 일관성 — external oracle 아님. vote 전까지 predicate 답은 안 보임.",
    scope: (s?: string) => `query · scope ${(s ?? "").toUpperCase()}`,
    frame: (cur: number, total: number | string) => `frame ${cur}/${total}`,
    frameStat: (speed: number, obs: number) => `${speed.toFixed(1)} m/s · obstacle ${obs}개`,
    frameHint: (scene: string) => `${scene} · ← → 로 프레임 이동`,
    prompt: "이 scene의 어느 한 프레임이라도 위 query에 해당하나?",
    yes: "예",
    no: "아니오",
    skip: "건너뜀",
    hidden: "vote 전까지 predicate 답 숨김",
    youVoted: "네 판정:",
    predicate: (retrieved: boolean, n: number) =>
      `predicate: ${retrieved ? "retrieved" : "not retrieved"}${n > 0 ? ` · ${n}개 프레임 매칭` : ""}`,
    agrees: "네 판정과 일치 (consistency)",
    disagrees: "불일치 — 진짜 signal, 그대로 둠 (재투표 압박 없음)",
    skipped: "건너뜀",
    overlayNote: "reachable free-space overlay 표시됨 (QA 전용)",
    undo: "되돌리기",
    next: "다음 →",
    save: (n: number) => `${n}개 판정 저장`,
    loading: "봉인된 pool 로딩 중…",
  },
} as const;

// Guidance that makes the task HUMAN-DOABLE: the labeler judges the spatial SHAPE qualitatively
// (clear-yes / clear-no / skip-if-borderline), never an exact metric or a speed by eye. Non-geometric
// gates (speed) are applied from data, not judged. This makes the human a COARSE oracle that catches
// gross predicate errors, not a sub-voxel ruler -- the honest ceiling on same-data human-vs-code.
export const GUIDE: Record<Lang, {
  lookForLabel: string;
  rule: string;
  coarse: string;
  speedAuto: (kmh: number) => string;
  views: { iso: string; top: string; side: string; front: string };
}> = {
  en: {
    lookForLabel: "What to look for",
    rule: "Yes = clearly there in some frame. No = clearly absent in every frame. Skip = borderline / can't tell — don't guess.",
    coarse: "You catch gross errors, not exact distances. If it's a close call, Skip.",
    speedAuto: (kmh) => `The ${kmh} km/h speed gate is applied automatically from data — judge ONLY the spatial geometry.`,
    views: { iso: "3/4", top: "Top", side: "Side", front: "Front" },
  },
  ko: {
    lookForLabel: "무엇을 볼지",
    rule: "예 = 어느 프레임에라도 분명히 있음. 아니오 = 모든 프레임에서 분명히 없음. 건너뜀 = 애매/모르겠음 — 찍지 마.",
    coarse: "정확한 거리가 아니라 gross error를 잡는 역할. 아슬아슬하면 건너뜀.",
    speedAuto: (kmh) => `${kmh} km/h 속도 게이트는 데이터에서 자동 적용 — 공간 geometry만 판정.`,
    views: { iso: "3/4", top: "위", side: "옆", front: "앞" },
  },
};

// Per-query: the human-judgeable geometric thing to look for (+ a speed gate that is auto-applied).
export const LOOK_FOR: Record<string, { en: string; ko: string; speedGateKmh?: number }> = {
  tight_clearance_at_speed: {
    en: "A solid object hugging the LEFT or RIGHT side of the red box (like a wall grazing the door). Yes only if it is to the SIDE — something only ahead or behind does NOT count for this query.",
    ko: "빨간 박스의 좌/우 옆면에 바짝 붙은 solid (문 옆을 스치는 벽 같은). 옆에 있을 때만 예 — 앞이나 뒤에만 있는 건 이 query 대상 아님.",
    speedGateKmh: 30,
  },
  free_path_is_blocked: {
    en: "Something blocking the path straight AHEAD of the red box (on its centerline). Yes only if the FORWARD path is blocked — a wall off to the side does NOT count.",
    ko: "빨간 박스 바로 앞(중심선)을 막는 것. 앞 길이 막혔을 때만 예 — 옆으로 비낀 벽은 대상 아님.",
  },
  corridor_narrows_below_vehicle_width: {
    en: "Ahead of the red box, the free lane pinched NARROWER than the box's own width — walls closing in on BOTH sides below car-width. Yes only if it's genuinely two-sided and narrower than the car.",
    ko: "빨간 박스 앞쪽에서 free 통로가 박스 자기 폭보다 좁아짐 — 양쪽 벽이 차 폭 미만으로 조여듦. 진짜 양쪽이 막히고 차보다 좁을 때만 예.",
  },
};

// Korean translations of the query NL + rationale, keyed by query_id. Faithful to the sealed English
// (same thresholds, same occupancy-vs-box argument). Missing keys fall back to the canonical English
// from pool.json, so an untranslated query still shows (degrades gracefully, never blank).
export const QUERY_KO: Record<string, { nl: string; rationale: string }> = {
  tight_clearance_at_speed: {
    nl: "ego가 30 km/h 넘게 달리는 동안 측면 여유 공간(side clearance)이 0.5 m 미만이었다",
    rationale:
      "body-side free gap을 dense occupancy로 잰 것(cuboid에 대응물 없음: 가장 가까운 occupied 공간이 box 없는 벽/curb일 수 있음) AND ego-speed gate. ego_speed 단독은 box로 표현 가능(cuboid velocity)하지만 occupancy 항과의 conjunction은 box+map으로 식별 불가. 8.33 m/s = 30 km/h는 pre-registered urban-speed knob, 0.5 m는 query 전반 공유되는 clearance tier.",
  },
  free_path_is_blocked: {
    nl: "ego의 직진 경로가 1초 도달 범위 안에서 (box 없는) free-space 장애물에 막혔다",
    rationale:
      "reachable-corridor 막힘을 표준 1.0 s horizon에서 판정. 막는 것이 cuboid가 없을 수 있음(돌출 적재물, 공사 barrier, free-space 벽). RefAV와 동일한 두 scene(box·map·velocity 같음)이 reach 안의 box 없는 장애물로만 달라질 수 있음. min_cluster_voxels=2가 lone-voxel sensor noise를 제거.",
  },
  corridor_narrows_below_vehicle_width: {
    nl: "앞쪽 유일한 free path가 차 폭(ego width)보다 좁게 좁아졌다",
    rationale:
      "free-space corridor 폭이 ego 자기 폭보다 작음(표준 2.0 s horizon) — 차체가 못 들어감. 두 (박스 없을 수 있는) 표면 사이의 폭은 RefAV box+map set에 primitive가 없음. corridor는 열려있고(>0, 진짜 two-sided) 그 폭이 ego_width 미만. ego_width chained compare가 self-relative '들어맞나' 척도. ego-width horizon sweep(1.0/2.0/4.0 s)의 중간 horizon.",
  },
};

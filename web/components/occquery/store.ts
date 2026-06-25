import { create } from "zustand";

export type ColorMode = "flat" | "height" | "forward" | "lateral" | "semantic" | "state";
export type VoxelShape = "cube" | "sphere";
export type Projection = "perspective" | "orthographic";
export type RenderMode = "voxel" | "points" | "both";

export type Settings = {
  renderMode: RenderMode;
  pointSize: number;
  showVoxels: boolean;
  showEgo: boolean;
  showForward: boolean;
  showGrid: boolean;
  showReachable: boolean;
  showBoxes: boolean;
  showStats: boolean;
  voxelShape: VoxelShape;
  voxelScale: number;
  voxelOpacity: number;
  wireframe: boolean;
  egoOpacity: number;
  colorMode: ColorMode;
  projection: Projection;
  playing: boolean;
  speed: number;
  loop: boolean;
  panelCollapsed: boolean;
  settingsOpen: boolean;
};

export const DEFAULTS: Settings = {
  renderMode: "voxel", pointSize: 0.04,
  showVoxels: true, showEgo: true, showForward: true, showGrid: true, showReachable: false, showBoxes: true,
  showStats: false, voxelShape: "cube", voxelScale: 0.85, voxelOpacity: 1, wireframe: false,
  egoOpacity: 0.55, colorMode: "height", projection: "perspective",
  playing: false, speed: 1, loop: true, panelCollapsed: false, settingsOpen: false,
};

type Cam = { preset: string; nonce: number };

type Store = Settings & {
  cam: Cam;
  set: <K extends keyof Settings>(k: K, v: Settings[K]) => void;
  toggle: (k: keyof Settings) => void;
  applyPreset: (preset: string) => void;
  reset: () => void;
};

export const useViewer = create<Store>((set) => ({
  ...DEFAULTS,
  cam: { preset: "iso", nonce: 0 },
  set: (k, v) => set({ [k]: v } as Partial<Settings>),
  toggle: (k) => set((s) => ({ [k]: !s[k] } as Partial<Settings>)),
  applyPreset: (preset) => set((s) => ({ cam: { preset, nonce: s.cam.nonce + 1 } })),
  reset: () => set({ ...DEFAULTS, cam: { preset: "iso", nonce: 0 } }),
}));

// color modes usable with current data; the rest are coming-soon (need export extension)
export const COLOR_MODES: { id: ColorMode; label: string; enabled: boolean }[] = [
  { id: "flat", label: "Flat", enabled: true },
  { id: "height", label: "Height", enabled: true },
  { id: "forward", label: "Forward", enabled: true },
  { id: "lateral", label: "Lateral", enabled: true },
  { id: "semantic", label: "Semantic", enabled: true },
  { id: "state", label: "Occ state", enabled: false },
];

export const CAMERA_PRESETS: { id: string; label: string; pos: [number, number, number] }[] = [
  { id: "iso", label: "3/4", pos: [-14, 11, 13] },
  { id: "bev", label: "Top", pos: [0.01, 42, 0] },
  { id: "front", label: "Front", pos: [30, 7, 0] },
  { id: "side", label: "Side", pos: [0, 7, 30] },
  { id: "rear", label: "Rear", pos: [-30, 7, 0] },
  { id: "driver", label: "Driver", pos: [-2.2, 2, 0] },
];

// shipped as disabled "coming soon" controls (need data/backend we don't have yet)
export const COMING_SOON: { group: string; items: string[] }[] = [
  { group: "Overlays", items: ["Measurement tool", "Occupancy-flow vectors"] },
  { group: "Search", items: ["Natural-language search", "Find similar scenes", "GT vs predicted diff"] },
  { group: "Panels", items: ["Embeddings (UMAP)", "HD-map underlay", "Multi-sensor panes", "SAM2 annotate"] },
];

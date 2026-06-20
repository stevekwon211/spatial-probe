"use client";

import type { ReactNode } from "react";
import { CAMERA_PRESETS, COLOR_MODES, COMING_SOON, useViewer, type Settings } from "./store";

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="space-y-1">
      <div className="px-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">{title}</div>
      {children}
    </div>
  );
}

function Toggle({ label, k, kbd }: { label: string; k: keyof Settings; kbd?: string }) {
  const v = useViewer((s) => s[k]) as boolean;
  const toggle = useViewer((s) => s.toggle);
  return (
    <button
      onClick={() => toggle(k)}
      className={`flex w-full items-center justify-between rounded px-2 py-1 text-xs ${v ? "bg-white/10 text-foreground" : "text-muted-foreground hover:bg-white/5"}`}
    >
      <span>{label}</span>
      {kbd && <kbd className="text-[10px] opacity-40">{kbd}</kbd>}
    </button>
  );
}

function Seg<T extends string>({ value, opts, on }: { value: T; opts: { id: T; label: string; enabled?: boolean }[]; on: (v: T) => void }) {
  return (
    <div className="grid grid-cols-3 gap-1 px-1">
      {opts.map((o) => (
        <button
          key={o.id}
          disabled={o.enabled === false}
          onClick={() => on(o.id)}
          title={o.enabled === false ? "coming soon (needs export)" : undefined}
          className={`rounded px-1.5 py-1 text-[11px] ${
            value === o.id ? "bg-blue-600 text-white" : o.enabled === false ? "cursor-not-allowed text-white/20" : "bg-white/5 text-muted-foreground hover:bg-white/10"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

function Slider({ label, v, min, max, step, on }: { label: string; v: number; min: number; max: number; step: number; on: (x: number) => void }) {
  return (
    <div className="px-1">
      <div className="flex justify-between text-[10px] text-muted-foreground">
        <span>{label}</span>
        <span>{v.toFixed(2)}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={v} onChange={(e) => on(Number(e.target.value))} className="w-full accent-blue-500" />
    </div>
  );
}

export function ControlPanel() {
  const set = useViewer((s) => s.set);
  const applyPreset = useViewer((s) => s.applyPreset);
  const colorMode = useViewer((s) => s.colorMode);
  const projection = useViewer((s) => s.projection);
  const voxelShape = useViewer((s) => s.voxelShape);
  const voxelScale = useViewer((s) => s.voxelScale);
  const voxelOpacity = useViewer((s) => s.voxelOpacity);

  return (
    <div className="space-y-3">
      <Section title="Show">
        <Toggle label="Obstacle voxels" k="showVoxels" kbd="O" />
        <Toggle label="Ego box" k="showEgo" kbd="E" />
        <Toggle label="Forward ray" k="showForward" />
        <Toggle label="Ground grid" k="showGrid" kbd="G" />
        <Toggle label="Axis gizmo" k="showGizmo" />
        <Toggle label="Stats (FPS)" k="showStats" />
        <Toggle label="Wireframe" k="wireframe" />
      </Section>

      <Section title="Color by">
        <Seg value={colorMode} opts={COLOR_MODES} on={(v) => set("colorMode", v)} />
      </Section>

      <Section title="Voxel">
        <Seg value={voxelShape} opts={[{ id: "cube", label: "Cube" }, { id: "sphere", label: "Sphere" }]} on={(v) => set("voxelShape", v)} />
        <Slider label="Size" v={voxelScale} min={0.3} max={1} step={0.05} on={(x) => set("voxelScale", x)} />
        <Slider label="Opacity" v={voxelOpacity} min={0.2} max={1} step={0.05} on={(x) => set("voxelOpacity", x)} />
      </Section>

      <Section title="Camera">
        <div className="grid grid-cols-3 gap-1 px-1">
          {CAMERA_PRESETS.map((p) => (
            <button key={p.id} onClick={() => applyPreset(p.id)} className="rounded bg-white/5 px-1.5 py-1 text-[11px] text-muted-foreground hover:bg-white/10">
              {p.label}
            </button>
          ))}
        </div>
        <Seg value={projection} opts={[{ id: "perspective", label: "Persp" }, { id: "orthographic", label: "Ortho" }]} on={(v) => set("projection", v)} />
      </Section>

      <Section title="Coming soon">
        {COMING_SOON.flatMap((g) => g.items).map((it) => (
          <button key={it} disabled className="flex w-full cursor-not-allowed items-center justify-between rounded px-2 py-1 text-xs text-white/20">
            <span>{it}</span>
            <span className="text-[9px]">soon</span>
          </button>
        ))}
      </Section>
    </div>
  );
}

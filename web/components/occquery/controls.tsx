"use client";

import type { ReactNode } from "react";
import { CAMERA_PRESETS, COLOR_MODES, COMING_SOON, useViewer, type Settings } from "./store";
import { Slider } from "@/components/ui/slider";
import { cn } from "@/lib/utils";

// Achromatic by design language: no accent color, active = white/10. Color belongs to data.

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="space-y-1.5">
      <div className="px-1 text-[10px] font-medium uppercase tracking-wider text-white/40">{title}</div>
      {children}
    </div>
  );
}

function ToggleRow({ label, k, kbd }: { label: string; k: keyof Settings; kbd?: string }) {
  const v = useViewer((s) => s[k]) as boolean;
  const toggle = useViewer((s) => s.toggle);
  return (
    <button
      onClick={() => toggle(k)}
      className={cn(
        "flex w-full items-center justify-between rounded-lg px-2.5 py-1.5 text-xs transition-colors",
        v ? "bg-white/10 text-white" : "text-white/50 hover:bg-white/[0.06] hover:text-white/80",
      )}
    >
      <span>{label}</span>
      {kbd && <kbd className="font-mono text-[10px] text-white/30">{kbd}</kbd>}
    </button>
  );
}

function Seg<T extends string>({ value, opts, on }: { value: T; opts: { id: T; label: string; enabled?: boolean }[]; on: (v: T) => void }) {
  return (
    <div className="flex flex-wrap gap-1 px-1">
      {opts.map((o) => {
        const disabled = o.enabled === false;
        const active = value === o.id;
        return (
          <button
            key={o.id}
            disabled={disabled}
            onClick={() => on(o.id)}
            title={disabled ? "coming soon (needs export)" : undefined}
            className={cn(
              "rounded-lg px-2.5 py-1 text-[11px] transition-colors",
              active
                ? "bg-white/10 text-white"
                : disabled
                  ? "cursor-not-allowed text-white/20"
                  : "text-white/50 hover:bg-white/[0.06] hover:text-white/80",
            )}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

function SliderRow({ label, k, min, max, step }: { label: string; k: "voxelScale" | "voxelOpacity"; min: number; max: number; step: number }) {
  const v = useViewer((s) => s[k]);
  const set = useViewer((s) => s.set);
  return (
    <div className="space-y-1.5 px-1">
      <div className="flex justify-between text-[10px] text-white/40">
        <span>{label}</span>
        <span className="font-mono text-white/60">{v.toFixed(2)}</span>
      </div>
      <Slider min={min} max={max} step={step} value={[v]} onValueChange={(val) => set(k, Array.isArray(val) ? val[0] : val)} />
    </div>
  );
}

export function ControlPanel() {
  const set = useViewer((s) => s.set);
  const applyPreset = useViewer((s) => s.applyPreset);
  const colorMode = useViewer((s) => s.colorMode);
  const projection = useViewer((s) => s.projection);
  const voxelShape = useViewer((s) => s.voxelShape);

  return (
    <div className="space-y-4">
      <Section title="Show">
        <ToggleRow label="Occupancy voxels" k="showVoxels" kbd="O" />
        <ToggleRow label="Ego vehicle" k="showEgo" kbd="E" />
        <ToggleRow label="Heading" k="showForward" />
        <ToggleRow label="Ground grid" k="showGrid" kbd="G" />
        <ToggleRow label="Reachable free-space" k="showReachable" />
        <ToggleRow label="Wireframe" k="wireframe" />
        <ToggleRow label="Stats (FPS)" k="showStats" />
      </Section>

      <Section title="Color by">
        <Seg value={colorMode} opts={COLOR_MODES} on={(v) => set("colorMode", v)} />
      </Section>

      <Section title="Voxel">
        <Seg value={voxelShape} opts={[{ id: "cube", label: "Cube" }, { id: "sphere", label: "Sphere" }]} on={(v) => set("voxelShape", v)} />
        <SliderRow label="Size" k="voxelScale" min={0.3} max={1} step={0.05} />
        <SliderRow label="Opacity" k="voxelOpacity" min={0.2} max={1} step={0.05} />
      </Section>

      <Section title="Camera">
        <div className="flex flex-wrap gap-1 px-1">
          {CAMERA_PRESETS.map((p) => (
            <button
              key={p.id}
              onClick={() => applyPreset(p.id)}
              className="rounded-lg px-2.5 py-1 text-[11px] text-white/50 transition-colors hover:bg-white/[0.06] hover:text-white/80"
            >
              {p.label}
            </button>
          ))}
        </div>
        <Seg value={projection} opts={[{ id: "perspective", label: "Persp" }, { id: "orthographic", label: "Ortho" }]} on={(v) => set("projection", v)} />
      </Section>

      <Section title="Coming soon">
        {COMING_SOON.flatMap((g) => g.items).map((it) => (
          <div key={it} className="flex w-full items-center justify-between rounded-lg px-2.5 py-1.5 text-xs text-white/20">
            <span>{it}</span>
            <span className="text-[9px] uppercase tracking-wider">soon</span>
          </div>
        ))}
      </Section>
    </div>
  );
}

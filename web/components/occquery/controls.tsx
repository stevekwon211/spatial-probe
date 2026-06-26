"use client";

import type { ReactNode } from "react";
import { useTranslations } from "next-intl";
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
  const t = useTranslations();
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
            title={disabled ? t("occquery.controls.comingSoonTitle") : undefined}
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

// Maps the stable English items in store.ts COMING_SOON to message keys (the list itself stays in
// the store as the source of truth for which controls are gated; only the display string is localized).
const COMING_SOON_ITEM_KEY: Record<string, string> = {
  "Measurement tool": "measurementTool",
  "Occupancy-flow vectors": "occupancyFlowVectors",
  "Natural-language search": "nlSearch",
  "Find similar scenes": "findSimilar",
  "GT vs predicted diff": "gtVsPredicted",
  "Embeddings (UMAP)": "embeddings",
  "HD-map underlay": "hdMap",
  "Multi-sensor panes": "multiSensor",
  "SAM2 annotate": "sam2",
};

export function ControlPanel() {
  const t = useTranslations();
  const set = useViewer((s) => s.set);
  const applyPreset = useViewer((s) => s.applyPreset);
  const colorMode = useViewer((s) => s.colorMode);
  const projection = useViewer((s) => s.projection);
  const voxelShape = useViewer((s) => s.voxelShape);

  const colorOpts = COLOR_MODES.map((m) => ({ ...m, label: t(`occquery.colorModes.${m.id}`) }));

  return (
    <div className="space-y-4">
      <Section title={t("occquery.controls.show")}>
        <ToggleRow label={t("occquery.controls.occupancyVoxels")} k="showVoxels" kbd="O" />
        <ToggleRow label={t("occquery.controls.egoVehicle")} k="showEgo" kbd="E" />
        <ToggleRow label={t("occquery.controls.heading")} k="showForward" />
        <ToggleRow label={t("occquery.controls.groundGrid")} k="showGrid" kbd="G" />
        <ToggleRow label={t("occquery.controls.reachableFreeSpace")} k="showReachable" />
        <ToggleRow label={t("occquery.controls.wireframe")} k="wireframe" />
        <ToggleRow label={t("occquery.controls.statsFps")} k="showStats" />
      </Section>

      <Section title={t("occquery.controls.colorBy")}>
        <Seg value={colorMode} opts={colorOpts} on={(v) => set("colorMode", v)} />
      </Section>

      <Section title={t("occquery.controls.voxel")}>
        <Seg value={voxelShape} opts={[{ id: "cube", label: t("occquery.voxelShape.cube") }, { id: "sphere", label: t("occquery.voxelShape.sphere") }]} on={(v) => set("voxelShape", v)} />
        <SliderRow label={t("occquery.controls.size")} k="voxelScale" min={0.3} max={1} step={0.05} />
        <SliderRow label={t("occquery.controls.opacity")} k="voxelOpacity" min={0.2} max={1} step={0.05} />
      </Section>

      <Section title={t("occquery.controls.camera")}>
        <div className="flex flex-wrap gap-1 px-1">
          {CAMERA_PRESETS.map((p) => (
            <button
              key={p.id}
              onClick={() => applyPreset(p.id)}
              className="rounded-lg px-2.5 py-1 text-[11px] text-white/50 transition-colors hover:bg-white/[0.06] hover:text-white/80"
            >
              {t(`occquery.cameraPresets.${p.id}`)}
            </button>
          ))}
        </div>
        <Seg value={projection} opts={[{ id: "perspective", label: t("occquery.projection.perspective") }, { id: "orthographic", label: t("occquery.projection.orthographic") }]} on={(v) => set("projection", v)} />
      </Section>

      <Section title={t("occquery.controls.comingSoon")}>
        {COMING_SOON.flatMap((g) => g.items).map((it) => (
          <div key={it} className="flex w-full items-center justify-between rounded-lg px-2.5 py-1.5 text-xs text-white/20">
            <span>{COMING_SOON_ITEM_KEY[it] ? t(`occquery.comingSoonItems.${COMING_SOON_ITEM_KEY[it]}`) : it}</span>
            <span className="text-[9px] uppercase tracking-wider">{t("occquery.controls.soon")}</span>
          </div>
        ))}
      </Section>
    </div>
  );
}

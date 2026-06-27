"use client";

// ---------------------------------------------------------------------------
// Aletheon interactive query panel — the new HERO interaction on /aletheon.
//
// 100% CLIENT-SIDE: the RSC (app/aletheon/page.tsx) passes the already-bundled,
// precomputed demo data in as props (serializable JSON). There is NO fetch, NO
// backend, NO new data file — selecting a query just filters data already in the
// React tree. That is what keeps /aletheon statically prerenderable + Vercel-static.
//
// HONESTY (do not overclaim): this is "precomputed-interactive" — canned queries
// over a bounded AV2 demo subset, NOT a live engine. The copy says so out loud.
// ---------------------------------------------------------------------------

import { useMemo, useState } from "react";
import { Search, X, ArrowRight } from "lucide-react";
import { useTranslations } from "next-intl";
import { GlassPanel } from "@/components/occquery/glass";
import { Metric } from "@/components/landing/metric";
import { cn } from "@/lib/utils";

// --- the subset of the bundled shapes this panel reads (props from the RSC) ---
export type SignatureKey = "missed_detection" | "path_blocked_no_box" | "box_in_free";

export type Cluster = {
  name: string;
  size: number;
  range_bin_m: number;
  category: string;
};
export type PanelSignature = {
  signature: string;
  description: string;
  honesty: string;
  n_logs: number;
  stride: number;
  n_frames_scanned: number;
  n_candidates: number;
  n_clusters: number;
  mean_forward_range_m: number;
  scope_note: string;
  top_clusters: Cluster[];
};
export type PanelFrame = {
  signature: string;
  log: string;
  frame_index: number;
  n_detected?: number;
  n_missed?: number;
  block_forward_m?: number;
  png: string;
  caption: string;
};

export type QueryPanelProps = {
  signatures: Record<SignatureKey, PanelSignature>;
  frames: PanelFrame[];
  honestCaveat: string;
};

// Each preset maps a natural-language query to one signature. `keywords` drives
// the optional free-text matcher (simple client-side scoring — see matchPreset).
// `external` decides whether the honesty tag reads as an independent check.
const PRESETS: { id: SignatureKey; external: boolean; keywords: string[] }[] = [
  {
    id: "missed_detection",
    external: true,
    keywords: ["pedestrian", "missed", "model", "detector", "miss", "person", "object", "see", "saw", "recall", "vehicle", "car"],
  },
  {
    id: "path_blocked_no_box",
    external: true,
    keywords: ["path", "blocked", "block", "no box", "unboxed", "obstacle", "free", "occupancy", "explain", "tracked", "drive", "driven"],
  },
  {
    id: "box_in_free",
    external: false,
    keywords: ["box", "free", "occupancy", "missed", "lidar", "marked", "empty", "footprint", "cuboid"],
  },
];

const ORDER: SignatureKey[] = ["missed_detection", "path_blocked_no_box", "box_in_free"];

// Lowercased keyword overlap score. Whole-query keyword hits beat single-token hits,
// so "path blocked but no box explains it" lands on path_blocked_no_box, not box_in_free.
function matchPreset(text: string): SignatureKey | null {
  const q = text.toLowerCase().trim();
  if (!q) return null;
  let best: SignatureKey | null = null;
  let bestScore = 0;
  for (const p of PRESETS) {
    let score = 0;
    for (const k of p.keywords) {
      if (q.includes(k)) score += k.includes(" ") ? 3 : 1; // multi-word phrase is a stronger signal
    }
    if (score > bestScore) {
      bestScore = score;
      best = p.id;
    }
  }
  return bestScore > 0 ? best : null;
}

export function PrismQueryPanel({ signatures, frames, honestCaveat }: QueryPanelProps) {
  const t = useTranslations("aletheon.query");
  const [active, setActive] = useState<SignatureKey>("missed_detection");
  const [text, setText] = useState("");
  const [matchedFrom, setMatchedFrom] = useState<"chip" | "text" | null>(null);
  const [lightbox, setLightbox] = useState<PanelFrame | null>(null);

  // Live client-side "query": filter the precomputed data for the active signature.
  // Pure derivation from props + active state — no effects, no fetch.
  const sig = signatures[active];
  const result = useMemo(() => {
    const sigFrames = frames.filter((f) => f.signature === active);
    const topClusters = [...sig.top_clusters].sort((a, b) => b.size - a.size).slice(0, 4);
    return { sigFrames, topClusters };
  }, [active, frames, sig]);

  const external = PRESETS.find((p) => p.id === active)?.external ?? false;

  const runText = () => {
    const hit = matchPreset(text);
    if (hit) {
      setActive(hit);
      setMatchedFrom("text");
    }
  };

  return (
    <div>
      {/* prompt-style header — looks like a tool, names the demo honestly */}
      <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
        <div>
          <div
            className="mb-2 font-mono uppercase"
            style={{ fontSize: "var(--fs-mono-sm)", letterSpacing: "var(--ls-wide)", color: "var(--faint)" }}
          >
            {t("label")}
          </div>
          <h2
            style={{ fontSize: "var(--fs-title)", fontWeight: "var(--w-title)", letterSpacing: "var(--ls-tight)", color: "var(--text)" }}
          >
            {t("heading")}
          </h2>
        </div>
        <span
          className="rounded-full border px-2.5 py-1 font-mono uppercase"
          style={{ fontSize: "var(--fs-mono-xs)", letterSpacing: "var(--ls-meta)", borderColor: "var(--hairline)", color: "var(--dim)" }}
        >
          {t("demoBadge")}
        </span>
      </div>

      <GlassPanel className="p-5 sm:p-6">
        {/* free-text matcher — a nicety; matches typed text to the nearest preset */}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            runText();
          }}
          className="flex items-center gap-2 rounded-xl border bg-black/40 px-3 py-2"
          style={{ borderColor: "var(--border-media)" }}
        >
          <Search className="size-4 shrink-0" style={{ color: "var(--dim)" }} aria-hidden />
          <input
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder={t("inputPlaceholder")}
            aria-label={t("inputAria")}
            className="w-full bg-transparent font-mono outline-none placeholder:opacity-60"
            style={{ fontSize: "var(--fs-mono-sm)", color: "var(--text)" }}
          />
          <button
            type="submit"
            className="inline-flex shrink-0 items-center gap-1 rounded-lg bg-white px-3 py-1.5 text-[12px] font-medium text-black transition-opacity hover:opacity-90 disabled:opacity-40"
            disabled={matchPreset(text) === null}
          >
            {t("findButton")} <ArrowRight className="size-3.5" />
          </button>
        </form>

        {/* preset chips — the must-have entry point; one per signature */}
        <div className="mt-3.5">
          <div
            className="mb-2 font-mono uppercase"
            style={{ fontSize: "var(--fs-mono-xs)", letterSpacing: "var(--ls-meta)", color: "var(--dim)" }}
          >
            {t("presetsLabel")}
          </div>
          <div className="flex flex-wrap gap-2">
            {PRESETS.map((p) => {
              const isActive = active === p.id;
              return (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => {
                    setActive(p.id);
                    setMatchedFrom("chip");
                  }}
                  aria-pressed={isActive}
                  className={cn(
                    "rounded-full border px-3.5 py-1.5 text-left font-mono transition-colors",
                    isActive ? "bg-white/[0.08]" : "bg-transparent hover:bg-white/[0.03]",
                  )}
                  style={{
                    fontSize: "var(--fs-mono-sm)",
                    borderColor: isActive ? "color-mix(in srgb,#2bb0a4 45%,transparent)" : "var(--hairline)",
                    color: isActive ? "var(--text)" : "var(--muted)",
                  }}
                >
                  {t(`presets.${p.id}`)}
                </button>
              );
            })}
          </div>
        </div>
      </GlassPanel>

      {/* RESULT CARD — updates live on query change (client filter over props) */}
      <div className="mt-4">
        <div
          className="mb-2 flex items-center gap-2 font-mono"
          style={{ fontSize: "var(--fs-mono-xs)", color: "var(--dim)" }}
        >
          <span style={{ color: "var(--data-text)" }}>{"›"}</span>
          <span>aletheon find &quot;{sig.signature}&quot;</span>
          {matchedFrom === "text" && (
            <span style={{ color: "var(--faint)" }}>{t("matchedNote")}</span>
          )}
        </div>

        <GlassPanel className="overflow-hidden">
          {/* result header: signature + honesty kind tag (the differentiator) */}
          <div
            className="flex flex-wrap items-center gap-2.5 border-b px-5 py-3.5"
            style={{ borderColor: "var(--hairline)" }}
          >
            <span className="font-mono" style={{ fontSize: "var(--fs-mono-sm)", color: "var(--text)" }}>
              {sig.signature}
            </span>
            <span
              className="rounded-full border px-2 py-0.5 font-mono uppercase"
              style={{
                fontSize: "var(--fs-mono-xs)",
                letterSpacing: "var(--ls-meta)",
                borderColor: external ? "color-mix(in srgb,#2bb0a4 45%,transparent)" : "var(--hairline)",
                color: external ? "var(--data-text)" : "var(--faint)",
              }}
            >
              {t(`kind.${active}`)}
            </span>
            <span
              className="ml-auto font-mono"
              style={{ fontSize: "var(--fs-mono-xs)", color: external ? "var(--muted)" : "var(--dim)" }}
            >
              {external ? t("independentCheck") : t("consistencyOnly")}
            </span>
          </div>

          <div className="px-5 py-4">
            <p className="mb-4" style={{ fontSize: "var(--fs-prose-sm)", lineHeight: "var(--lh-prose)", color: "var(--muted)" }}>
              {sig.description}
            </p>

            {/* count metrics — the "n found" of running the query */}
            <div className="grid grid-cols-3 gap-3">
              <ResultMetric value={sig.n_candidates.toLocaleString()} label={t("metricCandidates")} tone="data" />
              <ResultMetric value={sig.n_clusters.toLocaleString()} label={t("metricClusters")} />
              <ResultMetric value={sig.n_logs.toLocaleString()} label={t("metricLogs")} />
            </div>

            {/* top clusters by size — range bin + category, teal bar = relative size */}
            <div className="mt-5">
              <div
                className="mb-2 font-mono uppercase"
                style={{ fontSize: "var(--fs-mono-xs)", letterSpacing: "var(--ls-meta)", color: "var(--dim)" }}
              >
                {t("topClusters")}
              </div>
              <div className="grid gap-1.5">
                {result.topClusters.map((c) => {
                  const max = result.topClusters[0]?.size || 1;
                  const pct = Math.max(6, Math.round((c.size / max) * 100));
                  return (
                    <div
                      key={c.name}
                      className="flex items-center gap-3 rounded-lg border px-3 py-2"
                      style={{ borderColor: "var(--hairline)" }}
                    >
                      <span className="font-mono" style={{ fontSize: "var(--fs-mono-xs)", color: "var(--faint)", minWidth: "4.5rem" }}>
                        ~{c.range_bin_m.toFixed(0)} m
                      </span>
                      <span className="font-mono" style={{ fontSize: "var(--fs-mono-xs)", color: "var(--muted)", minWidth: "5.5rem" }}>
                        {c.category}
                      </span>
                      <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-white/[0.06]">
                        <span className="block h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: "var(--data)" }} />
                      </span>
                      <span className="font-mono" style={{ fontSize: "var(--fs-mono-xs)", color: "var(--dim)", minWidth: "2.5rem", textAlign: "right" }}>
                        <Metric>{c.size}</Metric>
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* matching annotated frames — click → lightbox + caption */}
            <div className="mt-5">
              <div
                className="mb-2 font-mono uppercase"
                style={{ fontSize: "var(--fs-mono-xs)", letterSpacing: "var(--ls-meta)", color: "var(--dim)" }}
              >
                {t("matchingFrames", { n: result.sigFrames.length })}
              </div>
              {result.sigFrames.length > 0 ? (
                <div className="grid gap-3 sm:grid-cols-3">
                  {result.sigFrames.map((f) => (
                    <button
                      key={`${f.log}-${f.frame_index}`}
                      type="button"
                      onClick={() => setLightbox(f)}
                      className="group overflow-hidden rounded-xl border bg-neutral-950 text-left transition-colors hover:border-white/25"
                      style={{ borderColor: "var(--border-media)" }}
                    >
                      <div
                        className="flex items-center gap-2 border-b border-[var(--hairline)] bg-white/[0.02] px-2.5 py-1.5 font-mono"
                        style={{ fontSize: "var(--fs-mono-xs)", color: "var(--dim)" }}
                      >
                        {f.n_missed != null ? (
                          <>
                            <span className="text-red-400">MISSED {f.n_missed}</span>
                            <span>/ saw {f.n_detected}</span>
                          </>
                        ) : f.block_forward_m != null ? (
                          <span style={{ color: "var(--data-text)" }}>BLOCK {f.block_forward_m.toFixed(1)} m</span>
                        ) : null}
                        <span className="ml-auto">{f.log.slice(0, 8)} · f{f.frame_index}</span>
                      </div>
                      {/* burned-in annotation; static asset under /data (CDN) */}
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={`/data/${f.png}`} alt={f.caption} loading="lazy" className="block w-full bg-black transition-opacity group-hover:opacity-90" />
                    </button>
                  ))}
                </div>
              ) : (
                <p
                  className="rounded-lg border border-dashed px-3 py-3 font-mono"
                  style={{ fontSize: "var(--fs-mono-xs)", lineHeight: "var(--lh-prose)", borderColor: "var(--hairline)", color: "var(--dim)" }}
                >
                  {t("noFrames")}
                </p>
              )}
            </div>

            {/* VERBATIM honesty tag — always visible, the per-result trust label */}
            <div
              className="mt-5 rounded-xl border p-4"
              style={{ borderColor: "var(--hairline)", backgroundColor: "rgba(255,255,255,0.015)" }}
            >
              <div
                className="mb-1.5 font-mono uppercase"
                style={{ fontSize: "var(--fs-mono-xs)", letterSpacing: "var(--ls-meta)", color: external ? "var(--data-text)" : "var(--dim)" }}
              >
                {t("honestyTag")}
              </div>
              <p className="font-mono" style={{ fontSize: "var(--fs-mono-xs)", lineHeight: "var(--lh-prose)", color: "var(--muted)" }}>
                {sig.honesty}
              </p>
            </div>

            {/* demo scope — stated honestly, per the no-overclaim rule */}
            <p
              className="mt-3 font-mono"
              style={{ fontSize: "var(--fs-mono-xs)", lineHeight: "var(--lh-prose)", color: "var(--dim)" }}
            >
              {t("scopePrefix")} {sig.scope_note} {t("scopeSep")} {honestCaveat}
            </p>
          </div>
        </GlassPanel>
      </div>

      {/* lightbox — larger frame + verbatim caption; achromatic overlay */}
      {lightbox && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label={lightbox.caption}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 p-4 backdrop-blur-sm sm:p-8"
          onClick={() => setLightbox(null)}
        >
          <div
            className="relative max-h-full w-full max-w-3xl overflow-auto rounded-2xl border bg-neutral-950"
            style={{ borderColor: "var(--border-media)" }}
            onClick={(e) => e.stopPropagation()}
          >
            <div
              className="sticky top-0 flex items-center gap-2 border-b bg-neutral-950/90 px-4 py-2.5 font-mono backdrop-blur"
              style={{ fontSize: "var(--fs-mono-xs)", color: "var(--dim)", borderColor: "var(--hairline)" }}
            >
              <span style={{ color: "var(--text)" }}>{lightbox.signature}</span>
              <span className="ml-auto">{lightbox.log.slice(0, 8)} · f{lightbox.frame_index}</span>
              <button
                type="button"
                onClick={() => setLightbox(null)}
                aria-label={t("closeAria")}
                className="ml-1 rounded-md p-1 transition-colors hover:bg-white/10"
              >
                <X className="size-4" style={{ color: "var(--muted)" }} />
              </button>
            </div>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={`/data/${lightbox.png}`} alt={lightbox.caption} className="block w-full bg-black" />
            <p
              className="px-4 py-3 font-mono"
              style={{ fontSize: "var(--fs-mono-xs)", lineHeight: "var(--lh-prose)", color: "var(--muted)" }}
            >
              {lightbox.caption}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

function ResultMetric({ value, label, tone }: { value: string; label: string; tone?: "data" }) {
  return (
    <div className="rounded-xl border px-3 py-3 text-center" style={{ borderColor: "var(--hairline)" }}>
      <div className="text-2xl">
        <Metric tone={tone === "data" ? "data" : "muted"}>{value}</Metric>
      </div>
      <div className="mt-1 font-mono uppercase" style={{ fontSize: "9.5px", letterSpacing: "var(--ls-meta)", color: "var(--dim)" }}>
        {label}
      </div>
    </div>
  );
}

import { readFileSync } from "node:fs";
import { join } from "node:path";
import Link from "next/link";
import { ArrowRight, ChevronLeft } from "lucide-react";
import { GlassPanel } from "@/components/occquery/glass";
import { Metric } from "@/components/landing/metric";
import { SectionLabel } from "@/components/landing/section-label";
import { StatusChip } from "@/components/ui/status-chip";

export const metadata = {
  title: "PRISM — find the scenes where the model is wrong",
  description:
    "An open spatial-failure-search engine whose answers carry a label-free correctness check. Find the frames that break a perception model, in a few lines.",
};

// ---------------------------------------------------------------------------
// Data shapes — only the fields this page reads, typed from the JSON it loads.
// The files live under web/public/data/ (also served at /data/*); we read them
// from disk server-side at render so the page is a static RSC with no client fetch.
// ---------------------------------------------------------------------------
type Honesty = string;
type SignatureScope = { logs: string[]; stride: number; note: string };
type Signature = {
  signature: string;
  description: string;
  honesty: Honesty;
  n_logs: number;
  stride: number;
  n_frames_scanned: number;
  n_candidates: number;
  n_clusters: number;
  mean_forward_range_m: number;
  scope_note: string;
};
type FailureCatalogue = {
  data_root: string;
  demo_scope: Record<string, SignatureScope> & { honest_caveat: string };
  signatures: Record<"missed_detection" | "path_blocked_no_box" | "box_in_free", Signature>;
};

type ManifestFrame = {
  signature: string;
  log: string;
  frame_index: number;
  n_detected?: number;
  n_missed?: number;
  block_forward_m?: number;
  png: string;
  caption: string;
};
type FramesManifest = { frames: ManifestFrame[] };

type FamilyExpr = {
  n_queries: number;
  occupancy_coverage_pct: number;
  box_only_coverage_pct: number;
  gap_pct: number;
};
type H3bExpressivity = {
  leg1_expressivity: {
    per_family: Record<string, FamilyExpr>;
    free_space_families_only: FamilyExpr & { families: string[] };
  };
  leg2_fp_denotation: {
    headline_free_path_agreement: number;
    headline_free_path_evals: { denotes_free: number; n: number };
    voxel_ribbon_fp: { true_fp_mean: number };
    box_only_free_space_denotation: string;
  };
  honest_bound: string;
};

type Oracle = {
  name: string;
  verdict: string;
  anchors: string;
  headline: string;
  external: boolean;
};
type OracleStatus = { note: string; oracles: Oracle[] };

function readData<T>(file: string): T {
  return JSON.parse(readFileSync(join(process.cwd(), "public", "data", file), "utf8")) as T;
}

// ---------------------------------------------------------------------------
// Layout tokens — reused verbatim from app/page.tsx so the page matches the site.
// ---------------------------------------------------------------------------
const WRAP = "mx-auto w-full px-[clamp(20px,5vw,40px)]";
const SECTION = { marginTop: "clamp(56px, 8vh, 88px)" } as const;

// Maps each demo signature to the independent check that backs it (section 4).
const HONESTY_KIND: Record<string, { label: string; external: boolean }> = {
  missed_detection: { label: "model-eval", external: true },
  path_blocked_no_box: { label: "external-fp", external: true },
  box_in_free: { label: "consistency-only", external: false },
};

export default function PrismPage() {
  const catalogue = readData<FailureCatalogue>("failure_catalogue.json");
  const manifest = readData<FramesManifest>("frames_manifest.json");
  const h3b = readData<H3bExpressivity>("h3b_expressivity.json");
  const oracleStatus = readData<OracleStatus>("oracle_status.json");

  const missedFrames = manifest.frames.filter((f) => f.signature === "missed_detection");
  const bevFrame = manifest.frames.find((f) => f.signature === "path_blocked_no_box");
  const freeSpace = h3b.leg1_expressivity.free_space_families_only;
  const leg2 = h3b.leg2_fp_denotation;
  const sig = catalogue.signatures;

  return (
    <div className="landing min-h-screen bg-[#080808] font-sans text-white antialiased">
      {/* nav — same chrome as the landing header */}
      <header className="sticky top-0 z-30 border-b border-[var(--hairline)] bg-[#080808]/70 backdrop-blur-xl">
        <div className={`${WRAP} flex items-center justify-between py-3.5`} style={{ maxWidth: "var(--col)" }}>
          <Link href="/" className="flex items-center gap-1.5 text-sm font-medium tracking-tight text-white/90 transition-colors hover:text-white">
            <ChevronLeft className="size-3.5 text-white/40" /> spatial-probe
          </Link>
          <nav className="flex items-center gap-1" style={{ fontSize: "var(--fs-mono-md)" }}>
            <Link href="/overview" className="ink-link rounded-md px-3 py-1.5 font-mono" style={{ color: "var(--muted)" }}>Pipeline</Link>
            <Link href="/evidence" className="ink-link rounded-md px-3 py-1.5 font-mono" style={{ color: "var(--muted)" }}>Evidence</Link>
            <Link href="/occquery" className="ml-1 inline-flex items-center gap-1.5 rounded-lg bg-white px-3.5 py-1.5 text-[13px] font-medium text-black transition-opacity hover:opacity-90">Open Explorer <ArrowRight className="size-3.5" /></Link>
          </nav>
        </div>
      </header>

      {/* 1 — hero */}
      <section className="relative overflow-hidden">
        <div aria-hidden className="pointer-events-none absolute inset-0 opacity-[0.18] [mask-image:radial-gradient(ellipse_at_top,black,transparent_70%)]"
          style={{ backgroundImage: "linear-gradient(rgba(255,255,255,.06) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.06) 1px,transparent 1px)", backgroundSize: "28px 28px" }} />
        <div className={`${WRAP} relative pt-20 pb-10 text-center sm:pt-28`} style={{ maxWidth: "var(--col)" }}>
          <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-[var(--hairline)] bg-white/[0.03] px-3 py-1 font-mono uppercase" style={{ fontSize: "var(--fs-mono-xs)", letterSpacing: "var(--ls-wide)", color: "var(--faint)" }}>
            <span className="size-1.5 rounded-full" style={{ backgroundColor: "var(--data)" }} /> PRISM
          </div>
          <h1 className="text-balance" style={{ fontSize: "var(--fs-display)", fontWeight: "var(--w-display)", lineHeight: "var(--lh-display)", letterSpacing: "var(--ls-tight)", color: "var(--text)" }}>
            Find the scenes where the model is wrong, in a few lines.
          </h1>
          <p className="mx-auto mt-6 text-balance" style={{ maxWidth: "var(--measure)", fontSize: "var(--fs-lead)", lineHeight: "var(--lh-prose)", color: "var(--muted)" }}>
            An open spatial-failure-search engine whose answers carry a label-free correctness check.
            You query for a failure mode; every hit comes back tagged with which independent check
            backs it, and how far to trust it.
          </p>
          <div className="mt-8 flex items-center justify-center gap-5">
            <a href="#frames" className="inline-flex items-center gap-1.5 rounded-lg bg-white px-5 py-2.5 text-sm font-medium text-black transition-opacity hover:opacity-90">See the failures <ArrowRight className="size-4" /></a>
            <Link href="/evidence" className="ink-link inline-flex items-center gap-1 text-sm" style={{ color: "var(--muted)" }}>See the evidence <ArrowRight className="size-3.5" /></Link>
          </div>
        </div>
      </section>

      {/* 2 — the wow: model-eval failure frames */}
      <section id="frames" className={WRAP} style={{ ...SECTION, maxWidth: "var(--col-wide)" }}>
        <SectionLabel>the wow · model-eval</SectionLabel>
        <div className="mb-1.5 flex items-baseline justify-between gap-4">
          <h2 style={{ fontSize: "var(--fs-title)", fontWeight: "var(--w-title)", letterSpacing: "var(--ls-tight)", color: "var(--text)" }}>Frames that break a detector</h2>
          <span className="font-mono" style={{ fontSize: "var(--fs-mono-sm)", color: "var(--faint)" }}>{sig.missed_detection.n_candidates} candidates · {sig.missed_detection.n_clusters} clusters</span>
        </div>
        <p className="mb-5" style={{ fontSize: "var(--fs-body-sm)", color: "var(--faint)" }}>
          Each scene scored against AV2 ground-truth boxes. <span style={{ color: "var(--data-text)" }}>Green</span> = the
          detector saw it. <span className="text-red-400">Red</span> = ground-truth object the model missed.
        </p>
        <div className="grid gap-4 sm:grid-cols-3">
          {missedFrames.map((f) => (
            <figure key={`${f.log}-${f.frame_index}`} className="overflow-hidden rounded-2xl border bg-neutral-950" style={{ borderColor: "var(--border-media)" }}>
              <div className="flex items-center gap-2 border-b border-[var(--hairline)] bg-white/[0.02] px-3 py-2 font-mono" style={{ fontSize: "var(--fs-mono-xs)", color: "var(--dim)" }}>
                <span className="text-red-400">MISSED {f.n_missed}</span>
                <span>/ saw {f.n_detected}</span>
                <span className="ml-auto">{f.log.slice(0, 8)} · f{f.frame_index}</span>
              </div>
              {/* burned-in green=saw / red=missed boxes; static asset under /data */}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={`/data/${f.png}`} alt={f.caption} loading="lazy" className="block w-full bg-black" />
              <figcaption className="px-3 py-2.5" style={{ fontSize: "var(--fs-prose-sm)", lineHeight: "var(--lh-prose)", color: "var(--muted)" }}>
                {/* the headline missed objects, parsed out of the verbatim caption */}
                <span className="font-mono" style={{ fontSize: "var(--fs-mono-xs)", color: "var(--faint)" }}>
                  detector saw {f.n_detected}, MISSED {f.n_missed}: {missedList(f.caption)}
                </span>
              </figcaption>
            </figure>
          ))}
        </div>
        <p className="mt-3 font-mono" style={{ fontSize: "var(--fs-mono-xs)", color: "var(--dim)", lineHeight: "var(--lh-prose)" }}>
          Honest: COCO-YOLOv8n detector (NOT trained on AV2) → cross-distribution recall, coarse
          car/bus/truck→vehicle class map, class-agnostic match. A miss = the detector&apos;s recall
          failure, not an occupancy claim.
        </p>
      </section>

      {/* 3 — the occupancy differentiator (H1) */}
      <section className={WRAP} style={{ ...SECTION, maxWidth: "var(--col)" }}>
        <SectionLabel>the differentiator · H1</SectionLabel>
        <h2 style={{ fontSize: "var(--fs-title)", fontWeight: "var(--w-title)", letterSpacing: "var(--ls-tight)", color: "var(--text)" }}>Occupancy sees what box-only structurally can&apos;t</h2>
        <p className="mb-6 mt-1.5" style={{ fontSize: "var(--fs-body-sm)", color: "var(--faint)" }}>
          One scene in the demo where the path is blocked but no tracked box explains it — the failure mode a box-only language has no words for.
        </p>
        <div className="grid items-start gap-5 sm:grid-cols-5">
          {bevFrame && (
            <figure className="overflow-hidden rounded-2xl border bg-neutral-950 sm:col-span-3" style={{ borderColor: "var(--border-media)" }}>
              <div className="flex items-center gap-2 border-b border-[var(--hairline)] bg-white/[0.02] px-3 py-2 font-mono" style={{ fontSize: "var(--fs-mono-xs)", color: "var(--dim)" }}>
                <span>path_blocked_no_box · BEV</span>
                <span className="ml-auto">{bevFrame.log.slice(0, 8)} · f{bevFrame.frame_index}</span>
              </div>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={`/data/${bevFrame.png}`} alt={bevFrame.caption} loading="lazy" className="block w-full bg-black" />
              <figcaption className="px-3 py-2.5" style={{ fontSize: "var(--fs-prose-sm)", lineHeight: "var(--lh-prose)", color: "var(--muted)" }}>
                Occupancy blocks the ego in-path band at{" "}
                <Metric tone="data">{bevFrame.block_forward_m?.toFixed(1)}&nbsp;m</Metric>, and no tracked box
                within 5&nbsp;m explains it.
              </figcaption>
            </figure>
          )}
          <div className="grid gap-3 sm:col-span-2">
            <div className="rounded-xl border bg-white/[0.03] p-5 text-center" style={{ borderColor: "var(--border-media)" }}>
              <div className="text-3xl"><Metric tone="data">{freeSpace.occupancy_coverage_pct.toFixed(0)}</Metric><span className="ml-0.5 text-base" style={{ color: "var(--faint)" }}>%</span></div>
              <div className="mt-1.5 text-[11px]" style={{ color: "var(--muted)" }}>occupancy</div>
              <div className="mt-0.5 font-mono text-[10px]" style={{ color: "var(--dim)" }}>free-space query coverage</div>
            </div>
            <div className="rounded-xl border bg-white/[0.015] p-5 text-center" style={{ borderColor: "var(--hairline)" }}>
              <div className="text-3xl"><Metric>{freeSpace.box_only_coverage_pct.toFixed(0)}</Metric><span className="ml-0.5 text-base" style={{ color: "var(--dim)" }}>%</span></div>
              <div className="mt-1.5 text-[11px]" style={{ color: "var(--faint)" }}>box-only</div>
              <div className="mt-0.5 font-mono text-[10px]" style={{ color: "var(--dim)" }}>free-space query coverage</div>
            </div>
            {/* +100pt expressivity gap bar */}
            <div className="rounded-xl border p-4" style={{ borderColor: "var(--hairline)" }}>
              <div className="mb-2 flex items-baseline justify-between font-mono" style={{ fontSize: "var(--fs-mono-xs)", color: "var(--dim)" }}>
                <span>expressivity gap</span>
                <span style={{ color: "var(--data-text)" }}>+{freeSpace.gap_pct.toFixed(0)}pt</span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-white/[0.06]">
                <div className="h-full rounded-full" style={{ width: "100%", backgroundColor: "var(--data)" }} />
              </div>
              <div className="mt-2 font-mono text-[10px]" style={{ color: "var(--dim)" }}>
                {freeSpace.n_queries} free-space queries · {freeSpace.families.join(", ")}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 4 — the honesty layer */}
      <section className={WRAP} style={{ ...SECTION, maxWidth: "var(--col)" }}>
        <SectionLabel>the honesty layer</SectionLabel>
        <h2 style={{ fontSize: "var(--fs-title)", fontWeight: "var(--w-title)", letterSpacing: "var(--ls-tight)", color: "var(--text)" }}>Every result says how much to trust it</h2>
        <p className="mb-6 mt-1.5" style={{ fontSize: "var(--fs-body-sm)", color: "var(--faint)" }}>
          Each failure type is tagged by which independent check backs it. This is the differentiator — no result is presented without its provenance.
        </p>
        <div className="grid gap-3">
          {(["missed_detection", "path_blocked_no_box", "box_in_free"] as const).map((key) => {
            const s = sig[key];
            const kind = HONESTY_KIND[key];
            return (
              <GlassPanel key={key} className="p-5">
                <div className="mb-2 flex flex-wrap items-center gap-2.5">
                  <span className="font-mono" style={{ fontSize: "var(--fs-mono-sm)", color: "var(--text)" }}>{s.signature}</span>
                  <span className="rounded-full border px-2 py-0.5 font-mono uppercase" style={{ fontSize: "var(--fs-mono-xs)", letterSpacing: "var(--ls-meta)", borderColor: kind.external ? "color-mix(in srgb,#2bb0a4 45%,transparent)" : "var(--hairline)", color: kind.external ? "var(--data-text)" : "var(--faint)" }}>
                    {kind.label}
                  </span>
                  <span className="ml-auto font-mono" style={{ fontSize: "var(--fs-mono-xs)", color: kind.external ? "var(--muted)" : "var(--dim)" }}>
                    {kind.external ? "independent check" : "consistency only"}
                  </span>
                </div>
                <p className="mb-2" style={{ fontSize: "var(--fs-prose-sm)", lineHeight: "var(--lh-prose)", color: "var(--muted)" }}>{s.description}</p>
                {/* verbatim honesty tag from src/prism/failure.py via failure_catalogue.json */}
                <p className="font-mono" style={{ fontSize: "var(--fs-mono-xs)", lineHeight: "var(--lh-prose)", color: "var(--dim)" }}>{s.honesty}</p>
              </GlassPanel>
            );
          })}
        </div>

        {/* oracle verdicts behind the tags */}
        <div className="mt-6">
          <div className="mb-2 font-mono uppercase" style={{ fontSize: "var(--fs-mono-xs)", letterSpacing: "var(--ls-label)", color: "var(--dim)" }}>oracles on file</div>
          <div className="grid gap-2.5 sm:grid-cols-3">
            {oracleStatus.oracles.map((o) => (
              <div key={o.name} className="rounded-xl border p-4" style={{ borderColor: o.external ? "color-mix(in srgb,#2bb0a4 35%,transparent)" : "var(--hairline)" }}>
                <div className="mb-1.5 flex items-center gap-2">
                  <span className="font-mono" style={{ fontSize: "var(--fs-mono-xs)", color: o.verdict === "INVALID-SCALE" ? "var(--dim)" : o.external ? "var(--data-text)" : "var(--faint)" }}>{o.verdict}</span>
                  {o.external && <span className="font-mono" style={{ fontSize: "9px", letterSpacing: "var(--ls-meta)", color: "var(--data-text)" }}>EXTERNAL</span>}
                </div>
                <div className="mb-1" style={{ fontSize: "var(--fs-prose-sm)", color: "var(--muted)" }}>{o.name}</div>
                <p className="font-mono" style={{ fontSize: "9.5px", lineHeight: 1.5, color: "var(--dim)" }}>{o.headline}</p>
              </div>
            ))}
          </div>
        </div>

        {/* the honest GPU-gated note — leg-2 FP-side claimed, BLOCKED-side recall is not */}
        <div className="mt-5 rounded-xl border p-4" style={{ borderColor: "var(--hairline)", backgroundColor: "rgba(255,255,255,0.015)" }}>
          <div className="mb-1.5 flex flex-wrap items-center gap-2 font-mono uppercase" style={{ fontSize: "var(--fs-mono-xs)", letterSpacing: "var(--ls-meta)", color: "var(--dim)" }}>
            <span>FP-side denotation</span>
            <span style={{ color: "var(--data-text)" }}>agreement {leg2.headline_free_path_agreement.toFixed(4)}</span>
            <span>· voxel-ribbon FP {leg2.voxel_ribbon_fp.true_fp_mean.toFixed(0)}</span>
            <span>· box-only {leg2.box_only_free_space_denotation}</span>
          </div>
          <p style={{ fontSize: "var(--fs-prose-sm)", lineHeight: "var(--lh-prose)", color: "var(--muted)" }}>{h3b.honest_bound}</p>
          <p className="mt-1.5 font-mono" style={{ fontSize: "var(--fs-mono-xs)", lineHeight: "var(--lh-prose)", color: "var(--dim)" }}>
            External (BLOCKED-side) unboxed-obstacle recall is GPU-gated — empirically closed solo/CPU this session. Not claimed.
          </p>
        </div>
      </section>

      {/* 5 — catalogue summary */}
      <section className={WRAP} style={{ ...SECTION, maxWidth: "var(--col)" }}>
        <SectionLabel>catalogue · demo scope</SectionLabel>
        <h2 style={{ fontSize: "var(--fs-title)", fontWeight: "var(--w-title)", letterSpacing: "var(--ls-tight)", color: "var(--text)" }}>What the demo actually swept</h2>
        <p className="mb-5 mt-1.5" style={{ fontSize: "var(--fs-body-sm)", color: "var(--faint)" }}>
          A bounded subset of the AV2 corpus, stated honestly — not a full-corpus sweep.
        </p>
        <div className="overflow-hidden rounded-2xl border" style={{ borderColor: "var(--border-media)" }}>
          <table className="w-full border-collapse text-left">
            <thead>
              <tr className="border-b border-[var(--hairline)] font-mono uppercase" style={{ fontSize: "var(--fs-mono-xs)", letterSpacing: "var(--ls-meta)", color: "var(--dim)" }}>
                <th className="px-4 py-2.5 font-normal">signature</th>
                <th className="px-3 py-2.5 text-right font-normal">candidates</th>
                <th className="px-3 py-2.5 text-right font-normal">clusters</th>
                <th className="px-3 py-2.5 text-right font-normal">logs</th>
                <th className="px-4 py-2.5 text-right font-normal">scope</th>
              </tr>
            </thead>
            <tbody>
              {(["missed_detection", "path_blocked_no_box", "box_in_free"] as const).map((key) => {
                const s = sig[key];
                return (
                  <tr key={key} className="border-b border-[var(--hairline)] last:border-0">
                    <td className="px-4 py-3 font-mono" style={{ fontSize: "var(--fs-mono-sm)", color: "var(--text)" }}>{s.signature}</td>
                    <td className="px-3 py-3 text-right"><Metric>{s.n_candidates.toLocaleString()}</Metric></td>
                    <td className="px-3 py-3 text-right"><Metric>{s.n_clusters}</Metric></td>
                    <td className="px-3 py-3 text-right"><Metric>{s.n_logs}</Metric></td>
                    <td className="px-4 py-3 text-right font-mono" style={{ fontSize: "var(--fs-mono-xs)", color: "var(--dim)" }}>
                      {s.n_logs} logs · stride {s.stride} · {s.n_frames_scanned.toLocaleString()} frames
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="mt-3 font-mono" style={{ fontSize: "var(--fs-mono-xs)", lineHeight: "var(--lh-prose)", color: "var(--dim)" }}>
          {catalogue.demo_scope.honest_caveat}
        </p>
      </section>

      {/* footer */}
      <footer className={`${WRAP} pb-12 pt-6`} style={{ ...SECTION, maxWidth: "var(--col)" }}>
        <div className="flex flex-col gap-3 pt-6 sm:flex-row sm:items-center sm:justify-between" style={{ borderTop: "1px solid var(--hairline)", fontSize: "var(--fs-prose-sm)", color: "var(--dim)" }}>
          <p style={{ maxWidth: "var(--measure)", lineHeight: "var(--lh-prose)" }}>
            Runs on Argoverse 2 sensor logs. No externally-validated recall is claimed where no independent oracle exists; the only external check on file is the traversal (FP-direction) oracle. Negatives are headlines here, not footnotes.
          </p>
          <div className="flex shrink-0 items-center gap-2.5">
            <StatusChip verdict="HOLDS" /> <span style={{ color: "var(--faint)" }}>H1 expressivity</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

// Pull the "FAILED to see ... (red): X; Y; Z" object list out of the verbatim caption
// so the headline shows the missed objects without restating the whole honesty paragraph.
function missedList(caption: string): string {
  const m = caption.match(/\(red\):\s*([^.]+)\./);
  return m ? m[1].trim() : "see caption";
}

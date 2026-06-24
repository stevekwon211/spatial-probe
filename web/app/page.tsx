import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { GlassPanel } from "@/components/occquery/glass";
import { StatusChip } from "@/components/ui/status-chip";
import { Metric } from "@/components/landing/metric";
import { SectionLabel } from "@/components/landing/section-label";
import { CapabilityRow } from "@/components/landing/capability-row";
import { ALL_MODULES, IN_PROGRESS, SHIPPED, THESIS, TOTAL } from "@/lib/pipeline";

export const metadata = {
  title: "spatial-probe — the spatial data engine for AV and robotics",
  description:
    "Query, measure, and trust 3D scene state by what it physically means — free-space, clearance, corridor width, dynamics — with every number you can re-run.",
};

function Logo({ className = "size-6" }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={`${className} shrink-0`} fill="none" aria-hidden="true">
      <g stroke="currentColor" strokeWidth={2}>
        <rect x="3" y="3" width="10" height="10" />
        <rect x="3" y="19" width="10" height="10" />
        <rect x="19" y="19" width="10" height="10" />
      </g>
      <rect x="18" y="2" width="12" height="12" fill="#2bb0a4" />
    </svg>
  );
}

// signature output, top-down (BEV): occupancy structure, the ego, and the free-space measurement
// (teal = the one live data value) that the box-only baseline (faint) is blind to. Deterministic SVG.
const RAMP = ["#4b6b63", "#5f7a52", "#7d8348", "#94795a", "#8a6450"];
function Structure({ x, y, cols, rows }: { x: number; y: number; cols: number; rows: number }) {
  const cells = [];
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      if ((r + c) % 7 === 0 && (r === 0 || r === rows - 1 || c === 0 || c === cols - 1)) continue;
      cells.push(<rect key={`${r}-${c}`} x={x + c * 11} y={y + r * 11} width={9.5} height={9.5} rx={1} fill={RAMP[r % RAMP.length]} opacity={0.92} />);
    }
  return <g>{cells}</g>;
}
function OccupancyDiagram() {
  return (
    <svg viewBox="0 0 800 380" className="block w-full" role="img" aria-label="Top-down occupancy: free-space measured where boxes are blind">
      <rect width="800" height="380" fill="#0b0b0b" />
      <g stroke="rgba(255,255,255,0.05)" strokeWidth="1">
        {Array.from({ length: 9 }, (_, i) => <line key={`v${i}`} x1={i * 100} y1={0} x2={i * 100} y2={380} />)}
        {Array.from({ length: 5 }, (_, i) => <line key={`h${i}`} x1={0} y1={i * 95} x2={800} y2={i * 95} />)}
      </g>
      <Structure x={84} y={150} cols={8} rows={10} />
      <Structure x={56} y={44} cols={7} rows={7} />
      <Structure x={604} y={132} cols={8} rows={11} />
      <Structure x={664} y={52} cols={6} rows={6} />
      <rect x={300} y={70} width={30} height={46} rx={3} fill="none" stroke="rgba(255,255,255,0.4)" strokeWidth="1.5" />
      <text x={315} y={62} textAnchor="middle" fontFamily="monospace" fontSize="10" fill="rgba(255,255,255,0.35)">vehicle</text>
      <rect x={388} y={300} width={16} height={28} rx={2} fill="#1a1a1a" stroke="#2bb0a4" strokeWidth="1.5" />
      <line x1={396} y1={300} x2={396} y2={284} stroke="#2bb0a4" strokeWidth="1.5" />
      <line x1={396} y1={300} x2={315} y2={93} stroke="rgba(255,255,255,0.28)" strokeWidth="1.25" strokeDasharray="4 4" />
      <text x={372} y={196} textAnchor="end" fontFamily="monospace" fontSize="11" fill="rgba(255,255,255,0.4)">9.49 m · box</text>
      <g stroke="#2bb0a4" strokeWidth="1.75">
        <line x1={388} y1={250} x2={172} y2={250} />
        <line x1={388} y1={244} x2={388} y2={256} />
        <line x1={172} y1={244} x2={172} y2={256} />
      </g>
      <text x={280} y={241} textAnchor="middle" fontFamily="monospace" fontSize="12" fill="#42c5b8">5.08 m · free space</text>
    </svg>
  );
}

const WRAP = "mx-auto w-full px-[clamp(20px,5vw,40px)]";
const SECTION = { marginTop: "clamp(56px, 8vh, 88px)" } as const;

export default function Landing() {
  return (
    <div className="landing min-h-screen bg-[#080808] font-sans text-white antialiased">
      {/* nav */}
      <header className="sticky top-0 z-30 border-b border-[var(--hairline)] bg-[#080808]/70 backdrop-blur-xl">
        <div className={`${WRAP} flex items-center justify-between py-3.5`} style={{ maxWidth: "var(--col)" }}>
          <div className="flex items-center gap-2 text-sm font-medium tracking-tight">
            <Logo className="size-5" /> spatial-probe
          </div>
          <nav className="flex items-center gap-1" style={{ fontSize: "var(--fs-mono-md)" }}>
            <Link href="/overview" className="ink-link rounded-md px-3 py-1.5 font-mono" style={{ color: "var(--muted)" }}>Pipeline</Link>
            <Link href="/evidence" className="ink-link rounded-md px-3 py-1.5 font-mono" style={{ color: "var(--muted)" }}>Evidence</Link>
            <Link href="/occquery" className="ml-1 inline-flex items-center gap-1.5 rounded-lg bg-white px-3.5 py-1.5 text-[13px] font-medium text-black transition-opacity hover:opacity-90">Open Explorer <ArrowRight className="size-3.5" /></Link>
          </nav>
        </div>
      </header>

      {/* hero */}
      <section className="relative overflow-hidden">
        <div aria-hidden className="pointer-events-none absolute inset-0 opacity-[0.18] [mask-image:radial-gradient(ellipse_at_top,black,transparent_70%)]"
          style={{ backgroundImage: "linear-gradient(rgba(255,255,255,.06) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.06) 1px,transparent 1px)", backgroundSize: "28px 28px" }} />
        <div className={`${WRAP} relative pt-24 pb-12 text-center sm:pt-32`} style={{ maxWidth: "var(--col)" }}>
          <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-[var(--hairline)] bg-white/[0.03] px-3 py-1 font-mono uppercase" style={{ fontSize: "var(--fs-mono-xs)", letterSpacing: "var(--ls-wide)", color: "var(--faint)" }}>
            <span className="size-1.5 rounded-full" style={{ backgroundColor: "var(--data)" }} /> state, not render
          </div>
          <h1 className="text-balance" style={{ fontSize: "var(--fs-display)", fontWeight: "var(--w-display)", lineHeight: "var(--lh-display)", letterSpacing: "var(--ls-tight)", color: "var(--text)" }}>
            Query, measure, and trust 3D scene state by what it physically means.
          </h1>
          <p className="mx-auto mt-6 text-balance" style={{ maxWidth: "var(--measure)", fontSize: "var(--fs-lead)", lineHeight: "var(--lh-prose)", color: "var(--muted)" }}>
            Free-space, clearance, corridor width, dynamics — geometry an object-box pipeline cannot
            express, with every number you can re-run. The spatial data engine for AV and robotics.
          </p>
          <div className="mt-9 flex items-center justify-center gap-5">
            <Link href="/occquery" className="inline-flex items-center gap-1.5 rounded-lg bg-white px-5 py-2.5 text-sm font-medium text-black transition-opacity hover:opacity-90">Open Explorer <ArrowRight className="size-4" /></Link>
            <Link href="/evidence" className="ink-link inline-flex items-center gap-1 text-sm" style={{ color: "var(--muted)" }}>See the evidence <ArrowRight className="size-3.5" /></Link>
          </div>
        </div>
        {/* product window — the signature output */}
        <div className={`${WRAP} relative pb-4`} style={{ maxWidth: "var(--col-wide)" }}>
          <Link href="/occquery" className="group block overflow-hidden rounded-2xl border bg-neutral-950 shadow-2xl shadow-black/60 transition-colors" style={{ borderColor: "var(--border-media)" }}>
            <div className="flex items-center gap-2 border-b border-[var(--hairline)] bg-white/[0.02] px-4 py-2.5 font-mono" style={{ fontSize: "var(--fs-mono-xs)", color: "var(--dim)" }}>
              <span className="size-2.5 rounded-full bg-white/15" /><span className="size-2.5 rounded-full bg-white/15" /><span className="size-2.5 rounded-full bg-white/15" />
              <span className="ml-3">explorer · scene-0061 · top-down</span>
              <span className="ml-auto inline-flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100">open live <ArrowRight className="size-3" /></span>
            </div>
            <OccupancyDiagram />
          </Link>
        </div>
      </section>

      {/* the gap */}
      <section className={WRAP} style={{ ...SECTION, maxWidth: "var(--col)" }}>
        <SectionLabel>the gap</SectionLabel>
        <div className="grid items-center gap-8 sm:grid-cols-2">
          <p style={{ fontSize: "var(--fs-body)", lineHeight: "var(--lh-prose)", color: "var(--muted)" }}>
            Object-box pipelines store boxes. Planners drive through free space. spatial-probe measures
            the free space — a wall <Metric tone="data">5.08&nbsp;m</Metric> beside the ego that box-only
            never sees, because its nearest object box reads <Metric>9.49&nbsp;m</Metric> away.
          </p>
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-xl border bg-white/[0.03] p-5 text-center" style={{ borderColor: "var(--border-media)" }}>
              <div className="text-3xl"><Metric tone="data">5.08</Metric><span className="ml-0.5 text-base" style={{ color: "var(--faint)" }}>m</span></div>
              <div className="mt-1.5 text-[11px]" style={{ color: "var(--muted)" }}>occupancy</div>
              <div className="mt-0.5 font-mono text-[10px]" style={{ color: "var(--dim)" }}>lateral_clearance</div>
            </div>
            <div className="rounded-xl border bg-white/[0.015] p-5 text-center" style={{ borderColor: "var(--hairline)" }}>
              <div className="text-3xl"><Metric>9.49</Metric><span className="ml-0.5 text-base" style={{ color: "var(--dim)" }}>m</span></div>
              <div className="mt-1.5 text-[11px]" style={{ color: "var(--faint)" }}>box-only</div>
              <div className="mt-0.5 font-mono text-[10px]" style={{ color: "var(--dim)" }}>box_distance</div>
            </div>
            <div className="col-span-2 text-center font-mono text-[10px]" style={{ color: "var(--dim)" }}>scene-0061 · frame 10 · re-runnable via probe_scene()</div>
          </div>
        </div>
      </section>

      {/* capability index — the one showcase section */}
      <section className={WRAP} style={{ ...SECTION, maxWidth: "var(--col)" }}>
        <SectionLabel>the pipeline</SectionLabel>
        <div className="mb-1.5 flex items-baseline justify-between gap-4">
          <h2 style={{ fontSize: "var(--fs-title)", fontWeight: "var(--w-title)", letterSpacing: "var(--ls-tight)", color: "var(--text)" }}>The pipeline, as capabilities</h2>
          <Link href="/overview" className="ink-link font-mono" style={{ fontSize: "var(--fs-mono-sm)", color: "var(--faint)" }}>{SHIPPED}/{TOTAL} shipped · {IN_PROGRESS} live →</Link>
        </div>
        <p className="mb-6" style={{ fontSize: "var(--fs-body-sm)", color: "var(--faint)" }}>Six axes of the data pipeline. Two are live. The rest are planned, and the rail says so.</p>
        <div style={{ borderTop: "1px solid var(--hairline)" }}>
          {ALL_MODULES.map((m, i) => <CapabilityRow key={m.id} module={m} index={i} />)}
        </div>
      </section>

      {/* auditable measurement */}
      <section className={WRAP} style={{ ...SECTION, maxWidth: "var(--col)" }}>
        <SectionLabel>auditable</SectionLabel>
        <h2 style={{ fontSize: "var(--fs-title)", fontWeight: "var(--w-title)", letterSpacing: "var(--ls-tight)", color: "var(--text)" }}>Measurement you can audit</h2>
        <p className="mb-6 mt-1.5" style={{ fontSize: "var(--fs-body-sm)", color: "var(--faint)" }}>A vendor whose own audit kills its own results is the one to trust.</p>
        <div className="grid gap-3 sm:grid-cols-3">
          {[
            { title: "Reproducible", body: "Every number carries its call. probe_scene(scene, frame, predicate) regenerates it bit-for-bit." },
            { title: "Pre-registered", body: "Hypothesis, analysis path, and kill criterion are committed before the data. HARKing shows in the diff." },
            { title: "Independently graded", body: "H1 holds oracle-free. H3 is inconclusive — audited twice and retracted in public.", href: "/evidence" },
          ].map((c) => {
            const inner = (
              <GlassPanel className="flex h-full flex-col p-5">
                <div style={{ fontSize: "var(--fs-body-sm)", fontWeight: "var(--w-title)", color: "var(--text)" }}>{c.title}</div>
                <p className="mt-1.5 flex-1" style={{ fontSize: "var(--fs-prose-sm)", lineHeight: "var(--lh-prose)", color: "var(--muted)" }}>{c.body}</p>
                {c.href && <div className="ink-link mt-3 inline-flex w-fit items-center gap-1 font-mono" style={{ fontSize: "var(--fs-mono-xs)", color: "var(--dim)" }}>see Evidence <ArrowRight className="size-3" /></div>}
              </GlassPanel>
            );
            return c.href ? <Link key={c.title} href={c.href}>{inner}</Link> : <div key={c.title}>{inner}</div>;
          })}
        </div>
        <div className="mt-5">
          <div className="mb-2 font-mono uppercase" style={{ fontSize: "var(--fs-mono-xs)", letterSpacing: "var(--ls-label)", color: "var(--dim)" }}>verdicts on file</div>
          <div className="flex flex-col gap-2 sm:flex-row sm:gap-6">
            <span className="inline-flex items-center gap-2.5" style={{ fontSize: "var(--fs-body-sm)", color: "var(--muted)" }}><StatusChip verdict="HOLDS" /> H1 expressivity</span>
            <span className="inline-flex items-center gap-2.5" style={{ fontSize: "var(--fs-body-sm)", color: "var(--muted)" }}><StatusChip verdict="INCONCLUSIVE" /> H3 denotation</span>
          </div>
        </div>
      </section>

      {/* access band */}
      <section className={WRAP} style={{ ...SECTION, maxWidth: "var(--col)" }}>
        <GlassPanel className="flex flex-col items-center gap-5 p-8 text-center sm:flex-row sm:justify-between sm:text-left">
          <div>
            <div style={{ fontSize: "var(--fs-title)", fontWeight: "var(--w-title)", color: "var(--text)" }}>One engine, three front doors.</div>
            <div className="mt-1" style={{ fontSize: "var(--fs-body-sm)", color: "var(--muted)" }}>Explorer for the eye. A typed predicate for the query. MCP for the agent.</div>
          </div>
          <Link href="/occquery" className="inline-flex shrink-0 items-center gap-1.5 rounded-lg bg-white px-5 py-2.5 text-sm font-medium text-black transition-opacity hover:opacity-90">Open Explorer <ArrowRight className="size-4" /></Link>
        </GlassPanel>
      </section>

      {/* footer */}
      <footer className={`${WRAP} pb-12 pt-6`} style={{ ...SECTION, maxWidth: "var(--col)" }}>
        <div className="flex flex-col gap-3 pt-6 sm:flex-row sm:items-center sm:justify-between" style={{ borderTop: "1px solid var(--hairline)", fontSize: "var(--fs-prose-sm)", color: "var(--dim)" }}>
          <p style={{ maxWidth: "var(--measure)", lineHeight: "var(--lh-prose)" }}>Runs on Occ3D-nuScenes and a deterministic predicate core. No validated F1 is claimed where no independent oracle exists. Negatives are headlines here, not footnotes.</p>
          <div className="flex shrink-0 items-center gap-2" title={THESIS}><Logo className="size-4" /> state, not render</div>
        </div>
      </footer>
    </div>
  );
}

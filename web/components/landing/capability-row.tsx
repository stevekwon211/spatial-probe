import Link from "next/link";
import { ArrowUpRight, Plus } from "lucide-react";
import type { ModuleItem } from "@/lib/pipeline";
import { cn } from "@/lib/utils";

const lifecycle = (s: string) => (s === "in-progress" ? "live" : s === "done" ? "shipped" : "planned");

// One row of the capability index — our single 3-layer/showcase section (kde SelectedWork pattern).
// Earns prominence through EXTRA LAYERS (index numeral · axis meta · status · live/planned dimming),
// not a bigger headline. Achromatic only: no teal anywhere (a row is chrome, not a measurement).
export function CapabilityRow({ module: m, index }: { module: ModuleItem; index: number }) {
  const live = m.status !== "planned";
  const inner = (
    <div
      className="grid items-baseline border-b py-4"
      style={{ gridTemplateColumns: "2.5ch minmax(0,1fr) auto", columnGap: "clamp(12px,2vw,20px)", borderColor: "var(--hairline)", opacity: live ? 1 : 0.6 }}
    >
      <span className="metric" style={{ fontSize: "var(--fs-body-sm)", color: "var(--faint)" }}>{String(index + 1).padStart(2, "0")}</span>
      <div className="min-w-0">
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <span className={cn("font-medium", m.href && "ink-link")} style={{ fontSize: "var(--fs-subtitle)", color: "var(--text)" }}>{m.title}</span>
          <span className="font-mono uppercase" style={{ fontSize: "var(--fs-mono-sm)", letterSpacing: "var(--ls-label)", color: "var(--dim)" }}>{m.axis}</span>
        </div>
        <p className="mt-1.5" style={{ fontSize: "var(--fs-prose-sm)", lineHeight: "var(--lh-prose)", color: "var(--muted)", maxWidth: "var(--measure)" }}>{m.oneLine}</p>
      </div>
      <div className="flex items-center gap-3 whitespace-nowrap">
        <span className="font-mono uppercase" style={{ fontSize: "var(--fs-mono-xs)", letterSpacing: "var(--ls-label)", color: live ? "var(--faint)" : "var(--dim)" }}>{lifecycle(m.status)}</span>
        {m.href ? <ArrowUpRight className="size-3.5" style={{ color: "var(--faint)" }} /> : <Plus className="size-3" style={{ color: "var(--dim)" }} />}
      </div>
    </div>
  );
  return m.href ? <Link href={m.href} className="group block">{inner}</Link> : inner;
}

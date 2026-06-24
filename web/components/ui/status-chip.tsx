import { cn } from "@/lib/utils";
import type { Verdict } from "@/lib/findings";

// Verdict chips are ACHROMATIC by the design law (color belongs to data only). The one exception is
// RETRACTED -- the single alarming state -- which reuses the live-data red used elsewhere for a TRUE
// blocked path. Everything else is white/X over the canvas.
const STYLES: Record<Verdict, string> = {
  HOLDS: "border-white/20 text-foreground",
  INCONCLUSIVE: "border-white/10 text-muted-foreground",
  RETRACTED: "border-red-400/30 text-red-400",
  "NOT-STARTED": "border-white/[0.07] text-muted-foreground/55",
};

export function StatusChip({ verdict, className }: { verdict: Verdict; className?: string }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider tabular-nums",
        STYLES[verdict],
        className,
      )}
    >
      {verdict}
    </span>
  );
}

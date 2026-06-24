import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

// Every number on the landing renders through this: mono + tabular-nums (the "machine voice").
// tone="data" = the ONE live measured value gets teal (--data-text); everything else inherits the
// surrounding chrome luminance. Color belongs to data only.
export function Metric({ children, tone = "muted", className }: { children: ReactNode; tone?: "data" | "muted"; className?: string }) {
  return (
    <span className={cn("metric", className)} style={tone === "data" ? { color: "var(--data-text)" } : undefined}>
      {children}
    </span>
  );
}

import { cn } from "@/lib/utils";
import type { ComponentProps } from "react";

/**
 * Liquid-glass surface — the query layer floating over the 3D state.
 * Achromatic by design: no accent color, hairline border, deep blur.
 */
export function GlassPanel({ className, ...props }: ComponentProps<"div">) {
  return (
    <div
      className={cn(
        "rounded-2xl border border-white/10 bg-neutral-900/55 backdrop-blur-2xl",
        "shadow-xl shadow-black/40",
        className,
      )}
      {...props}
    />
  );
}

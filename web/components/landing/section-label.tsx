"use client";

import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

// The mono machine-eyebrow that opens every section (the identical header tier — kde invariant 2).
// Pixel-reveal: a one-shot left-to-right mask wipe when it scrolls into view (achromatic; it reveals
// the existing faint-white label, no color). Labels already on-screen at mount stay static (no flash).
// Progressive-enhanced: with JS off, the static label renders.
export function SectionLabel({ children, className }: { children: React.ReactNode; className?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [armed, setArmed] = useState(false);
  const [shown, setShown] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const top = el.getBoundingClientRect().top;
    if (top < window.innerHeight && top > 0) return; // in view at mount -> static, no flash
    setArmed(true);
    const io = new IntersectionObserver(
      ([e]) => { if (e.isIntersecting) { setShown(true); io.unobserve(el); } },
      { threshold: 0.6 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  return (
    <div
      ref={ref}
      className={cn("sec-label mb-6 font-mono uppercase", shown && "is-in", className)}
      style={{ fontSize: "var(--fs-mono-sm)", letterSpacing: "var(--ls-wide)", color: "var(--faint)" }}
      {...(armed ? { "data-reveal": "" } : {})}
    >
      {children}
    </div>
  );
}

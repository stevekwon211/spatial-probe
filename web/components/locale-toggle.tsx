"use client";

import { useLocale } from "next-intl";
import { useRouter } from "next/navigation";
import { useTransition } from "react";

import { setLocale } from "@/i18n/actions";
import type { Locale } from "@/i18n/config";
import { cn } from "@/lib/utils";

// Mirrors mode-toggle.tsx: a compact switch that flips a single piece of UI state and re-renders.
// Where mode-toggle calls setTheme(), this calls the setLocale() server action to persist the
// NEXT_LOCALE cookie, then router.refresh() so the server re-renders with the new messages — the
// next-intl cookie / no-i18n-routing pattern (the URL never changes).
const OPTIONS: { value: Locale; label: string }[] = [
  { value: "en", label: "EN" },
  { value: "ko", label: "KO" },
];

export function LocaleToggle({ className }: { className?: string }) {
  const active = useLocale();
  const router = useRouter();
  const [pending, startTransition] = useTransition();

  const choose = (locale: Locale) => {
    if (locale === active || pending) return;
    startTransition(async () => {
      await setLocale(locale);
      router.refresh();
    });
  };

  return (
    <div
      className={cn("inline-flex items-center rounded-lg border border-white/10 p-0.5 font-mono", className)}
      role="group"
      aria-label="Language"
    >
      {OPTIONS.map((o) => (
        <button
          key={o.value}
          type="button"
          onClick={() => choose(o.value)}
          aria-pressed={active === o.value}
          disabled={pending}
          className={cn(
            "rounded-md px-2 py-0.5 text-[10px] uppercase tracking-wide transition-colors",
            active === o.value ? "bg-white/15 text-white" : "text-white/45 hover:text-white/80",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

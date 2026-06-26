// Shared locale constants for the cookie-driven (no-i18n-routing) next-intl setup.
// English is canonical; Korean values in messages/ko.json start as English placeholders
// and are translated separately. Switching language sets NEXT_LOCALE and refreshes — the URL
// never changes (no /ko /en segments).

export const locales = ["en", "ko"] as const;
export type Locale = (typeof locales)[number];

export const defaultLocale: Locale = "en";

// The cookie next-intl reads on every request to pick the active locale.
export const LOCALE_COOKIE = "NEXT_LOCALE";

export function isLocale(value: string | undefined): value is Locale {
  return value === "en" || value === "ko";
}
